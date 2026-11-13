"""MuJoCo scene for the GARMI lift-the-bottle task: the same SceneSpec as the Isaac Sim scene (scene.py).

Robot: garmi_description's own MuJoCo model (garmi_description/mujoco/garmi.xml), whose joints, tool-centre
sites and camera frames match the URDF exactly (checked against kinematics.py: 0.00 mm). Changes for the task:
  * the base is fixed with base_link at the world origin (SceneSpec positions are in base_link);
  * the arms are gravity compensated (the real FR3 is torque controlled with gravity compensation);
  * each gripper is a position servo on the finger opening with 2000 N/m and 70 N per finger, like the Isaac
    recordings (Franka Hand: 70 N continuous grasp force) instead of menagerie's weak default;
  * the head fisheye (left Olive lens) uses the lens and head pose calibrated on the v16 recordings
    (head_camera.py): five 90 deg cube cameras at the lens, remapped at HEAD_RENDER px and downsampled to 256 like
    v16; the wrist cameras use the D435i intrinsics from the robot (camera_models.wrist_view), warped to 640x480 and
    downsampled to 256x192, letterboxed to 256x256.
Room, table, bottle and lights follow scene.py; Isaac's lights map to MuJoCo ones (sun -> directional light,
ceiling panels -> downward spot lights with inverse-square falloff, dome -> ambient), with gains calibrated
against the Isaac renders of the same seeds (scripts/mj_expert_check.py --compare-images).

The model is compiled per scene (~0.3 s), so nothing of one episode carries over to the next.
"""

from __future__ import annotations

import functools
import json
import os

import cv2
import mujoco
import numpy as np

from garmi_bench import hdri_mj
from garmi_bench import head_camera
from garmi_bench import v16_lab
from garmi_bench import kinematics as kin
from garmi_bench import camera_models
from garmi_bench.camera_models import CUBE_FACES
from garmi_bench.camera_models import FthetaFromCube
from garmi_bench.camera_models import letterbox
from garmi_bench.camera_models import tone_inverse
from garmi_bench.camera_models import tone_lut
from garmi_bench.spec import LIFT_SUCCESS_M
from garmi_bench.spec import START_FINGER
from garmi_bench.spec import START_JOINTS
from garmi_bench.spec import SceneSpec

GARMI_XML = os.path.expanduser("~/sim/garmi_description/garmi_description/mujoco/garmi.xml")
BOTTLE_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "bottles_mj")

CONTROL_HZ = 15.0
SUBSTEPS = 34  # physics steps per 15 Hz control tick
TIMESTEP = 1.0 / (CONTROL_HZ * SUBSTEPS)  # 1.96 ms
SETTLE_S = 0.25  # as scene.py's reset (30 PhysX steps of 1/120 s) before the bottle's rest height is taken
GROUND_Z = -0.05  # as scene.py
HEAD_RENDER = 512  # fisheye remap resolution before downsampling to 256 (cube faces at the same size)
FINGER_KP, FINGER_KV, FINGER_MAX_FORCE = 2000.0, 100.0, 70.0  # per finger, as record_demos.py / robot.drive_gains
HEAD_NEAR, WRIST_NEAR = 0.08, 0.01  # as scene.py's clipping ranges (head: 8 cm drops the camera's own housing)
SCENE_EXTENT = 2.5  # fixed model extent (m): the 20 m HDRI dome must not coarsen shadow maps or clip planes
GRIP_FRICTION = (1.0, 0.005, 0.0001)  # scene.py's grip material: static 1.2 / dynamic 1.0

# Isaac light intensities -> MuJoCo light colors, and the tone curve applied to MuJoCo's output. Fitted by
# scripts/mj_calibrate_lights.py to Isaac's renders of the same seeds (assets/mj_lighting.json); these
# defaults are only the starting point of that fit.
LIGHTING_FILE = os.path.join(os.path.dirname(__file__), "..", "assets", "mj_lighting.json")
DEFAULT_LIGHTING = {
    "ambient_gain": 1.0e-3,  # dome intensity -> headlight ambient
    "sun_gain": 3.5e-4,  # distant-light intensity -> directional diffuse
    "ceiling_gain": 4.0e-4,  # rect-light intensity x area (m^2) -> spot diffuse at 1 m (inverse-square falloff)
    "tone_k": 1.5,  # camera_models.tone_lut
    # HDRI rooms (hdri_mj.py): x the room's mean brightness x dome intensity / 1000.
    "hdri_ambient_gain": 0.5,  # headlight ambient (the room's diffuse light)
    "hdri_key_gain": 1.0,  # directional key light from the brightest direction (x its share of the light)
    "hdri_sky_gain": 0.5,  # the dome texture's linear values -> 8 bit
    # The v16 lab twin (v16_lab.py): fitted to real v16 head images at the recorded joint angles
    # (scripts/mj_calibrate_lab.py).
    "lab_ambient_gain": 0.5,
    "lab_key_gain": 0.5,
    "lab_sky_gain": 1.0,
}


def lighting() -> dict:
    if os.path.exists(LIGHTING_FILE):
        with open(LIGHTING_FILE) as f:
            return {**DEFAULT_LIGHTING, **{k: v for k, v in json.load(f).items() if k in DEFAULT_LIGHTING}}
    return dict(DEFAULT_LIGHTING)


def isaac_background(spec: SceneSpec) -> float:
    """Gray level (0-255) of Isaac's background (the dome light seen directly, e.g. outside the fisheye circle):
    0.2248 x dome intensity + 97.5, fitted on 40 recorded episodes (residual std 2.9)."""
    return float(np.clip(0.2248 * spec.light_intensity + 97.5, 0, 255))


def kelvin_rgb(kelvin: float) -> np.ndarray:
    """Approximate blackbody color, normalized to 1 at 6500 K."""

    def raw(t):
        t = t / 100.0
        r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
        g = 99.4708025861 * np.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
        b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307)
        return np.clip(np.array([r, g, b]), 0, 255) / 255.0

    return raw(kelvin) / raw(6500.0)


def light_color(spec: SceneSpec, name: str) -> np.ndarray:
    """A light's color before its gain: Isaac intensity (x panel area for ceiling panels) x blackbody color."""
    if name == "sun":
        _, _, intensity, temp = spec.sun
        return intensity * kelvin_rgb(temp)
    _, _, _, edge, intensity, temp = spec.ceiling[int(name.split("_")[1])]
    return intensity * edge * edge * kelvin_rgb(temp)


def _robot_spec() -> mujoco.MjSpec:
    s = mujoco.MjSpec.from_file(GARMI_XML)
    s.delete(s.joint("base_free"))
    for key in list(s.keys):  # the home keyframe no longer fits the joint count
        s.delete(key)
    s.body("base_link").pos = [0.0, 0.0, 0.0]
    return s


@functools.lru_cache(maxsize=None)
def _robot_info() -> dict:
    """Arm body names (for gravity compensation) and the left head lens's frame in its body."""
    m = _robot_spec().compile()
    arm_bodies = []
    for side in ("left", "right"):
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_fr3_link1")
        for b in range(m.nbody):
            p = b
            while p not in (0, root):
                p = m.body_parentid[p]
            if p == root:
                arm_bodies.append(m.body(b).name)
    cam = m.cam("olive_camera_left")
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, cam.quat)
    return {"arm_bodies": arm_bodies, "head_body": m.body(cam.bodyid[0]).name, "head_pos": cam.pos.copy(),
            "head_rot": rot.reshape(3, 3)}


def _quat_from_axes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rot = np.stack([x, y, np.cross(x, y)], axis=1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, rot.flatten())
    return q


def _box(body, name, center, half, material, *, collide=True, friction=None):
    # No geom rgba: an explicitly set geom rgba overrides the material's color in MuJoCo.
    g = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(center), size=list(half), material=material)
    g.contype, g.conaffinity = (1, 1) if collide else (0, 0)
    if friction is not None:
        g.friction = list(friction)
    return g


def _add_light(body, *, name, directional, pos, direction, diffuse, cutoff=90.0, exponent=1.0, attenuation=(1, 0, 0),
               castshadow=True):
    light = body.add_light(name=name, pos=list(pos), dir=list(direction), diffuse=list(diffuse), specular=[0.1, 0.1, 0.1],
                           castshadow=castshadow, cutoff=cutoff, exponent=exponent, attenuation=list(attenuation))
    if hasattr(light, "type"):  # MuJoCo >= 3.3: light type enum
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL if directional else mujoco.mjtLightType.mjLIGHT_SPOT
    else:
        light.directional = directional
    return light


def build_spec(spec: SceneSpec, face_size: int = HEAD_RENDER, light: dict | None = None) -> mujoco.MjSpec:
    light = light or lighting()
    s = _robot_spec()
    info = _robot_info()
    s.option.timestep = TIMESTEP
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    s.option.impratio = 10.0
    for name in info["arm_bodies"]:
        s.body(name).gravcomp = 1.0

    # Grippers: tendon = mean finger opening; joint force = 0.5 x tendon force per finger.
    for side in ("left", "right"):
        act = s.actuator(f"{side}_gripper")
        act.gainprm = [2 * FINGER_KP] + [0.0] * 9
        act.biasprm = [0.0, -2 * FINGER_KP, -2 * FINGER_KV] + [0.0] * 7
        act.ctrlrange = [0.0, 0.04]
        act.forcerange = [-2 * FINGER_MAX_FORCE, 2 * FINGER_MAX_FORCE]
        (w, h), hfov = camera_models.wrist_render()
        vfov = 2 * np.degrees(np.arctan(np.tan(np.radians(hfov) / 2) * h / w))
        s.camera(f"camera_d435i_{side}_color").fovy = float(vfov)

    # Head fisheye: cube cameras at the left lens, axes from the lens's optical frame.
    head = s.body(info["head_body"])
    r = info["head_rot"]  # MuJoCo camera axes in the body: x right, y up, looking along -z
    optical = np.stack([r[:, 0], -r[:, 1], -r[:, 2]], axis=1)  # optical x right, y down, z forward
    for name, (_, right, up) in CUBE_FACES.items():
        head.add_camera(name=f"head_{name}", pos=list(info["head_pos"]),
                        quat=list(_quat_from_axes(optical @ np.array(right, float), optical @ np.array(up, float))),
                        fovy=90.0)

    # Rendering.
    s.visual.global_.offwidth = max(face_size, camera_models.wrist_render()[0][0])
    s.visual.global_.offheight = max(face_size, camera_models.wrist_render()[0][1])
    s.visual.quality.shadowsize = 4096
    s.visual.quality.offsamples = 8
    s.visual.headlight.diffuse = [0.0, 0.0, 0.0]
    s.visual.headlight.specular = [0.0, 0.0, 0.0]
    s.visual.headlight.ambient = [light["ambient_gain"] * spec.light_intensity] * 3
    s.visual.map.zfar = 50.0
    sky = tone_inverse(isaac_background(spec) / 255.0, light["tone_k"])  # Isaac's gray after the tone curve
    s.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                  rgb1=[sky] * 3, rgb2=[sky] * 3, width=32, height=32)

    def material(name, rgba):
        s.add_material(name=name, rgba=list(rgba) + [1.0] if len(rgba) == 3 else list(rgba), specular=0.15,
                       shininess=0.2, reflectance=0.0)
        return name

    world = s.worldbody
    s.stat.extent = SCENE_EXTENT
    s.stat.center = [1.0, 0.0, 0.8]
    # Room: floor slab (top at GROUND_Z), back wall, side wall -- as scene.py (walls only without an HDRI room).
    lab = spec.hdri == v16_lab.ROOM
    floor = _box(world, "floor", (1.0, 0.0, GROUND_Z - 0.05), (6.0, 6.0, 0.05), material("floor", spec.floor_color))
    if lab:  # the lab panorama shows the real floor and the space under the table: floor and legs collide only (group 3)
        floor.group = 3
    if spec.walls:
        wall = material("wall", spec.wall_color)
        _box(world, "back_wall", (spec.back_wall_x + 0.05, 0.0, GROUND_Z + 1.3), (0.05, 4.0, 1.3), wall)
        _box(world, "side_wall", (1.0, spec.side_wall_y + np.sign(spec.side_wall_y) * 0.05, GROUND_Z + 1.3), (3.0, 0.05, 1.3), wall)
    # Table: top slab on four square legs.
    if lab:  # wood grain, colored as the real table in the wrist images (v16_lab.py)
        tex = cv2.imread(v16_lab.wood_texture(), cv2.IMREAD_GRAYSCALE)
        s.add_texture(name="wood", type=mujoco.mjtTexture.mjTEXTURE_2D, file=os.path.abspath(v16_lab.wood_texture()))
        rgb = np.clip(np.array(spec.table_color) * v16_lab.WRIST_TABLE_GAIN / (tex.mean() / 255.0), 0, 1)
        mat = s.add_material(name="table", rgba=list(rgb) + [1.0], specular=0.15, shininess=0.2, reflectance=0.0,
                             texuniform=True, texrepeat=[1 / v16_lab.WOOD_TEXTURE_M] * 2)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "wood"
        table = "table"
    else:
        table = material("table", spec.table_color)
    _box(world, "table_top", spec.table_top_center, (spec.table_depth / 2, spec.table_width / 2, spec.table_thickness / 2),
         table, friction=GRIP_FRICTION)
    leg_h = spec.table_height - spec.table_thickness - GROUND_Z
    for i, (x, y) in enumerate(spec.table_leg_xy):
        leg = _box(world, f"table_leg_{i}", (x, y, GROUND_Z + leg_h / 2),
                   (spec.table_leg_width / 2, spec.table_leg_width / 2, leg_h / 2), table)
        if lab:
            leg.group = 3

    # Lights.
    if lab:  # the real v16 lab (v16_lab.py): its panorama as the dome, light from its windows
        s.add_mesh(name="hdri_dome", file=os.path.abspath(hdri_mj.dome_obj()))
        s.add_texture(name="hdri", type=mujoco.mjtTexture.mjTEXTURE_2D,
                      file=os.path.abspath(v16_lab.dome_texture(light["tone_k"], light["lab_sky_gain"])))
        dome_mat = s.add_material(name="hdri", rgba=[1, 1, 1, 1], emission=1.0, specular=0.0, shininess=0.0, reflectance=0.0)
        dome_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "hdri"
        g = world.add_geom(name="hdri_dome", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="hdri_dome", material="hdri",
                           pos=[0.0, 0.0, 0.0], density=0.0)
        g.contype = g.conaffinity = 0
        to_key = v16_lab.key_light().copy()
        to_key[2] = max(to_key[2], 0.15)
        to_key /= np.linalg.norm(to_key)
        s.visual.headlight.ambient = [light["lab_ambient_gain"]] * 3
        _add_light(world, name="key", directional=True, pos=1.0 + 5.0 * to_key, direction=-to_key,
                   diffuse=[light["lab_key_gain"]] * 3)
    elif spec.hdri:  # the room photo around the scene and light estimated from it (hdri_mj.py); no sun / ceiling panels
        scale = spec.hdri_intensity / 1000.0
        rot = np.radians(spec.hdri_rotation_deg + hdri_mj.AZIMUTH_OFFSET_DEG)
        key_dir, share, mean = hdri_mj.key_light(spec.hdri)
        s.add_mesh(name="hdri_dome", file=os.path.abspath(hdri_mj.dome_obj()))
        s.add_texture(name="hdri", type=mujoco.mjtTexture.mjTEXTURE_2D,
                      file=os.path.abspath(hdri_mj.texture_png(spec.hdri, float(f"{light['hdri_sky_gain'] * scale:.3g}"))))
        dome_mat = s.add_material(name="hdri", rgba=[1, 1, 1, 1], emission=1.0, specular=0.0, shininess=0.0, reflectance=0.0)
        dome_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "hdri"
        g = world.add_geom(name="hdri_dome", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="hdri_dome", material="hdri",
                           pos=[0.0, 0.0, 0.0], quat=[np.cos(rot / 2), 0.0, 0.0, np.sin(rot / 2)], density=0.0)
        g.contype = g.conaffinity = 0
        c, sn = np.cos(rot), np.sin(rot)
        to_key = np.array([c * key_dir[0] - sn * key_dir[1], sn * key_dir[0] + c * key_dir[1], key_dir[2]])
        to_key[2] = max(to_key[2], 0.15)  # light from below the horizon (a bright floor) would not shine onto the table
        to_key /= np.linalg.norm(to_key)
        s.visual.headlight.ambient = [light["hdri_ambient_gain"] * (1 - share) * scale] * 3
        _add_light(world, name="key", directional=True, pos=1.0 + 5.0 * to_key, direction=-to_key,
                   diffuse=[light["hdri_key_gain"] * share * scale] * 3)
    elif spec.sun is not None:
        az, el, intensity, temp = spec.sun
        to_sun = np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)), np.cos(np.radians(el)) * np.sin(np.radians(az)),
                           np.sin(np.radians(el))])
        _add_light(world, name="sun", directional=True, pos=1.0 + 5.0 * to_sun, direction=-to_sun,
                   diffuse=light["sun_gain"] * light_color(spec, "sun"))
    for i in range(0 if spec.hdri else len(spec.ceiling)):
        x, y, z = spec.ceiling[i][:3]
        # No shadow maps for the panels: an 85 deg spot's shadow map is too coarse (blocky dark patches on the
        # table), and Isaac's area lights only cast soft shadows; the sun keeps its sharp shadow.
        _add_light(world, name=f"ceiling_{i}", directional=False, pos=(x, y, z), direction=(0, 0, -1), cutoff=85.0,
                   exponent=1.0, attenuation=(0, 0, 1), diffuse=light["ceiling_gain"] * light_color(spec, f"ceiling_{i}"),
                   castshadow=False)

    if v16_lab.is_pet(spec.bottle_asset):
        _add_pet_bottle(s, world, spec)
        return s
    # Bottle: visual meshes (textured) and convex collision parts, mass and inertia from the collision hull.
    name = os.path.splitext(spec.bottle_asset)[0]
    folder = os.path.join(BOTTLE_DIR, name)
    parts = json.load(open(os.path.join(folder, "parts.json")))
    k = spec.bottle_scale
    yaw = spec.bottle_yaw
    body = world.add_body(name="bottle", pos=[spec.bottle_x, spec.bottle_y, spec.table_height + 0.003],
                          quat=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    body.add_freejoint(name="bottle_free")
    body.explicitinertial = True
    body.mass = spec.bottle_mass
    body.ipos = list(np.array(parts["hull_com"]) * k)
    inertia = np.array(parts["hull_inertia_unit_mass"]) * spec.bottle_mass * k * k
    body.fullinertia = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    for i, vis in enumerate(parts["visual"]):
        s.add_mesh(name=f"bottle_vis_{i}", file=os.path.join(folder, vis["mesh"]), scale=[k, k, k])
        mat = s.add_material(name=f"bottle_mat_{i}", rgba=vis["rgba"], specular=0.4, shininess=0.5)
        if vis["texture"]:
            s.add_texture(name=f"bottle_tex_{i}", type=mujoco.mjtTexture.mjTEXTURE_2D, file=os.path.join(folder, vis["texture"]))
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"bottle_tex_{i}"
        g = body.add_geom(name=f"bottle_vis_{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"bottle_vis_{i}",
                          material=f"bottle_mat_{i}", group=2, density=0.0)
        g.contype = g.conaffinity = 0
    for i, col in enumerate(parts["collision"]):
        s.add_mesh(name=f"bottle_col_{i}", file=os.path.join(folder, col), scale=[k, k, k])
        g = body.add_geom(name=f"bottle_col_{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"bottle_col_{i}", group=3,
                          density=0.0, condim=4, friction=list(GRIP_FRICTION), rgba=[0.5, 0.5, 0.5, 0.0])
        g.contype = g.conaffinity = 1
    return s


def _add_pet_bottle(s: mujoco.MjSpec, world, spec: SceneSpec) -> None:
    """A clear PET water bottle (v16_lab.py): transparent lathe-turned body, opaque colored cap; solid-cylinder mass."""
    h, r = spec.bottle_height, v16_lab.PET_RADIUS
    yaw = spec.bottle_yaw
    body = world.add_body(name="bottle", pos=[spec.bottle_x, spec.bottle_y, spec.table_height + 0.003],
                          quat=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    body.add_freejoint(name="bottle_free")
    body.explicitinertial = True
    m = spec.bottle_mass
    body.mass = m
    body.ipos = [0.0, 0.0, 0.4 * h]
    body.fullinertia = [m * (3 * r * r + h * h) / 12, m * (3 * r * r + h * h) / 12, m * r * r / 2, 0.0, 0.0, 0.0]
    s.add_mesh(name="pet_body", file=os.path.abspath(v16_lab.lathe_obj(h)))
    # No refraction in MuJoCo: a whitish shell with strong highlights and the water inside (a smaller copy of the
    # body, up to the shoulder), as clear bottles look in the real wrist images.
    s.add_material(name="pet_plastic", rgba=[0.90, 0.94, 0.97, 0.45], specular=1.0, shininess=1.0, reflectance=0.0)
    s.add_mesh(name="pet_water", file=os.path.abspath(v16_lab.lathe_obj(h)), scale=[0.9, 0.9, 0.7])
    s.add_material(name="pet_water", rgba=[0.72, 0.84, 0.92, 0.35], specular=0.6, shininess=0.8, reflectance=0.0)
    g = body.add_geom(name="bottle_vis_water", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="pet_water", material="pet_water",
                      pos=[0, 0, 0.004], group=2, density=0.0)
    g.contype = g.conaffinity = 0
    s.add_material(name="pet_cap", rgba=list(v16_lab.CAP_COLORS[v16_lab.pet_cap(spec.bottle_asset)]) + [1.0],
                   specular=0.3, shininess=0.4)
    g = body.add_geom(name="bottle_vis_body", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="pet_body", material="pet_plastic",
                      group=2, density=0.0)
    g.contype = g.conaffinity = 0
    cap_z = h - v16_lab.CAP_HEIGHT / 2
    cap_r = spec.cap_radius or v16_lab.CAP_RADIUS
    g = body.add_geom(name="bottle_vis_cap", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[cap_r, v16_lab.CAP_HEIGHT / 2, 0],
                      pos=[0, 0, cap_z], material="pet_cap", group=2, density=0.0)
    g.contype = g.conaffinity = 0
    cols = [("bottle_col_cap", dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[cap_r, v16_lab.CAP_HEIGHT / 2, 0], pos=[0, 0, cap_z]))]
    for i, part in enumerate(v16_lab.collision_parts(h)):  # convex parts: body, shoulder, neck
        if "mesh" in part:
            s.add_mesh(name=f"pet_col_{i}", file=os.path.abspath(part["mesh"]))
            cols.append((f"bottle_col_{i}", dict(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"pet_col_{i}")))
        else:
            r_c, z0, z1 = part["cylinder"]
            cols.append((f"bottle_col_{i}", dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r_c, (z1 - z0) / 2, 0],
                                                 pos=[0, 0, (z0 + z1) / 2])))
    for name, geom in cols:
        g = body.add_geom(name=name, group=3, density=0.0, condim=4, friction=list(GRIP_FRICTION), rgba=[0.5, 0.5, 0.5, 0.0], **geom)
        g.contype = g.conaffinity = 1


class MjLiftBottleScene:
    """One scene (one SceneSpec). step() takes v16's 16 commands: 14 arm joint targets, 2 finger openings (m)."""

    def __init__(self, spec: SceneSpec, face_size: int = HEAD_RENDER, light: dict | None = None):
        self.spec = spec
        self.light = light or lighting()
        self.model = build_spec(spec, face_size, self.light).compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.arm_qpos = {s: [m.joint(n).qposadr[0] for n in kin.arm_names(s)] for s in ("left", "right")}
        self.finger_qpos = {s: [m.joint(f"{s}_fr3_finger_joint{f}").qposadr[0] for f in (1, 2)] for s in ("left", "right")}
        self.arm_act = {s: [m.actuator(n).id for n in kin.arm_names(s)] for s in ("left", "right")}
        self.grip_act = {s: m.actuator(f"{s}_gripper").id for s in ("left", "right")}
        self.bottle_body = m.body("bottle").id
        rays, valid = head_camera.pixel_rays(face_size)
        self.fisheye = FthetaFromCube(size=face_size, face_size=face_size, rays=rays, valid=valid)
        self._face_renderer = mujoco.Renderer(m, face_size, face_size)
        (ww, wh), _ = camera_models.wrist_render()
        self._wrist_renderer = mujoco.Renderer(m, wh, ww)
        self._lut = tone_lut(self.light["tone_k"])
        self._background = int(round(isaac_background(spec)))
        self._reset()

    def set_light_gains(self, **gains: float) -> None:
        """Change light gains of the compiled model (calibration). tone_k and hdri_sky_gain are fixed at build time."""
        m, spec = self.model, self.spec
        light = {**self.light, **gains}
        if spec.hdri == v16_lab.ROOM:
            m.vis.headlight.ambient[:] = light["lab_ambient_gain"]
            m.light_diffuse[m.light("key").id] = light["lab_key_gain"]
        elif spec.hdri:
            scale = spec.hdri_intensity / 1000.0
            _, share, mean = hdri_mj.key_light(spec.hdri)
            m.vis.headlight.ambient[:] = light["hdri_ambient_gain"] * (1 - share) * scale
            m.light_diffuse[m.light("key").id] = light["hdri_key_gain"] * share * scale
        else:
            m.vis.headlight.ambient[:] = light["ambient_gain"] * spec.light_intensity
            for i in range(m.nlight):
                name = m.light(i).name
                if name == "sun" or name.startswith("ceiling_"):
                    m.light_diffuse[i] = light["sun_gain" if name == "sun" else "ceiling_gain"] * light_color(spec, name)
        self.light = light

    def _reset(self) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        offsets = np.array(self.spec.start_offsets) if len(self.spec.start_offsets) == 14 else np.zeros(14)
        start = {"left": np.array(START_JOINTS["left"]) + offsets[:7], "right": np.array(START_JOINTS["right"]) + offsets[7:]}
        for side in ("left", "right"):
            d.qpos[self.arm_qpos[side]] = start[side]
            d.qpos[self.finger_qpos[side]] = START_FINGER
        head = head_camera.head_joints()  # calibrated on v16
        for joint, actuator in (("o1_motor_2", "head_tilt"), ("o1_motor_1", "head_pan")):
            d.qpos[m.joint(joint).qposadr[0]] = head[joint]
            d.ctrl[m.actuator(actuator).id] = head[joint]
        self.q_start = np.concatenate([start["left"], start["right"], [START_FINGER, START_FINGER]])
        self._apply(self.q_start)
        mujoco.mj_forward(m, d)
        for _ in range(int(round(SETTLE_S / TIMESTEP))):
            mujoco.mj_step(m, d)
        self.bottle_rest_z = float(d.xpos[self.bottle_body][2])

    def _apply(self, cmd: np.ndarray) -> None:
        for i, side in enumerate(("left", "right")):
            self.data.ctrl[self.arm_act[side]] = cmd[7 * i:7 * i + 7]
            self.data.ctrl[self.grip_act[side]] = np.clip(cmd[14 + i], 0.0, 0.04)

    def step(self, cmd: np.ndarray) -> None:
        """Hold the 16 commands for one 15 Hz tick."""
        self._apply(np.asarray(cmd, dtype=np.float64))
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def joints(self) -> np.ndarray:
        """Measured 14 arm joints + 2 finger positions (finger_joint1 per hand), v16 order."""
        q = self.data.qpos
        return np.concatenate([q[self.arm_qpos["left"]], q[self.arm_qpos["right"]],
                               [q[self.finger_qpos["left"][0]], q[self.finger_qpos["right"][0]]]])

    def state(self) -> np.ndarray:
        """v16's 34-dim state, computed as record_demos.py does."""
        j = self.joints()
        return np.concatenate([j, kin.v16_eef("left", j[:7]), kin.v16_eef("right", j[7:14])]).astype(np.float32)

    def images(self) -> dict[str, np.ndarray]:
        """head: 256x256 fisheye (black outside the image circle, like the real lens); left_wrist / right_wrist:
        256x192 letterboxed to 256x256 (uint8 RGB). Rendered at higher resolution and downsampled, like v16."""
        m, d = self.model, self.data
        extent = m.stat.extent
        m.vis.map.znear = HEAD_NEAR / extent
        faces = {}
        for name in CUBE_FACES:
            self._face_renderer.update_scene(d, camera=f"head_{name}")
            faces[name] = self._lut[self._face_renderer.render()]
        size = head_camera.OUT_SIZE
        out = {"head": head_camera.camera_frame(self.fisheye.remap(faces, outside=0), (size, size))}
        m.vis.map.znear = WRIST_NEAR / extent
        for side in ("left", "right"):
            self._wrist_renderer.update_scene(d, camera=f"camera_d435i_{side}_color")
            view = camera_models.wrist_view(self._lut[self._wrist_renderer.render()], side)
            wrist = head_camera.camera_frame(view, head_camera.WRIST_OUT)
            out[f"{side}_wrist"] = letterbox(wrist)
        return out

    def bottle_pos(self) -> np.ndarray:
        return self.data.xpos[self.bottle_body].copy()

    def bottle_lift(self) -> float:
        """How far the bottle is above its resting height on the table (m), as scene.py."""
        return float(self.data.xpos[self.bottle_body][2] - self.bottle_rest_z)

    def bottle_tilt_deg(self) -> float:
        """Angle between the bottle's axis and world up."""
        zaxis = self.data.xmat[self.bottle_body].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(zaxis[2], -1.0, 1.0))))

    def success(self) -> bool:
        return self.bottle_lift() >= LIFT_SUCCESS_M

    def close(self) -> None:
        self._face_renderer.close()
        self._wrist_renderer.close()
