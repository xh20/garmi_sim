"""Isaac Sim scene for the GARMI lift-the-bottle task, built once and re-randomized per SceneSpec.

Must be imported after `isaacsim.SimulationApp` has started.
"""

from __future__ import annotations

import os

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.types import ArticulationAction
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.rotations as rot
from pxr import Usd
from pxr import UsdPhysics

from garmi_bench import robot as _robot
from garmi_bench.spec import LIFT_SUCCESS_M
from garmi_bench.spec import SceneSpec

PHYSICS_DT = 1.0 / 120.0  # 8 physics steps per 15 Hz control step
RENDER_DT = 1.0 / 15.0
# The fixed base's convex-hull collision shapes reach below z=0; touching the ground throws the arms off.
GROUND_Z = -0.05
BOTTLE_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "bottles")
MAX_CEILING_LIGHTS = 4


def letterbox(img: np.ndarray, size: int = 256) -> np.ndarray:
    """Pad a (size*3/4, size) image to (size, size) with black bars, like v16's wrist images."""
    h = img.shape[0]
    out = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - h) // 2
    out[top:top + h] = img[..., :3]
    return out


def _xform_op(prim, op_type):
    """The prim's existing transform op of this type, or a new one (ops cannot be added twice)."""
    from pxr import UsdGeom

    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == op_type:
            return op
    return xf.AddXformOp(op_type, UsdGeom.XformOp.PrecisionDouble)


class LiftBottleScene:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=RENDER_DT)
        # Room: floor slab (top at GROUND_Z), a back wall behind the table and one side wall. Fixed colliders,
        # recolored and moved per scene.
        self.floor = self.world.scene.add(FixedCuboid(prim_path="/World/room/floor", name="floor", size=1.0,
                                                      position=np.array([1.0, 0.0, GROUND_Z - 0.05]), scale=np.array([12.0, 12.0, 0.1]),
                                                      color=np.array([0.5, 0.5, 0.5])))
        self.back_wall = self.world.scene.add(FixedCuboid(prim_path="/World/room/back_wall", name="back_wall", size=1.0,
                                                          position=np.array([3.0, 0.0, 1.25]), scale=np.array([0.1, 8.0, 2.6]),
                                                          color=np.array([0.9, 0.9, 0.9])))
        self.side_wall = self.world.scene.add(FixedCuboid(prim_path="/World/room/side_wall", name="side_wall", size=1.0,
                                                          position=np.array([1.0, 2.0, 1.25]), scale=np.array([6.0, 0.1, 2.6]),
                                                          color=np.array([0.9, 0.9, 0.9])))
        # Lights: ambient dome, a sun (distant light) and up to MAX_CEILING_LIGHTS ceiling panels; set per scene.
        self.dome = prim_utils.create_prim("/World/DomeLight", "DomeLight", attributes={"inputs:intensity": 300.0})
        self.sun = prim_utils.create_prim("/World/Lights/sun", "DistantLight",
                                          attributes={"inputs:intensity": 0.0, "inputs:angle": 0.53, "inputs:enableColorTemperature": True})
        self.ceiling = [prim_utils.create_prim(f"/World/Lights/ceiling_{i}", "RectLight",
                                               attributes={"inputs:intensity": 0.0, "inputs:width": 0.6, "inputs:height": 0.6,
                                                           "inputs:enableColorTemperature": True})
                        for i in range(MAX_CEILING_LIGHTS)]

        garmi = prim_utils.create_prim("/World/garmi", usd_path=_robot.import_garmi_usd())
        garmi.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
        self.robot = self.world.scene.add(SingleArticulation(prim_path=f"/World/garmi/{_robot.ARTICULATION_ROOT}", name="garmi"))

        grip = PhysicsMaterial("/World/Materials/grip", static_friction=1.2, dynamic_friction=1.0, restitution=0.0)
        # Table: a top slab on four square legs (all fixed colliders).
        self.table = self.world.scene.add(FixedCuboid(prim_path="/World/table/top", name="table", size=1.0,
                                                      position=np.array([1.1, 0.1, 0.83]), scale=np.array([0.7, 1.4, 0.04]),
                                                      color=np.array([0.55, 0.40, 0.28]), physics_material=grip))
        self.legs = [self.world.scene.add(FixedCuboid(prim_path=f"/World/table/leg_{i}", name=f"table_leg_{i}", size=1.0,
                                                      position=np.array([1.1, 0.1, 0.4]), scale=np.array([0.05, 0.05, 0.8]),
                                                      color=np.array([0.55, 0.40, 0.28])))
                     for i in range(4)]
        self.grip = grip
        self.bottle = None  # SingleRigidPrim of the current bottle asset (swapped in reset())
        self._bottle_asset: str | None = None
        self.world.reset()
        self.names = self.robot.dof_names
        self._init_robot()
        self.q_start = _robot.start_positions(self.names, self.robot.get_joint_positions())
        self.spec: SceneSpec | None = None

    def _init_robot(self) -> None:
        kp, kd = _robot.drive_gains(self.names)
        self.robot.get_articulation_controller().set_gains(kps=kp, kds=kd)

    def _load_bottle(self, spec: SceneSpec) -> None:
        """Swap in the spec's bottle asset (a USD rigid body, origin at its bottom center)."""
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.core.utils.prims import delete_prim
        from pxr import UsdShade

        # Scale and mass are authored before the physics reset: changing a collider's scale while simulating
        # makes PhysX rebuild it and invalidates every physics view ("Simulation view object is invalidated").
        key = (spec.bottle_asset, round(spec.bottle_scale, 6), round(spec.bottle_mass, 6))
        if self._bottle_asset == key:
            return
        if self.bottle is not None:
            self.world.stop()  # no prim removal while the simulation runs (invalidates the physics views)
            self.world.scene.remove_object("bottle")
            delete_prim("/World/bottle")
        prim = prim_utils.create_prim("/World/bottle", usd_path=os.path.join(BOTTLE_DIR, spec.bottle_asset),
                                      scale=np.full(3, spec.bottle_scale))
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(spec.bottle_mass)
        mat = UsdShade.Material(self.world.stage.GetPrimAtPath(self.grip.prim_path))
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        self.bottle = self.world.scene.add(SingleRigidPrim(prim_path="/World/bottle", name="bottle"))
        self._bottle_asset = key
        self.world.reset()  # (re)build physics with the new body
        self._init_robot()

    def find_prim(self, name: str) -> str:
        stage = self.world.stage
        for p in Usd.PrimRange(stage.GetPrimAtPath("/World/garmi")):
            if p.GetName() == name:
                return str(p.GetPath())
        raise KeyError(name)

    def reset(self, spec: SceneSpec, settle_steps: int = 30) -> None:
        """Apply a scene spec: room, table, bottle and light; robot back to the start pose; let things settle."""
        self.spec = spec
        # First: swapping the bottle runs world.reset(), which puts every scene object back to its initial
        # pose. Everything placed before it would be undone.
        self._load_bottle(spec)
        self.table.set_world_pose(position=np.array(spec.table_top_center))
        self.table.set_local_scale(np.array([spec.table_depth, spec.table_width, spec.table_thickness]))
        self.table.get_applied_visual_material().set_color(np.array(spec.table_color))
        leg_h = spec.table_height - spec.table_thickness - GROUND_Z
        for leg, (x, y) in zip(self.legs, spec.table_leg_xy, strict=True):
            leg.set_world_pose(position=np.array([x, y, GROUND_Z + leg_h / 2]))
            leg.set_local_scale(np.array([spec.table_leg_width, spec.table_leg_width, leg_h]))
            leg.get_applied_visual_material().set_color(np.array(spec.table_color))

        self.floor.get_applied_visual_material().set_color(np.array(spec.floor_color))
        # Without walls (an HDRI room around the scene) they are parked far below the floor.
        away = 0.0 if spec.walls else -50.0
        self.back_wall.set_world_pose(position=np.array([spec.back_wall_x + 0.05, 0.0, GROUND_Z + 1.3 + away]))
        self.side_wall.set_world_pose(position=np.array([1.0, spec.side_wall_y + np.sign(spec.side_wall_y) * 0.05, GROUND_Z + 1.3 + away]))
        for wall in (self.back_wall, self.side_wall):
            wall.get_applied_visual_material().set_color(np.array(spec.wall_color))
        self._set_lights(spec)

        self.bottle.set_world_pose(position=np.array([spec.bottle_x, spec.bottle_y, spec.table_height + 0.003]),
                                   orientation=rot.euler_angles_to_quat(np.array([0.0, 0.0, spec.bottle_yaw])))
        self.bottle.set_linear_velocity(np.zeros(3))
        self.bottle.set_angular_velocity(np.zeros(3))

        self.robot.set_joint_positions(self.q_start)
        self.robot.set_joint_velocities(np.zeros(len(self.names)))
        for _ in range(settle_steps):
            self.step(self.q_start, render=False)
        # Lift is measured from where the bottle came to rest (its convex collision shapes can sit a few
        # centimeters off the visual mesh's bottom), so success does not depend on the collision approximation.
        self.bottle_rest_z = float(self.bottle.get_world_pose()[0][2])

    def _set_lights(self, spec: SceneSpec) -> None:
        from pxr import Gf
        from pxr import Sdf
        from pxr import UsdGeom

        if spec.hdri:
            # A real room photo as the dome: background and main light. The extra sun / ceiling panels are off.
            from garmi_bench.spec import HDRI_DIR

            self.dome.CreateAttribute("inputs:texture:file", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(os.path.abspath(os.path.join(HDRI_DIR, spec.hdri))))
            self.dome.CreateAttribute("inputs:texture:format", Sdf.ValueTypeNames.Token).Set("latlong")
            self.dome.GetAttribute("inputs:intensity").Set(spec.hdri_intensity)
            op = _xform_op(self.dome, UsdGeom.XformOp.TypeRotateXYZ)
            vec = Gf.Vec3d if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble else Gf.Vec3f
            op.Set(vec(0.0, 0.0, spec.hdri_rotation_deg))  # latlong in a Z-up stage: rotate about z
            self.sun.GetAttribute("inputs:intensity").Set(0.0)
            for light in self.ceiling:
                light.GetAttribute("inputs:intensity").Set(0.0)
            return
        if self.dome.GetAttribute("inputs:texture:file"):
            self.dome.GetAttribute("inputs:texture:file").Set(Sdf.AssetPath(""))
        self.dome.GetAttribute("inputs:intensity").Set(spec.light_intensity)
        if spec.sun is None:
            self.sun.GetAttribute("inputs:intensity").Set(0.0)
        else:
            az, el, intensity, temp = spec.sun
            # A distant light shines along its -z axis: point +z at the sun.
            s = np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)), np.cos(np.radians(el)) * np.sin(np.radians(az)),
                          np.sin(np.radians(el))])
            x = np.cross([0.0, 0.0, 1.0], s)
            x /= np.linalg.norm(x)
            m = Gf.Matrix3d(*np.stack([x, np.cross(s, x), s], axis=0).flatten().tolist())  # rows = local axes in world
            q = m.ExtractRotation().GetQuat()
            op = _xform_op(self.sun, UsdGeom.XformOp.TypeOrient)
            quat = Gf.Quatd if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble else Gf.Quatf
            vec = Gf.Vec3d if quat is Gf.Quatd else Gf.Vec3f
            op.Set(quat(q.GetReal(), vec(*q.GetImaginary())))
            self.sun.GetAttribute("inputs:intensity").Set(intensity)
            self.sun.GetAttribute("inputs:colorTemperature").Set(temp)
        for i, light in enumerate(self.ceiling):
            if i >= len(spec.ceiling):
                light.GetAttribute("inputs:intensity").Set(0.0)
                continue
            x, y, z, edge, intensity, temp = spec.ceiling[i]
            op = _xform_op(light, UsdGeom.XformOp.TypeTranslate)  # a RectLight faces -z: down
            op.Set(Gf.Vec3d(x, y, z) if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble else Gf.Vec3f(x, y, z))
            light.GetAttribute("inputs:width").Set(edge)
            light.GetAttribute("inputs:height").Set(edge)
            light.GetAttribute("inputs:intensity").Set(intensity)
            light.GetAttribute("inputs:colorTemperature").Set(temp)

    def step(self, q_target: np.ndarray, render: bool = True) -> None:
        self.robot.apply_action(ArticulationAction(joint_positions=q_target))
        self.world.step(render=render)

    def bottle_lift(self) -> float:
        """How far the bottle is above its resting height on the table (m)."""
        pos, _ = self.bottle.get_world_pose()
        return float(pos[2] - self.bottle_rest_z)

    def success(self) -> bool:
        return self.bottle_lift() >= LIFT_SUCCESS_M

    def link_world_pose(self, link_name: str) -> tuple[np.ndarray, np.ndarray]:
        return SingleXFormPrim(self.find_prim(link_name)).get_world_pose()

    def add_wrist_camera(self, side: str):
        """The D435i color camera on a wrist: a centred pinhole of camera_models.wrist_render(), which
        camera_models.wrist_view() warps to the robot's 640x480 and intrinsics; then downsample to 256x192 and letterbox
        to 256x256 like v16 (head_camera.camera_frame, letterbox())."""
        from isaacsim.sensors.camera import Camera
        from pxr import Gf
        from pxr import UsdGeom

        from garmi_bench import camera_models

        resolution, hfov_deg = camera_models.wrist_render()
        path = self.find_prim(f"camera_d435i_{side}_color_optical_frame") + "/wrist_cam"
        xf = UsdGeom.Xformable(UsdGeom.Camera.Define(self.world.stage, path))
        xf.ClearXformOpOrder()
        xf.AddOrientOp().Set(Gf.Quatf(0.0, 1.0, 0.0, 0.0))  # ROS optical (z forward, y down) -> USD camera
        cam = Camera(prim_path=path, resolution=resolution)
        cam.initialize()
        cam.set_clipping_range(0.01, 100.0)
        cam.set_focal_length(cam.get_horizontal_aperture() / (2 * np.tan(np.radians(hfov_deg) / 2)))
        return cam

    def finger_collision_prims(self) -> list[str]:
        stage = self.world.stage
        return [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath("/World/garmi"))
                if p.HasAPI(UsdPhysics.CollisionAPI) and "finger" in str(p.GetPath())]

    def add_head_camera(self, size: int | None = None):
        """GARMI's head camera (left lens of the Olive stereo fisheye), riding the head's pan/tilt.

        The camera prim is a child of the URDF's `olive_camera_left_optical_frame` (ROS optical axes:
        z forward, y down), turned 180 deg about x into USD's camera axes (looks along -z, y up), then by the
        calibrated image roll about the optical axis. The lens is the f-theta curve calibrated on the v16
        recordings (head_camera.py), at the camera's native 1000x1000 per eye; downsample to 256 like v16 and black
        out everything outside the image circle (head_camera.downsample / circle_mask). Set the head joints to
        head_camera.head_joints() (record_demos.py, isaac_env.py).
        """
        from isaacsim.sensors.camera import Camera
        from pxr import Gf
        from pxr import UsdGeom

        from garmi_bench import head_camera

        size = size or head_camera.RENDER_SIZE
        path = self.find_prim("olive_camera_left_optical_frame") + "/head_cam"
        cam_prim = UsdGeom.Camera.Define(self.world.stage, path)
        xf = UsdGeom.Xformable(cam_prim)
        xf.ClearXformOpOrder()
        # 180 deg about x (optical -> USD camera axes), then the image roll about the viewing axis. The image rotates
        # opposite to the camera, and USD's y is up where the image's is down, so the camera turns by +roll about
        # its own z (checked against the v16 frames: scripts/calibrate_head_camera.py overlays).
        half = head_camera.roll() / 2
        q = Gf.Quatf(0.0, 1.0, 0.0, 0.0) * Gf.Quatf(float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))
        xf.AddOrientOp().Set(q)
        cam = Camera(prim_path=path, resolution=(size, size))
        cam.initialize()
        # Isaac's default near plane is 1.0 m, which cuts away the arms and the table. 3 cm skips the camera's
        # own housing and lens cover (closer: the image shows the housing; farther: the head's inside).
        # With the real lens's ~200 deg field of view, the CAD camera body, housing and cover (all within ~3 cm of the
        # lens) fill one side of the image black; the real lens sits in front of them and does not see them. A near
        # plane of 8 cm drops them; the closest real content, the arms, is > 20 cm away. (Hiding their prims instead
        # also hides instanced collision shapes and breaks the physics views.)
        cam.set_clipping_range(0.08, 100.0)
        cam.set_ftheta_properties(**head_camera.isaac_ftheta(size))
        return cam
