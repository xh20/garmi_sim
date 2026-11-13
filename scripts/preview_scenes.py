"""Render the first N randomized lift-the-bottle scenes: an overview camera and GARMI's head camera.

    OMNI_KIT_ACCEPT_EULA=YES ~/isaac/.venv/bin/python scripts/preview_scenes.py --n 10 --out /tmp/garmi_scenes

The head camera is the calibrated one (head_camera.py: lens, head tilt / pan, roll), downsampled to 256 and black
outside the image circle like v16's images; the overview camera looks back at GARMI from beyond the table.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--first-seed", type=int, default=0)
parser.add_argument("--split", choices=("train", "eval", "unseen"), default="train")
parser.add_argument("--out", default="/tmp/garmi_scenes")
parser.add_argument("--head-size", type=int, default=1000, help="head render resolution (lower when the GPU is shared)")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
import isaacsim.core.utils.rotations as rot  # noqa: E402
from PIL import Image  # noqa: E402
from PIL import ImageDraw  # noqa: E402

from garmi_bench import head_camera  # noqa: E402
from garmi_bench import spec as _spec  # noqa: E402
from garmi_bench.scene import LiftBottleScene  # noqa: E402


def look_at(eye, target):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = (target - eye) / np.linalg.norm(target - eye)
    return eye, rot.euler_angles_to_quat(np.array([0.0, -np.arcsin(f[2]), np.arctan2(f[1], f[0])]))


def set_fov(cam: Camera, fov_deg: float) -> None:
    cam.set_focal_length(cam.get_horizontal_aperture() / (2 * np.tan(np.radians(fov_deg) / 2)))


def main():
    os.makedirs(args.out, exist_ok=True)
    scene = LiftBottleScene()
    overview = Camera(prim_path="/World/overview", resolution=(640, 480))
    scene.world.reset()
    overview.initialize()
    overview.set_clipping_range(0.01, 100.0)
    set_fov(overview, 60.0)
    head = scene.add_head_camera(args.head_size)
    for joint, value in head_camera.head_joints().items():
        scene.q_start[scene.names.index(joint)] = value
    circle = head_camera.circle_mask(head_camera.OUT_SIZE)[..., None]
    ranges = {"train": _spec.TRAIN, "eval": _spec.EVAL_NOMINAL, "unseen": _spec.EVAL_UNSEEN_OBJECTS}[args.split]

    tiles_over, tiles_head = [], []
    for i in range(args.n):
        seed = args.first_seed + i
        spec = _spec.sample_scene(seed, ranges)
        scene.reset(spec)
        # Beyond the table (inside the back wall), on the side away from the side wall, looking back at GARMI.
        eye_x = spec.table_front_x + spec.table_depth + 1.0
        if spec.walls:
            eye_x = min(eye_x, spec.back_wall_x - 0.15)
        eye_y = spec.table_center_y - np.sign(spec.side_wall_y) * 1.25
        eye, quat = look_at([eye_x, eye_y, spec.table_height + 0.9],
                            [0.55 * (spec.bottle_x + 0.3), 0.5 * spec.bottle_y + 0.05, spec.table_height + 0.15])
        overview.set_world_pose(position=eye, orientation=quat, camera_axes="world")
        for _ in range(8):
            scene.step(scene.q_start, render=True)
        size = head_camera.OUT_SIZE
        img_o = Image.fromarray(overview.get_rgb()[..., :3])
        img_h = Image.fromarray((head_camera.camera_frame(head.get_rgb(), (size, size)) * circle).astype(np.uint8))
        if spec.hdri:
            light = f"room {spec.hdri.removesuffix('_2k.hdr')}"
        else:
            light = ("no sun" if spec.sun is None else f"sun el {spec.sun[1]:.0f} deg {spec.sun[3]:.0f}K") + \
                f", {len(spec.ceiling)} ceiling light{'s' if len(spec.ceiling) != 1 else ''}"
        label = (f"seed {seed}  {spec.arm} arm  table {spec.table_height:.2f} m  {spec.bottle_asset.removesuffix('.usda')}  "
                 f"{light}")
        ImageDraw.Draw(img_o).rectangle([0, 0, 640, 18], fill=(0, 0, 0))
        ImageDraw.Draw(img_o).text((5, 3), label, fill=(255, 255, 255))
        ImageDraw.Draw(img_h).text((4, 4), f"seed {seed}", fill=(255, 255, 0))
        img_o.save(f"{args.out}/scene_{seed:03d}_overview.png")
        img_h.save(f"{args.out}/scene_{seed:03d}_head.png")
        open(f"{args.out}/scene_{seed:03d}.json", "w").write(spec.to_json())
        tiles_over.append(img_o)
        tiles_head.append(img_h)
        w, x, y, z = scene.bottle.get_world_pose()[1]
        tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))
        print(f"[scene] {label} | rests {(scene.bottle_rest_z - spec.table_height) * 100:+.1f} cm from the table top, "
              f"tilt {tilt:.0f} deg, success={scene.success()}", flush=True)

    cols = 5
    rows = (len(tiles_over) + cols - 1) // cols
    for tiles, name, (w, h) in ((tiles_over, "overview", (640, 480)), (tiles_head, "head", (256, 256))):
        grid = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
        for k, t in enumerate(tiles):
            grid.paste(t, ((k % cols) * w, (k // cols) * h))
        grid.save(f"{args.out}/grid_{name}.png")
    print(f"[scene] wrote {args.out}/grid_overview.png and grid_head.png", flush=True)
    app.close()


main()
