"""Calibrate GARMI's head fisheye (as v16 saw it) from the v16 recordings themselves.

    ~/sim/mujoco_venv/bin/python scripts/calibrate_head_camera.py --episodes 12

No intrinsics or head angles are recorded for the real head camera, but the arms' joint angles are, and the head
does not move within an episode. So the pixels that change over an episode are where the arms swept (plus the
bottle), and projecting the arm meshes through a candidate camera for every recorded joint state predicts that
area. Fitted parameters, for v16's 256x256 left-eye image:
  tilt, pan   head joints o1_motor_2 / o1_motor_1 (rad), constant over the dataset;
  roll        rotation of the image about the optical axis (rad), for a lens / mount roll;
  f, k2       equidistant fisheye with a cubic term: r = f (theta + k2 theta^3), r in pixels, theta from the axis;
  cx, cy      optical centre (pixels).
Objective: 1 - mean IoU of predicted and observed swept areas over the episodes (grid search, then Nelder-Mead).
The image circle (black outside) is measured from the mean real frame. Writes assets/head_camera_calibration.json
and overlay images (real frame + projected arm points) for checking.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from garmi_bench import kinematics as kin  # noqa: E402
from garmi_bench.mj_scene import _robot_info  # noqa: E402
from garmi_bench.mj_scene import _robot_spec  # noqa: E402

V16 = os.path.expanduser("~/garmi_vlapi/data/garmi_grasp_bottle_v16")
OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "head_camera_calibration.json")
SIZE = 256
PARAMS = ("tilt", "pan", "roll", "f", "k2", "cx", "cy")


class ArmPoints:
    """Surface points of both arms (links, hands, fingers, wrist cameras) in the world for a joint state."""

    def __init__(self, per_geom: int = 300):
        self.m = _robot_spec().compile()
        self.d = mujoco.MjData(self.m)
        arm = set(_robot_info()["arm_bodies"])
        rng = np.random.default_rng(0)
        self.geoms = []
        for g in range(self.m.ngeom):
            if self.m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or self.m.geom_group[g] != 2:
                continue
            if self.m.body(self.m.geom_bodyid[g]).name not in arm:
                continue
            mid = self.m.geom_dataid[g]
            v = self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid] + self.m.mesh_vertnum[mid]]
            if len(v) > per_geom:
                v = v[rng.choice(len(v), per_geom, replace=False)]
            self.geoms.append((g, v.copy()))
        self.adr = {s: [self.m.joint(n).qposadr[0] for n in kin.arm_names(s)] for s in ("left", "right")}
        self.fing = {s: [self.m.joint(f"{s}_fr3_finger_joint{k}").qposadr[0] for k in (1, 2)] for s in ("left", "right")}

    def __call__(self, state: np.ndarray) -> np.ndarray:
        d = self.d
        d.qpos[self.adr["left"]], d.qpos[self.adr["right"]] = state[0:7], state[7:14]
        d.qpos[self.fing["left"]], d.qpos[self.fing["right"]] = state[14], state[15]
        mujoco.mj_kinematics(self.m, d)
        return np.concatenate([d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T for g, v in self.geoms])


class HeadCamera:
    """Pose of the left Olive lens's optical frame (x right, y down, z forward) for given head pan / tilt."""

    def __init__(self):
        self.m = _robot_spec().compile()
        self.d = mujoco.MjData(self.m)
        self.tilt, self.pan = self.m.joint("o1_motor_2").qposadr[0], self.m.joint("o1_motor_1").qposadr[0]
        self.cam = self.m.cam("olive_camera_left").id

    def __call__(self, tilt: float, pan: float):
        self.d.qpos[self.tilt], self.d.qpos[self.pan] = tilt, pan
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)
        r = self.d.cam_xmat[self.cam].reshape(3, 3)
        return self.d.cam_xpos[self.cam].copy(), np.stack([r[:, 0], -r[:, 1], -r[:, 2]], axis=1)  # optical axes


def project(points, pose, p):
    pos, rot = pose
    c = (points - pos) @ rot  # optical coordinates
    theta = np.arctan2(np.hypot(c[:, 0], c[:, 1]), c[:, 2])
    phi = np.arctan2(c[:, 1], c[:, 0]) + p["roll"]
    r = p["f"] * (theta + p["k2"] * theta**3)
    return np.stack([p["cx"] + r * np.cos(phi), p["cy"] + r * np.sin(phi)], axis=1), theta


def occupancy(uv, theta, theta_max):
    img = np.zeros((SIZE, SIZE), np.uint8)
    ok = (theta < theta_max) & (uv[:, 0] >= 0) & (uv[:, 0] < SIZE) & (uv[:, 1] >= 0) & (uv[:, 1] < SIZE)
    iu = uv[ok].astype(int)
    img[iu[:, 1], iu[:, 0]] = 1
    img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=2)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def load_episode(ep: int, stride: int):
    df = pd.read_parquet(sorted(glob.glob(f"{V16}/data/chunk-*/episode_{ep:06d}.parquet"))[0], columns=["observation.state"])
    states = np.stack(df["observation.state"].to_numpy())
    video = sorted(glob.glob(f"{V16}/videos/chunk-*/observation.images.head/episode_{ep:06d}.mp4"))[0]
    frames = iio.imread(video, plugin="pyav")
    n = min(len(states), len(frames))
    return states[:n:stride], frames[:n:stride]


def observed_change(frames: np.ndarray) -> np.ndarray:
    gray = frames.astype(np.float32).mean(axis=3)
    std = gray.std(axis=0)
    mask = (std > 14.0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--overlays", default=os.path.expanduser("~/sim/calib_head"))
    args = ap.parse_args()
    os.makedirs(args.overlays, exist_ok=True)
    n_eps = len(glob.glob(f"{V16}/data/chunk-*/episode_*.parquet"))
    eps = np.linspace(0, n_eps - 1, args.episodes).astype(int).tolist()
    arm, cam = ArmPoints(), HeadCamera()

    data, all_frames = [], []
    for ep in eps:
        states, frames = load_episode(ep, args.stride)
        data.append({"ep": ep, "points": [arm(s) for s in states], "observed": observed_change(frames), "frame": frames[len(frames) // 2],
                     "mid_state": states[len(states) // 2]})
        all_frames.append(frames.astype(np.float32).mean(axis=0))
        print(f"[calib] episode {ep}: {len(states)} frames, observed swept area {data[-1]['observed'].mean() * 100:.1f} % of the image", flush=True)

    # Image circle: radius where the mean frame turns black, around the image centre.
    mean = np.mean(all_frames, axis=0).mean(axis=2)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    rr = np.hypot(xx + 0.5 - SIZE / 2, yy + 0.5 - SIZE / 2)
    radial = [mean[(rr >= r) & (rr < r + 1)].mean() for r in range(0, 182)]
    r_circle = float(next((r for r in range(100, 182) if radial[r] < 15.0), 181))
    print(f"[calib] image circle radius ~{r_circle:.0f} px (black beyond; corner brightness {mean[:8, :8].mean():.1f})", flush=True)

    def loss(p):
        pose = cam(p["tilt"], p["pan"])
        theta_max = np.pi  # the lens edge is applied after the fit; arms are well inside
        ious = []
        for e in data:
            occ = [occupancy(*project(pts, pose, p), theta_max) for pts in e["points"]]
            union, inter = np.max(occ, axis=0), np.min(occ, axis=0)
            pred = union & (1 - inter)
            obs = e["observed"]
            ious.append((pred & obs).sum() / max((pred | obs).sum(), 1))
        return 1.0 - float(np.mean(ious))

    base = {"tilt": 0.5, "pan": 0.0, "roll": 0.0, "f": 100.0, "k2": 0.0, "cx": SIZE / 2, "cy": SIZE / 2}
    best = (1.0, None)
    for tilt in np.arange(-0.2, 0.86, 0.05):
        for f in np.arange(60.0, 201.0, 10.0):
            p = {**base, "tilt": float(tilt), "f": float(f)}
            v = loss(p)
            if v < best[0]:
                best = (v, p)
    print(f"[calib] grid: 1 - IoU {best[0]:.3f} at tilt {best[1]['tilt']:.2f} rad, f {best[1]['f']:.0f} px/rad", flush=True)

    x0 = np.array([best[1][k] for k in PARAMS])
    scale = np.array([0.05, 0.05, 0.05, 10.0, 0.05, 5.0, 5.0])
    res = minimize(lambda z: loss(dict(zip(PARAMS, x0 + z * scale, strict=True))), np.zeros(len(PARAMS)), method="Nelder-Mead",
                   options={"maxiter": 1500, "xatol": 1e-3, "fatol": 1e-4, "initial_simplex": np.vstack([np.zeros(7), np.eye(7)])})
    p = dict(zip(PARAMS, (x0 + res.x * scale).tolist(), strict=True))
    per_ep = []
    pose = cam(p["tilt"], p["pan"])
    for e in data:
        occ = [occupancy(*project(pts, pose, p), np.pi) for pts in e["points"]]
        pred = np.max(occ, axis=0) & (1 - np.min(occ, axis=0))
        per_ep.append(float((pred & e["observed"]).sum() / max((pred | e["observed"]).sum(), 1)))
        over = e["frame"].copy()
        uv, th = project(arm(e["mid_state"]), pose, p)
        for u, v in uv[(uv[:, 0] >= 0) & (uv[:, 0] < SIZE) & (uv[:, 1] >= 0) & (uv[:, 1] < SIZE)].astype(int)[::3]:
            over[v, u] = (255, 0, 255)
        edge = np.zeros((SIZE, SIZE, 3), np.uint8)
        edge[..., 1] = e["observed"] * 255
        edge[..., 0] = pred * 255
        cv2.imwrite(os.path.join(args.overlays, f"overlay_ep{e['ep']:03d}.png"),
                    cv2.cvtColor(np.concatenate([over, edge], axis=1), cv2.COLOR_RGB2BGR))
    theta_max = r_circle / p["f"]  # equidistant part only; refined below with k2
    for _ in range(20):
        theta_max = r_circle / (p["f"] * (1 + p["k2"] * theta_max**2))
    out = {**{k: round(v, 5) for k, v in p.items()}, "image_size": SIZE, "image_circle_px": r_circle,
           "fov_deg": round(float(np.degrees(2 * theta_max)), 1), "iou_mean": round(float(np.mean(per_ep)), 3),
           "iou_per_episode": {str(e["ep"]): round(v, 3) for e, v in zip(data, per_ep, strict=True)},
           "model": "r = f (theta + k2 theta^3) around (cx, cy), image rotated by roll; tilt / pan = o1_motor_2 / o1_motor_1",
           "data": f"{V16}, episodes {eps}, every {args.stride}th frame", "optimizer": res.message}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[calib] result: {json.dumps(out)}", flush=True)


if __name__ == "__main__":
    main()
