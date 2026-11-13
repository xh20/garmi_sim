"""Do the checkerboard intrinsics of the Olive's left lens (Kannala-Brandt / OpenCV equidistant, 1000x1000) explain v16's
head images better than our silhouette-fitted cubic model? Same data and objective as calibrate_head_camera.py
(IoU of the arms' predicted and observed swept areas), with the intrinsics fixed and only the head pose (tilt, pan,
roll) fitted.

    ~/sim/mujoco_venv/bin/python scripts/check_head_intrinsics.py --episodes 12
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import cv2  # noqa: E402
import glob  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import calibrate_head_camera as cal  # noqa: E402

SIZE = cal.SIZE
S = SIZE / 1000.0  # checkerboard calibration at 1000 x 1000 per eye
OLIVE_LEFT = {"fx": 422.174969 * S, "fy": 421.004097 * S, "cx": 492.072730 * S, "cy": 492.618757 * S,
              "k": [-0.038748, -0.007903, 0.003115, -0.001121]}
OLIVE_RIGHT = {"fx": 420.793412 * S, "fy": 419.812752 * S, "cx": 511.627142 * S, "cy": 498.494729 * S,
               "k": [-0.036728, -0.006926, 0.000835, -0.000321]}
# The pixel centre convention: OpenCV puts pixel (0, 0)'s centre at 0.0, our cubic model at 0.5 (pixel_rays: +0.5).
# At 256 px that is half a pixel; the KB centres are shifted by +0.5 px to our convention.


def project_kb(points, pose, p, kb):
    pos, rot = pose
    c = (points - pos) @ rot
    theta = np.arctan2(np.hypot(c[:, 0], c[:, 1]), c[:, 2])
    phi = np.arctan2(c[:, 1], c[:, 0]) + p["roll"]
    t2 = theta * theta
    k1, k2, k3, k4 = kb["k"]
    td = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    return np.stack([kb["cx"] + 0.5 + kb["fx"] * td * np.cos(phi), kb["cy"] + 0.5 + kb["fy"] * td * np.sin(phi)], axis=1), theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", default=os.path.expanduser("~/sim/calib_head_kb"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n_eps = len(glob.glob(f"{cal.V16}/data/chunk-*/episode_*.parquet"))
    eps = np.linspace(0, n_eps - 1, args.episodes).astype(int).tolist()
    arm, cam = cal.ArmPoints(), cal.HeadCamera()
    data = []
    for ep in eps:
        states, frames = cal.load_episode(ep, args.stride)
        data.append({"ep": ep, "points": [arm(s) for s in states], "observed": cal.observed_change(frames),
                     "frame": frames[len(frames) // 2], "mid_state": states[len(states) // 2]})
    print(f"[kb] {len(data)} episodes loaded", flush=True)

    def ious(project, p):
        pose = cam(p["tilt"], p["pan"])
        out = []
        for e in data:
            occ = [cal.occupancy(*project(pts, pose, p), 1.15 * 1.4) for pts in e["points"]]
            pred = np.max(occ, axis=0) & (1 - np.min(occ, axis=0))
            out.append(float((pred & e["observed"]).sum() / max((pred | e["observed"]).sum(), 1)))
        return out

    cur = json.load(open(cal.OUT))
    res = {"current_cubic": {"pose": {k: cur[k] for k in ("tilt", "pan", "roll")}}}
    res["current_cubic"]["iou"] = ious(cal.project, cur)
    for name, kb in (("olive_left", OLIVE_LEFT), ("olive_right", OLIVE_RIGHT)):
        proj = lambda pts, pose, p, kb=kb: project_kb(pts, pose, p, kb)  # noqa: E731
        start = {k: cur[k] for k in ("tilt", "pan", "roll")}
        same_pose = ious(proj, start)
        x0, scale = np.array([start["tilt"], start["pan"], start["roll"]]), np.array([0.03, 0.03, 0.03])

        def loss(z):
            return 1.0 - float(np.mean(ious(proj, dict(zip(("tilt", "pan", "roll"), x0 + z * scale)))))

        r = minimize(loss, np.zeros(3), method="Nelder-Mead",
                     options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-4, "initial_simplex": np.vstack([np.zeros(3), np.eye(3)])})
        pose = dict(zip(("tilt", "pan", "roll"), (x0 + r.x * scale).tolist()))
        res[name] = {"iou_same_pose": same_pose, "pose": pose, "iou": ious(proj, pose)}
        print(f"[kb] {name}: IoU at the current pose {np.mean(same_pose):.3f}, refitted {np.mean(res[name]['iou']):.3f} "
              f"pose {json.dumps({k: round(v, 4) for k, v in pose.items()})}", flush=True)
        # overlays: real frame + projected arm points (current model magenta, this model green)
        for e in data[::3]:
            over = e["frame"].copy()
            pts = arm(e["mid_state"])
            for prj, p, col in ((cal.project, cur, (255, 0, 255)), (proj, pose, (0, 255, 0))):
                uv, _ = prj(pts, cam(p["tilt"], p["pan"]), p)
                for u, v in uv[(uv[:, 0] >= 0) & (uv[:, 0] < SIZE) & (uv[:, 1] >= 0) & (uv[:, 1] < SIZE)].astype(int)[::4]:
                    over[v, u] = col
            cv2.imwrite(os.path.join(args.out, f"{name}_ep{e['ep']:03d}.png"),
                        cv2.cvtColor(cv2.resize(over, (512, 512), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR))
    print(f"[kb] current cubic model: IoU {np.mean(res['current_cubic']['iou']):.3f}", flush=True)
    json.dump({**res, "episodes": eps}, open(os.path.join(args.out, "result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
