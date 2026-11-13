"""Side-by-side statistics of the real v16 demos and a simulated dataset (both LeRobot, v16 layout).

    ~/sim/mujoco_venv/bin/python scripts/compare_real_sim.py --sim ~/sim/demos/lerobot/garmi/garmi_sim_lift_bottle_v3

Per episode, from states and actions only (no simulator): the grasping arm (the gripper that closes), the holding
onset (action column 17), the fingertip pose at that moment in base_link (kinematics.tcp_pose, lift at 0), approach
direction, gripper opening, lift height, the other arm's motion, joint speeds and phase timing. From the videos
(first, middle and last frame of each episode): brightness, contrast, saturation and sharpness per camera.
Prints a table and writes <out>.json.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from garmi_bench import kinematics as kin  # noqa: E402

FPS = 15.0
CAMS = ("head", "left_wrist", "right_wrist")


def episode_stats(states: np.ndarray, actions: np.ndarray) -> dict:
    n = len(states)
    hold = actions[:, 17] > 0.5
    k = int(np.argmax(hold)) if hold.any() else n - 1
    drop = states[0, 14:16] - states[k, 14:16]
    side = "left" if drop[0] > drop[1] else "right"
    j0, f = (0, 14) if side == "left" else (7, 15)
    other = 7 if side == "left" else 0
    tcp = [kin.tcp_pose(side, q[j0:j0 + 7]) for q in states]
    p = np.array([t[:3, 3] for t in tcp])
    z_axis = tcp[k][:3, 2]
    dq = np.abs(np.diff(states[:, :14], axis=0))
    after = p[k:, 2] - p[k, 2]
    lifted = np.flatnonzero(after > 0.02)
    finger = states[:, f]
    open_max = float(finger[:k + 1].max())
    k_open = int(np.argmax(finger[:k + 1] >= open_max - 1e-3))
    closing = np.flatnonzero((np.arange(n) < k) & (finger < open_max - 0.003) & (np.arange(n) > k_open))
    return {
        "arm": side, "frames": n, "t_grasp_s": k / FPS, "t_after_grasp_s": (n - k) / FPS,
        "grasp_x": p[k, 0], "grasp_y": p[k, 1], "grasp_z": p[k, 2],
        "approach_az_deg": float(np.degrees(np.arctan2(z_axis[1], z_axis[0]))),
        "approach_el_deg": float(np.degrees(np.arcsin(np.clip(z_axis[2], -1, 1)))),
        "finger_start_mm": 1000 * float(finger[0]), "finger_open_max_mm": 1000 * open_max,
        "finger_at_hold_mm": 1000 * float(finger[k]), "finger_end_mm": 1000 * float(finger[-1]),
        "t_close_s": (k - int(closing[0])) / FPS if len(closing) else 0.0,
        "lift_cm": 100 * float(after.max()), "t_lift_start_s": (int(lifted[0]) / FPS) if len(lifted) else None,
        "other_arm_max_dev_rad": float(np.abs(states[:, other:other + 7] - states[0, other:other + 7]).max()),
        "max_joint_step_rad": float(dq[:, j0:j0 + 7].max()), "mean_joint_speed_rad_s": float(dq[:, j0:j0 + 7].mean() * FPS * 7 / 7),
        "tcp_speed_max_m_s": float(np.linalg.norm(np.diff(p, axis=0), axis=1).max() * FPS),
        "start_q": states[0, :14].tolist(),
    }


def image_stats(img: np.ndarray, cam: str) -> dict:
    if cam != "head":
        img = img[32:224]  # inside v16's letterbox
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return {"brightness": float(gray.mean()), "contrast": float(gray.std()), "saturation": float(hsv[..., 1].mean()),
            "sharpness": float(cv2.Laplacian(gray, cv2.CV_32F).var())}


def dataset(root: str, max_eps: int):
    files = sorted(glob.glob(f"{root}/data/chunk-*/episode_*.parquet"))[:max_eps]
    eps, imgs = [], {c: [] for c in CAMS}
    for f in files:
        df = pd.read_parquet(f, columns=["observation.state", "action"])
        s, a = np.stack(df["observation.state"].to_numpy()), np.stack(df["action"].to_numpy())
        eps.append(episode_stats(s, a))
        ep = os.path.basename(f).replace(".parquet", "")
        for cam in CAMS:
            v = sorted(glob.glob(f"{root}/videos/chunk-*/observation.images.{cam}/{ep}.mp4"))[0]
            frames = iio.imread(v, plugin="pyav")
            for i in (0, len(frames) // 2, len(frames) - 1):
                imgs[cam].append(image_stats(frames[i], cam))
    return eps, imgs


def summary(eps, imgs):
    out = {}
    keys = [k for k in eps[0] if k not in ("arm", "start_q")]
    for arm in ("all", "left", "right"):
        sel = [e for e in eps if arm == "all" or e["arm"] == arm]
        out[arm] = {k: (float(np.nanmedian([np.nan if e[k] is None else e[k] for e in sel])),
                        float(np.nanpercentile([np.nan if e[k] is None else e[k] for e in sel], 10)),
                        float(np.nanpercentile([np.nan if e[k] is None else e[k] for e in sel], 90))) for k in keys}
        out[arm]["episodes"] = len(sel)
    q = np.array([e["start_q"] for e in eps])
    out["start_q_spread_rad"] = float((np.percentile(q, 90, axis=0) - np.percentile(q, 10, axis=0)).max())
    out["images"] = {c: {k: float(np.median([x[k] for x in imgs[c]])) for k in imgs[c][0]} for c in CAMS}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=os.path.expanduser("~/garmi_vlapi/data/garmi_grasp_bottle_v16"))
    ap.add_argument("--sim", required=True)
    ap.add_argument("--max-episodes", type=int, default=400)
    ap.add_argument("--out", default=os.path.expanduser("~/sim/compare_real_sim"))
    args = ap.parse_args()
    real, sim = summary(*dataset(args.real, args.max_episodes)), summary(*dataset(args.sim, args.max_episodes))
    json.dump({"real": real, "sim": sim}, open(args.out + ".json", "w"), indent=1)
    lines = [f"{'metric (median, p10-p90)':34s} {'arm':5s} {'real v16':>24s} {'sim':>24s}"]
    for arm in ("left", "right"):
        for k, v in real[arm].items():
            if k == "episodes":
                continue
            (rm, r10, r90), (sm, s10, s90) = v, sim[arm][k]
            lines.append(f"{k:34s} {arm:5s} {rm:8.3f} ({r10:7.3f}..{r90:7.3f}) {sm:8.3f} ({s10:7.3f}..{s90:7.3f})")
    lines.append(f"{'episodes':34s} {'':5s} {real['all']['episodes']:>24d} {sim['all']['episodes']:>24d}")
    lines.append(f"{'start pose spread (p90-p10, max)':34s} {'':5s} {real['start_q_spread_rad']:24.3f} {sim['start_q_spread_rad']:24.3f}")
    for c in CAMS:
        for k in real["images"][c]:
            lines.append(f"{'image ' + c + ' ' + k:34s} {'':5s} {real['images'][c][k]:24.1f} {sim['images'][c][k]:24.1f}")
    text = "\n".join(lines)
    open(args.out + ".txt", "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
