"""Fit the v16 lab twin's light gains (mj_scene.py lab_*) to the real v16 camera images.

    MUJOCO_GL=egl ~/sim/mujoco_venv/bin/python scripts/mj_calibrate_lab.py --n 16

Uses the first frame of real v16 episodes: the arms are driven to the recorded joint angles and fingers, in a lab
scene (spec.EVAL_V16_LAB) with the bottle on the same side. As mj_calibrate_lights.py: for each dome gain
(`--sky-gain`), coordinate descent (log space) on the ambient and key gains minimizes the squared difference of
per-channel mean color and 90th-percentile brightness over all three cameras (head inside its image circle). Writes the
best into assets/mj_lighting.json and side-by-side images (real | twin) to --out.
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from garmi_bench import mj_scene  # noqa: E402
from garmi_bench import spec as _spec  # noqa: E402
from mj_calibrate_lights import CAMS  # noqa: E402
from mj_calibrate_lights import loss  # noqa: E402
from mj_calibrate_lights import masks  # noqa: E402
from mj_calibrate_lights import stats  # noqa: E402

V16 = os.path.expanduser("~/garmi_vlapi/data/garmi_grasp_bottle_v16")
GAINS = ("lab_ambient_gain", "lab_key_gain")


def real_first_frame(ep: int) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    df = pd.read_parquet(sorted(glob.glob(f"{V16}/data/chunk-*/episode_{ep:06d}.parquet"))[0], columns=["observation.state"])
    states = np.stack(df["observation.state"].to_numpy())
    imgs = {}
    for cam in CAMS:
        video = sorted(glob.glob(f"{V16}/videos/chunk-*/observation.images.{cam}/episode_{ep:06d}.mp4"))[0]
        imgs[cam] = next(iter(iio.imiter(video, plugin="pyav")))
    drop = states[0, 14:16] - states.min(0)[14:16]  # the gripper that closes
    return states[0, :16], imgs, "left" if drop[0] > drop[1] else "right"


def lab_scene_for(ep: int, arm: str) -> _spec.SceneSpec:
    for seed in range(700000 + 100 * ep, 700000 + 100 * ep + 100):  # apart from the v16_lab suite (600000+)
        sc = _spec.sample_scene(seed, _spec.EVAL_V16_LAB)
        if sc.arm == arm:
            return sc
    raise RuntimeError(f"no lab scene with the {arm} arm")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--sky-gain", type=float, nargs="+", default=[0.7, 0.85, 1.0, 1.2])
    p.add_argument("--out", default=os.path.expanduser("~/sim/lab_twin_calibration"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n_eps = len(glob.glob(f"{V16}/data/chunk-*/episode_*.parquet"))
    eps = np.linspace(0, n_eps - 1, args.n).astype(int).tolist()
    episodes = []
    for ep in eps:
        q, imgs, arm = real_first_frame(ep)
        episodes.append((ep, lab_scene_for(ep, arm), q, imgs))
    cam_masks = masks()
    targets = [{c: stats(imgs[c], cam_masks[c]) for c in CAMS} for *_, imgs in episodes]

    best = None
    for sky in args.sky_gain:
        t0 = time.time()
        light = {**mj_scene.lighting(), "lab_sky_gain": sky}
        scenes = [mj_scene.MjLiftBottleScene(sc, light=light) for _, sc, _, _ in episodes]
        for scene, (_, _, q, _) in zip(scenes, episodes, strict=True):
            for _ in range(30):
                scene.step(q)
        gains = {g: light[g] for g in GAINS}

        def evaluate(g):
            for scene in scenes:
                scene.set_light_gains(**g)
            return loss(scenes, targets, cam_masks)

        current = evaluate(gains)
        for factor in (2.0, 1.4, 1.15, 1.05):
            improved = True
            while improved:
                improved = False
                for g in GAINS:
                    for f in (factor, 1 / factor):
                        trial = {**gains, g: gains[g] * f}
                        value = evaluate(trial)
                        if value < current:
                            gains, current, improved = trial, value, True
        rms = float(np.sqrt(current))
        print(f"[lab] sky gain {sky}: rms {rms:.1f} gray levels, " + ", ".join(f"{g} {v:.3g}" for g, v in gains.items())
              + f" ({time.time() - t0:.0f} s)", flush=True)
        if best is None or rms < best["rms"]:
            best = {"lab_sky_gain": sky, **gains, "rms": rms}
            evaluate(gains)
            for scene, (ep, _, _, real) in zip(scenes, episodes, strict=True):
                sim = scene.images()
                rows = [np.concatenate([real[c], sim[c]], axis=1) for c in CAMS]
                cv2.imwrite(os.path.join(args.out, f"lab_ep{ep:03d}.png"), np.concatenate(rows, axis=0)[..., ::-1])
        for scene in scenes:
            scene.close()

    out = {}
    if os.path.exists(mj_scene.LIGHTING_FILE):
        with open(mj_scene.LIGHTING_FILE) as f:
            out = json.load(f)
    out.update({k: v for k, v in best.items() if k != "rms"})
    out["fit_lab"] = {"rms": best["rms"], "v16_episodes": eps}
    with open(mj_scene.LIGHTING_FILE, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[lab] best: {json.dumps(best)} -> {mj_scene.LIGHTING_FILE}; images in {args.out}", flush=True)


if __name__ == "__main__":
    main()
