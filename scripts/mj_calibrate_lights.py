"""Fit MuJoCo's light gains and tone curve to Isaac Sim's renders of the same scenes.

    MUJOCO_GL=egl ~/sim/mujoco_venv/bin/python scripts/mj_calibrate_lights.py \
        --isaac-raw ~/sim/demos/lift_bottle_v3/raw_train --n 24

Uses the first frame of recorded Isaac episodes (start pose, after the same 10 warm-up ticks). Scenes with an HDRI
room fit the HDRI gains (ambient, key light; for each background brightness `--sky-gain`), scenes without one the
dome / sun / ceiling gains. For each tone-curve strength, coordinate descent (in log space) minimizes the squared
difference of per-channel mean color and of the 90th-percentile brightness, over all three cameras (head fisheye
inside its image circle only). Writes the best into assets/mj_lighting.json, which mj_scene.py loads.
"""

import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from garmi_bench import head_camera  # noqa: E402
from garmi_bench import mj_scene  # noqa: E402
from garmi_bench import spec as _spec  # noqa: E402

CAMS = ("head", "left_wrist", "right_wrist")


def stats(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    px = img[mask].astype(np.float64)
    return np.concatenate([px.mean(0), [np.percentile(px.mean(1), 90)]])


def masks() -> dict[str, np.ndarray]:
    wrist = np.zeros((256, 256), bool)
    wrist[32:224] = True  # inside the letterbox
    return {"head": head_camera.circle_mask(head_camera.OUT_SIZE), "left_wrist": wrist, "right_wrist": wrist}


def loss(scenes, targets, cam_masks) -> float:
    err = []
    for scene, target in zip(scenes, targets, strict=True):
        imgs = scene.images()
        for cam in CAMS:
            err.append(stats(imgs[cam], cam_masks[cam]) - target[cam])
    return float(np.mean(np.square(err)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--isaac-raw", required=True)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--tone-k", type=float, nargs="+", default=[0.0, 1.5, 3.5])
    p.add_argument("--sky-gain", type=float, nargs="+", default=[0.15, 0.3, 0.6], help="HDRI scenes: dome texture gains")
    args = p.parse_args()
    raw = os.path.expanduser(args.isaac_raw)
    files = sorted(f for f in os.listdir(raw) if f.startswith("episode_") and f.endswith(".npz"))[:args.n]
    episodes = []
    for f in files:
        ep = np.load(os.path.join(raw, f))
        seed = json.loads(str(ep["meta"]))["spec"]["seed"]
        episodes.append((_spec.sample_scene(seed, _spec.TRAIN), {c: ep[c][0] for c in CAMS}))
    hdri = bool(episodes[0][0].hdri)
    gains_names = ("hdri_ambient_gain", "hdri_key_gain") if hdri else ("ambient_gain", "sun_gain", "ceiling_gain")
    outer = list(itertools.product(args.tone_k, args.sky_gain if hdri else [None]))
    cam_masks = masks()
    targets = [{c: stats(imgs[c], cam_masks[c]) for c in CAMS} for _, imgs in episodes]

    best = None
    for k, sky in outer:
        t0 = time.time()
        light = {**mj_scene.lighting(), "tone_k": k, **({"hdri_sky_gain": sky} if hdri else {})}
        scenes = [mj_scene.MjLiftBottleScene(spec, light=light) for spec, _ in episodes]
        for scene in scenes:
            for _ in range(10):
                scene.step(scene.q_start)
        gains = {g: light[g] for g in gains_names}

        def evaluate(g):
            for scene in scenes:
                scene.set_light_gains(**g)
            return loss(scenes, targets, cam_masks)

        current = evaluate(gains)
        for factor in (2.0, 1.4, 1.15, 1.05):
            improved = True
            while improved:
                improved = False
                for g in gains_names:
                    for f in (factor, 1 / factor):
                        trial = {**gains, g: gains[g] * f}
                        value = evaluate(trial)
                        if value < current:
                            gains, current, improved = trial, value, True
        for scene in scenes:
            scene.close()
        rms = float(np.sqrt(current))
        print(f"[calibrate] tone_k {k}" + (f", sky gain {sky}" if hdri else "") + f": rms {rms:.1f} gray levels, gains "
              + ", ".join(f"{g} {v:.3g}" for g, v in gains.items()) + f" ({time.time() - t0:.0f} s)", flush=True)
        if best is None or rms < best["rms"]:
            best = {"tone_k": k, **gains, **({"hdri_sky_gain": sky} if hdri else {}), "rms": rms}

    out = {}
    if os.path.exists(mj_scene.LIGHTING_FILE):
        with open(mj_scene.LIGHTING_FILE) as f:
            out = json.load(f)
    key = "hdri" if hdri else "plain"
    out.update({k: v for k, v in best.items() if k != "rms"})
    out[f"fit_{key}"] = {"rms": best["rms"], "isaac_raw": raw, "episodes": files}
    with open(mj_scene.LIGHTING_FILE, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[calibrate] best ({key}): {json.dumps(best)} -> {mj_scene.LIGHTING_FILE}", flush=True)


if __name__ == "__main__":
    main()
