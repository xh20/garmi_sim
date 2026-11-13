"""Run the scripted expert in MuJoCo on seeds recorded in Isaac Sim and compare outcomes and images.

    MUJOCO_GL=egl ~/sim/mujoco_venv/bin/python scripts/mj_expert_check.py \
        --isaac-raw ~/sim/demos/lift_bottle_v1/raw_train --n 40 --out ~/sim/mj_check

Per seed (same SceneSpec, same expert commands; the expert is open loop, so its commands do not depend on the
simulator): success in MuJoCo vs Isaac, final lift, and the joint tracking error. With --compare-images, the
first frame of each kept Isaac episode is shown next to MuJoCo's (head, left wrist, right wrist) in
compare_<seed>.png, and mean brightness per camera is written to brightness.json (light-gain calibration).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from garmi_bench import spec as _spec  # noqa: E402
from garmi_bench.expert import LiftBottleExpert  # noqa: E402
from garmi_bench.mj_scene import CONTROL_HZ  # noqa: E402
from garmi_bench.mj_scene import MjLiftBottleScene  # noqa: E402

WARMUP_TICKS = 10  # record_demos.py holds the start pose for 10 rendered ticks before recording


def run_expert(spec, seed, max_seconds, want_first_images):
    expert = LiftBottleExpert(spec, seed)
    if not expert.ik_ok:
        return {"seed": seed, "ik_ok": False}, None
    scene = MjLiftBottleScene(spec)
    for _ in range(WARMUP_TICKS):
        scene.step(scene.q_start)
    first = scene.images() if want_first_images else None
    t0, ticks, track, max_lift = time.time(), 0, [], 0.0
    while not expert.done and ticks < max_seconds * CONTROL_HZ:
        cmd = expert.step()
        scene.step(cmd)
        track.append(np.abs(cmd[:14] - scene.joints()[:14]).max())
        max_lift = max(max_lift, scene.bottle_lift())
        ticks += 1
    ok = bool(expert.done and scene.success())
    result = {"seed": seed, "ik_ok": True, "arm": spec.arm, "bottle": spec.bottle_asset, "success": ok, "ticks": ticks,
              "final_lift_cm": round(scene.bottle_lift() * 100, 1), "max_lift_cm": round(max_lift * 100, 1),
              "final_tilt_deg": round(scene.bottle_tilt_deg(), 1), "max_track_err_rad": round(float(max(track)), 3),
              "wall_s": round(time.time() - t0, 1)}
    scene.close()
    return result, first


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--isaac-raw", required=True, help="an Isaac record_demos.py output folder (record_log.json, episodes)")
    p.add_argument("--n", type=int, default=40, help="number of seeds from its record_log.json (in order)")
    p.add_argument("--split", choices=("train", "eval", "unseen"), default="train", help="ranges the Isaac run used")
    p.add_argument("--compare-images", type=int, default=0, help="save Isaac/MuJoCo first frames for this many kept seeds")
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--out", default=os.path.expanduser("~/sim/mj_check"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ranges = {"train": _spec.TRAIN, "eval": _spec.EVAL_NOMINAL, "unseen": _spec.EVAL_UNSEEN_OBJECTS}[args.split]
    raw = os.path.expanduser(args.isaac_raw)
    isaac_log = {e["seed"]: e for e in json.load(open(os.path.join(raw, "record_log.json"))) if "frames" in e or "reason" in e}
    episodes = {}
    for name in sorted(os.listdir(raw)):
        if name.startswith("episode_") and name.endswith(".npz"):
            meta = json.loads(str(np.load(os.path.join(raw, name))["meta"]))
            episodes[meta["spec"]["seed"]] = os.path.join(raw, name)

    results, brightness, compared = [], [], 0
    for seed in list(isaac_log)[:args.n]:
        spec = _spec.sample_scene(seed, ranges)
        want = compared < args.compare_images and seed in episodes
        r, first = run_expert(spec, seed, args.max_seconds, want)
        isaac = isaac_log[seed]
        r["isaac_success"] = bool(isaac.get("kept", False)) if "frames" in isaac else None
        r["isaac_final_lift_cm"] = isaac.get("final_lift_cm")
        results.append(r)
        if r["ik_ok"]:
            print(f"[mj_check] seed {seed} {spec.arm:5s} {spec.bottle_asset:22s} MuJoCo {'OK  ' if r['success'] else 'FAIL'} "
                  f"(lift {r['final_lift_cm']:+.1f} cm, tilt {r['final_tilt_deg']:.0f} deg, track {r['max_track_err_rad']:.3f}) "
                  f"| Isaac {'OK' if r['isaac_success'] else 'FAIL'} ({r['isaac_final_lift_cm']} cm) | {r['wall_s']} s", flush=True)
        if first is not None:
            ep = np.load(episodes[seed])
            isaac_imgs = {"head": ep["head"][0], "left_wrist": ep["left_wrist"][0], "right_wrist": ep["right_wrist"][0]}
            rows = [np.concatenate([isaac_imgs[k] for k in ("head", "left_wrist", "right_wrist")], axis=1),
                    np.concatenate([first[k] for k in ("head", "left_wrist", "right_wrist")], axis=1)]
            Image.fromarray(np.concatenate(rows, axis=0)).save(os.path.join(args.out, f"compare_{seed:05d}.png"))
            brightness.append({"seed": seed, "sun": spec.sun is not None, "n_ceiling": len(spec.ceiling),
                               **{f"isaac_{k}": float(isaac_imgs[k].mean()) for k in isaac_imgs},
                               **{f"mj_{k}": float(first[k].mean()) for k in first}})
            compared += 1

    run = [r for r in results if r["ik_ok"]]
    both = [r for r in run if r["isaac_success"] is not None]
    summary = {
        "seeds": len(results), "ik_ok": len(run),
        "mujoco_success": sum(r["success"] for r in run), "isaac_success": sum(bool(r["isaac_success"]) for r in both),
        "agree": sum(r["success"] == r["isaac_success"] for r in both),
        "mujoco_success_per_arm": {a: f"{sum(r['success'] for r in run if r['arm'] == a)}/{sum(r['arm'] == a for r in run)}"
                                   for a in ("left", "right")},
    }
    json.dump({"summary": summary, "results": results}, open(os.path.join(args.out, "expert_check.json"), "w"), indent=1)
    if brightness:
        json.dump(brightness, open(os.path.join(args.out, "brightness.json"), "w"), indent=1)
    print(f"[mj_check] {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
