"""Why does a policy fail in the v16 lab twin but not in the nominal suite? Swap one part of the scene at a time.

    MUJOCO_GL=egl ~/sim/mujoco_venv/bin/python scripts/ablate_lab.py --name v16_real_smoothest_27000 --n 16 \
        --out ~/sim/eval/ablate_lab/v16_real_smoothest_27000

Needs a policy server (as eval_policy.py). Variants, each on the first --n seeds of a suite (balanced arms):
  lab               v16_lab suite as is
  lab_room_only     nominal suite scenes (table, bottle, placement) with the lab panorama as the room
  lab_with_catalog  v16_lab scenes with the nominal scene's catalog bottle (same seed index) instead of the PET bottle
  nominal           nominal suite as is
"""

import argparse
import dataclasses
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--variants", nargs="+", default=["lab", "lab_room_only", "lab_with_catalog", "nominal"])
ap.add_argument("--out", required=True)
ap.add_argument("--delay-steps", type=int, default=3)
mine, rest = ap.parse_known_args()
sys.argv = [sys.argv[0], "--sim", "mujoco", "--name", mine.name, "--out", mine.out, "--delay-steps", str(mine.delay_steps)] + rest
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import eval_policy as ev  # noqa: E402

from garmi_bench import spec as _spec  # noqa: E402
from garmi_bench import v16_lab  # noqa: E402


def balanced(entries, n):
    left = [e for e in entries if e["arm"] == "left"][:(n + 1) // 2]
    right = [e for e in entries if e["arm"] == "right"][:n // 2]
    return left + right


def main():
    os.makedirs(mine.out, exist_ok=True)
    suites = json.load(open(ev.SUITES_FILE))["suites"]
    lab = [_spec.sample_scene(e["seed"], _spec.EVAL_V16_LAB) for e in balanced(suites["v16_lab"]["episodes"], mine.n)]
    nom = [_spec.sample_scene(e["seed"], _spec.EVAL_NOMINAL) for e in balanced(suites["nominal"]["episodes"], mine.n)]
    variants = {
        "lab": lab,
        "nominal": nom,
        "lab_room_only": [dataclasses.replace(s, hdri=v16_lab.ROOM, hdri_rotation_deg=0.0, hdri_intensity=1000.0, walls=False)
                          for s in nom],
        "lab_with_catalog": [dataclasses.replace(s, bottle_asset=c.bottle_asset, bottle_scale=c.bottle_scale,
                                                 bottle_height=c.bottle_height, bottle_width=c.bottle_width,
                                                 bottle_mass=c.bottle_mass,
                                                 cap_z=round(s.table_height + c.bottle_height - _spec.CAP_BELOW_TOP, 4))
                             for s, c in zip(lab, nom)],
    }
    policy = ev.Policy(ev.args.host, ev.args.port)
    scene = ev.make_scene(lab[0])
    for _ in range(ev.WARMUP_TICKS):
        scene.step(scene.q_start)
    horizon = policy(ev.make_obs_fn(scene, ev.args, {})()).shape[0]
    scene.close()
    results = {}
    for name in mine.variants:
        eps = []
        for spec in variants[name]:
            r = ev.run_episode(policy, spec, ev.args, mine.delay_steps, horizon,
                               video_path=os.path.join(mine.out, f"{name}_{spec.seed}.mp4"))
            eps.append(r)
            print(f"[ablate] {name:17s} seed {spec.seed} {spec.arm:5s} {spec.bottle_asset:22s} {r['outcome']}", flush=True)
        k = sum(e["success"] for e in eps)
        results[name] = {"success": k, "n": len(eps), "episodes": eps}
        print(f"[ablate] {name}: {k}/{len(eps)}", flush=True)
    with open(os.path.join(mine.out, "ablate.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    print("[ablate] " + ", ".join(f"{k} {v['success']}/{v['n']}" for k, v in results.items()), flush=True)


if __name__ == "__main__":
    main()
