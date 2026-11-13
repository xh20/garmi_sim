"""Freeze the lift-the-bottle evaluation suites: seed lists every policy is evaluated on.

    MUJOCO_GL=egl ~/sim/mujoco_venv/bin/python scripts/make_eval_suites.py --per-arm 25

Suites (seed ranges disjoint from the demos: train 0-306, val 100000+):
  nominal         EVAL_NOMINAL ranges (fixed table, training bottles), seeds from 200000
  unseen_objects  EVAL_UNSEEN_OBJECTS (fixed table, the held-out bottle models), seeds from 300000
  unseen_rooms    EVAL_UNSEEN_ROOMS (fixed table, the 6 held-out HDRI rooms), seeds from 500000
  train_ranges    TRAIN ranges (randomized table/room as in the demos), seeds from 400000
  v16_lab         EVAL_V16_LAB: the real v16 lab rebuilt from the recordings (v16_lab.py), clear PET bottles,
                  seeds from 600000
Seeds go in order; a seed is used if the expert's grasp plan is feasible (expert.ik_ok) and its arm still has
room (per-arm balance). The scripted expert is also run in MuJoCo on every suite seed, as a reference
(`expert_success`): failures there mark scenes that are hard even for the oracle. Output:
assets/eval_suites.json (with --only, the named suites are (re)made and the others kept).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from garmi_bench import spec as _spec  # noqa: E402
from garmi_bench.expert import LiftBottleExpert  # noqa: E402

SUITES = {"nominal": ("EVAL_NOMINAL", 200000), "unseen_objects": ("EVAL_UNSEEN_OBJECTS", 300000),
          "unseen_rooms": ("EVAL_UNSEEN_ROOMS", 500000), "train_ranges": ("TRAIN", 400000),
          "v16_lab": ("EVAL_V16_LAB", 600000)}
OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "eval_suites.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-arm", type=int, default=25)
    p.add_argument("--no-expert", action="store_true", help="skip the MuJoCo expert reference run")
    p.add_argument("--only", nargs="+", choices=list(SUITES), help="make only these suites, keep the others in the file")
    args = p.parse_args()
    if not args.no_expert:
        from mj_expert_check import run_expert  # noqa: E402  (scripts/ is on sys.path as the script's dir)
    suites = json.load(open(OUT))["suites"] if args.only else {}
    for name, (ranges_name, first_seed) in SUITES.items():
        if args.only and name not in args.only:
            continue
        ranges = getattr(_spec, ranges_name)
        entries, count, seed = [], {"left": 0, "right": 0}, first_seed
        while min(count.values()) < args.per_arm:
            spec = _spec.sample_scene(seed, ranges)
            if count[spec.arm] < args.per_arm and LiftBottleExpert(spec, seed).ik_ok:
                entry = {"seed": seed, "arm": spec.arm, "bottle": spec.bottle_asset}
                if not args.no_expert:
                    r, _ = run_expert(spec, seed, 30.0, False)
                    entry["expert_success"], entry["expert_final_lift_cm"] = r["success"], r["final_lift_cm"]
                entries.append(entry)
                count[spec.arm] += 1
            seed += 1
        ok = sum(e.get("expert_success", False) for e in entries)
        suites[name] = {"ranges": ranges_name, "first_seed": first_seed, "last_seed": seed - 1, "episodes": entries}
        print(f"[suites] {name}: {len(entries)} seeds ({first_seed}..{seed - 1}), expert in MuJoCo {ok}/{len(entries)}", flush=True)
    with open(OUT, "w") as f:
        json.dump({"task": "grasp the bottle", "per_arm": args.per_arm, "suites": suites}, f, indent=1)
    print(f"[suites] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
