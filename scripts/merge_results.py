"""Merge eval_policy.py results of one policy run in parallel parts (e.g. one process per suite) into one results.json.

    ~/sim/mujoco_venv/bin/python scripts/merge_results.py --out <dir>/results.json <part>/results.json ...

Episodes are keyed by (suite, seed); a later part does not replace an episode already taken. The header (name,
settings, provenance) is the first part's; the summary is recomputed as eval_policy.py does.
"""

import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("parts", nargs="+")
ap.add_argument("--out", required=True)
mine = ap.parse_args()
sys.argv = [sys.argv[0], "--name", "merge", "--out", os.path.dirname(os.path.abspath(mine.out))]
sys.path.insert(0, os.path.dirname(__file__))

import eval_policy as ev  # noqa: E402


def main():
    runs = [json.load(open(p)) for p in mine.parts]
    seen, episodes = set(), []
    for run in runs:
        for e in run["episodes"]:
            if (e["suite"], e["seed"]) not in seen:
                seen.add((e["suite"], e["seed"]))
                episodes.append(e)
    suites = json.load(open(ev.SUITES_FILE))["suites"]
    header = {k: v for k, v in runs[0].items() if k not in ("summary", "episodes")}
    header["merged_from"] = [os.path.abspath(p) for p in mine.parts]
    os.makedirs(os.path.dirname(os.path.abspath(mine.out)), exist_ok=True)
    summary = ev.summarize(episodes, suites)
    with open(mine.out, "w") as f:
        json.dump({**header, "summary": summary, "episodes": episodes}, f, indent=1)
    print(f"[merge] {len(episodes)} episodes -> {mine.out}: "
          + ", ".join(f"{k} {v['success']}/{v['episodes']}" for k, v in summary.items()), flush=True)


if __name__ == "__main__":
    main()
