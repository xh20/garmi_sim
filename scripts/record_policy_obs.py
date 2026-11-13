"""Run closed-loop episodes as eval_policy.py and save every observation sent to the policy (for attention maps).

    MUJOCO_GL=egl ~/sim/mujoco_venv/bin/python scripts/record_policy_obs.py --sim mujoco --host <server> --port 8003 \
        --name sim_v3_smoothest_23000 --suite nominal --seeds 200000 200001 --out ~/sim/eval/attention/obs

Per seed: <out>/<sim>_<suite>_<seed>.npz with the state and the three 224x224 images of each policy call (as sent),
the call's control tick, the episode outcome, and the episode video. Isaac: run with ~/isaac/.venv/bin/python.
"""

import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--sim", choices=("mujoco", "isaac"), default="mujoco")
ap.add_argument("--host", default="localhost")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--name", required=True)
ap.add_argument("--suite", default="nominal")
ap.add_argument("--seeds", type=int, nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--delay-steps", type=int, default=3)
mine = ap.parse_args()
sys.argv = [sys.argv[0], "--sim", mine.sim, "--host", mine.host, "--port", str(mine.port), "--name", mine.name,
            "--out", mine.out, "--delay-steps", str(mine.delay_steps), "--video", "all"]
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

import eval_policy as ev  # noqa: E402

from garmi_bench import spec as _spec  # noqa: E402

KEYS = ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image")


class RecordingPolicy(ev.Policy):
    def __init__(self, host, port):
        super().__init__(host, port)
        self.calls = []

    def __call__(self, obs):
        self.calls.append({"state": np.asarray(obs["observation/state"], np.float32),
                           **{k: np.asarray(obs[k]) for k in KEYS}})
        return super().__call__(obs)


def main():
    os.makedirs(mine.out, exist_ok=True)
    suites = json.load(open(ev.SUITES_FILE))["suites"]
    ranges = getattr(_spec, suites[mine.suite]["ranges"])
    policy = RecordingPolicy(mine.host, mine.port)
    for seed in mine.seeds:
        spec = _spec.sample_scene(seed, ranges)
        policy.calls = []
        stem = os.path.join(mine.out, f"{mine.sim}_{mine.suite}_{seed}")
        r = ev.run_episode(policy, spec, ev.args, mine.delay_steps, 32, video_path=stem + ".mp4")
        calls = policy.calls
        np.savez_compressed(stem + ".npz", state=np.stack([c["state"] for c in calls]),
                            **{k.split("/")[1]: np.stack([c[k] for c in calls]) for k in KEYS},
                            meta=json.dumps({"sim": mine.sim, "suite": mine.suite, "seed": seed, "arm": spec.arm,
                                             "bottle": spec.bottle_asset, "policy": mine.name, "outcome": r["outcome"],
                                             "max_lift_cm": r["max_lift_cm"]}))
        print(f"[obs] {mine.sim} {mine.suite} {seed} {spec.arm} {r['outcome']}: {len(calls)} calls -> {stem}.npz", flush=True)
    if mine.sim == "isaac":
        ev.app.close()


if __name__ == "__main__":
    main()
