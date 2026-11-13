"""Does a policy's action chunk follow the bottle in the images, or replay a trajectory from the robot state?

    MUJOCO_GL=egl ~/sim/mujoco_venv/bin/python scripts/probe_bottle_shift.py --host <server> --port 8001 \
        --name v16_real_smoothest_27000 --out ~/sim/eval/probe/v16_real_smoothest_27000

Causal test in MuJoCo, nominal suite. Per scene, one normal closed-loop episode first records the robot state at a few
policy calls of the approach (--calls). At each of these states the scene is rebuilt with the bottle moved by
--offsets-cm along x (forward) or y (sideways), or removed (below the floor), the arm and fingers are set to exactly the
recorded joint angles (no physics: only the images change, the state sent is the recorded one), and the policy is
queried --samples times. Response: where the grasping arm's fingertip would be at the end of the predicted chunk
(forward kinematics of its joint targets), mean over the samples. Per call: the slope of that endpoint (along the
shift axis) against the bottle shift (1 = follows the bottle, 0 = ignores it), the endpoint change when the bottle is
removed, and the sampling spread at the unchanged scene. Writes <out>/probe.json and prints a summary.
"""

import argparse
import dataclasses
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="localhost")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--name", required=True)
ap.add_argument("--n-scenes", type=int, default=8)
ap.add_argument("--calls", type=int, nargs="+", default=[0, 3, 6])
ap.add_argument("--offsets-cm", type=float, nargs="+", default=[-8, -4, 4, 8])
ap.add_argument("--samples", type=int, default=4)
ap.add_argument("--state-dim", type=int, default=0)
ap.add_argument("--out", required=True)
mine = ap.parse_args()
sys.argv = [sys.argv[0], "--sim", "mujoco", "--host", mine.host, "--port", str(mine.port), "--name", mine.name,
            "--out", mine.out, "--delay-steps", "3", "--video", "none", "--state-dim", str(mine.state_dim)]
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import eval_policy as ev  # noqa: E402

from garmi_bench import kinematics as kin  # noqa: E402
from garmi_bench import spec as _spec  # noqa: E402
from garmi_bench.mj_scene import MjLiftBottleScene  # noqa: E402


class RecordingPolicy(ev.Policy):
    def __init__(self, host, port):
        super().__init__(host, port)
        self.states = []

    def __call__(self, obs):
        self.states.append(np.asarray(obs["observation/state"], np.float64))
        return super().__call__(obs)


def observation(scene: MjLiftBottleScene, state: np.ndarray, remove_bottle: bool) -> dict:
    """Images of `scene` with the arms at `state` (14 joints, 2 fingers), and the recorded state itself."""
    m, d = scene.model, scene.data
    for i, side in enumerate(("left", "right")):
        d.qpos[scene.arm_qpos[side]] = state[7 * i:7 * i + 7]
        d.qpos[scene.finger_qpos[side]] = state[14 + i]
    if remove_bottle:
        adr = m.jnt_qposadr[m.joint("bottle_free").id]
        d.qpos[adr:adr + 3] = [0.0, 0.0, -2.0]
    mujoco.mj_forward(m, d)
    obs = ev.make_obs_fn(scene, ev.args, {})()
    obs["observation/state"] = state[:mine.state_dim] if mine.state_dim else state.astype(np.float32)
    return obs


def endpoint(policy, obs, side: str) -> np.ndarray:
    """Fingertip of the grasping arm at the end of the predicted chunk, mean over --samples queries; and their spread."""
    j0 = 0 if side == "left" else 7
    pts = np.array([kin.tcp_pose(side, policy(obs)[-1, j0:j0 + 7])[:3, 3] for _ in range(mine.samples)])
    return pts.mean(axis=0), float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).mean())


def main():
    os.makedirs(mine.out, exist_ok=True)
    suites = json.load(open(ev.SUITES_FILE))["suites"]
    entries = suites["nominal"]["episodes"]
    entries = [e for e in entries if e["arm"] == "left"][:mine.n_scenes // 2] + [e for e in entries if e["arm"] == "right"][:mine.n_scenes // 2]
    policy = RecordingPolicy(mine.host, mine.port)
    first = MjLiftBottleScene(_spec.sample_scene(entries[0]["seed"], _spec.EVAL_NOMINAL))
    horizon = len(policy(observation(first, first.state().astype(np.float64), False)))
    first.close()
    results = []
    for e in entries:
        base = _spec.sample_scene(e["seed"], _spec.EVAL_NOMINAL)
        policy.states = []
        ev.run_episode(policy, base, ev.args, 3, horizon)  # the unperturbed rollout: states along the approach
        for call in mine.calls:
            if call >= len(policy.states):
                continue
            state = policy.states[call]  # as sent; its first 16 values are the joints and fingers set below
            row = {"seed": e["seed"], "arm": base.arm, "call": call}
            scene = MjLiftBottleScene(base)
            p0, spread = endpoint(policy, observation(scene, state, False), base.arm)
            row["spread_cm"] = 100 * spread
            row["removed_cm"] = 100 * float(np.linalg.norm(endpoint(policy, observation(scene, state, True), base.arm)[0] - p0))
            scene.close()
            for axis, key in ((0, "bottle_x"), (1, "bottle_y")):
                shifts, moves = [], []
                for off in mine.offsets_cm:
                    spec = dataclasses.replace(base, **{key: getattr(base, key) + off / 100.0})
                    scene = MjLiftBottleScene(spec)
                    p, _ = endpoint(policy, observation(scene, state, False), base.arm)
                    scene.close()
                    shifts.append(off / 100.0)
                    moves.append(float(p[axis] - p0[axis]))
                s, mv = np.array(shifts), np.array(moves)
                row[f"slope_{'xy'[axis]}"] = float((s @ mv) / (s @ s))  # least squares through the origin
                row[f"moves_{'xy'[axis]}_cm"] = [round(100 * v, 2) for v in moves]
            results.append(row)
            print(f"[probe] seed {e['seed']} {base.arm:5s} call {call}: slope x {row['slope_x']:+.2f}, y {row['slope_y']:+.2f}, "
                  f"bottle removed {row['removed_cm']:.1f} cm, sampling spread {row['spread_cm']:.1f} cm", flush=True)
    with open(os.path.join(mine.out, "probe.json"), "w") as f:
        json.dump({"name": mine.name, "offsets_cm": mine.offsets_cm, "samples": mine.samples, "rows": results}, f, indent=1)
    for call in mine.calls:
        rows = [r for r in results if r["call"] == call]
        if rows:
            print(f"[probe] {mine.name} call {call} ({len(rows)} scenes): slope x {np.mean([r['slope_x'] for r in rows]):+.2f}, "
                  f"slope y {np.mean([r['slope_y'] for r in rows]):+.2f}, bottle removed {np.mean([r['removed_cm'] for r in rows]):.1f} cm, "
                  f"spread {np.mean([r['spread_cm'] for r in rows]):.1f} cm", flush=True)


if __name__ == "__main__":
    main()
