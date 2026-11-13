"""Closed-loop evaluation of a served policy on the lift-the-bottle suites, in MuJoCo or Isaac Sim.

Start the policy server (openpi repo, any machine), then:
    MUJOCO_GL=egl ~/sim/mujoco_venv/bin/python scripts/eval_policy.py --host localhost --port 8000 \
        --name sim_v1_27000 --out ~/sim/eval/mujoco/sim_v1_27000
    OMNI_KIT_ACCEPT_EULA=YES ~/isaac/.venv/bin/python scripts/eval_policy.py --sim isaac \
        --name sim_v1_27000 --out ~/sim/eval/isaac/sim_v1_27000
Both simulators build the same scene from a suite seed and run the identical loop below, so a model trained on
one simulator's demos can be scored in-domain and across simulators on the same seeds.

Per episode (assets/eval_suites.json, made by make_eval_suites.py): the scene is built from its seed, the robot
holds its start pose for 10 ticks (as in the demos), then runs at 15 Hz for at most --max-seconds:
  observation  = v16 layout: 34-dim state from the measured joints, head fisheye + both wrist images, each
                 resized with padding to --client-resize (what the robot client sends), prompt "grasp the bottle";
  execution    = the robot client's real-time chunking with a 5-step cross-fade and its safety limits, in sim
                 time (garmi_bench/rtc_sim.py); latency = --delay-steps ticks, measured from the server if 0;
  success      = bottle >= 20 cm above its resting height for --hold-seconds without interruption.
An episode also ends early when the bottle falls off the table. Results (per episode and per suite, with Wilson
95% intervals) go to <out>/results.json after every episode; --resume skips episodes already in it.
"""

import argparse
import collections
import glob
import hashlib
import importlib.metadata as md
import json
import math
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

p = argparse.ArgumentParser()
p.add_argument("--sim", choices=("mujoco", "isaac"), default="mujoco")
p.add_argument("--host", default="localhost")
p.add_argument("--port", type=int, default=8000)
p.add_argument("--name", required=True, help="label of the evaluated policy (checkpoint), stored in results.json")
p.add_argument("--out", required=True)
p.add_argument("--suites", nargs="+", default=["nominal", "unseen_objects", "unseen_rooms", "train_ranges"])
p.add_argument("--episodes-per-suite", type=int, default=0, help="0 = all (quick tests: e.g. 4)")
p.add_argument("--max-seconds", type=float, default=30.0)
p.add_argument("--wrist-camera", choices=("real", "legacy"), default="real",
               help="wrist camera model (garmi_bench.camera_models): the robot's intrinsics, or the one used up to v7obs")
p.add_argument("--hold-seconds", type=float, default=1.0)
p.add_argument("--delay-steps", type=int, default=0, help="inference latency in 15 Hz ticks; 0 = measure the server")
p.add_argument("--blend-steps", type=int, default=5)
p.add_argument("--execute-steps", type=int, default=0, help="replan after this many executed steps per chunk (0 = use the chunk up to its end)")
p.add_argument("--prefix-horizon", type=int, default=0, help="guide only this many leading steps of the new chunk toward the old tail (0 = all)")
p.add_argument("--binary-gripper", type=float, default=0.0, help="binary gripper: close fully when the predicted finger drops below this (mm); 0 = off")
p.add_argument("--binary-open", type=float, default=30.0, help="binary gripper: open fully again above this (mm)")
p.add_argument("--client-resize", type=int, default=224)
p.add_argument("--wrist-flow-gap", type=int, default=0,
               help="also send the wrist frames this many ticks ago (observation/*_wrist_image_prev) for wrist-flow "
                    "policies (openpi.shared.wrist_flow.FLOW_GAP = 3); 0 = off")
p.add_argument("--prompt", default="grasp the bottle")
p.add_argument("--video", choices=("none", "failures", "all"), default="failures")
p.add_argument("--resume", action="store_true")
p.add_argument("--state-dim", type=int, default=0, help="send only the first N state values (16 for policies trained before v16); 0 = all")
args = p.parse_args()

if args.sim == "isaac":  # must run before any Isaac import
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402

from garmi_bench import camera_models as _camera_models  # noqa: E402
from garmi_bench import kinematics as _kin  # noqa: E402
from garmi_bench import spec as _spec  # noqa: E402
from garmi_bench.expert import CONTROL_HZ  # noqa: E402
from garmi_bench.rtc_sim import ClientLimits  # noqa: E402
from garmi_bench.rtc_sim import ClientSafety  # noqa: E402
from garmi_bench.rtc_sim import SimTimeBlendRTC  # noqa: E402

_camera_models.set_wrist_model(args.wrist_camera)

if args.sim == "isaac":
    from garmi_bench.isaac_env import IsaacLiftBottleEnv  # noqa: E402

    _ISAAC = IsaacLiftBottleEnv()

    def make_scene(spec):
        return _ISAAC.load(spec)
else:
    from garmi_bench.mj_scene import MjLiftBottleScene  # noqa: E402

    def make_scene(spec):
        return MjLiftBottleScene(spec)

BASE = os.path.join(os.path.dirname(__file__), "..")
SUITES_FILE = os.path.join(BASE, "assets", "eval_suites.json")
WARMUP_TICKS = 10
EXTRA_NAMES = ("progress", "holding")
FELL_OFF_M = 0.15  # bottle this far below the table surface: it fell off, the episode cannot succeed any more


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 3), round((c + h) / d, 3))


def provenance(args) -> dict:
    files = sorted(glob.glob(os.path.join(BASE, "garmi_bench", "*.py")) + glob.glob(os.path.join(BASE, "scripts", "*.py"))
                   + [SUITES_FILE, os.path.join(BASE, "assets", "mj_lighting.json"), os.path.join(BASE, "assets", "bottles", "catalog.json")])
    record = {"command": " ".join([os.path.basename(sys.executable)] + sys.argv), "args": vars(args),
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": platform.node(),
              "versions": {args.sim: md.version("isaacsim" if args.sim == "isaac" else "mujoco"), "numpy": np.__version__,
                           "python": platform.python_version()},
              "sha256": {os.path.relpath(p, BASE): hashlib.sha256(open(p, "rb").read()).hexdigest() for p in files if os.path.exists(p)}}
    if args.sim == "mujoco":
        from garmi_bench.mj_scene import lighting

        record["lighting"] = lighting()
    return record


class Policy:
    """Websocket client; appends the extra outputs (progress, holding) as columns so they are blended too."""

    def __init__(self, host: str, port: int):
        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.latencies_ms: list[float] = []

    def __call__(self, obs: dict) -> np.ndarray:
        t0 = time.monotonic()
        result = self.client.infer(obs)
        self.latencies_ms.append(1e3 * (time.monotonic() - t0))
        actions = np.asarray(result["actions"], dtype=np.float64)
        extras = [np.asarray(result[n], dtype=np.float64) for n in EXTRA_NAMES if n in result]
        return np.concatenate([actions, np.stack(extras, axis=-1)], axis=-1) if extras else actions


def make_obs_fn(scene, args, record):
    def make_obs():
        imgs = record.get("tick_images") or scene.images()
        record["last_images"] = imgs
        def send(img):
            return image_tools.resize_with_pad(img, args.client_resize, args.client_resize) if args.client_resize > 0 else img
        state = scene.state()
        obs = {"observation/state": state[:args.state_dim] if args.state_dim else state, "observation/image": send(imgs["head"]),
               "observation/left_wrist_image": send(imgs["left_wrist"]),
               "observation/right_wrist_image": send(imgs["right_wrist"]), "prompt": args.prompt}
        if args.wrist_flow_gap:  # wrist frames wrist_flow_gap ticks ago (the oldest kept; the first frame at the start)
            history = record.get("wrist_history")
            prev = history[0] if history else imgs  # no history (the latency probe): zero flow, as at an episode start
            obs["observation/left_wrist_image_prev"] = send(prev["left_wrist"])
            obs["observation/right_wrist_image_prev"] = send(prev["right_wrist"])
        return obs
    return make_obs


def run_episode(policy, spec, args, delay_steps, horizon, video_path=None) -> dict:
    scene = make_scene(spec)
    for _ in range(WARMUP_TICKS):
        scene.step(scene.q_start)
    record = {}
    make_obs = make_obs_fn(scene, args, record)
    rtc = SimTimeBlendRTC(policy, horizon=horizon, delay_steps=delay_steps, blend_steps=args.blend_steps,
                          execute_steps=args.execute_steps, prefix_horizon=args.prefix_horizon)
    safety = ClientSafety(scene.q_start, ClientLimits(binary_close=args.binary_gripper / 1000.0,
                                                       binary_open=args.binary_open / 1000.0))
    n_lat = len(policy.latencies_ms)
    hold_ticks = int(round(args.hold_seconds * CONTROL_HZ))
    above, max_lift, max_tilt, frames, cmds, signals = 0, 0.0, 0.0, [], [], []
    outcome, success_tick = "timeout", None
    # Grasp diagnostics: where the active arm's fingertip TCP is relative to the cap at every binary close, and how
    # close it got overall. along: cap position along the TCP's approach axis (> 0: the cap is still ahead, the
    # gripper closed short of it); lateral: distance off that axis; closing: component along the closing axis.
    side_i = 0 if spec.arm == "left" else 1
    bottle0 = scene.bottle_pos()
    close_events, min_cap, was_closed = [], np.inf, False
    wrist_history = collections.deque(maxlen=args.wrist_flow_gap + 1)
    record["wrist_history"] = wrist_history
    for tick in range(int(args.max_seconds * CONTROL_HZ)):
        if args.wrist_flow_gap:  # wrist-flow policies need the wrist frames of every tick (rendered once per tick)
            imgs = scene.images()
            record["tick_images"] = imgs
            wrist_history.append({"left_wrist": imgs["left_wrist"], "right_wrist": imgs["right_wrist"]})
        action, _ = rtc.step(tick, make_obs)
        cmd = safety(tick, action, scene.joints())
        tcp = _kin.tcp_pose(spec.arm, scene.joints()[7 * side_i:7 * side_i + 7])
        d = np.array([spec.bottle_x, spec.bottle_y, spec.cap_z]) + (scene.bottle_pos() - bottle0) - tcp[:3, 3]
        min_cap = min(min_cap, float(np.linalg.norm(d)))
        is_closed = bool(getattr(safety, "closed", [False, False])[side_i])
        if is_closed and not was_closed:
            along = float(d @ tcp[:3, 2])
            close_events.append({"tick": tick, "along_mm": round(along * 1000, 1),
                                 "lateral_mm": round(float(np.linalg.norm(d - along * tcp[:3, 2])) * 1000, 1),
                                 "closing_mm": round(float(d @ tcp[:3, 1]) * 1000, 1),
                                 "dist_mm": round(float(np.linalg.norm(d)) * 1000, 1),
                                 "pred_finger_mm": round(float(action[14 + side_i]) * 1000, 1)})
        was_closed = is_closed
        cmds.append(cmd)
        signals.append(action[16:16 + len(EXTRA_NAMES)])
        if video_path is not None:
            imgs = record.get("last_images") or record.get("tick_images") or scene.images()
            frames.append(np.concatenate([imgs["head"], imgs["left_wrist"], imgs["right_wrist"]], axis=1))
            record.pop("last_images", None)
        scene.step(cmd)
        lift, tilt = scene.bottle_lift(), scene.bottle_tilt_deg()
        max_lift, max_tilt = max(max_lift, lift), max(max_tilt, tilt)
        above = above + 1 if lift >= _spec.LIFT_SUCCESS_M else 0
        if above >= hold_ticks:
            outcome, success_tick = "success", tick + 1 - hold_ticks
            break
        if scene.bottle_pos()[2] < spec.table_height - FELL_OFF_M:
            outcome = "fell_off_table"
            break
    if outcome == "timeout":
        if max_lift >= _spec.LIFT_SUCCESS_M:
            outcome = "lifted_not_held"
        elif max_tilt > 60.0:
            outcome = "knocked_over"
        elif max_lift >= 0.02:
            outcome = "lifted_too_little"
        else:
            outcome = "not_lifted"
    cmds = np.asarray(cmds)
    accel = np.diff(cmds[:, :14], n=2, axis=0) * CONTROL_HZ**2 if len(cmds) > 2 else np.zeros((1, 14))
    lat = policy.latencies_ms[n_lat:]
    sig = np.asarray(signals) if len(signals) and len(signals[0]) else None
    result = {
        "seed": spec.seed, "arm": spec.arm, "bottle": spec.bottle_asset, "success": outcome == "success", "outcome": outcome,
        "time_to_success_s": None if success_tick is None else round(success_tick / CONTROL_HZ, 2),
        "ticks": len(cmds), "max_lift_cm": round(max_lift * 100, 1), "final_lift_cm": round(scene.bottle_lift() * 100, 1),
        "max_tilt_deg": round(max_tilt, 1), "n_infer": rtc.n_infer, "n_switches": rtc.n_switches,
        "stall_ticks": rtc.n_stall_ticks, "mean_overlap": round(float(np.mean(rtc.overlaps)), 2) if rtc.overlaps else None,
        "latency_ms_median": round(float(np.median(lat)), 1) if lat else None,
        "cmd_accel_rms": round(float(np.sqrt(np.mean(accel**2))), 4), **safety.counts,
        "final_progress": None if sig is None else round(float(sig[-1, 0]), 3),
        "max_holding": None if sig is None or sig.shape[1] < 2 else round(float(sig[:, 1].max()), 3),
        "min_tcp_cap_mm": round(min_cap * 1000, 1), "close_events": close_events,
    }
    scene.close()
    if video_path is not None and frames:
        import imageio

        imageio.mimwrite(video_path, frames, fps=int(CONTROL_HZ), macro_block_size=1)
    return result


def summarize(episodes: list[dict], suites: dict) -> dict:
    out = {}
    for name in suites:
        eps = [e for e in episodes if e["suite"] == name]
        if not eps:
            continue
        k = sum(e["success"] for e in eps)
        expert = [s.get("expert_success") for s in suites[name]["episodes"] if s["seed"] in {e["seed"] for e in eps}]
        times = [e["time_to_success_s"] for e in eps if e["success"]]
        out[name] = {
            "episodes": len(eps), "success": k, "success_rate": round(k / len(eps), 3), "wilson95": wilson(k, len(eps)),
            "per_arm": {a: f"{sum(e['success'] for e in eps if e['arm'] == a)}/{sum(e['arm'] == a for e in eps)}"
                        for a in ("left", "right")},
            "expert_reference": f"{sum(bool(x) for x in expert)}/{len(expert)}" if None not in expert else None,
            "mean_time_to_success_s": round(float(np.mean(times)), 2) if times else None,
            "outcomes": {o: sum(e["outcome"] == o for e in eps) for o in sorted({e["outcome"] for e in eps})},
            "cmd_accel_rms_mean": round(float(np.mean([e["cmd_accel_rms"] for e in eps])), 4),
        }
    all_eps = [e for e in episodes]
    k = sum(e["success"] for e in all_eps)
    out["all"] = {"episodes": len(all_eps), "success": k, "success_rate": round(k / max(len(all_eps), 1), 3),
                  "wilson95": wilson(k, len(all_eps))}
    return out


def main():
    out_dir = os.path.expanduser(args.out)
    os.makedirs(os.path.join(out_dir, "videos"), exist_ok=True)
    suites = json.load(open(SUITES_FILE))["suites"]
    results_path = os.path.join(out_dir, "results.json")
    done = json.load(open(results_path))["episodes"] if args.resume and os.path.exists(results_path) else []
    done_keys = {(e["suite"], e["seed"]) for e in done}

    policy = Policy(args.host, args.port)
    # Chunk length and latency from a few calls on the first scene's start observation.
    first = suites[args.suites[0]]["episodes"][0]
    scene = make_scene(_spec.sample_scene(first["seed"], getattr(_spec, suites[args.suites[0]]["ranges"])))
    for _ in range(WARMUP_TICKS):  # as in every episode: Isaac's cameras have no image before the first renders
        scene.step(scene.q_start)
    obs = make_obs_fn(scene, args, {})()
    horizon = policy(obs).shape[0]
    for _ in range(3):
        policy(obs)
    scene.close()
    latency = float(np.median(policy.latencies_ms[1:]))
    delay_steps = args.delay_steps or max(1, math.ceil(latency / (1e3 / CONTROL_HZ)))
    print(f"[eval] {args.name}: chunk {horizon} steps, server latency {latency:.0f} ms -> delay {delay_steps} ticks, "
          f"blend {args.blend_steps}", flush=True)
    header = {"name": args.name, "horizon": horizon, "measured_latency_ms": round(latency, 1), "delay_steps": delay_steps,
              "server_metadata": policy.client.get_server_metadata(), "provenance": provenance(args)}

    episodes = list(done)
    for suite in args.suites:
        entries = suites[suite]["episodes"]
        if args.episodes_per_suite:  # keep the arm balance in quick tests
            left = [e for e in entries if e["arm"] == "left"][:(args.episodes_per_suite + 1) // 2]
            right = [e for e in entries if e["arm"] == "right"][:args.episodes_per_suite // 2]
            entries = sorted(left + right, key=lambda e: e["seed"])
        ranges = getattr(_spec, suites[suite]["ranges"])
        for entry in entries:
            if (suite, entry["seed"]) in done_keys:
                continue
            spec = _spec.sample_scene(entry["seed"], ranges)
            video = os.path.join(out_dir, "videos", f"{suite}_{entry['seed']}.mp4") if args.video != "none" else None
            t0 = time.time()
            r = run_episode(policy, spec, args, delay_steps, horizon, video)
            if video is not None and args.video == "failures" and r["success"] and os.path.exists(video):
                os.remove(video)
            r = {"suite": suite, **r, "expert_success": entry.get("expert_success"), "wall_s": round(time.time() - t0, 1)}
            episodes.append(r)
            print(f"[eval] {suite:14s} seed {r['seed']} {r['arm']:5s} {r['bottle']:22s} {r['outcome']:17s} "
                  f"lift {r['max_lift_cm']:5.1f} cm, {r['ticks']} ticks, {r['n_infer']} calls, {r['wall_s']} s", flush=True)
            json.dump({**header, "summary": summarize(episodes, suites), "episodes": episodes}, open(results_path, "w"), indent=1)
    summary = summarize(episodes, suites)
    json.dump({**header, "summary": summary, "episodes": episodes}, open(results_path, "w"), indent=1)
    for name, s in summary.items():
        print(f"[eval] {name:14s} {s['success']}/{s['episodes']} = {100 * s['success_rate']:.0f}% "
              f"(95% CI {100 * s['wilson95'][0]:.0f}-{100 * s['wilson95'][1]:.0f}%)"
              + (f", per arm {s['per_arm']}, outcomes {s['outcomes']}" if name != "all" else ""), flush=True)


if __name__ == "__main__":
    main()
    if args.sim == "isaac":
        app.close()
