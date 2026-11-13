"""Record scripted lift-the-bottle demos in Isaac Sim; keep the successful ones as raw .npz episodes.

    OMNI_KIT_ACCEPT_EULA=YES ~/isaac/.venv/bin/python scripts/record_demos.py --episodes 20 --out ~/sim/demos/raw

Per 15 Hz tick, in v16's layout: the three camera images (head fisheye 256x256; wrists 4:3 letterboxed
to 256x256), the 34-dim state (14 arm joints, 2 finger positions, both v16 EEF poses) and the expert's
16 commands. Actions are derived later, as in v16 (action[k] = state[k+1], arm columns low-passed), by
scripts/convert_to_lerobot.py. An episode is kept if the bottle is >= 20 cm above its resting height at
the end of the hold.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=20, help="successful episodes to record")
parser.add_argument("--first-seed", type=int, default=0)
parser.add_argument("--max-attempts", type=int, default=0, help="0 = 3 x episodes")
parser.add_argument("--split", choices=("train", "eval", "unseen"), default="train")
parser.add_argument("--out", default=os.path.expanduser("~/sim/demos/raw"))
parser.add_argument("--max-seconds", type=float, default=20.0, help="per episode")
parser.add_argument("--wrist-camera", choices=("real", "legacy"), default="real",
                    help="wrist camera model (garmi_bench.camera_models): the robot's intrinsics, or the one used up to v7obs")
parser.add_argument("--debug-video", action="store_true", help="also save a close overview GIF per episode")
parser.add_argument("--arm", choices=("left", "right"), default=None,
                    help="only record seeds whose sampled scene uses this arm (others are skipped, not simulated)")
parser.add_argument("--aim-error", type=float, default=0.0,
                    help="corrective reach: std (m) of the expert's wrong first aim, corrected during the move (0 = off)")
parser.add_argument("--start-jitter", type=float, default=0.0,
                    help="start pose: uniform +- this (rad) per arm joint instead of the scene's v16-like jitter (0 = off)")
parser.add_argument("--start-real", action="store_true",
                    help="start pose: the offsets of a random real v16 start pose (assets/v16_start_offsets.json)")
parser.add_argument("--bow", type=float, default=0.0, help="curved move: largest sideways/upward detour (m), 0 = off")
parser.add_argument("--close-ramp", type=float, default=0.0,
                    help="gradual close: longest open -> closed finger ramp (s), 0 = instant")
parser.add_argument("--grasp-jitter", type=float, default=0.02, help="bottle position std around the copied real grasp (m)")
parser.add_argument("--cap-jitter", type=float, default=0.01, help="cap (and so table) height std around the real grasp (m)")
parser.add_argument("--orient-jitter", type=float, default=2.0, help="hand orientation std around the real grasp (deg)")
parser.add_argument("--approach-curve", type=float, default=0.0,
                    help="curved approach: largest off-axis offset of the pre-grasp (m), 0 = straight approach")
parser.add_argument("--speed-jitter", type=float, default=0.0, help="move / approach duration spread (0.3 = x0.77..x1.3)")
parser.add_argument("--approach-tilt", type=float, default=0.0,
                    help="final approach direction tilted against the fingertip axis, up to this (deg), 0 = along it")
parser.add_argument("--standoff", type=float, default=0.0,
                    help="staged approach (v6): curve to a standoff this far (m) back along the fingertip axis, align there on "
                         "the measured pose, insert straight and slowly (0 = off, the v5 approach)")
parser.add_argument("--insert-speed", type=float, nargs=2, default=(0.03, 0.05), help="insertion speed range (m/s)")
parser.add_argument("--recover-prob", type=float, default=0.0,
                    help="staged approach: share of episodes whose first insertion is 3-4.5 cm off sideways, so the "
                         "expert touches / misses the cap, backs off, re-aligns and inserts again")
parser.add_argument("--balance-arms", action="store_true", help="keep half of the episodes per arm")
parser.add_argument("--bottle-y", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="keep only scenes whose bottle sits in this y band (m, torso frame). With --grasp-jitter "
                         "0.08 the bottle spreads over y -0.70 .. 0.98 and its tertiles fall at -0.05 and 0.35, so "
                         "0.35 1.00 / -0.05 0.35 / -0.70 -0.05 take a third of the draws each: left, middle (which "
                         "straddles the arm boundary at 0.10, so both arms appear in it) and right")
parser.add_argument("--observer", action="store_true",
                    help="v7: the other arm moves its wrist camera to a side view of the cap (LiftBottleExpert observer)")
parser.add_argument("--max-track-err", type=float, default=0.08,
                    help="drop the episode if either arm ever ends up this far (rad) from the joint position it was "
                         "commanded; 0 = off. Catches an arm blocked by the scene or by the robot itself, and one "
                         "asked to move faster than it can")
parser.add_argument("--observer-max-err", type=float, default=15.0,
                    help="drop the episode if the observer camera ends up this far (deg) off the cap (blocked on the way)")
parser.add_argument("--low-cap-frac", type=float, default=0.0,
                    help="keep this share of the episodes with the cap below --low-cap-below (0 = no quota)")
parser.add_argument("--low-cap-below", type=float, default=0.954, help="cap height (m) of the --low-cap-frac quota")
# DAgger (v6a): a policy served by scripts/serve_policy.py drives first (as in eval_policy.py: real-time chunking with a
# fixed delay, the robot client's limits, binary gripper); the expert takes over at a trigger and finishes the grasp from
# wherever the policy left the hand. Needs --standoff (the staged expert). The frames before the takeover are kept in
# the .npz (meta takeover_tick); convert_to_lerobot.py trains only on the expert's part.
parser.add_argument("--dagger-port", type=int, default=0, help="policy server port (0 = off: the expert drives throughout)")
parser.add_argument("--dagger-host", default="localhost")
parser.add_argument("--dagger-mix", type=float, nargs=3, default=(0.4, 0.4, 0.2),
                    help="share of the takeover triggers: at the policy's close / in the insertion corridor / early")
parser.add_argument("--dagger-late", type=float, default=25.0, help="take over at this time (s) at the latest")
parser.add_argument("--dagger-keep-grasp", action="store_true",
                    help="also keep episodes where the policy closed on the cap itself (dropped by default: no correction)")
parser.add_argument("--delay-steps", type=int, default=3, help="DAgger: inference delay in ticks (as the evaluation)")
parser.add_argument("--horizon", type=int, default=32, help="DAgger: action chunk length of the policy")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from pxr import UsdShade  # noqa: E402

from garmi_bench import camera_models  # noqa: E402
from garmi_bench import head_camera  # noqa: E402
from garmi_bench import kinematics as kin  # noqa: E402
from garmi_bench import spec as _spec  # noqa: E402
from garmi_bench.expert import CONTROL_HZ  # noqa: E402
from garmi_bench.expert import LiftBottleExpert  # noqa: E402
from garmi_bench.rtc_sim import ClientLimits  # noqa: E402
from garmi_bench.rtc_sim import ClientSafety  # noqa: E402
from garmi_bench.rtc_sim import SimTimeBlendRTC  # noqa: E402
from garmi_bench.spec import START_FINGER  # noqa: E402
from garmi_bench.scene import PHYSICS_DT  # noqa: E402
from garmi_bench.scene import LiftBottleScene  # noqa: E402
from garmi_bench.scene import letterbox  # noqa: E402

camera_models.set_wrist_model(args.wrist_camera)

FINGER_MAX_FORCE = 70.0  # N per finger drive (Franka Hand: 70 N continuous grasp force)


def rgb(camera, tries: int = 5) -> np.ndarray:
    """The camera's image; Isaac occasionally returns none for a frame (seen once in ~30000 renders): render again."""
    from isaacsim.core.api import World

    for _ in range(tries):
        img = camera.get_rgb()
        if img is not None and img.size:
            return img
        World.instance().render()
    raise RuntimeError(f"{camera.prim_path}: no image after {tries} renders")


def write_generation_record() -> None:
    """Everything needed to regenerate this output: command, software versions, checksums of all inputs."""
    import glob
    import hashlib
    import importlib.metadata as md
    import platform
    import scipy

    base = os.path.join(os.path.dirname(__file__), "..")
    inputs = sorted(glob.glob(os.path.join(base, "garmi_bench", "*.py")) + glob.glob(os.path.join(base, "scripts", "*.py"))
                    + [os.path.join(base, "assets", f) for f in ("reach_map.json", "grasp_map.json", "bottle_survey.json",
                                                                 "v16_grasps.json", "head_camera_calibration.json")]
                    + glob.glob(os.path.join(base, "assets", "hdri", "*"))
                    + [os.path.join(base, "assets", "bottles", f) for f in ("catalog.json", "stability.json")]
                    + [kin.URDF])
    record = {
        "command": " ".join([os.path.basename(sys.executable)] + sys.argv),
        "args": vars(args),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "versions": {"isaacsim": md.version("isaacsim"), "numpy": np.__version__, "scipy": scipy.__version__,
                     "python": platform.python_version()},
        "sha256": {os.path.relpath(p, base): hashlib.sha256(open(p, "rb").read()).hexdigest() for p in inputs if os.path.exists(p)},
        "note": "Episode i stores its scene seed in its .npz meta (spec.seed); record_log.json lists every attempted seed. "
                "Rerunning this command with the same inputs (checksums above) regenerates the same scenes and demos: "
                "physics, states and commands bit-identically (tested); images up to RTX real-time rendering noise "
                "(median 0.3-0.7 of 255 per pixel).",
    }
    with open(os.path.join(args.out, "generation.json"), "w") as f:
        json.dump(record, f, indent=1)


def main():
    os.makedirs(args.out, exist_ok=True)
    write_generation_record()
    ranges = {"train": _spec.TRAIN, "eval": _spec.EVAL_NOMINAL, "unseen": _spec.EVAL_UNSEEN_OBJECTS}[args.split]
    scene = LiftBottleScene()
    head = scene.add_head_camera()
    wrist = {s: scene.add_wrist_camera(s) for s in ("left", "right")}
    names = scene.names
    for joint, value in head_camera.head_joints().items():  # calibrated on v16 (not recorded there)
        scene.q_start[names.index(joint)] = value
    circle = head_camera.circle_mask(head_camera.OUT_SIZE)[..., None]
    idx = {
        "arm": {s: [names.index(n) for n in kin.arm_names(s)] for s in ("left", "right")},
        "finger": {s: [names.index(f"{s}_fr3_finger_joint{k}") for k in (1, 2)] for s in ("left", "right")},
    }
    grip = UsdShade.Material(scene.world.stage.GetPrimAtPath(scene.grip.prim_path))
    for path in scene.finger_collision_prims():
        UsdShade.MaterialBindingAPI.Apply(scene.world.stage.GetPrimAtPath(path)).Bind(
            grip, UsdShade.Tokens.weakerThanDescendants, "physics")
    overview = None
    if args.debug_video:
        from isaacsim.sensors.camera import Camera

        overview = Camera(prim_path="/World/debug_view", resolution=(480, 360))
        overview.initialize()
        overview.set_clipping_range(0.01, 100.0)
        overview.set_focal_length(overview.get_horizontal_aperture() / (2 * np.tan(np.radians(55.0) / 2)))

    kept, attempts, seed = 0, 0, args.first_seed
    max_attempts = args.max_attempts or 3 * args.episodes
    log = []
    kept_arm = {"left": 0, "right": 0}
    kept_low = {True: 0, False: 0}
    n_low = int(round(args.episodes * args.low_cap_frac))
    policy = None
    if args.dagger_port:
        assert args.standoff > 0, "--dagger-port needs the staged expert (--standoff)"
        from openpi_client import image_tools
        from openpi_client import websocket_client_policy

        client = websocket_client_policy.WebsocketClientPolicy(args.dagger_host, args.dagger_port)

        def policy(obs):
            return np.asarray(client.infer(obs)["actions"], dtype=np.float64)

        def resize(img):
            return image_tools.resize_with_pad(img, 224, 224)
    while kept < args.episodes and attempts < max_attempts:
        spec = _spec.sample_scene(seed, ranges, grasp_jitter_m=args.grasp_jitter, cap_jitter_m=args.cap_jitter)
        if args.start_jitter > 0 or args.start_real:
            import dataclasses

            rng_s = np.random.default_rng([seed, 9])
            if args.start_real:
                with open(os.path.join(os.path.dirname(__file__), "..", "assets", "v16_start_offsets.json")) as f:
                    real = json.load(f)["offsets"]
                jit = np.array(real[int(rng_s.integers(len(real)))])
            else:
                jit = rng_s.uniform(-args.start_jitter, args.start_jitter, 14)
            spec = dataclasses.replace(spec, start_offsets=tuple(round(float(v), 5) for v in jit))
        if args.arm and spec.arm != args.arm:
            log.append({"seed": seed, "kept": False, "reason": f"{spec.arm} arm (recording --arm {args.arm} only)"})
            seed += 1
            continue
        if args.balance_arms and kept_arm[spec.arm] >= (args.episodes + 1) // 2:
            log.append({"seed": seed, "kept": False, "reason": f"{spec.arm} arm already has its half (--balance-arms)"})
            seed += 1
            continue
        if args.bottle_y and not args.bottle_y[0] <= spec.bottle_y <= args.bottle_y[1]:
            log.append({"seed": seed, "kept": False,
                        "reason": f"bottle y {spec.bottle_y:.3f} outside --bottle-y "
                                  f"{args.bottle_y[0]:.2f} {args.bottle_y[1]:.2f}"})
            seed += 1
            continue
        low = bool(spec.cap_z < args.low_cap_below)
        if args.low_cap_frac > 0 and kept_low[low] >= (n_low if low else args.episodes - n_low):
            log.append({"seed": seed, "kept": False, "reason": f"{'low' if low else 'high'} cap quota full (--low-cap-frac)"})
            seed += 1
            continue
        expert = LiftBottleExpert(spec, seed, aim_error_m=args.aim_error, bow_m=args.bow, close_ramp_s=args.close_ramp,
                                  orient_jitter_deg=args.orient_jitter, approach_curve_m=args.approach_curve,
                                  speed_jitter=args.speed_jitter, approach_tilt_deg=args.approach_tilt,
                                  standoff_m=args.standoff, insert_speed=tuple(args.insert_speed),
                                  recover_prob=args.recover_prob, observer=args.observer)
        attempts += 1
        if not expert.ik_ok:
            why = "no reachable observer view" if args.observer and expert.q_view is None else "pre-grasp unreachable"
            log.append({"seed": seed, "kept": False, "reason": why})
            seed += 1
            continue
        for i, s in enumerate(("left", "right")):  # this episode's start pose (v16-like jitter)
            scene.q_start[idx["arm"][s]] = expert.q_start[s]
        scene.reset(spec)
        efforts = scene.robot._articulation_view.get_max_efforts()
        for s in ("left", "right"):
            efforts[:, idx["finger"][s]] = FINGER_MAX_FORCE
        scene.robot._articulation_view.set_max_efforts(efforts)

        q_target = scene.q_start.copy()
        if overview is not None:  # close to the grasp, from the outside of the grasping arm
            import isaacsim.core.utils.rotations as rot

            sgn = 1.0 if spec.arm == "left" else -1.0
            eye = np.array([spec.bottle_x + 0.55, spec.bottle_y + sgn * 0.75, spec.table_height + 0.45])
            tgt = np.array([spec.bottle_x - 0.05, spec.bottle_y, spec.table_height + 0.12])
            f = (tgt - eye) / np.linalg.norm(tgt - eye)
            overview.set_world_pose(position=eye, camera_axes="world",
                                    orientation=rot.euler_angles_to_quat(np.array([0.0, -np.arcsin(f[2]), np.arctan2(f[1], f[0])])))
        for _ in range(10):  # first frames after a reset: no image yet, then the RTX denoiser settles
            scene.step(q_target, render=True)
        frames = {"head": [], "left": [], "right": [], "state": [], "cmd": [], "lift": [], "bottle": []}
        t0, t_sim0 = time.time(), scene.world.current_time
        dagger = None
        if policy is not None:
            rng_d = np.random.default_rng([seed, 23])
            dagger = {"mode": str(rng_d.choice(["close", "insert", "early"], p=np.asarray(args.dagger_mix) / sum(args.dagger_mix))),
                      "t_early": float(rng_d.uniform(3.0, 9.0)), "insert_delay": float(rng_d.uniform(0.0, 1.5)),
                      "tick": None, "reason": None, "route": None}
            rtc = SimTimeBlendRTC(policy, horizon=args.horizon, delay_steps=args.delay_steps, blend_steps=5)
            last_cmd = np.concatenate([expert.q_start["left"], expert.q_start["right"], [START_FINGER, START_FINGER]])
            safety = ClientSafety(last_cmd, ClientLimits(binary_close=0.025, binary_open=0.030))
            side_i = 0 if spec.arm == "left" else 1
            standoff_m = float(np.linalg.norm(expert.plan.grasp[:3, 3] - expert.plan.standoff[:3, 3]))
            corridor_since, bottle0 = None, None
        abort = None
        while not expert.done and len(frames["state"]) < args.max_seconds * CONTROL_HZ:
            q = scene.robot.get_joint_positions()
            arms = {s: q[idx["arm"][s]] for s in ("left", "right")}
            state = np.concatenate([arms["left"], arms["right"], [q[idx["finger"]["left"][0]], q[idx["finger"]["right"][0]]],
                                    kin.v16_eef("left", arms["left"]), kin.v16_eef("right", arms["right"])])
            frames["state"].append(state.astype(np.float32))
            # v16's image path: native resolution (head: one 1000x1000 eye of the stereo frame; wrists: the 960x720
            # 4:3 crop of the D435i's 1280x720) -> the camera's JPEG -> downsampled to v16's sizes (head 256, black
            # outside the image circle; wrists 256x192, letterboxed to 256x256).
            size = head_camera.OUT_SIZE
            frames["head"].append((head_camera.camera_frame(rgb(head), (size, size)) * circle).astype(np.uint8))
            for s in ("left", "right"):
                view = camera_models.wrist_view(rgb(wrist[s]), s)
                frames[s].append(letterbox(head_camera.camera_frame(view, head_camera.WRIST_OUT)))
            frames["lift"].append(scene.bottle_lift())
            b_pos, b_quat = scene.bottle.get_world_pose()
            b_pos, b_quat = np.asarray(b_pos, float), np.asarray(b_quat, float)
            frames["bottle"].append(np.concatenate([b_pos, b_quat]).astype(np.float32))

            tick = len(frames["state"]) - 1
            if dagger is not None and dagger["tick"] is None:
                if bottle0 is None:
                    bottle0 = (b_pos.copy(), b_quat.copy())
                reason = None
                moved, rot = expert._bottle_moved((b_pos, b_quat), bottle0)
                if moved > 0.005 or rot > 5.0:
                    reason = "touch"
                elif safety.closed[side_i]:  # (the policy's close, commanded last tick)
                    reason = "close"
                elif tick >= args.dagger_late * CONTROL_HZ:
                    reason = "late"
                elif dagger["mode"] == "early" and tick >= dagger["t_early"] * CONTROL_HZ:
                    reason = "early"
                elif dagger["mode"] == "insert":
                    g = expert.plan.grasp
                    rel = kin.tcp_pose(spec.arm, arms[spec.arm])[:3, 3] - (g[:3, 3] + b_pos - bottle0[0])
                    remaining = -float(rel @ g[:3, 2])
                    lat = float(np.linalg.norm(rel + remaining * g[:3, 2]))
                    if 0.0 < remaining < standoff_m + 0.04 and lat < 0.05:
                        corridor_since = tick if corridor_since is None else corridor_since
                        if tick - corridor_since >= dagger["insert_delay"] * CONTROL_HZ:
                            reason = "insert"
                if reason is not None:
                    dagger.update(tick=tick, reason=reason,
                                  route=expert.takeover(last_cmd, arms[spec.arm], bottle0, (b_pos, b_quat), tick))
                    if dagger["route"] == "grasp" and not args.dagger_keep_grasp:
                        abort = "policy grasped the cap itself (nothing to correct)"
            if dagger is not None and dagger["tick"] is None:  # the policy drives
                def make_obs():
                    return {"observation/state": state.astype(np.float32), "observation/image": resize(frames["head"][-1]),
                            "observation/left_wrist_image": resize(frames["left"][-1]),
                            "observation/right_wrist_image": resize(frames["right"][-1]), "prompt": "grasp the bottle"}
                action, _ = rtc.step(tick, make_obs)
                cmd = safety(tick, action, state[:14])
                last_cmd = cmd
            else:
                cmd = expert.step(q_measured=arms[spec.arm], bottle=(b_pos, b_quat))
            frames["cmd"].append(cmd.astype(np.float32))
            for i, s in enumerate(("left", "right")):
                q_target[idx["arm"][s]] = cmd[7 * i:7 * i + 7]
                q_target[idx["finger"][s]] = cmd[14 + i]
            # One rendered world step advances a full rendering_dt (1/15 s = `substeps` physics steps).
            scene.step(q_target, render=True)
            if overview is not None:
                frames.setdefault("overview", []).append(overview.get_rgb()[..., :3])
            if abort:
                break

        success = expert.done and scene.success() and not abort
        n = len(frames["state"])
        dt_sim = (scene.world.current_time - t_sim0) / max(n, 1)
        aim_mm = None if expert.plan.aim_error is None else (expert.plan.aim_error * 1000).round(1).tolist()
        entry = {"seed": seed, "kept": bool(success), "frames": n, "arm": spec.arm, "bottle": spec.bottle_asset, "aim_error_mm": aim_mm,
                 "final_lift_cm": round(scene.bottle_lift() * 100, 1), "phase_ticks": expert.phase_ticks,
                 "wall_s": round(time.time() - t0, 1), "sim_dt": round(dt_sim, 4)}
        if "close" in expert.phase_ticks and expert.phase_ticks["close"] < n:
            k = expert.phase_ticks["close"]
            entry["finger_at_lift_mm"] = round(float(frames["state"][k][14 if spec.arm == "left" else 15]) * 1000, 1)
            entry["half_width_mm"] = round(spec.bottle_width * 500, 1)
        if "approach" in expert.phase_ticks and expert.phase_ticks["approach"] < n:
            k, j0 = expert.phase_ticks["approach"], (0 if spec.arm == "left" else 7)
            tcp = kin.tcp_pose(spec.arm, frames["state"][k][j0:j0 + 7])
            entry["tcp_err_at_grasp_mm"] = round(float(np.linalg.norm(tcp[:3, 3] - expert.cur_grasp[:3, 3])) * 1000, 1)
            # how far the bottle was pushed sideways before the close command (any phase), and its rotation
            b = np.stack(frames["bottle"][:k + 1])
            entry["bottle_shift_before_close_mm"] = round(float(np.linalg.norm(b[:, :2] - b[0, :2], axis=1).max()) * 1000, 1)
            dots = np.clip(np.abs(b[:, 3:] @ b[0, 3:]), 0.0, 1.0)
            entry["bottle_rot_before_close_deg"] = round(float(np.degrees(2 * np.arccos(dots)).max()), 1)
        if expert.staged:
            entry["staged"] = {**expert.log, "retries": expert.retries, "insert_speed": round(expert.insert_speed, 3),
                               "events": expert.events}
        if "move_pregrasp" in expert.phase_ticks and "approach" in expert.phase_ticks:
            # Sideways deviation from the straight approach line (what makes a fingertip hit the bottle),
            # for the commanded and the measured fingertip; and the lag along the line at the end.
            k0, k1, j0 = expert.phase_ticks["move_pregrasp"], expert.phase_ticks["approach"], (0 if spec.arm == "left" else 7)
            p0, p1 = expert.plan.pregrasp[:3, 3], expert.plan.grasp[:3, 3]
            u = (p1 - p0) / np.linalg.norm(p1 - p0)
            def lateral(q_arm):
                d = kin.tcp_pose(spec.arm, q_arm)[:3, 3] - p0
                return float(np.linalg.norm(d - (d @ u) * u))
            entry["approach_lateral_cmd_mm"] = round(1000 * max(lateral(frames["cmd"][k][j0:j0 + 7]) for k in range(k0, min(k1, n))), 1)
            entry["approach_lateral_meas_mm"] = round(1000 * max(lateral(frames["state"][k][j0:j0 + 7]) for k in range(k0, min(k1 + 1, n))), 1)
        st, cm = np.stack(frames["state"]), np.stack(frames["cmd"])
        track = np.abs(cm[:-1, :14] - st[1:, :14])  # command at k vs measured at k+1
        entry["max_track_err_rad"] = round(float(track.max()), 3)
        entry["worst_joint"] = kin.arm_names("left")[int(track.max(0).argmax()) % 7].replace("left", "left" if track.max(0).argmax() < 7 else "right")
        if dagger is not None:
            entry["dagger"] = {k: dagger[k] for k in ("mode", "reason", "tick", "route")}
            entry["cap_z"] = spec.cap_z
        if args.observer:  # where the observer camera points once it should have arrived (+0.7 s), and at the close
            k_view = min(int(expert.obs_move_s * CONTROL_HZ) + 10, n - 1)
            k_close = min(expert.phase_ticks.get("approach", n - 1), n - 1)
            o = 0 if expert.obs_side == "left" else 7
            err_view = expert.view_error_deg(frames["state"][k_view][o:o + 7])
            entry["observer"] = {**expert.obs_log, "side": expert.obs_side, "err_deg_arrived": round(err_view, 1),
                                 "err_deg_at_close": round(expert.view_error_deg(frames["state"][k_close][o:o + 7]), 1)}
            # Diagnostics: first tick the observer lags its command by > 0.08 rad (blocked), the grasping arm's phase
            # then, and the closest the two hands (fingertip TCP / flange) come to each other.
            st_o = np.stack(frames["state"])[1:, o:o + 7]
            lag = np.abs(np.stack(frames["cmd"])[:-1, o:o + 7] - st_o).max(axis=1)
            k_lag = int(np.argmax(lag > 0.08)) + 1 if (lag > 0.08).any() else None
            phase_at = None
            if k_lag is not None:
                phase_at = "move_pregrasp"
                for t_ev, ph in expert.events:
                    if t_ev <= k_lag:
                        phase_at = ph
            g = 0 if spec.arm == "left" else 7
            def pts(side, q):
                names = dict(zip(kin.arm_names(side), q, strict=True))
                return [kin.fk(f"{side}_fr3_{lk}", "base_link", names)[:3, 3] for lk in ("hand_tcp", "link8", "link6", "link4")]
            dmin = min(min(np.linalg.norm(a - b) for a in pts(expert.obs_side, frames["state"][k][o:o + 7])
                           for b in pts(spec.arm, frames["state"][k][g:g + 7])) for k in range(0, n, 3))
            entry["observer"].update(lag_tick=k_lag, lag_phase=phase_at, min_hand_dist_mm=round(dmin * 1000))
            if success and err_view > args.observer_max_err:
                success = entry["kept"] = False
                abort = f"observer camera {err_view:.0f} deg off the cap (blocked)"
        # An arm that does not go where it was told did not demonstrate what the recorded actions say it did, so the
        # episode teaches a state the policy cannot reproduce. Episodes separate cleanly: over the v7b/c/d/e smokes
        # the clean ones stay under 0.059 rad and the bad ones start at 0.092. The original v7 training set, recorded
        # with no such check, carried 22 of 83 episodes over this threshold, the worst at 0.565 rad -- a quarter of
        # the demonstrations had an arm somewhere other than where the action said. v5/v6/v6a are clean (2-5%).
        #
        # The failures that remain (7-10% of episodes, always the observer arm, always joint 2) are a CONSTANT offset
        # of about 0.2 rad held for seconds and released when the arm next moves away, which is a blockage rather
        # than a tracking lag. What blocks it is NOT established. Ruled out with measurements on 2026-11-13, so as
        # not to be re-tested: the arrival sync (identical lag to the tick before and after it), the retreat speed
        # (unchanged once rate-limited), the robot's own body (the stalled poses sit 391-448 mm from the torso mesh
        # while 24 of 28 clean ones are closer), joint limits (joint 2 at -0.73 of a +-1.84 range) and joint-2
        # gravity torque (16 of 28 clean episodes carry a larger moment). Identifying it needs Isaac contact
        # reporting on the observer's links, which nobody has run yet; until then this check is what keeps those
        # episodes out of the data.
        if success and args.max_track_err > 0 and entry["max_track_err_rad"] > args.max_track_err:
            success = entry["kept"] = False
            abort = f"{entry['worst_joint']} {entry['max_track_err_rad']:.3f} rad behind its command (blocked or too fast)"
        if abort:
            entry["reason"] = abort
        elif dagger is not None and dagger["tick"] is None:
            entry["reason"] = "no takeover"
        elif not expert.done:
            entry["reason"] = "timeout"
        elif not success:
            entry["reason"] = f"bottle not lifted ({entry['final_lift_cm']} cm)"
        log.append(entry)
        json.dump(log, open(os.path.join(args.out, "record_log.json"), "w"), indent=1)  # after every episode: crash-safe
        if overview is not None and frames.get("overview"):
            from PIL import Image

            imgs = [Image.fromarray(np.asarray(f)) for f in frames["overview"][::2]]
            tag = "keep" if success else "drop"
            imgs[0].save(os.path.join(args.out, f"debug_seed{seed:05d}_{tag}.gif"), save_all=True, append_images=imgs[1:],
                         duration=int(2000 / CONTROL_HZ), loop=0)
        print(f"[demo] seed {seed} {spec.arm:5s} {spec.bottle_asset:22s} {n} frames, lift {entry['final_lift_cm']:+.1f} cm "
              f"-> {'KEEP' if success else 'drop: ' + entry.get('reason', '')} ({entry['wall_s']} s, sim dt {dt_sim:.4f})"
              + ("" if dagger is None else f" | takeover {dagger['mode']}/{dagger['reason']} at {dagger['tick']} -> {dagger['route']}"
                 f" {expert.log.get('takeover', {})} retries {expert.retries}"), flush=True)
        if success:
            np.savez_compressed(
                os.path.join(args.out, f"episode_{kept:06d}.npz"),
                head=np.stack(frames["head"]), left_wrist=np.stack(frames["left"]), right_wrist=np.stack(frames["right"]),
                state=np.stack(frames["state"]), cmd=np.stack(frames["cmd"]), lift=np.array(frames["lift"], np.float32),
                bottle=np.stack(frames["bottle"]),
                meta=json.dumps({"spec": json.loads(spec.to_json()), "phase_ticks": expert.phase_ticks, "fps": CONTROL_HZ, "aim_error_mm": aim_mm,
                                 "wrist_camera": args.wrist_camera,
                                 "task": "grasp the bottle", "staged": entry.get("staged"), "dagger": entry.get("dagger"),
                                 "takeover_tick": None if dagger is None else dagger["tick"],
                                 "observer": entry.get("observer")}),
            )
            kept += 1
            kept_arm[spec.arm] += 1
            kept_low[low] += 1
        seed += 1
    json.dump(log, open(os.path.join(args.out, "record_log.json"), "w"), indent=1)
    rate = kept / max(attempts, 1)
    print(f"[demo] kept {kept} of {attempts} attempts ({100 * rate:.0f}%) -> {args.out}", flush=True)
    app.close()


main()
