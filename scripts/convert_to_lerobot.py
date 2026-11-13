"""Convert raw simulated episodes (record_demos.py .npz) into a LeRobot v2.1 dataset laid out like v16.

Run with the openpi environment (its lerobot writes codebase v2.1):
    HF_LEROBOT_HOME=~/sim/demos/lerobot ~/garmi_vlapi/.venv/bin/python scripts/convert_to_lerobot.py \
        --raw ~/sim/demos/check20 --repo-id garmi/garmi_sim_lift_bottle_check20

Matches fabiomanz/garmi_grasp_bottle_v16 (feature names, 15 fps, 256x256 videos, 34-dim state, 36-dim
action) and its action definition (conversion_report.json):
  action[k] = state[k+1] (last repeated); arm columns 0-13 then zero-phase Gaussian low-pass at -3 dB / 2 Hz;
  grippers unfiltered; then progress, holding, and the next state's EEF poses.
progress: linear in time from 0 at the first frame to 1 when the bottle is lifted (>= 2 cm), 1 afterwards.
holding: 1 from the frame the grasping fingers clamp (after the close command: closed by >= 3 mm, then stopped
short of fully closed), 0 before.
--aux-targets appends 8 gripper-centric targets (action dims 36-43), exact from the simulator:
  target_{left,right}_{x,y,z}: the arm's next target point in its fingertip TCP frame (m; z = the approach axis).
    Grasping arm: the cap's grasp point (moving with the bottle) until the fingers clamp, then the place where the
    held grasp point ends up at the end of the episode. The other arm has no target: 0.
  between_{left,right}: 1 while the cap's grasp point is between that arm's fingertips (inside BETWEEN_BOX_M around
    the fingertip centre, in the fingertip frame), else 0.
--aux16 instead appends 16 (action dims 36-51; with the 16 commands they fill pi0.5's 32 action dims). Per arm the
  arm that does not grasp gets 0 everywhere; the target is defined as for --aux-targets:
  target_{left,right}_{du,dv}: where the target appears in that arm's wrist image relative to the fingertip centre,
    as fractions of the image width / height (camera_models, with the camera model the episode was recorded with;
    clipped to +-1; 0, 0 = the target is on the fingertip centre's line of sight).
  target_{left,right}_z: the target's distance along the approach axis (fingertip TCP frame, m).
  between_{left,right}, holding_{left,right}: cap between the fingertips (as above); the fingers clamp an object.
  phase: the expert's phase (meta.staged.events) / 6: move 0, curve 1, align 2, insert 3, close 4, lift 5, hold 6.
    A recovery's retreat counts as align, with phase_progress running back from 1 to 0 (the graph's back edge).
  phase_progress: 0 -> 1 through the current phase.
  axis_{left,right}_{x,y}: x and y of the bottle's axis (unit vector, towards the cap) in the fingertip TCP frame.
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

parser = argparse.ArgumentParser()
parser.add_argument("--raw", required=True, nargs="+",
                    help="raw episode folders, as split=path (e.g. train=raw_train val=raw_val); splits in order of first "
                         "appearance, episodes within a split ordered by seed")
parser.add_argument("--max-per-arm", nargs="*", default=[],
                    help="split=N: keep at most N episodes per arm in that split (lowest seeds first), e.g. train=100")
parser.add_argument("--exclude-route", nargs="*", default=[],
                    help="drop DAgger episodes whose takeover took one of these routes (expert.takeover), e.g. reopen")
parser.add_argument("--aux-targets", action="store_true",
                    help="append the per-arm target point and cap-between-fingers labels as action dims 36-43")
parser.add_argument("--aux16", action="store_true",
                    help="append the 16 auxiliary labels (image-space target, phase, holding, bottle axis) as dims 36-51")
parser.add_argument("--repo-id", required=True)
parser.add_argument("--v16-meta", default=os.path.expanduser("~/garmi_vlapi/data/garmi_grasp_bottle_v16/meta"))
args = parser.parse_args()

import functools  # noqa: E402

import lerobot.common.datasets.lerobot_dataset as _lerobot_dataset  # noqa: E402
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

# Videos like v16's: h264 (v16: High profile, yuv420p, ~1.1 Mbit/s at 256x256 15 fps) instead of LeRobot's AV1
# default; CRF 18 lands in that bitrate range.
VIDEO_CODEC, VIDEO_CRF = "h264", 18
_lerobot_dataset.encode_video_frames = functools.partial(_lerobot_dataset.encode_video_frames, vcodec=VIDEO_CODEC, crf=VIDEO_CRF)

FPS = 15
SIGMA_FRAMES = 0.8326 / (2 * np.pi * 2.0) * FPS  # Gaussian with -3 dB at 2 Hz: sigma = 0.0663 s ~ 0.99 frames
LIFTED_M = 0.02
AUX_NAMES = ["target_left_x", "target_left_y", "target_left_z", "target_right_x", "target_right_y", "target_right_z",
             "between_left", "between_right"]
# |x| across the fingers, |y| along the closing direction (the closing fingers centre the cap), |z| along the approach
# axis (cap height + neck). Contains 219 of the 220 clamps of v6 (10 mm in every axis: 195).
BETWEEN_BOX_M = np.array([0.012, 0.020, 0.020])
AUX16_NAMES = ["target_left_du", "target_left_dv", "target_left_z", "target_right_du", "target_right_dv",
               "target_right_z", "between_left", "between_right", "holding_left", "holding_right", "phase",
               "phase_progress", "axis_left_x", "axis_left_y", "axis_right_x", "axis_right_y"]
PHASES = {"move": 0, "curve": 1, "align": 2, "approach": 3, "retreat": 2, "close": 4, "lift": 5, "hold": 6, "done": 6}
N_PHASES = 6  # largest phase index (phase label = index / N_PHASES)


def v16_features() -> dict:
    info = json.load(open(os.path.join(args.v16_meta, "info.json")))
    f = info["features"]
    feats = {k: {"dtype": f[k]["dtype"], "shape": tuple(f[k]["shape"]), "names": f[k]["names"]}
             for k in ("observation.state", "action", "observation.images.head", "observation.images.left_wrist",
                       "observation.images.right_wrist")}
    for k in ("observation.images.head", "observation.images.left_wrist", "observation.images.right_wrist"):
        feats[k]["dtype"] = "video"
    if args.aux_targets and args.aux16:
        raise SystemExit("--aux-targets and --aux16 are alternatives")
    names = AUX_NAMES if args.aux_targets else AUX16_NAMES if args.aux16 else []
    if names:
        a = feats["action"]
        a["shape"] = (a["shape"][0] + len(names),)
        a["names"] = list(a["names"]) + names if isinstance(a["names"], list) else a["names"]
    return feats


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def aux_targets(state: np.ndarray, bottle: np.ndarray, spec: dict, k_clamp: int) -> np.ndarray:
    """(n, 8) labels, see the module docstring. bottle: (n, 7) world position + wxyz quaternion per frame."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from garmi_bench import kinematics  # only needed with --aux-targets

    n, arm = len(state), spec["arm"]
    # The expert's grasp point at the start (spec), carried along with the bottle's pose (translation and tilt).
    grasp0 = np.array([spec["bottle_x"], spec["bottle_y"], spec["cap_z"]])
    offset = _quat_wxyz_to_rot(bottle[0, 3:]).T @ (grasp0 - bottle[0, :3])
    cap = np.array([bottle[k, :3] + _quat_wxyz_to_rot(bottle[k, 3:]) @ offset for k in range(n)])
    q = state[:, :7] if arm == "left" else state[:, 7:14]
    tcp = np.array([kinematics.tcp_pose(arm, qk) for qk in q])
    target = np.where(np.arange(n)[:, None] < k_clamp, cap, cap[-1])
    rel_target = np.einsum("kji,kj->ki", tcp[:, :3, :3], target - tcp[:, :3, 3])
    rel_cap = np.einsum("kji,kj->ki", tcp[:, :3, :3], cap - tcp[:, :3, 3])
    between = np.all(np.abs(rel_cap) < BETWEEN_BOX_M, axis=1)
    out = np.zeros((n, len(AUX_NAMES)), np.float32)
    i = 0 if arm == "left" else 1
    out[:, 3 * i:3 * i + 3] = rel_target
    out[:, 6 + i] = between
    return out


def phase_labels(events: list, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(phase / N_PHASES, progress through the phase) per frame from the expert's [[tick, name], ...] events."""
    segs = [(0, "move")] + [(int(t), name) for t, name in events]
    phase, progress = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for j, (t0, name) in enumerate(segs):
        t1 = segs[j + 1][0] if j + 1 < len(segs) else n
        if t1 <= t0 or t0 >= n:
            continue
        t1 = min(t1, n)
        frac = np.arange(t1 - t0) / max(t1 - t0, 1)
        phase[t0:t1] = PHASES[name] / N_PHASES
        progress[t0:t1] = 1.0 if name == "done" else 1.0 - frac if name == "retreat" else frac
    return phase, progress


def aux16(state: np.ndarray, bottle: np.ndarray, meta: dict, k_clamp: int) -> np.ndarray:
    """(n, 16) labels, see the module docstring."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from garmi_bench import camera_models  # only needed with --aux16
    from garmi_bench import kinematics

    spec, n = meta["spec"], len(state)
    arm, i = spec["arm"], 0 if spec["arm"] == "left" else 1
    targets8 = aux_targets(state, bottle, spec, k_clamp)  # target in the TCP frame, between
    rel = targets8[:, 3 * i:3 * i + 3].astype(np.float64)
    # The wrist camera is rigid with the fingertip TCP: its optical frame relative to the TCP, from the URDF.
    q0 = dict(zip(kinematics.arm_names(arm), state[0, 7 * i:7 * i + 7], strict=True))
    t_cam_tcp = np.linalg.inv(kinematics.fk(f"camera_d435i_{arm}_color_optical_frame", "base_link", q0)) @ \
        kinematics.fk(f"{arm}_fr3_hand_tcp", "base_link", q0)
    camera_models.set_wrist_model(meta.get("wrist_camera", "legacy"))  # recorded before 2026-11-11: legacy
    fx, fy, cx, cy = camera_models.wrist_intrinsics(arm)
    w, h = camera_models.wrist_size()

    def pixel(p_tcp: np.ndarray) -> np.ndarray:  # (k, 3) TCP frame -> (k, 2) fractions of the image size
        p = p_tcp @ t_cam_tcp[:3, :3].T + t_cam_tcp[:3, 3]
        z = np.maximum(p[:, 2], 0.02)  # targets at or behind the camera plane end up clipped at the image border
        return np.stack([(fx * p[:, 0] / z + cx) / w, (fy * p[:, 1] / z + cy) / h], axis=1)

    duv = np.clip(pixel(rel) - pixel(np.zeros((1, 3))), -1.0, 1.0)
    q = state[:, 7 * i:7 * i + 7]
    axis = np.array([kinematics.tcp_pose(arm, qk)[:3, :3].T @ _quat_wxyz_to_rot(bottle[k, 3:])[:, 2]
                     for k, qk in enumerate(q)])
    out = np.zeros((n, len(AUX16_NAMES)), np.float32)
    out[:, 3 * i:3 * i + 2] = duv
    out[:, 3 * i + 2] = rel[:, 2]
    out[:, 6 + i] = targets8[:, 6 + i]
    out[k_clamp:, 8 + i] = 1.0
    out[:, 10], out[:, 11] = phase_labels(meta["staged"]["events"], n)
    out[:, 12 + 2 * i:14 + 2 * i] = axis[:, :2]
    return out


def actions_v16(state: np.ndarray, lift: np.ndarray, arm: str, close_start: int) -> np.ndarray:
    n = len(state)
    nxt = np.concatenate([state[1:], state[-1:]], axis=0)
    a = np.zeros((n, 36), np.float32)
    a[:, :14] = gaussian_filter1d(nxt[:, :14], sigma=SIGMA_FRAMES, axis=0, mode="nearest")
    a[:, 14:16] = nxt[:, 14:16]
    lifted = np.flatnonzero(lift >= LIFTED_M)
    k_done = int(lifted[0]) if len(lifted) else n - 1
    a[:, 16] = np.clip(np.arange(n) / max(k_done, 1), 0.0, 1.0)
    finger = state[:, 14 if arm == "left" else 15]
    open_at_close = finger[min(close_start, n - 1)]
    k_clamp = min(close_start + 3, n - 1)
    for k in range(close_start + 1, n):  # first frame after the close command where the fingers closed, then stopped
        if finger[k] < open_at_close - 0.003 and abs(finger[k] - finger[k - 1]) < 5e-4 and finger[k] > 0.002:
            k_clamp = k
            break
    a[k_clamp:, 17] = 1.0
    a[:, 18:36] = nxt[:, 16:34]
    return a


def main():
    feats = v16_features()
    root = os.path.join(os.environ.get("HF_LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot")), args.repo_id)
    if os.path.exists(root):
        shutil.rmtree(root)
    ds = LeRobotDataset.create(args.repo_id, fps=FPS, robot_type="garmi_bimanual", features=feats, use_videos=True,
                               image_writer_threads=8)
    sources = [r.split("=", 1) if "=" in r else ("train", r) for r in args.raw]
    caps = {s: int(n) for s, n in (c.split("=", 1) for c in args.max_per_arm)}
    by_split, generation, excluded = {}, {}, []
    for split, raw in sources:
        raw = os.path.expanduser(raw)
        for path in sorted(glob.glob(os.path.join(raw, "episode_*.npz"))):
            meta = json.loads(str(np.load(path)["meta"]))
            spec, route = meta["spec"], (meta.get("dagger") or {}).get("route")
            if route is not None and route in args.exclude_route:
                excluded.append({"split": split, "seed": spec["seed"], "arm": spec["arm"], "raw": os.path.abspath(path),
                                 "reason": f"--exclude-route {route}"})
                continue
            by_split.setdefault(split, []).append((spec["seed"], spec["arm"], path))
        gen = os.path.join(raw, "generation.json")
        if os.path.exists(gen):
            generation.setdefault(split, {})[os.path.basename(raw)] = json.load(open(gen))
    episodes = []
    for split, eps in by_split.items():
        count = {}
        for seed, arm, path in sorted(eps):
            count[arm] = count.get(arm, 0) + 1
            if split in caps and count[arm] > caps[split]:
                excluded.append({"split": split, "seed": seed, "arm": arm, "raw": os.path.abspath(path),
                                 "reason": f"over --max-per-arm {split}={caps[split]}"})
            else:
                episodes.append((split, path))
    splits, provenance = {}, []
    for index, (split, path) in enumerate(episodes):
        splits.setdefault(split, []).append(index)
        ep = np.load(path)
        meta = json.loads(str(ep["meta"]))
        provenance.append({"episode_index": index, "split": split, "seed": meta["spec"]["seed"], "arm": meta["spec"]["arm"],
                           "bottle": meta["spec"]["bottle_asset"], "raw": os.path.abspath(path)})
        state, lift = ep["state"].astype(np.float32), ep["lift"]
        arm = meta["spec"]["arm"]
        # The close command starts when the approach phase ends (expert.phase_ticks["approach"]).
        act = actions_v16(state, lift, arm, meta["phase_ticks"].get("approach", len(state) - 1))
        if args.aux_targets or args.aux16:
            k_clamp = int(np.argmax(act[:, 17])) if act[:, 17].any() else len(state)
            aux = aux16(state, ep["bottle"], meta, k_clamp) if args.aux16 else \
                aux_targets(state, ep["bottle"], meta["spec"], k_clamp)
            act = np.concatenate([act, aux], axis=1)
        # each ep[key] access decompresses the whole array again, so read every image array once
        head, left, right = ep["head"], ep["left_wrist"], ep["right_wrist"]
        # DAgger episodes (record_demos.py --dagger-port): a policy drove until takeover_tick; only the expert's part is
        # trained on (actions and progress computed on the whole episode, so they are continuous at the cut).
        k0 = meta.get("takeover_tick") or 0
        if k0:
            state, act, head, left, right = state[k0:], act[k0:], head[k0:], left[k0:], right[k0:]
            provenance[-1]["takeover_tick"] = k0
        for k in range(len(state)):
            ds.add_frame({"observation.state": state[k], "action": act[k], "observation.images.head": head[k],
                          "observation.images.left_wrist": left[k], "observation.images.right_wrist": right[k],
                          "task": meta["task"]})
        ds.save_episode()
        print(f"[convert] {index:3d} {split:5s} seed {meta['spec']['seed']}: {len(state)} frames, arm {arm}, "
              f"holding from {int(np.argmax(act[:, 17]))}, progress 1 at {int(np.argmax(act[:, 16] >= 1.0))}", flush=True)
    shutil.copy(os.path.join(args.v16_meta, "modality.json"), os.path.join(root, "meta", "modality.json"))
    with open(os.path.join(root, "meta", "splits.json"), "w") as f:
        json.dump(splits, f)
    with open(os.path.join(root, "meta", "sim_source.json"), "w") as f:
        json.dump({"episodes": provenance, "excluded": excluded, "command": " ".join(["python"] + sys.argv),
                   "generation": generation, "action_definition": __doc__}, f, indent=1)
    check_fps(root, len(episodes))
    print(f"[convert] {len(episodes)} episodes ({ {k: len(v) for k, v in splits.items()} }) -> {root}", flush=True)


def check_fps(root: str, n_episodes: int) -> None:
    """Every video has exactly one frame per data row, 1/FPS apart."""
    import av
    import pandas as pd

    for i in range(n_episodes):
        rows = len(pd.read_parquet(os.path.join(root, f"data/chunk-000/episode_{i:06d}.parquet"), columns=["timestamp"]))
        for cam in ("head", "left_wrist", "right_wrist"):
            with av.open(os.path.join(root, f"videos/chunk-000/observation.images.{cam}/episode_{i:06d}.mp4")) as c:
                stream = c.streams.video[0]
                ts = [float(fr.pts * stream.time_base) for fr in c.decode(video=0)]
            dt = np.diff(ts)
            if len(ts) != rows or not np.allclose(dt, 1.0 / FPS, atol=1e-3):
                raise SystemExit(f"episode {i} {cam}: {len(ts)} frames for {rows} rows, frame spacing {dt.min():.4f}-{dt.max():.4f} s")
    print(f"[convert] fps check: all {n_episodes} x 3 videos have one frame per row at {FPS} fps", flush=True)


main()
