"""Per-episode grasp facts of the real v16 demos, used as the reference distribution for the simulated demos.

    ~/sim/mujoco_venv/bin/python scripts/extract_v16_grasps.py

From states and actions (no video): grasping arm (the gripper that closes), holding onset k (action column 17), the
fingertip centre (kinematics.tcp_pose, base_link, lift at 0) and its approach direction at k, the finger opening at
k, time to k, time from k until the fingertip has risen 2 cm, lift height (fingertip z rise after k) and time from k
to the episode end. Writes assets/v16_grasps.json.
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from garmi_bench import kinematics as kin  # noqa: E402

V16 = os.path.expanduser("~/garmi_vlapi/data/garmi_grasp_bottle_v16")
OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "v16_grasps.json")
FPS = 15.0


def main():
    grasps = []
    for f in sorted(glob.glob(f"{V16}/data/chunk-*/episode_*.parquet")):
        df = pd.read_parquet(f, columns=["observation.state", "action"])
        s, a = np.stack(df["observation.state"].to_numpy()), np.stack(df["action"].to_numpy())
        hold = a[:, 17] > 0.5
        if not hold.any():
            continue
        k = int(np.argmax(hold))
        side = "left" if s[0, 14] - s[k, 14] > s[0, 15] - s[k, 15] else "right"
        j0, fi = (0, 14) if side == "left" else (7, 15)
        tcp = [kin.tcp_pose(side, q[j0:j0 + 7]) for q in s]
        p = np.array([t[:3, 3] for t in tcp])
        z = tcp[k][:3, 2]
        rise = p[k:, 2] - p[k, 2]
        lifted = np.flatnonzero(rise > 0.02)
        k_peak = k + int(np.argmax(rise))
        grasps.append({
            "episode": int(os.path.basename(f)[8:14]), "arm": side, "frames": len(s),
            "x": round(float(p[k, 0]), 4), "y": round(float(p[k, 1]), 4), "z": round(float(p[k, 2]), 4),
            "rotation": np.round(tcp[k][:3, :3], 5).tolist(),  # fingertip frame in base_link at the grasp
            "q_grasp": np.round(s[k, j0:j0 + 7], 5).tolist(),  # the arm's joints at the grasp
            "q_lift_peak": np.round(s[k_peak, j0:j0 + 7], 5).tolist(),  # ... at the top of the lift
            "t_peak_s": round((k_peak - k) / FPS, 3),  # from the grasp to the top of the lift
            "approach_az_deg": round(float(np.degrees(np.arctan2(z[1], z[0]))), 2),
            "approach_el_deg": round(float(np.degrees(np.arcsin(np.clip(z[2], -1, 1)))), 2),
            "finger_at_hold_m": round(float(s[k, fi]), 4), "t_grasp_s": round(k / FPS, 3),
            "t_lift_start_s": round(int(lifted[0]) / FPS, 3) if len(lifted) else None,
            "lift_m": round(float(rise.max()), 4), "t_after_grasp_s": round((len(s) - k) / FPS, 3),
        })
    with open(OUT, "w") as f:
        json.dump({"source": V16, "grasps": grasps}, f, indent=1)
    for arm in ("left", "right"):
        g = [x for x in grasps if x["arm"] == arm]
        print(f"[v16] {arm}: {len(g)} grasps, x {np.median([x['x'] for x in g]):.3f}, y {np.median([x['y'] for x in g]):+.3f}, "
              f"z {np.median([x['z'] for x in g]):.3f}, el {np.median([x['approach_el_deg'] for x in g]):.1f} deg, "
              f"lift {np.median([x['lift_m'] for x in g]):.3f} m", flush=True)
    print(f"[v16] {len(grasps)} grasps -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
