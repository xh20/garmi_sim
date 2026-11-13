"""Kinematic reachability of bottle positions on the table, per arm (NumPy only, no simulator).

    ~/isaac/.venv/bin/python scripts/reach_map.py

A position (x, y) at table height h is reachable for an arm if damped-least-squares IK finds all three
expert poses -- pre-grasp, grasp, and lift (+25 cm) -- with the nominal real-demo grasp orientation
(no random offsets), within 3 mm / 3 deg, with every joint at least 0.1 rad inside its limits.
Output: assets/reach_map.json and assets/reach_map.png.
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from garmi_bench import expert as E  # noqa: E402
from garmi_bench import kinematics as K  # noqa: E402
from garmi_bench.spec import START_JOINTS  # noqa: E402

XS = np.round(np.arange(0.70, 1.451, 0.04), 3)
YS = np.round(np.arange(-0.74, 0.901, 0.04), 3)
HEIGHTS = (0.81, 0.85, 0.89)
GRASP_FRACTION = 0.42  # of a 0.22 m bottle
BOTTLE_H = 0.22
OUT = os.path.join(os.path.dirname(__file__), "..", "assets")

root = ET.parse(K.URDF).getroot()
LIMITS = {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
          for j in root.findall("joint") if j.find("limit") is not None and "fr3_joint" in j.get("name")}


def solve(side, q0, target, q_rest, iters=150):
    q = q0.copy()
    for _ in range(iters):
        dq, err = K.ik_step(side, q, target, q_rest, null_gain=0.05)
        q = q + np.clip(dq, -0.2, 0.2)
        if np.linalg.norm(err[:3]) < 1e-3 and np.linalg.norm(err[3:]) < 1e-2:
            break
    return q, np.linalg.norm(err[:3]), np.degrees(np.linalg.norm(err[3:]))


def reachable(side, x, y, h):
    shoulder = K.fk(f"{side}_fr3_link0", "base_link", {})[:3, 3]
    az = np.arctan2(y - shoulder[1], x - shoulder[0]) + np.radians(-45.0 if side == "left" else 55.0)
    rot = E.tcp_rotation(az, -np.radians(22.0))
    grasp = np.eye(4)
    grasp[:3, :3], grasp[:3, 3] = rot, [x, y, h + BOTTLE_H * GRASP_FRACTION]
    pre, lift = grasp.copy(), grasp.copy()
    pre[:3, 3] -= rot[:, 2] * 0.125
    lift[2, 3] += 0.25
    q_rest = np.array(START_JOINTS[side])
    q, ok = q_rest, True
    for target in (pre, grasp, lift):
        q, ep, er = solve(side, q, target, q_rest)
        lim_ok = all(LIMITS[n][0] + 0.1 <= v <= LIMITS[n][1] - 0.1 for n, v in zip(K.arm_names(side), q, strict=True))
        ok = ok and ep < 3.0e-3 and er < 3.0 and lim_ok
        if not ok:
            break
    return ok


def _cell(task):
    return reachable(*task)


def main():
    import multiprocessing

    result = {"xs": XS.tolist(), "ys": YS.tolist(), "heights": list(HEIGHTS), "maps": {}}
    tasks = [(side, float(x), float(y), h) for side in ("left", "right") for h in HEIGHTS for x in XS for y in YS]
    with multiprocessing.Pool(max(1, os.cpu_count() - 2)) as pool:
        flags = iter(pool.map(_cell, tasks, chunksize=16))
    for side in ("left", "right"):
        for h in HEIGHTS:
            grid = np.array([[next(flags) for _ in YS] for _ in XS], dtype=bool)
            result["maps"][f"{side}@{h}"] = grid.astype(int).tolist()
            print(f"[reach] {side:5s} table {h:.2f} m: {grid.sum()} of {grid.size} cells reachable", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(result, open(os.path.join(OUT, "reach_map.json"), "w"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(HEIGHTS), figsize=(5 * len(HEIGHTS), 5.5), sharey=True)
    for ax, h in zip(axes, HEIGHTS, strict=True):
        lft = np.array(result["maps"][f"left@{h}"]); rgt = np.array(result["maps"][f"right@{h}"])
        img = np.zeros(lft.shape + (3,))
        img[..., 0] = rgt; img[..., 2] = lft; img[(lft + rgt) == 2] = (0.6, 0.2, 0.8)
        ext = [YS[0], YS[-1], XS[0], XS[-1]]
        ax.imshow(img, origin="lower", extent=ext, aspect="equal")
        ax.invert_xaxis()
        ax.set_title(f"table {h:.2f} m  (red right, blue left, purple both)")
        ax.set_xlabel("y [m] (left +)")
    axes[0].set_ylabel("x [m] (forward)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "reach_map.png"), dpi=90)


main()
