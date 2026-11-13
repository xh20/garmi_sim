"""Grasp-success map: run the scripted expert in Isaac Sim (physics only) at kinematically reachable
bottle positions on the nominal table, per arm.

    OMNI_KIT_ACCEPT_EULA=YES ~/isaac/.venv/bin/python scripts/grasp_map.py --step 0.05 --trials 2

Catches what kinematics cannot: the arm hitting GARMI's own torso or tablet, the hand knocking the bottle
over on the approach, the bottle slipping. Uses the nominal table (0.85 m) and one typical bottle scaled to
70 mm; each trial varies yaw and the expert's grasp offsets (different seeds). Output: assets/grasp_map.json
(success fraction per cell and arm) and assets/grasp_map.png.
"""

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--step", type=float, default=0.05)
parser.add_argument("--trials", type=int, default=2)
parser.add_argument("--bottle", default="001_bottle_0.usda")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402

from garmi_bench import kinematics as kin  # noqa: E402
from garmi_bench import spec as _spec  # noqa: E402
from garmi_bench.expert import LiftBottleExpert  # noqa: E402
from garmi_bench.scene import LiftBottleScene  # noqa: E402

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
SUBSTEPS = 8  # physics steps per 15 Hz control tick when not rendering
FINGER_MAX_FORCE = 70.0


def run_trial(scene, idx, spec, seed):
    expert = LiftBottleExpert(spec, seed)
    if not expert.ik_ok:
        return False, "ik"
    scene.reset(spec)
    efforts = scene.robot._articulation_view.get_max_efforts()
    for s in ("left", "right"):
        efforts[:, idx["finger"][s]] = FINGER_MAX_FORCE
    scene.robot._articulation_view.set_max_efforts(efforts)
    q_target = scene.q_start.copy()
    ticks = 0
    while not expert.done and ticks < 20 * 15:
        cmd = expert.step()
        for i, s in enumerate(("left", "right")):
            q_target[idx["arm"][s]] = cmd[7 * i:7 * i + 7]
            q_target[idx["finger"][s]] = cmd[14 + i]
        for _ in range(SUBSTEPS):
            scene.step(q_target, render=False)
        ticks += 1
    return bool(expert.done and scene.success()), f"lift {scene.bottle_lift() * 100:.1f} cm"


def main():
    reach = json.load(open(os.path.join(ASSETS, "reach_map.json")))
    rx, ry = np.array(reach["xs"]), np.array(reach["ys"])
    cat = {a["asset"]: a for a in json.load(open(_spec.CATALOG))["assets"]}[args.bottle]
    diameter = {b["asset"] + ".usda": b["max_diameter_mm"] / 1000.0 for b in json.load(open(_spec.SURVEY))["bottles"]}[args.bottle]
    scale = min(1.0, _spec.MAX_BOTTLE_DIAMETER / diameter)
    base = _spec.sample_scene(0, _spec.EVAL_NOMINAL)
    base = dataclasses.replace(base, table_height=0.85, table_front_x=0.70, table_depth=0.80, table_width=1.80, table_center_y=0.10,
                               bottle_asset=args.bottle, bottle_scale=scale, bottle_height=cat["height"] * scale,
                               bottle_width=diameter * scale, bottle_mass=0.4)
    scene = LiftBottleScene()
    names = scene.names
    idx = {"arm": {s: [names.index(n) for n in kin.arm_names(s)] for s in ("left", "right")},
           "finger": {s: [names.index(f"{s}_fr3_finger_joint{k}") for k in (1, 2)] for s in ("left", "right")}}

    xs = np.round(np.arange(0.80, 1.401, args.step), 3)
    ys = np.round(np.arange(-0.60, 0.801, args.step), 3)
    out = {"xs": xs.tolist(), "ys": ys.tolist(), "step": args.step, "trials": args.trials, "bottle": args.bottle, "maps": {}}
    path = os.path.join(ASSETS, "grasp_map.json")
    if os.path.exists(path):  # resume: keep every cell already measured on the same grid
        old = json.load(open(path))
        if old["xs"] == out["xs"] and old["ys"] == out["ys"] and old["trials"] == args.trials:
            out["maps"] = old["maps"]
    for side in ("left", "right"):
        rmap = np.array(reach["maps"][f"{side}@0.85"])
        prev = out["maps"].get(side)
        grid = np.array([[np.nan if v is None else v for v in row] for row in prev], float) if prev else np.full((len(xs), len(ys)), np.nan)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                if not rmap[np.abs(rx - x).argmin(), np.abs(ry - y).argmin()] or not np.isnan(grid[i, j]):
                    continue
                wins = 0
                for t in range(args.trials):
                    seed = 900000 + 1000 * i + 10 * j + t + (500000 if side == "right" else 0)
                    spec = dataclasses.replace(base, bottle_x=float(x), bottle_y=float(y), bottle_yaw=float(np.random.default_rng(seed).uniform(-np.pi, np.pi)))
                    object.__setattr__(spec, "seed", seed)
                    ok, why = run_trial(scene, idx, dataclasses.replace(spec, arm=side), seed)
                    wins += ok
                grid[i, j] = wins / args.trials
                print(f"[grasp] {side:5s} ({x:.2f}, {y:+.2f}): {wins}/{args.trials}", flush=True)
                out["maps"][side] = [[None if np.isnan(v) else float(v) for v in row] for row in grid]
                json.dump(out, open(path, "w"))  # after every cell, so a crash loses nothing
        out["maps"][side] = [[None if np.isnan(v) else float(v) for v in row] for row in grid]
        json.dump(out, open(path, "w"))
    plot(out)
    app.close()


def plot(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    for ax, side in zip(axes, ("left", "right"), strict=True):
        g = np.array([[np.nan if v is None else v for v in row] for row in out["maps"][side]], float)
        xs, ys = np.array(out["xs"]), np.array(out["ys"])
        im = ax.imshow(g, origin="lower", cmap="RdYlGn", vmin=0, vmax=1, aspect="equal",
                       extent=[ys[0] - out["step"] / 2, ys[-1] + out["step"] / 2, xs[0] - out["step"] / 2, xs[-1] + out["step"] / 2])
        ax.invert_xaxis()
        ax.set_title(f"{side} arm: grasp success (grey = kinematically unreachable)")
        ax.set_xlabel("y [m] (left +)")
    axes[0].set_ylabel("x [m] (forward)")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(os.path.join(ASSETS, "grasp_map.png"), dpi=90)


main()
