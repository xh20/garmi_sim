"""Does each bottle model stand by itself on the table? (Isaac Sim, 2 s settle on the nominal table)

    OMNI_KIT_ACCEPT_EULA=YES ~/isaac/.venv/bin/python scripts/check_bottle_stability.py

Every catalog model is tested at the scale the scene sampler uses (widest horizontal diameter scaled to at most
spec.MAX_BOTTLE_DIAMETER; models needing less than spec.MIN_BOTTLE_SCALE are skipped, they are never used). All of
them are tested, not just spec.bottle_assets(), because bottle_assets() itself filters by this file's result.
Writes assets/bottles/stability.json: tilt (deg) and drift (cm) after 2 s; tilt > 10 deg = falls over.
"""

import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402

from garmi_bench import spec as _spec  # noqa: E402
from garmi_bench.scene import LiftBottleScene  # noqa: E402

OUT = os.path.join(os.path.dirname(_spec.CATALOG), "stability.json")


def main():
    with open(_spec.SURVEY) as f:
        diameter = {b["asset"] + ".usda": b["max_diameter_mm"] / 1000.0 for b in json.load(f)["bottles"]}
    with open(_spec.CATALOG) as f:
        catalog = json.load(f)["assets"]
    scene = LiftBottleScene()
    base = _spec.sample_scene(0, _spec.EVAL_NOMINAL)
    res = {}
    for a in catalog:
        s = min(1.0, _spec.MAX_BOTTLE_DIAMETER / diameter[a["asset"]])
        if s < _spec.MIN_BOTTLE_SCALE:
            print(f"[stab] {a['asset']:24s} skipped: would need scale {s:.2f} < {_spec.MIN_BOTTLE_SCALE}", flush=True)
            continue
        spec = dataclasses.replace(base, bottle_asset=a["asset"], bottle_scale=s, bottle_height=a["height"] * s,
                                   bottle_width=diameter[a["asset"]] * s, bottle_x=1.15, bottle_y=-0.1, bottle_yaw=0.0)
        scene.reset(spec, settle_steps=240)  # 2 s
        p, q = scene.bottle.get_world_pose()
        w, x, y, z = q
        tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))
        drift = float(np.hypot(p[0] - 1.15, p[1] + 0.1)) * 100
        res[a["asset"]] = {"scale": round(s, 3), "tilt_deg": round(tilt, 1), "drift_cm": round(drift, 1)}
        print(f"[stab] {a['asset']:24s} scale {s:.2f} tilt {tilt:5.1f} deg, drift {drift:4.1f} cm "
              f"{'UNSTABLE' if tilt > 10 else ''}", flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[stab] {len(res)} models tested, {sum(v['tilt_deg'] > 10 for v in res.values())} fall over -> {OUT}", flush=True)


main()
app.close()
