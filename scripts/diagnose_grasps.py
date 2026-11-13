"""Where does a policy's grasp go wrong? Closed-loop MuJoCo episodes with per-tick logging of the grasping hand.

    MUJOCO_GL=egl ~/sim/mujoco_venv/bin/python scripts/diagnose_grasps.py --name v16_real_smoothest_27000 \
        --suite nominal --n 16 --out ~/sim/eval/diagnose/v16_real_smoothest_27000

Same execution as eval_policy.py (real-time chunking in sim time, client limits). Per episode it records:
  first contact      tick, which robot part touched the bottle first (fingertip / finger / hand / arm link), and
                     the fingertip centre (hand_tcp site) relative to the bottle at that moment: radial distance
                     from the bottle axis, height above the table as a fraction of the bottle height, and the
                     offset in the robot frame (forward x / left y / up z);
  close              when the grasping gripper is first commanded below 30 mm: the fingertip centre's offset from
                     the bottle axis and its height fraction, plus the approach direction (fingertip z axis) vs. the
                     scripted expert's planned one (azimuth and elevation errors);
  outcome            as eval_policy.py.
The expert grasps at 35-50 % of the bottle height with the fingertip centre on the bottle axis, so a systematic
offset at close points to misjudged bottle position (distance), a large approach-angle error to a wrong hand pose.
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402

from garmi_bench import spec as _spec  # noqa: E402
from garmi_bench.expert import CONTROL_HZ  # noqa: E402
from garmi_bench.expert import LiftBottleExpert  # noqa: E402
from garmi_bench.mj_scene import MjLiftBottleScene  # noqa: E402
from garmi_bench.rtc_sim import ClientLimits  # noqa: E402
from garmi_bench.rtc_sim import ClientSafety  # noqa: E402
from garmi_bench.rtc_sim import SimTimeBlendRTC  # noqa: E402

SUITES = os.path.join(os.path.dirname(__file__), "..", "assets", "eval_suites.json")
CLOSE_BELOW = 0.030  # m per finger: the gripper command counts as "closing"


def part_of(name: str) -> str:
    if "tip" in name:
        return "fingertip"
    if "finger" in name:
        return "finger"
    if "hand" in name or "link8" in name:
        return "hand"
    if "fr3_link" in name:
        return "arm_link"
    return "other:" + name


def bottle_contacts(scene) -> list[str]:
    m, d = scene.model, scene.data
    parts = []
    for c in d.contact[:d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if scene.bottle_body in (b1, b2):
            other = b2 if b1 == scene.bottle_body else b1
            name = m.body(other).name
            if not name.startswith(("table", "world")) and other != 0:
                parts.append(name)
    return parts


def hand_vs_bottle(scene, spec) -> dict:
    d = scene.data
    site = d.site(f"{spec.arm}_fr3_hand_tcp")
    tcp, rot = site.xpos.copy(), site.xmat.reshape(3, 3)
    bottle = scene.bottle_pos()
    rel = tcp - bottle
    approach = rot[:, 2]
    return {"radial_mm": round(1000 * float(np.hypot(rel[0], rel[1])), 1),
            "height_frac": round(float((tcp[2] - spec.table_height) / spec.bottle_height), 3),
            "dx_mm": round(1000 * float(rel[0]), 1), "dy_mm": round(1000 * float(rel[1]), 1),
            "dz_mm": round(1000 * float(tcp[2] - (spec.table_height + 0.425 * spec.bottle_height)), 1),
            "approach_az_deg": round(math.degrees(math.atan2(approach[1], approach[0])), 1),
            "approach_el_deg": round(math.degrees(math.asin(np.clip(approach[2], -1, 1))), 1)}


def run(client, spec, args, delay_steps, horizon, model_guard):
    scene = MjLiftBottleScene(spec)
    for _ in range(10):
        scene.step(scene.q_start)
    side = 0 if spec.arm == "left" else 1
    plan = LiftBottleExpert(spec, spec.seed).plan
    plan_z = plan.grasp[:3, 2]
    plan_az, plan_el = math.degrees(math.atan2(plan_z[1], plan_z[0])), math.degrees(math.asin(plan_z[2]))

    def infer(obs):
        model_guard()
        r = client.infer(obs)
        return np.asarray(r["actions"], dtype=np.float64)

    def make_obs():
        imgs = scene.images()
        pad = lambda im: image_tools.resize_with_pad(im, 224, 224)
        return {"observation/state": scene.state(), "observation/image": pad(imgs["head"]),
                "observation/left_wrist_image": pad(imgs["left_wrist"]), "observation/right_wrist_image": pad(imgs["right_wrist"]),
                "prompt": "grasp the bottle"}

    rtc = SimTimeBlendRTC(infer, horizon=horizon, delay_steps=delay_steps, blend_steps=5)
    safety = ClientSafety(scene.q_start, ClientLimits())
    first_contact = close = None
    above, outcome, max_lift, max_tilt = 0, "timeout", 0.0, 0.0
    for tick in range(int(args.max_seconds * CONTROL_HZ)):
        action, _ = rtc.step(tick, make_obs)
        cmd = safety(tick, action, scene.joints())
        if close is None and cmd[14 + side] < CLOSE_BELOW:
            close = {"tick": tick, **hand_vs_bottle(scene, spec)}
        scene.step(cmd)
        if first_contact is None:
            parts = bottle_contacts(scene)
            if parts:
                first_contact = {"tick": tick, "parts": sorted({part_of(p) for p in parts}), "bodies": sorted(set(parts)),
                                 "bottle_tilt_deg": round(scene.bottle_tilt_deg(), 1), **hand_vs_bottle(scene, spec)}
        lift, tilt = scene.bottle_lift(), scene.bottle_tilt_deg()
        max_lift, max_tilt = max(max_lift, lift), max(max_tilt, tilt)
        above = above + 1 if lift >= _spec.LIFT_SUCCESS_M else 0
        if above >= 15:
            outcome = "success"
            break
        if scene.bottle_pos()[2] < spec.table_height - 0.15:
            outcome = "fell_off_table"
            break
    if outcome == "timeout":
        outcome = ("lifted_not_held" if max_lift >= 0.2 else "knocked_over" if max_tilt > 60 else
                   "lifted_too_little" if max_lift >= 0.02 else "not_lifted")
    scene.close()
    for rec in (first_contact, close):
        if rec is not None:
            rec["approach_az_err_deg"] = round(((rec["approach_az_deg"] - plan_az + 180) % 360) - 180, 1)
            rec["approach_el_err_deg"] = round(rec["approach_el_deg"] - plan_el, 1)
    return {"seed": spec.seed, "arm": spec.arm, "bottle": spec.bottle_asset, "bottle_width_mm": round(1000 * spec.bottle_width, 1),
            "outcome": outcome, "first_contact": first_contact, "close": close}


def summarize(eps):
    def stats(key, field, sel):
        v = [e[key][field] for e in eps if sel(e) and e[key] is not None]
        return f"{np.mean(v):+.1f} +/- {np.std(v):.1f} (n={len(v)})" if v else "-"
    lines = []
    for label, sel in (("success", lambda e: e["outcome"] == "success"), ("failure", lambda e: e["outcome"] != "success")):
        lines.append(f"{label}: {sum(sel(e) for e in eps)} episodes")
        for key in ("first_contact", "close"):
            for field in ("radial_mm", "height_frac", "dx_mm", "dy_mm", "dz_mm", "approach_az_err_deg", "approach_el_err_deg"):
                lines.append(f"  {key:13s} {field:20s} {stats(key, field, sel)}")
        parts = {}
        for e in eps:
            if sel(e) and e["first_contact"]:
                for p in e["first_contact"]["parts"]:
                    parts[p] = parts.get(p, 0) + 1
        lines.append(f"  first-contact parts: {parts}; no contact at all: {sum(sel(e) and e['first_contact'] is None for e in eps)}")
        lines.append(f"  never closed: {sum(sel(e) and e['close'] is None for e in eps)}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--name", required=True)
    p.add_argument("--suite", default="nominal")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--delay-steps", type=int, default=4)
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--guard-log", default="", help="abort if this runner log reports the served model is done")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    def model_guard():
        if args.guard_log and f"done {args.name}" in open(os.path.expanduser(args.guard_log)).read():
            raise SystemExit(f"served model {args.name} finished; aborting to avoid mixing models")

    suite = json.load(open(SUITES))["suites"][args.suite]
    ranges = getattr(_spec, suite["ranges"])
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    entries = suite["episodes"][:args.n]
    eps = []
    for entry in entries:
        spec = _spec.sample_scene(entry["seed"], ranges)
        t0 = time.time()
        e = run(client, spec, args, args.delay_steps, 32, model_guard)
        eps.append(e)
        fc, cl = e["first_contact"], e["close"]
        print(f"[diag] {e['seed']} {e['arm']:5s} {e['bottle']:22s} {e['outcome']:17s} first contact "
              + (f"{fc['parts']} radial {fc['radial_mm']} mm h {fc['height_frac']} dx {fc['dx_mm']} dy {fc['dy_mm']} dz {fc['dz_mm']}" if fc else "none")
              + " | close " + (f"radial {cl['radial_mm']} mm h {cl['height_frac']} az_err {cl['approach_az_err_deg']} el_err {cl['approach_el_err_deg']}" if cl else "never")
              + f" ({time.time() - t0:.0f} s)", flush=True)
        json.dump({"name": args.name, "suite": args.suite, "episodes": eps}, open(os.path.join(out, "diagnose.json"), "w"), indent=1)
    text = summarize(eps)
    open(os.path.join(out, "diagnose_summary.txt"), "w").write(f"{args.name} on {args.suite} ({len(eps)} episodes)\n{text}\n")
    print(text)


if __name__ == "__main__":
    main()
