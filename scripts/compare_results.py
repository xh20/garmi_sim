"""Comparison table of closed-loop evaluations (results.json files written by eval_policy.py).

    ~/sim/mujoco_venv/bin/python scripts/compare_results.py ~/sim/eval/mujoco/*/results.json --out ~/sim/eval/mujoco/comparison

Writes <out>.txt (aligned text tables: per suite, then per arm and failure modes) and <out>.csv (one row per
model x suite). The scripted expert's success on the same seeds (eval_suites.json) is listed as a reference.
"""

import argparse
import csv
import json
import os


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(c.ljust(w) for c, w in zip(rows[0], widths, strict=True)), line]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in rows[1:]]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+")
    p.add_argument("--out", required=True, help="output path without extension")
    args = p.parse_args()
    runs = [json.load(open(os.path.expanduser(r))) for r in args.results]
    suite_names = [s for s in ("nominal", "unseen_objects", "unseen_rooms", "train_ranges", "v16_lab") if any(s in r["summary"] for r in runs)]

    main_rows = [["model", "simulator", *[f"{s} (95% CI)" for s in suite_names], "all (95% CI)"]]
    detail_rows = [["model", "suite", "success", "left", "right", "expert", "mean t_success", "accel rms", "outcomes"]]
    csv_rows = []
    for r in runs:
        sim = r["provenance"]["args"].get("sim", "mujoco")
        cells = []
        for s in suite_names + ["all"]:
            x = r["summary"].get(s)
            cells.append("-" if x is None else f"{x['success']}/{x['episodes']} = {pct(x['success_rate'])} "
                                                f"({pct(x['wilson95'][0])}-{pct(x['wilson95'][1])})")
        main_rows.append([r["name"], sim, *cells])
        for s in suite_names:
            x = r["summary"].get(s)
            if x is None:
                continue
            outcomes = ", ".join(f"{k} {v}" for k, v in sorted(x["outcomes"].items(), key=lambda kv: -kv[1]))
            t = "-" if x["mean_time_to_success_s"] is None else f"{x['mean_time_to_success_s']:.1f} s"
            detail_rows.append([r["name"], s, f"{x['success']}/{x['episodes']}", x["per_arm"]["left"], x["per_arm"]["right"],
                                x["expert_reference"] or "-", t, f"{x['cmd_accel_rms_mean']:.3f}", outcomes])
            csv_rows.append({"model": r["name"], "simulator": sim, "suite": s, "episodes": x["episodes"], "success": x["success"],
                             "success_rate": x["success_rate"], "ci95_low": x["wilson95"][0], "ci95_high": x["wilson95"][1],
                             "left": x["per_arm"]["left"], "right": x["per_arm"]["right"], "expert_reference": x["expert_reference"],
                             "mean_time_to_success_s": x["mean_time_to_success_s"], "cmd_accel_rms": x["cmd_accel_rms_mean"],
                             "outcomes": outcomes, "delay_steps": r["delay_steps"], "latency_ms": r["measured_latency_ms"]})

    settings = sorted({(r["horizon"], r["delay_steps"], r["provenance"]["args"]["blend_steps"], r["provenance"]["args"]["max_seconds"],
                        r["provenance"]["args"]["hold_seconds"]) for r in runs})
    text = "\n".join([
        "GARMI lift-the-bottle benchmark: closed-loop success (bottle >= 20 cm above its rest height, held 1 s)",
        "",
        table(main_rows),
        "",
        "Per suite: success per arm, scripted-expert reference on the same seeds, time to success, smoothness of the",
        "executed joint commands (rad/s^2 RMS), outcomes.",
        "",
        table(detail_rows),
        "",
        "Settings (chunk, delay ticks, blend steps, max s, hold s): " + "; ".join(str(s) for s in settings),
        "Latency per run: " + ", ".join(f"{r['name']} {r['measured_latency_ms']:.0f} ms" for r in runs),
        "Sources: " + ", ".join(os.path.expanduser(r) for r in args.results),
    ])
    out = os.path.expanduser(args.out)
    with open(out + ".txt", "w") as f:
        f.write(text + "\n")
    with open(out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    print(text)


if __name__ == "__main__":
    main()
