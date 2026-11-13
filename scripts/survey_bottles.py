"""Survey every RoboTwin bottle-like object for graspability by GARMI's Franka Hand (original size, no rescaling).

    ~/sim/mujoco_venv/bin/python scripts/survey_bottles.py

Per instance, from the collision glb (meters via model_data `scale`, upright with glTF's +Y up):
  footprint   x and y extents, and the widest horizontal diameter in any direction (the hand may approach from
              any side), over the whole height;
  mass        estimated full: 90 % of the convex-hull volume filled with liquid (1.0 kg/L) plus the container
              (plastic 0.03 kg, glass 0.30 kg by category); RoboTwin has no masses;
  stable      stands by itself (assets/bottles/stability.json, Isaac, tilt <= 10 deg), where measured.
Graspable: widest diameter <= 70 mm (>= 5 mm clearance per finger at the 80 mm opening), mass <= 1.5 kg (FR3
payload 3 kg; holding needs friction >= m g / 140 N), and stable. Writes assets/bottle_survey.json and .txt.
"""

import glob
import json
import os
import re

import numpy as np
import trimesh

SRC = os.path.expanduser("~/sim/assets/robotwin/objects")
ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
CATEGORIES = ("001_bottle", "114_bottle", "029_olive-oil", "065_soy-sauce", "066_vinegar", "080_pillbottle")
GLASS = {"029_olive-oil", "065_soy-sauce", "066_vinegar"}
OPENING_MM, MAX_DIAM_MM, MAX_MASS_KG = 80.0, 70.0, 1.5


def load_upright(glb: str, scale: float) -> np.ndarray:
    mesh = trimesh.load(glb, force="scene").to_geometry()
    mesh.apply_scale(scale)
    v = mesh.vertices
    if np.ptp(v[:, 1]) > np.ptp(v[:, 2]):  # glTF +Y up -> +Z (same rule as the fixed asset builds)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    return mesh


def main():
    stability = json.load(open(os.path.join(ASSETS, "bottles", "stability.json")))
    rows = []
    for cat in CATEGORIES:
        for glb in sorted(glob.glob(f"{SRC}/{cat}/collision/base*.glb"), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0])):
            idx = re.findall(r"\d+", os.path.basename(glb))[0]
            meta = json.load(open(f"{SRC}/{cat}/model_data{idx}.json"))
            scale = float(meta["scale"][0]) if "scale" in meta else 1.0
            mesh = load_upright(glb, scale)
            hull = mesh.convex_hull
            v = hull.vertices
            diam = max(np.ptp(v[:, 0] * np.cos(a) + v[:, 1] * np.sin(a)) for a in np.linspace(0, np.pi, 180, endpoint=False))
            # Cap: widest horizontal diameter of the top 15 mm of the collision parts' own convex hulls -- what the
            # simulators collide with (assets/bottles_mj, upright, original size); the hull of the whole bottle
            # would bridge from the cap edge to the shoulders.
            mj_dir = os.path.join(ASSETS, "bottles_mj", f"{cat}_{idx}")
            if os.path.exists(os.path.join(mj_dir, "parts.json")):
                surf = np.concatenate([trimesh.sample.sample_surface(
                    trimesh.load(os.path.join(mj_dir, c), force="mesh", process=False).convex_hull, 20000, seed=0)[0]
                    for c in json.load(open(os.path.join(mj_dir, "parts.json")))["collision"]])
            else:  # no simulator asset (category never used): the whole hull, an upper bound
                surf = np.concatenate([trimesh.sample.sample_surface(hull, 20000, seed=0)[0], v])
            top = surf[surf[:, 2] >= surf[:, 2].max() - 0.015]
            cap = max(np.ptp(top[:, 0] * np.cos(a) + top[:, 1] * np.sin(a)) for a in np.linspace(0, np.pi, 90, endpoint=False))
            liquid = 0.9 * hull.volume * 1000.0  # kg at 1.0 kg/L
            mass = liquid + (0.30 if cat in GLASS else 0.03)
            name = f"{cat}_{idx}"
            tilt = stability.get(f"{name}.usda", {}).get("tilt_deg")
            stable = None if tilt is None else tilt <= 10.0
            reasons = []
            if 1000 * diam > MAX_DIAM_MM:
                reasons.append(f"too wide ({1000 * diam:.0f} mm)")
            if mass > MAX_MASS_KG:
                reasons.append(f"too heavy (~{mass:.2f} kg)")
            if stable is False:
                reasons.append(f"falls over ({tilt:.0f} deg)")
            rows.append({"asset": name, "category": cat, "x_mm": round(1000 * np.ptp(v[:, 0]), 1), "y_mm": round(1000 * np.ptp(v[:, 1]), 1),
                         "max_diameter_mm": round(1000 * diam, 1), "cap_diameter_mm": round(1000 * cap, 1),
                         "height_mm": round(1000 * np.ptp(v[:, 2]), 1),
                         "volume_l": round(hull.volume * 1000, 3), "est_mass_kg": round(mass, 2), "stable": stable,
                         "clearance_per_finger_mm": round((OPENING_MM - 1000 * diam) / 2, 1),
                         "graspable": not reasons, "reasons": reasons})
    json.dump({"criteria": {"max_diameter_mm": MAX_DIAM_MM, "max_mass_kg": MAX_MASS_KG, "gripper_opening_mm": OPENING_MM},
               "bottles": rows}, open(os.path.join(ASSETS, "bottle_survey.json"), "w"), indent=1)
    lines = [f"{'asset':18s} {'x':>6s} {'y':>6s} {'diam':>6s} {'height':>6s} {'mass*':>6s} {'clear':>6s} stable  graspable",
             f"{'':18s} {'mm':>6s} {'mm':>6s} {'mm':>6s} {'mm':>6s} {'kg':>6s} {'mm':>6s}"]
    for r in rows:
        st = "-" if r["stable"] is None else ("yes" if r["stable"] else "NO")
        lines.append(f"{r['asset']:18s} {r['x_mm']:6.1f} {r['y_mm']:6.1f} {r['max_diameter_mm']:6.1f} {r['height_mm']:6.0f} "
                     f"{r['est_mass_kg']:6.2f} {r['clearance_per_finger_mm']:6.1f} {st:6s}  "
                     + ("YES" if r["graspable"] else "no: " + "; ".join(r["reasons"])))
    ok = [r for r in rows if r["graspable"]]
    lines += ["", f"graspable: {len(ok)} of {len(rows)}; per category: "
              + ", ".join(f"{c} {sum(r['category'] == c for r in ok)}/{sum(r['category'] == c for r in rows)}" for c in CATEGORIES),
              "* mass estimated: 90 % of hull volume x 1.0 kg/L + container (plastic 0.03 kg, glass 0.30 kg)",
              "stable '-': not measured (category not in the Isaac stability check)"]
    open(os.path.join(ASSETS, "bottle_survey.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
