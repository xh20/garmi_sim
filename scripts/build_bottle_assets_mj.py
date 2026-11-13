"""Convert the RoboTwin-OD bottles into MuJoCo assets, matching the Isaac Sim ones (build_bottle_assets.py).

    ~/sim/mujoco_venv/bin/python scripts/build_bottle_assets_mj.py

Same source files and the same rules as the USD build: scaled to meters with the instance's `scale`, the long
(bottle) axis turned to z, the visual and the collision mesh each shifted so that its bottom center is the
origin. Output per catalog asset, assets/bottles_mj/<category>_<N>/: visual_<k>.obj (+ texture_<k>.png) per
glTF primitive, collision_<k>.obj per collision primitive (MuJoCo collides with each one's convex hull, like
PhysX's convexHull approximation), and parts.json. Every asset's collision size is checked against the Isaac
catalog (catalog.json): height and width must agree to 1 mm.
"""

import json
import os

import numpy as np
import trimesh

SRC = os.path.expanduser("~/sim/assets/robotwin/objects")
ISAAC_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "bottles")
OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "bottles_mj")


def load_parts(path: str, scale: float) -> list[trimesh.Trimesh]:
    """All primitives of a glb in meters (scene transforms applied), before the upright/origin fix."""
    scene = trimesh.load(path, force="scene")
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        mesh.apply_transform(transform)
        mesh.apply_scale(scale)
        parts.append(mesh)
    return parts


def upright(parts: list[trimesh.Trimesh]) -> None:
    """Long axis y -> z (rotate +90 deg about x) when the mesh is taller along y, as the USD build decides."""
    lo, hi = np.min([p.bounds[0] for p in parts], 0), np.max([p.bounds[1] for p in parts], 0)
    size = hi - lo
    if size[1] > size[2]:
        rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        for p in parts:
            p.apply_transform(rot)


def to_bottom_center(parts: list[trimesh.Trimesh]) -> np.ndarray:
    lo, hi = np.min([p.bounds[0] for p in parts], 0), np.max([p.bounds[1] for p in parts], 0)
    shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    for p in parts:
        p.apply_translation(shift)
    return hi - lo


def export_visual(mesh: trimesh.Trimesh, out_dir: str, k: int) -> dict:
    entry = {"mesh": f"visual_{k}.obj", "texture": None, "rgba": [0.8, 0.8, 0.8, 1.0]}
    material = getattr(mesh.visual, "material", None)
    image = None
    if material is not None:
        image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
        factor = getattr(material, "baseColorFactor", None)
        if factor is not None:
            entry["rgba"] = [round(float(c) / (255.0 if np.max(factor) > 1 else 1.0), 4) for c in factor]
    has_uv = getattr(mesh.visual, "uv", None) is not None
    if image is not None and has_uv:
        image.convert("RGB").save(os.path.join(out_dir, f"texture_{k}.png"))
        entry["texture"] = f"texture_{k}.png"
        entry["rgba"] = [1.0, 1.0, 1.0, 1.0]  # the texture carries the color
    # OBJ with uv only (MuJoCo ignores the .mtl; the material is set in the MJCF).
    obj = trimesh.exchange.obj.export_obj(mesh, include_normals=True, include_texture=has_uv, write_texture=False,
                                          mtl_name=None)
    with open(os.path.join(out_dir, entry["mesh"]), "w") as f:
        f.write("\n".join(line for line in obj.splitlines() if not line.startswith(("mtllib", "usemtl"))))
    return entry


def build(asset: dict) -> dict:
    cat, idx = asset["category"], asset["instance"]
    meta = json.load(open(f"{SRC}/{cat}/model_data{idx}.json"))
    scale = float(meta.get("scale", [1.0])[0]) if "scale" in meta else 1.0
    out_dir = os.path.join(OUT, f"{cat}_{idx}")
    os.makedirs(out_dir, exist_ok=True)

    visual = load_parts(f"{SRC}/{cat}/visual/base{idx}.glb", scale)
    collision = load_parts(f"{SRC}/{cat}/collision/base{idx}.glb", scale)
    upright(visual)
    upright(collision)
    to_bottom_center(visual)
    size = to_bottom_center(collision)

    parts = {"visual": [export_visual(m, out_dir, k) for k, m in enumerate(visual)], "collision": []}
    for k, m in enumerate(collision):
        name = f"collision_{k}.obj"
        trimesh.Trimesh(m.vertices, m.faces, process=False).export(os.path.join(out_dir, name))
        parts["collision"].append(name)
    hull = trimesh.util.concatenate([m.convex_hull for m in collision]).convex_hull
    parts["hull_volume"] = float(hull.volume)
    parts["hull_com"] = hull.center_mass.tolist()
    parts["hull_inertia_unit_mass"] = (hull.moment_inertia / hull.volume).tolist()  # density 1 -> per kg
    parts["height"], parts["width"] = round(float(size[2]), 4), round(float(max(size[0], size[1])), 4)
    with open(os.path.join(out_dir, "parts.json"), "w") as f:
        json.dump(parts, f, indent=1)
    return parts


def main():
    catalog = json.load(open(os.path.join(ISAAC_DIR, "catalog.json")))["assets"]
    worst = 0.0
    for asset in catalog:
        parts = build(asset)
        dh, dw = abs(parts["height"] - asset["height"]), abs(parts["width"] - asset["width"])
        worst = max(worst, dh, dw)
        flag = "" if max(dh, dw) <= 0.001 else "  <-- differs from the Isaac asset"
        print(f"[assets_mj] {asset['asset']:22s} {parts['height'] * 100:5.1f} x {parts['width'] * 100:4.1f} cm "
              f"(Isaac {asset['height'] * 100:5.1f} x {asset['width'] * 100:4.1f}), {len(parts['visual'])} visual / "
              f"{len(parts['collision'])} collision parts{flag}", flush=True)
    print(f"[assets_mj] {len(catalog)} assets -> {OUT}; largest size difference to Isaac {worst * 1000:.2f} mm", flush=True)


if __name__ == "__main__":
    main()
