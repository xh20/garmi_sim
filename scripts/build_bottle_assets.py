"""Convert RoboTwin-OD bottle-like objects (MIT license) into Z-up, metric USD rigid bodies + a catalog.

    OMNI_KIT_ACCEPT_EULA=YES ~/isaac/.venv/bin/python scripts/build_bottle_assets.py

Source: https://huggingface.co/datasets/TianxingChen/RoboTwin2.0 objects.zip, unpacked to
~/sim/assets/robotwin/objects. Each instance has visual/baseN.glb, collision/baseN.glb and
model_dataN.json (Y-up mesh, `scale` to meters where given, `extents`, `center`).

Output, per instance, assets/bottles/<category>_<N>.usda: an Xform with the visual mesh and an
invisible convex-decomposition collider, both scaled to meters, turned Y-up -> Z-up and shifted so the
origin is the bottom center. catalog.json lists height/width per asset (no simulator needed to read it).
"""

import asyncio
import glob
import json
import os
import re

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("omni.kit.asset_converter")
app.update()
import omni.kit.asset_converter as converter  # noqa: E402
from pxr import Gf  # noqa: E402
from pxr import Sdf  # noqa: E402
from pxr import Usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

SRC = os.path.expanduser("~/sim/assets/robotwin/objects")
OUT = os.path.expanduser("~/sim/garmi_bench/assets/bottles")
CATEGORIES = ("001_bottle", "114_bottle", "029_olive-oil", "065_soy-sauce", "066_vinegar")


def _top_is_wider(points, axis: int) -> bool:
    """Whether the mesh is wider at the high end of `axis` than at the low end (95th-percentile radius of the
    vertices in the outer 15 % of its length at either end). A bottle is wider at its base than at its neck."""
    import numpy as np

    pts = np.asarray(points, dtype=float)
    a = pts[:, axis]
    lo, hi = a.min(), a.max()
    others = [i for i in range(3) if i != axis]
    r = np.hypot(*(pts[:, i] - pts[:, i].mean() for i in others))
    return np.percentile(r[a > hi - 0.15 * (hi - lo)], 95) > np.percentile(r[a < lo + 0.15 * (hi - lo)], 95)


def glb_top_is_wider(glb: str) -> bool:
    """The same test on the source glb along glTF's up axis (+Y), the authoritative orientation."""
    import trimesh

    return _top_is_wider(trimesh.load(glb, force="scene").dump(concatenate=True).vertices, axis=1)


def usd_points(prim):
    import numpy as np

    cache, pts = UsdGeom.XformCache(), []
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Mesh):
            v = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(), float)
            m = np.asarray(cache.GetLocalToWorldTransform(p), float)
            pts.append(v @ m[:3, :3] + m[3, :3])
    return np.concatenate(pts)


async def glb_to_usd(src: str, dst: str) -> None:
    ctx = converter.AssetConverterContext()
    ctx.ignore_animations = True
    ctx.ignore_camera = True
    ctx.ignore_light = True
    ctx.embed_textures = True
    task = converter.get_instance().create_converter_task(src, dst, None, ctx)
    if not await task.wait_until_finished():
        raise RuntimeError(f"conversion failed: {src}: {task.get_status()} {task.get_error_message()}")


def build_asset(cat: str, idx: str, meta: dict, visual_usd: str, collision_usd: str, out: str, sources: dict) -> dict:
    """The converter writes centimeter USD and turns *some* glbs to Z-up, so each mesh's own upAxis and
    metersPerUnit decide rotation and scale; the result is then measured and shifted so that the bottom
    center of the collision mesh is the origin. Height/width in the catalog are of the collision mesh."""
    s = float(meta.get("scale", [1.0])[0]) if "scale" in meta else 1.0

    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/bottle")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.4)

    ops = {}
    all_purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy]
    for name, ref, collide in (("visual", visual_usd, False), ("collision", collision_usd, True)):
        mpu = UsdGeom.GetStageMetersPerUnit(Usd.Stage.Open(ref))
        x = UsdGeom.Xform.Define(stage, f"/bottle/{name}")
        ops[name] = x.AddTranslateOp()
        rot_op = x.AddRotateXOp()
        x.AddScaleOp().Set(Gf.Vec3f(s * mpu, s * mpu, s * mpu))
        x.GetPrim().GetReferences().AddReference(os.path.relpath(ref, os.path.dirname(out)))
        # Neither the converter's upAxis metadata nor the source file's own bounds tell how the referenced mesh
        # ends up oriented, so measure it here, as referenced: a bottle is taller than wide (long axis -> z).
        rot_op.Set(0.0)
        r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), all_purposes).ComputeWorldBound(x.GetPrim()).ComputeAlignedRange()
        size = r.GetMax() - r.GetMin()
        angle = 90.0 if size[1] > size[2] else 0.0  # long axis y -> z
        rot_op.Set(angle)
        # Which way up: the long axis alone leaves some bottles standing on their caps (v1/v2 of the assets: 19 of
        # 40, the ones the converter had already turned Z-up). Keep the glb's +Y top at +z: compare which end is
        # wider with the source glb and turn 180 deg if they disagree.
        if _top_is_wider(usd_points(x.GetPrim()), axis=2) != glb_top_is_wider(sources[name]):
            rot_op.Set(angle + 180.0)
        if collide:
            UsdGeom.Imageable(x.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.guide)  # never rendered
            for p in Usd.PrimRange(x.GetPrim()):
                if p.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(p)
                    UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")  # bottles are near-convex: one hull gives a flat, stable bottom

    def bounds(name, purpose):
        return UsdGeom.BBoxCache(Usd.TimeCode.Default(), [purpose]).ComputeWorldBound(
            stage.GetPrimAtPath(f"/bottle/{name}")).ComputeAlignedRange()

    for name, purpose in (("visual", UsdGeom.Tokens.default_), ("collision", UsdGeom.Tokens.guide)):
        ops[name].Set(Gf.Vec3d(0, 0, 0))
        r = bounds(name, purpose)
        lo, hi = r.GetMin(), r.GetMax()
        ops[name].Set(Gf.Vec3d(-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
    r = bounds("collision", UsdGeom.Tokens.guide)
    size = r.GetMax() - r.GetMin()
    stage.GetRootLayer().Save()
    return {"asset": os.path.relpath(out, OUT), "category": cat, "instance": int(idx), "height": round(size[2], 4),
            "width": round(max(size[0], size[1]), 4)}


async def main():
    if os.path.exists(os.path.join(OUT, "meshes")) and any(f.endswith(".usd") for f in os.listdir(os.path.join(OUT, "meshes"))):
        raise SystemExit(f"{OUT}/meshes holds USDs of the old flat layout; move assets/bottles away first")
    os.makedirs(os.path.join(OUT, "meshes"), exist_ok=True)
    catalog = []
    for cat in CATEGORIES:
        for vis in sorted(glob.glob(f"{SRC}/{cat}/visual/base*.glb"), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0])):
            idx = re.findall(r"\d+", os.path.basename(vis))[0]
            meta_path, col = f"{SRC}/{cat}/model_data{idx}.json", f"{SRC}/{cat}/collision/base{idx}.glb"
            if not (os.path.exists(meta_path) and os.path.exists(col)):
                print(f"[assets] skip {cat}/{idx}: missing metadata or collision mesh", flush=True)
                continue
            # One folder per instance: the converter writes a glb's textures to textures/ next to the USD under
            # generic names (texture_0.png, Image_0.png, ...), so a shared folder lets every conversion overwrite
            # the previous one's textures (v1 of the assets ended up with 5 texture sets for 40 bottles).
            mesh_dir = f"{OUT}/meshes/{cat}_{idx}"
            os.makedirs(mesh_dir, exist_ok=True)
            vis_usd, col_usd = f"{mesh_dir}/visual.usd", f"{mesh_dir}/collision.usd"
            await glb_to_usd(vis, vis_usd)
            await glb_to_usd(col, col_usd)
            entry = build_asset(cat, idx, json.load(open(meta_path)), vis_usd, col_usd, f"{OUT}/{cat}_{idx}.usda",
                                {"visual": vis, "collision": col})
            catalog.append(entry)
            print(f"[assets] {entry['asset']}: {entry['height'] * 100:.1f} cm tall, {entry['width'] * 100:.1f} cm wide", flush=True)
    json.dump({"source": "RoboTwin-OD (TianxingChen/RoboTwin2.0 objects.zip), MIT license", "assets": catalog},
              open(f"{OUT}/catalog.json", "w"), indent=1)
    print(f"[assets] {len(catalog)} assets -> {OUT}/catalog.json", flush=True)


asyncio.get_event_loop().run_until_complete(main())
app.close()
