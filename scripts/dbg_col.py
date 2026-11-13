import os, sys
sys.path.insert(0, os.path.expanduser("~/sim/garmi_bench"))
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdPhysics, UsdGeom
from garmi_bench.scene import LiftBottleScene
scene = LiftBottleScene()
stage = scene.world.stage
for side in ("left", "right"):
    rows = []
    for p in Usd.PrimRange(stage.GetPrimAtPath("/World/garmi"), Usd.TraverseInstanceProxies()):
        path = str(p.GetPath())
        if f"/{side}_fr3_" in path and any(k in path for k in ("finger", "hand")) and p.HasAPI(UsdPhysics.CollisionAPI):
            en = UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
            approx = UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() if p.HasAPI(UsdPhysics.MeshCollisionAPI) else "-"
            rows.append(f"{path.split('/Geometry/')[-1][-70:]} enabled={en} approx={approx}")
    print(f"[col] {side}: {len(rows)} finger/hand colliders", flush=True)
    for r in rows: print("[col]    ", r, flush=True)
app.close()
