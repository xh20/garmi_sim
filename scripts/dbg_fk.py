import os, sys
sys.path.insert(0, os.path.expanduser("~/sim/garmi_bench"))
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import numpy as np
from garmi_bench import spec as _spec, kinematics as kin
from garmi_bench.scene import LiftBottleScene
scene = LiftBottleScene()
scene.reset(_spec.sample_scene(0))
names = scene.names
rng = np.random.default_rng(0)
for trial in range(3):
    q = scene.q_start.copy()
    qa = {}
    for s in ("left", "right"):
        qa[s] = np.array(_spec.START_JOINTS[s]) + rng.uniform(-0.4, 0.4, 7)
        for i, n in enumerate(kin.arm_names(s)): q[names.index(n)] = qa[s][i]
    scene.robot.set_joint_positions(q)
    scene.world.step(render=False)
    q_now = scene.robot.get_joint_positions()
    for s in ("left", "right"):
        qs = np.array([q_now[names.index(n)] for n in kin.arm_names(s)])
        p_sim, _ = scene.link_world_pose(f"{s}_fr3_hand_tcp")
        p_np = kin.tcp_pose(s, qs)[:3, 3]
        base_sim, _ = scene.link_world_pose("base_link")
        print(f"[fk] trial {trial} {s:5s}: sim {np.round(p_sim, 3)} numpy {np.round(p_np, 3)} | diff {np.linalg.norm(p_sim - p_np) * 1000:.1f} mm | base_link in sim {np.round(base_sim, 3)} | joints applied err {np.abs(qs - qa[s]).max():.4f}", flush=True)
app.close()
