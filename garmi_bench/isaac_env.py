"""Isaac Sim version of the closed-loop scene, with MjLiftBottleScene's interface (for scripts/eval_policy.py).

Set up exactly as the demos were recorded (scripts/record_demos.py): head fisheye and wrist cameras, head tilt,
70 N finger drives, grip material on the fingers, 10 rendered warm-up ticks, one rendered world step per 15 Hz
tick. Must be imported after `isaacsim.SimulationApp` has started. Isaac builds its world once, so `load(spec)`
re-randomizes the same scene instead of building a new one per episode.
"""

from __future__ import annotations

import numpy as np
from pxr import UsdShade

from garmi_bench import camera_models
from garmi_bench import head_camera
from garmi_bench import kinematics as kin
from garmi_bench.scene import LiftBottleScene
from garmi_bench.scene import letterbox
from garmi_bench.spec import LIFT_SUCCESS_M
from garmi_bench.spec import START_JOINTS
from garmi_bench.spec import SceneSpec

FINGER_MAX_FORCE = 70.0


class IsaacLiftBottleEnv:
    def __init__(self):
        self.scene = LiftBottleScene()
        self.head = self.scene.add_head_camera()
        self.wrist = {s: self.scene.add_wrist_camera(s) for s in ("left", "right")}
        names = self.scene.names
        for joint, value in head_camera.head_joints().items():  # calibrated on v16, as record_demos.py
            self.scene.q_start[names.index(joint)] = value
        self._circle = head_camera.circle_mask(head_camera.OUT_SIZE)[..., None]
        self.idx = {
            "arm": {s: [names.index(n) for n in kin.arm_names(s)] for s in ("left", "right")},
            "finger": {s: [names.index(f"{s}_fr3_finger_joint{k}") for k in (1, 2)] for s in ("left", "right")},
        }
        stage = self.scene.world.stage
        grip = UsdShade.Material(stage.GetPrimAtPath(self.scene.grip.prim_path))
        for path in self.scene.finger_collision_prims():
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(grip, UsdShade.Tokens.weakerThanDescendants, "physics")
        self.spec: SceneSpec | None = None

    def load(self, spec: SceneSpec) -> IsaacLiftBottleEnv:
        """Reset to a scene (as record_demos.py before its warm-up)."""
        self.spec = spec
        offsets = np.array(spec.start_offsets) if len(spec.start_offsets) == 14 else np.zeros(14)
        for i, s in enumerate(("left", "right")):  # the scene's start pose, as the expert's
            self.scene.q_start[self.idx["arm"][s]] = np.array(START_JOINTS[s]) + offsets[7 * i:7 * i + 7]
        self.scene.reset(spec)
        view = self.scene.robot._articulation_view
        efforts = view.get_max_efforts()
        for s in ("left", "right"):
            efforts[:, self.idx["finger"][s]] = FINGER_MAX_FORCE
        view.set_max_efforts(efforts)
        self._q_target = self.scene.q_start.copy()
        q = self.scene.q_start
        self.q_start = np.concatenate([q[self.idx["arm"]["left"]], q[self.idx["arm"]["right"]],
                                       [q[self.idx["finger"]["left"][0]], q[self.idx["finger"]["right"][0]]]])
        return self

    def step(self, cmd: np.ndarray) -> None:
        """Hold the 16 commands for one 15 Hz tick (one rendered world step)."""
        for i, s in enumerate(("left", "right")):
            self._q_target[self.idx["arm"][s]] = cmd[7 * i:7 * i + 7]
            self._q_target[self.idx["finger"][s]] = cmd[14 + i]
        self.scene.step(self._q_target, render=True)

    def joints(self) -> np.ndarray:
        q = self.scene.robot.get_joint_positions()
        return np.concatenate([q[self.idx["arm"]["left"]], q[self.idx["arm"]["right"]],
                               [q[self.idx["finger"]["left"][0]], q[self.idx["finger"]["right"][0]]]])

    def state(self) -> np.ndarray:
        j = self.joints()
        return np.concatenate([j, kin.v16_eef("left", j[:7]), kin.v16_eef("right", j[7:14])]).astype(np.float32)

    def images(self) -> dict[str, np.ndarray]:
        size = head_camera.OUT_SIZE
        return {"head": (head_camera.camera_frame(self._rgb(self.head), (size, size)) * self._circle).astype(np.uint8),
                **{f"{s}_wrist": letterbox(head_camera.camera_frame(camera_models.wrist_view(self._rgb(self.wrist[s]), s),
                                                                    head_camera.WRIST_OUT))
                   for s in ("left", "right")}}

    def _rgb(self, camera, tries: int = 5) -> np.ndarray:
        """As record_demos.rgb: Isaac occasionally returns no image for a frame; render again."""
        for _ in range(tries):
            img = camera.get_rgb()
            if img is not None and img.size:
                return img
            self.scene.world.render()
        raise RuntimeError(f"{camera.prim_path}: no image after {tries} renders")

    def bottle_pos(self) -> np.ndarray:
        return np.asarray(self.scene.bottle.get_world_pose()[0], dtype=np.float64)

    def bottle_lift(self) -> float:
        return self.scene.bottle_lift()

    def bottle_tilt_deg(self) -> float:
        w, x, y, z = self.scene.bottle.get_world_pose()[1]  # quaternion (w, x, y, z)
        zz = 1.0 - 2.0 * (x * x + y * y)  # world-z component of the bottle's z axis
        return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))

    def success(self) -> bool:
        return self.bottle_lift() >= LIFT_SUCCESS_M

    def close(self) -> None:  # the world is reused for the next episode
        pass
