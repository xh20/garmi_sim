"""GARMI arm kinematics from the URDF, in NumPy (no simulator): forward kinematics and Jacobians.

Frames, verified against the real v16 recordings (0.00 mm residual over 121 frames):
  * `base_link` is GARMI's base frame (the simulated robot is fixed at the world origin, so it is
    also the world frame); the lift joints are at 0 (v16 does not record them).
  * v16's end-effector values are the flange `<side>_fr3_link8` pose plus 103.4 mm along its z axis
    (Franka's standard hand TCP), expressed in the arm base `<side>_fr3_link0`.
  * The real fingertip is `<side>_fr3_hand_tcp` (129.5 mm from the flange: GARMI's long fingertips);
    grasps are planned for it.
"""

from __future__ import annotations

import functools
import os
import xml.etree.ElementTree as ET

import numpy as np

URDF = os.path.expanduser("~/sim/garmi_description/garmi_description/urdf/garmi.urdf")
V16_EEF_OFFSET = 0.1034


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, q):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * k + (1 - np.cos(q)) * k @ k


@functools.lru_cache(maxsize=None)
def _joints():
    root = ET.parse(URDF).getroot()
    return {j.find("child").get("link"): j for j in root.findall("joint")}


@functools.lru_cache(maxsize=None)
def _chain(link: str, base: str):
    """[(origin 4x4, joint type, axis, joint name)] from base to link."""
    joints, out = _joints(), []
    while link != base:
        j = joints[link]
        o = j.find("origin")
        xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0.0] * 3
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0.0] * 3
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = _rpy(*rpy), xyz
        ax = j.find("axis")
        axis = tuple(float(v) for v in ax.get("xyz").split()) if ax is not None else (1.0, 0.0, 0.0)
        out.append((t, j.get("type"), axis, j.get("name")))
        link = j.find("parent").get("link")
    return tuple(out[::-1])


def fk(link: str, base: str, q: dict[str, float]) -> np.ndarray:
    t = np.eye(4)
    for origin, jtype, axis, name in _chain(link, base):
        t = t @ origin
        if jtype in ("revolute", "continuous"):
            m = np.eye(4)
            m[:3, :3] = _axis_angle(axis, q.get(name, 0.0))
            t = t @ m
        elif jtype == "prismatic":
            m = np.eye(4)
            m[:3, 3] = np.asarray(axis) * q.get(name, 0.0)
            t = t @ m
    return t


def arm_names(side: str) -> list[str]:
    return [f"{side}_fr3_joint{i}" for i in range(1, 8)]


def tcp_pose(side: str, q_arm: np.ndarray) -> np.ndarray:
    """4x4 pose of the real fingertip TCP in base_link."""
    return fk(f"{side}_fr3_hand_tcp", "base_link", dict(zip(arm_names(side), q_arm, strict=True)))


def v16_eef(side: str, q_arm: np.ndarray) -> np.ndarray:
    """v16's 9 EEF values: [x, y, z, r11, r12, r13, r21, r22, r23] of flange + 103.4 mm, in the arm base."""
    t = fk(f"{side}_fr3_link8", f"{side}_fr3_link0", dict(zip(arm_names(side), q_arm, strict=True)))
    p = t[:3, 3] + V16_EEF_OFFSET * t[:3, 2]
    return np.concatenate([p, t[0, :3], t[1, :3]])


def pose_error(t_cur: np.ndarray, t_goal: np.ndarray) -> np.ndarray:
    """6-vector [position error, rotation error (axis * angle)] in base_link."""
    dp = t_goal[:3, 3] - t_cur[:3, 3]
    r = t_goal[:3, :3] @ t_cur[:3, :3].T
    angle = np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))
    if angle < 1e-9:
        return np.concatenate([dp, np.zeros(3)])
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2 * np.sin(angle))
    return np.concatenate([dp, w * angle])


def jacobian(side: str, q_arm: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """6x7 numerical Jacobian of the TCP (position + rotation) w.r.t. the arm joints."""
    t0 = tcp_pose(side, q_arm)
    jac = np.zeros((6, 7))
    for i in range(7):
        dq = np.array(q_arm, float)
        dq[i] += eps
        jac[:, i] = pose_error(t0, tcp_pose(side, dq)) / eps
    return jac


def ik_step(side: str, q: np.ndarray, t_goal: np.ndarray, q_rest: np.ndarray, *, damping: float = 0.05,
            null_gain: float = 0.2, rot_weight: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """One damped-least-squares IK step towards t_goal, with a null-space pull towards q_rest.

    Returns (dq, error). Rotation error is weighted by rot_weight (rad counts like rot_weight m).
    """
    err = pose_error(tcp_pose(side, q), t_goal)
    w = np.diag([1, 1, 1, rot_weight, rot_weight, rot_weight])
    jac = w @ jacobian(side, q)
    jjt = jac @ jac.T + damping**2 * np.eye(6)
    j_pinv = jac.T @ np.linalg.solve(jjt, np.eye(6))
    dq = j_pinv @ (w @ err) + (np.eye(7) - j_pinv @ jac) @ (null_gain * (q_rest - q))
    return dq, err
