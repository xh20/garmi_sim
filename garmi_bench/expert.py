"""Scripted expert for the GARMI lift-the-bottle task (simulator-independent: NumPy kinematics only).

Commands absolute joint targets for both arms and both grippers at 15 Hz. It grasps the way the real v16
demonstrations do; every scene copies one real episode (SceneSpec.ref_episode, assets/v16_grasps.json):
  * the cap is grasped (fingertip centre CAP_BELOW_TOP below the bottle's top), with the gripper kept at the start
    opening (35 mm per finger, as v16 never opens wider) and closed onto the cap;
  * approach direction (fingertip z) and its downward tilt from the reference episode (+ a few degrees), fingers
    closing horizontally (fingertip x up);
  * timing from the reference: time to the grasp, lift starting ~0.6 s after it, lift height, episode end;
  * joint speed up to MAX_JOINT_STEP per tick (the human demos reach 0.035-0.05 rad per tick).
Phases: joint-space move to a pre-grasp pose 11-14 cm back along the approach -> straight approach -> close ->
straight lift -> hold. Each phase is minimum-jerk in time.

Corrective reach (aim_error_m > 0, off by default): the move first heads for the pre-grasp of a wrong bottle position
(the true one plus a random horizontal error, as a human's coarse first reach) and corrects to the true pre-grasp
during the move (min-jerk, starting at a random point of it). The state trajectory then points at the wrong target
until the correction, which only the images can explain -- unlike the open-loop plan, whose first steps already give
the target away. Also off by default: a curved move (bow_m, a random sideways / upward detour, as the human paths
are ~18% longer than straight) and a gradual close (close_ramp_s, as the operator closes over ~0.2-0.5 s).

Staged approach (standoff_m > 0, v6; off by default): move -> curve (the curved / tilted approach, ending at a standoff
standoff_m back along the fingertip axis) -> align (closed loop on the measured fingertip pose until within 3 mm / 3 deg)
-> approach (straight, slow insertion along the fingertip axis, closed loop across it) -> close (at once) -> lift -> hold.
If the bottle moves during the insertion (> 5 mm or > 5 deg), or a deliberately offset insertion (recover_prob) misses the
cap, the expert backs off along the axis, re-aims at the bottle's current position, aligns and inserts again (<= 2x).

Observer arm (v7, observer=True): the arm that does not grasp moves its wrist camera to a side view of the cap, across
the grasping arm's approach direction, while the grasping arm reaches; it holds there to the end (_plan_view).

DAgger takeover (v6a, takeover()): a policy drives first; at a trigger the expert continues from the policy's last command
and finishes the grasp the staged way from wherever the hand is (see takeover() for the routes), so the recorded
corrections start in the states the policy itself reaches.
"""

from __future__ import annotations

import dataclasses
import itertools

from garmi_bench import kinematics as kin
from garmi_bench.spec import START_FINGER
from garmi_bench.spec import START_JOINTS
from garmi_bench.spec import SceneSpec
from garmi_bench.spec import v16_grasp
from garmi_bench.spec import v16_grasps
import numpy as np

CONTROL_HZ = 15.0
# rad per tick, both arms. The v16 teleop demos reach 0.035-0.05, but the real client caps every tick at
# --max-joint-step 0.02: demonstrating faster than that teaches a motion the robot then executes clipped, so the
# commanded pose lags further behind every tick, the follower's position error grows and the arm throws a
# cartesian_reflex -- which is what the v7 observer arm did on the robot on 2026-11-12 at 0.57 rad/s.
MAX_JOINT_STEP = 0.02  # = 0.30 rad/s at 15 Hz, the real client's limit
FINGER_CLOSED = 0.0
FINGER_WIDE = START_FINGER  # v16 never opens the gripper beyond its start opening (35 mm per finger)
PHASES = ("move_pregrasp", "approach", "close", "lift", "hold", "done")


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s**3 * (10 - 15 * s + 6 * s * s)


def tcp_rotation(azimuth: float, elevation: float) -> np.ndarray:
    """Fingertip frame: z = approach direction, x = as close to world up as possible, y = z cross x."""
    z = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    up = np.array([0.0, 0.0, 1.0])
    x = up - z * (up @ z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def slerp_rot(r0: np.ndarray, r1: np.ndarray, s: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    from scipy.spatial.transform import Slerp

    s = min(max(float(s), 0.0), 1.0)  # min_jerk can round a hair past 1.0; Slerp rejects that
    return Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([r0, r1])))([s]).as_matrix()[0]


@dataclasses.dataclass
class GraspPlan:
    side: str
    grasp: np.ndarray  # 4x4 fingertip pose at the grasp
    pregrasp: np.ndarray
    lift: np.ndarray
    q_pregrasp: np.ndarray  # 7 arm joints
    approach_s: float
    close_s: float
    lift_s: float
    hold_s: float
    move_s: float = 0.0  # joint-space move to the pre-grasp pose (at least what the speed limit needs)
    dq_lift: np.ndarray | None = None  # joint-space lift (the reference demo's change from grasp to lift top)
    q_aim: np.ndarray | None = None  # pre-grasp joints of the wrong first aim (corrective reach), else None
    aim_error: np.ndarray | None = None  # its position error (m)
    correct_from: float = 1.0  # fraction of the move where the correction starts
    correct_frac: float = 0.0  # ... and how much of the move it takes
    # Curved approach: control point of a quadratic Bezier from the (off-axis) pre-grasp to the grasp. It lies on the
    # approach axis, so the fingertips still arrive along the axis (the curve's end tangent) and do not hit the cap.
    approach_ctrl: np.ndarray | None = None
    # Staged approach (standoff_m > 0, v6): the curved / tilted approach ends at a standoff pose standoff_m back along
    # the fingertip axis; the hand is aligned there on the measured pose, then inserted straight and slowly.
    standoff: np.ndarray | None = None
    q_standoff: np.ndarray | None = None
    insert_need: float = 0.0  # minimum duration of the straight insertion under the rate limit


def _ik(side: str, target: np.ndarray, q_rest: np.ndarray) -> np.ndarray:
    q = q_rest.copy()
    for _ in range(300):
        dq, err = kin.ik_step(side, q, target, q_rest, null_gain=0.05)
        q = q + np.clip(dq, -0.2, 0.2)
        if np.linalg.norm(err[:3]) < 1e-3 and np.linalg.norm(err[3:]) < 1e-2:
            break
    return q


def _joint_limits(side: str) -> list[tuple[float, float]]:
    import xml.etree.ElementTree as ET

    if side not in _LIMITS:
        joints = {j.get("name"): j for j in ET.parse(kin.URDF).getroot().findall("joint")}
        _LIMITS[side] = [
            (float(joints[n].find("limit").get("lower")), float(joints[n].find("limit").get("upper")))
            for n in kin.arm_names(side)
        ]
    return _LIMITS[side]


_LIMITS: dict[str, list[tuple[float, float]]] = {}

MAX_APPROACH_DOWN_DEG = 12.0  # how far below horizontal the hand may travel on its way into the cap
# How far the bottle is raised off the table; the v16 references lift 25-55 cm, far more than the task needs.
# spec.LIFT_SUCCESS_M (0.20) is the success threshold, so the target carries a margin above it: the grasp is never
# perfectly rigid and the bottle settles a little in the fingers on the way up.
LIFT_HEIGHT_M = 0.24


def _lift_scale(side: str, q_grasp: np.ndarray, dq: np.ndarray, height_m: float) -> float:
    """Fraction of the reference lift that raises this arm's fingertips by height_m.

    The reference lift is a joint-space change, and only part of it is vertical, so scaling it by
    height_m / ref["lift_m"] is off by tens of percent (measured: a 20 cm command lifting 12.6 cm). The arm it is
    replayed on is not the reference arm either. Bisecting on the forward kinematics costs ~20 FK calls at plan
    time and lands within a millimetre.
    """
    names = kin.arm_names(side)

    def z_at(s: float) -> float:
        return float(kin.fk(f"{side}_fr3_hand_tcp", "base_link", dict(zip(names, q_grasp + s * dq, strict=True)))[2, 3])

    z0 = z_at(0.0)
    if z_at(1.0) - z0 <= height_m:  # the whole reference lift is still short of the target: use all of it
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if z_at(mid) - z0 < height_m:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _level_approach(z: np.ndarray, max_down_deg: float = MAX_APPROACH_DOWN_DEG) -> np.ndarray:
    """Lift a direction that descends more steeply than `max_down_deg` back towards the horizontal, same heading."""
    z = np.asarray(z, dtype=float)
    z = z / np.linalg.norm(z)
    limit = np.sin(np.radians(max_down_deg))
    if -float(z[2]) <= limit:
        return z
    flat = z[:2]
    n = float(np.linalg.norm(flat))
    if n < 1e-9:  # straight down: no heading left to keep
        return z
    c = np.cos(np.radians(max_down_deg))
    out = np.array([flat[0] / n * c, flat[1] / n * c, -limit])
    return out / np.linalg.norm(out)


def _level_grasp(rot: np.ndarray, max_down_deg: float = MAX_APPROACH_DOWN_DEG) -> np.ndarray:
    """Turn a grasp frame until its fingertip axis is no steeper than `max_down_deg` below horizontal.

    The straight insertion runs from the standoff to the grasp along the fingertip axis, so levelling the
    approach direction alone would not help: the hand would still be pointed down and the cap would no longer
    slide into the slot between the fingers. Rotating the whole frame about the horizontal axis perpendicular
    to the fingertip axis keeps the fingers around the cap and brings the hand in from the side instead.

    Diving grasps are what the v16 references plus `--approach-tilt 40` produced, up to 40 deg below
    horizontal: descending onto a standing bottle, the last centimetres run along the bottle's own axis, where
    a distance misjudged by the 2 cm measured on the robot topples it rather than sliding past it.
    """
    from scipy.spatial.transform import Rotation

    rot = np.asarray(rot, dtype=float)
    z = rot[:, 2]
    target = _level_approach(z, max_down_deg)
    axis = np.cross(z, target)
    s = float(np.linalg.norm(axis))
    if s < 1e-9:
        return rot
    angle = float(np.arctan2(s, float(z @ target)))
    return Rotation.from_rotvec(axis / s * angle).as_matrix() @ rot


def plan(
    spec: SceneSpec,
    rng: np.random.Generator,
    aim_error_m: float = 0.0,
    orient_jitter_deg: float = 2.0,
    approach_curve_m: float = 0.0,
    approach_tilt_deg: float = 0.0,
    standoff_m: float = 0.0,
) -> GraspPlan:
    side = spec.arm
    ref = v16_grasp(spec.ref_episode)
    # The real hand's orientation at the grasp, turned by a small random rotation (~2 deg); the real robot reached
    # exactly this orientation within GRASP_JITTER_M of the scene's bottle position.
    from scipy.spatial.transform import Rotation

    jitter = Rotation.from_rotvec(rng.normal(0.0, np.radians(orient_jitter_deg), 3)).as_matrix()
    # Levelled first, so the standoff, the straight insertion along the fingertip axis and the approach
    # direction derived from it all come in from the side rather than diving onto the cap.
    rot = _level_grasp(jitter @ np.array(ref["rotation"]))

    grasp = np.eye(4)
    grasp[:3, :3], grasp[:3, 3] = rot, [spec.bottle_x, spec.bottle_y, spec.cap_z]
    pregrasp = grasp.copy()
    back = rng.uniform(0.11, 0.14)
    z = rot[:, 2]  # direction of the final approach: the fingertip axis ...
    if approach_tilt_deg > 0:
        # ... or tilted against it, as the human hand arrives (v16: final approach directions spread ~30 deg per arm
        # around the mean, the sim's straight approach ~15). Within the fingers' plane (about the closing axis y) up to
        # the full tilt: the cap slides along the slot; towards a finger (about x) up to 0.6 x: with 35 mm per finger
        # around a 14 mm cap radius, ~25 deg over the last 45 mm of the fingertips still clears the cap.
        rng_t = np.random.default_rng([spec.seed, 13])
        a, b = np.radians(rng_t.uniform(-1.0, 1.0, 2) * approach_tilt_deg * np.array([1.0, 0.6]))
        z = rot @ np.array([np.sin(a) * np.cos(b), -np.sin(b), np.cos(a) * np.cos(b)])
    z = _level_approach(z)
    # Staged approach: the curve ends at the standoff (on the fingertip axis), not at the grasp.
    standoff = None
    end = grasp[:3, 3]
    if standoff_m > 0:
        standoff = grasp.copy()
        standoff[:3, 3] -= rot[:, 2] * standoff_m
        end = standoff[:3, 3]
        pregrasp[:3, 3] = end
    pregrasp[:3, 3] -= z * back
    approach_ctrl = None
    if approach_tilt_deg > 0:
        approach_ctrl = end - z * 0.5 * back  # (a straight line unless the curve below moves the pre-grasp)
    if approach_curve_m > 0:  # curved approach: pre-grasp off the approach axis (sideways / up), see GraspPlan
        rng_c = np.random.default_rng([spec.seed, 12])
        d = rng_c.normal(size=3)
        d -= (d @ z) * z
        d *= rng_c.uniform(0.3, 1.0) * approach_curve_m / np.linalg.norm(d)
        if d[2] < 0:  # never from below the cap's height plane
            d = -d
        pregrasp[:3, 3] += d
        approach_ctrl = end - z * 0.5 * back
    lift = grasp.copy()
    lift[2, 3] += LIFT_HEIGHT_M

    # IK seeded with, and its null space pulled towards, the real arm's joints at the grasp: the same arm
    # configuration as the human demo (the start pose is far from it and leads to other, often worse, solutions).
    q_rest = np.array(ref["q_grasp"])
    q = _ik(side, pregrasp, q_rest)
    # Straight-line segments must be slow enough that no joint exceeds the rate limit: clipping joints one by
    # one bends the fingertip path sideways (30-45 mm on some approaches), and the long fingertips then hit and
    # topple the bottle. Duration >= MIN_JERK_PEAK * max|dq/ds| / (limit * 15 Hz).
    q_standoff, insert_need = None, 0.0
    if standoff is not None:
        q_standoff, approach_need = _segment_speed(side, q, pregrasp, standoff, q)
        q_grasp, insert_need = _segment_speed(side, q_standoff, standoff, grasp, q_standoff)
    else:
        q_grasp, approach_need = _segment_speed(side, q, pregrasp, grasp, q)
    # The lift repeats the real arm's motion from the grasp to the top of the lift (a joint-space change applied to
    # this arm's grasp configuration, which is seeded from the real one): the real lift height, and the hand turning
    # on the way up as it did -- a straight vertical line at a fixed orientation runs into joint limits.
    # The reference demos lift 25-55 cm, far more than the task needs; scaling the joint-space change keeps the
    # hand's turn on the way up (a straight vertical line at a fixed orientation runs into joint limits) while
    # stopping at LIFT_HEIGHT_M. A shorter lift also keeps the bottle inside the wrist camera's view and leaves
    # the observer arm, which backs off at the same moment, less to clear. The scale is solved on this arm's own
    # forward kinematics, not from the reference's lift_m: see _lift_scale.
    dq_lift = np.array(ref["q_lift_peak"]) - np.array(ref["q_grasp"])
    dq_lift = dq_lift * _lift_scale(side, q_grasp, dq_lift, LIFT_HEIGHT_M)
    lift_need = MIN_JERK_PEAK * np.abs(dq_lift).max() / (SEGMENT_STEP_LIMIT * CONTROL_HZ)
    # Timing after the reference episode: holding starts ~CLAMP_S after the close command, the lift starts CLOSE_S
    # after it and peaks t_peak_s after holding (as in the demo), the episode ends t_after_grasp_s after holding.
    lift_s = max(ref["t_peak_s"] - (CLOSE_S - CLAMP_S), lift_need, 0.8)
    hold_s = max(0.4, ref["t_after_grasp_s"] - (CLOSE_S - CLAMP_S) - lift_s)
    approach_s = max(rng.uniform(1.4, 2.4), approach_need)
    move_s = ref["t_grasp_s"] - approach_s - CLAMP_S
    lift[:3, :3] = grasp[:3, :3]
    lift[2, 3] = grasp[2, 3] + LIFT_HEIGHT_M  # nominal (diagnostics); the commanded lift is dq_lift
    out = GraspPlan(
        side,
        grasp,
        pregrasp,
        lift,
        q,
        approach_s=approach_s,
        close_s=CLOSE_S,
        lift_s=lift_s,
        hold_s=hold_s,
        move_s=move_s,
        dq_lift=dq_lift,
        approach_ctrl=approach_ctrl,
        standoff=standoff,
        q_standoff=q_standoff,
        insert_need=insert_need,
    )
    if aim_error_m > 0:  # (drawn last, so the plan above is the same as without it)
        rng_aim = np.random.default_rng(rng.integers(2**32))
        err = np.r_[
            np.clip(rng_aim.normal(0.0, aim_error_m, 2), -2 * aim_error_m, 2 * aim_error_m),
            np.clip(rng_aim.normal(0.0, aim_error_m / 3), -aim_error_m, aim_error_m),
        ]
        aim = pregrasp.copy()
        aim[:3, 3] += err
        out.q_aim, out.aim_error = _ik(side, aim, q), err
        out.correct_from = float(rng_aim.uniform(0.35, 0.7))
        out.correct_frac = float(rng_aim.uniform(0.25, 1.0 - out.correct_from))
    return out


MIN_JERK_PEAK = 1.875  # peak of d/ds min_jerk(s)
MAX_SEGMENT_S = 8.0  # longest acceptable straight approach / lift; slower means a near-singular path
SEGMENT_STEP_LIMIT = 0.018  # rad per tick, a little under MAX_JOINT_STEP; the observer arm uses it too, so both
# arms move at one speed (until 2026-11-12 the grasping arm was limited here and the observer separately at
# MAX_JOINT_STEP, so the two moved at visibly different speeds in the same episode)
CLAMP_S = 0.2  # fingers reach the cap ~0.2 s after the close command (v16: holding 0.2 s after closing starts)
CLOSE_S = 0.45  # close command -> lift start; with the lift's slow start, the 2 cm rise comes ~0.6 s after holding


def _segment_speed(side: str, q0: np.ndarray, t0: np.ndarray, t1: np.ndarray, q_rest: np.ndarray, n: int = 40):
    """IK along the straight segment t0 -> t1 (constant orientation of t1). Returns (q at t1, minimum duration)."""
    q, qs = q0.copy(), [q0.copy()]
    for k in range(1, n + 1):
        t = t1.copy()
        t[:3, 3] = t0[:3, 3] + (t1[:3, 3] - t0[:3, 3]) * k / n
        for _ in range(20):
            dq, err = kin.ik_step(side, q, t, q_rest, null_gain=0.05)
            q = q + dq
            if np.linalg.norm(err[:3]) < 2e-4:
                break
        qs.append(q.copy())
    dq_ds = np.abs(np.diff(np.array(qs), axis=0)).max() * n
    return q, MIN_JERK_PEAK * dq_ds / (SEGMENT_STEP_LIMIT * CONTROL_HZ)


class LiftBottleExpert:
    """Per-tick controller. step() returns the 16 joint/gripper targets in v16 order."""

    def __init__(
        self,
        spec: SceneSpec,
        seed: int,
        aim_error_m: float = 0.0,
        bow_m: float = 0.0,
        close_ramp_s: float = 0.0,
        orient_jitter_deg: float = 2.0,
        approach_curve_m: float = 0.0,
        speed_jitter: float = 0.0,
        approach_tilt_deg: float = 0.0,
        standoff_m: float = 0.0,
        insert_speed: tuple = (0.03, 0.05),
        recover_prob: float = 0.0,
        max_retries: int = 2,
        *,
        observer: bool = False,
    ):
        self._spec = spec
        self.rng = np.random.default_rng([seed, 7])
        self.plan = plan(
            spec, self.rng, aim_error_m, orient_jitter_deg, approach_curve_m, approach_tilt_deg, standoff_m
        )
        # Staged approach (v6): align at the standoff on the measured pose, slow straight insertion, and recovery when the
        # bottle is touched (back off, re-align on the bottle's new position, insert again). Own random stream.
        self.staged = self.plan.standoff is not None
        rng_v6 = np.random.default_rng([seed, 17])
        self.insert_speed = float(rng_v6.uniform(*insert_speed))
        self.max_retries = max_retries
        self.retries = 0
        self.offset = np.zeros(3)  # deliberate sideways error of the first insertion (to record a recovery)
        if self.staged and rng_v6.random() < recover_prob:
            # mostly along the closing axis (y: a fingertip runs into the cap), a little up/down (x)
            sideways = rng_v6.choice([-1.0, 1.0]) * rng_v6.uniform(0.03, 0.045)
            self.offset = self.plan.grasp[:3, :3] @ np.array([rng_v6.normal(0.0, 0.005), sideways, 0.0])
        self.retreat_m = float(rng_v6.uniform(0.05, 0.07))
        self.cur_grasp = self.plan.grasp.copy()  # current targets (moved with the bottle after a recovery)
        self.cur_standoff = None if not self.staged else self.plan.standoff.copy()
        if self.staged and np.any(self.offset):
            self.cur_grasp[:3, 3] += self.offset
            self.cur_standoff[:3, 3] += self.offset
        self.q_bias = np.zeros(7)
        self._q_ref: np.ndarray | None = None
        self._desired_prev: np.ndarray | None = None
        self._bottle_plan: np.ndarray | None = None  # bottle position at the first tick
        self._bottle_insert: tuple | None = None  # (position, quaternion) at the start of the current insertion
        self._quat_plan: np.ndarray | None = None
        self._ok_ticks = 0
        self._sub = "line"  # insertion sub-phase: "line", then "settle"
        self.events: list[tuple[int, str]] = []  # (tick, phase entered), every transition (phase_ticks keeps the last)
        self.log: dict = {
            "offset_mm": (self.offset * 1000).round(1).tolist(),
            "align_s": [],
            "contacts": [],
            "align_timeouts": 0,
            "disturbed_before_insert_mm": 0.0,
        }
        self.side = self.plan.side
        offsets = np.array(spec.start_offsets) if len(spec.start_offsets) == 14 else np.zeros(14)
        self.q_start = {
            "left": np.array(START_JOINTS["left"]) + offsets[:7],
            "right": np.array(START_JOINTS["right"]) + offsets[7:],
        }
        self.q_cmd = {s: self.q_start[s].copy() for s in ("left", "right")}
        self.finger = dict.fromkeys(("left", "right"), START_FINGER)
        self.phase, self.t_phase, self.tick = "move_pregrasp", 0, 0
        self.phase_ticks: dict[str, int] = {}
        dist = np.abs(self.plan.q_pregrasp - self.q_cmd[self.side]).max()
        if self.plan.q_aim is not None:
            dist = max(dist, np.abs(self.plan.q_aim - self.q_cmd[self.side]).max())
        # The reference episode's time to the grasp, but never faster than the speed limit allows.
        self.move_s = max(self.plan.move_s, MIN_JERK_PEAK * dist / (0.9 * MAX_JOINT_STEP * CONTROL_HZ), 1.0)
        rng_h = np.random.default_rng([seed, 11])  # (own stream: the plan above is the same with or without these)
        # Curved move (bow_m > 0): a Cartesian detour of up to bow_m, perpendicular to the start -> pre-grasp line and
        # never downwards, added in joint space as dq_bow * sin(pi * s^p) (the apex early or late in the move, p).
        self.dq_bow, self.bow_p, self.bow_offset = None, 1.0, None
        if bow_m > 0:
            q0, q1 = self.q_start[self.side], self.plan.q_pregrasp
            q_mid = 0.5 * (q0 + q1)
            line = kin.tcp_pose(self.side, q1)[:3, 3] - kin.tcp_pose(self.side, q0)[:3, 3]
            u = line / np.linalg.norm(line)
            d = rng_h.normal(size=3)
            d -= (d @ u) * u
            d *= rng_h.uniform(0.2, 1.0) * bow_m / np.linalg.norm(d)
            d[2] = abs(d[2])
            target = kin.tcp_pose(self.side, q_mid)
            target[:3, 3] += d
            self.dq_bow = _ik(self.side, target, q_mid) - q_mid
            self.bow_offset = d
            self.bow_p = float(np.exp(rng_h.uniform(np.log(0.6), np.log(1.6))))
        # Gradual close (close_ramp_s > 0): the finger command runs from open to closed over U(0.5, 1) x close_ramp_s,
        # as the human operator's; the close phase is longer by the time the fingers need to reach the cap.
        self.close_ramp = float(rng_h.uniform(0.5, 1.0) * close_ramp_s) if close_ramp_s > 0 else 0.0
        if self.close_ramp:
            self.plan.close_s += 0.6 * self.close_ramp
        # Speed (speed_jitter > 0): the move takes x U(1 / (1 + j), 1 + j) of the reference timing (never faster than the
        # speed limit allows), the approach x U(1, 1 + j) (only slower: it is already at its rate-limited minimum).
        if speed_jitter > 0:
            lo, hi = np.log(1.0 / (1.0 + speed_jitter)), np.log(1.0 + speed_jitter)
            need = MIN_JERK_PEAK * dist / (0.9 * MAX_JOINT_STEP * CONTROL_HZ)
            self.move_s = max(self.move_s * float(np.exp(rng_h.uniform(lo, hi))), need, 1.0)
            self.plan.approach_s = min(self.plan.approach_s * float(np.exp(rng_h.uniform(0.0, hi))), MAX_SEGMENT_S)
        self._seg_start: np.ndarray | None = None
        self._lift_from: np.ndarray | None = None
        # Observer arm (v7, observer=True): the other arm brings its wrist camera to a side view of the cap, across the
        # grasping arm's approach direction (see _plan_view), while the grasping arm moves; it holds there to the end.
        self.observer = observer
        self.obs_side = "right" if self.side == "left" else "left"
        self.q_view: np.ndarray | None = None
        self.obs_move_s = 0.0
        self.obs_start_s = 0.0
        self.obs_retreat_s = self.VIEW_RETREAT_S
        self.obs_log: dict = {}
        if observer:
            self._plan_view(np.random.default_rng([seed, 29]))
            if self.q_view is not None:
                self._sync_observer_arrival()
        # DAgger takeover (v6a, see takeover()): overrides of the curve segment and the insertion, None = the plan's.
        self.took_over = False
        self._curve: tuple | None = None  # (duration s, Bezier control point or None)
        self._insert_dur: float | None = None
        self._lat0 = np.zeros(3)  # sideways offset at the start of the insertion, corrected early in it

    # --- observer arm (v7) -----------------------------------------------------------------------------------------
    # Wrist camera -> cap. The far end is generous on purpose: a side view of a bottle on the grasping arm's own
    # far side (y around 0.5, where most of the v16 left-arm demos put it) and keeping the observer out of the
    # grasping arm's way are in conflict close up, and the way out is distance -- the camera backs off along the
    # view direction, which moves the whole observer arm towards its own side and away from the grasp and lift
    # path. The pick stays weighted towards VIEW_DIST_M[0] (VIEW_PREFER_DIST_M), so the far end is only used when
    # nothing closer is feasible.
    VIEW_DIST_M = (0.25, 0.65)
    # A side view, only a little above the cap: the insertion depth the grasping wrist cannot see is a sideways
    # offset from the side, and vanishes again from overhead. Until 2026-11-12 these were 30-65 deg of elevation
    # and +-70 deg of yaw, so a draw could sit almost behind the hand looking down the approach -- "from the
    # middle", where the view adds nothing.
    VIEW_ELEV_DEG = (15.0, 40.0)  # camera above the cap (level: the arm hit the table)
    VIEW_YAW_DEG = 25.0  # spread of the view direction around the perpendicular to the approach
    VIEW_DRAWS = 48
    # Preference among the feasible draws (a random pick weighted by these Gaussians): closer and more perpendicular
    # views show the insertion depth best, but every feasible view can be picked, so the observer's pose varies.
    VIEW_PREFER_YAW_DEG = 12.0
    VIEW_PREFER_DIST_M = 0.15  # (beyond VIEW_DIST_M[0])
    VIEW_CLEAR_M = 0.10  # the observer's fingertips stay this far from the bottle's axis (horizontally)
    VIEW_CROSS_M = 0.45  # the camera at most this far past the centre plane (y = 0) towards the grasping side

    def view_error_deg(self, q_obs: np.ndarray) -> float | None:
        """Angle between the observer camera's optical axis (at joints q_obs) and the direction to the planned cap."""
        if self.q_view is None:
            return None
        cam = kin.fk(
            f"camera_d435i_{self.obs_side}_color_optical_frame",
            "base_link",
            dict(zip(kin.arm_names(self.obs_side), q_obs, strict=True)),
        )
        d = self.plan.grasp[:3, 3] - cam[:3, 3]
        return float(np.degrees(np.arccos(np.clip(cam[:3, 2] @ d / np.linalg.norm(d), -1.0, 1.0))))

    def _plan_view(self, rng: np.random.Generator) -> None:
        """Pick the observer's wrist-camera pose: on the observer's side of the bottle, looking down at the cap from
        above and from the side, across the grasping arm's approach direction (the insertion depth along it, which the
        grasping wrist camera hardly sees, becomes a sideways offset in this view), VIEW_DIST_M away at VIEW_ELEV_DEG
        below the horizontal, VIEW_YAW_DEG around the perpendicular. Of VIEW_DRAWS random draws, those the IK reaches with
        the fingertips clear of the bottle, joint margins, the whole arm clear of the table top on the way (Isaac
        blocked the elbow / forearm on the table for low side views) and of the grasping arm are feasible; one of them
        is picked at random, weighted towards close and perpendicular views (VIEW_PREFER_*). q_view stays None if none."""
        obs, g = self.obs_side, self.plan.grasp
        cap = g[:3, 3]
        a = g[:3, 2].copy()
        a[2] = 0.0
        a /= np.linalg.norm(a)
        n = np.array([-a[1], a[0], 0.0])
        base = kin.fk(f"{obs}_fr3_link0", "base_link", {})[:3, 3]
        if n @ (base - cap) < 0:  # the observer's side of the bottle
            n = -n
        cam_link = f"camera_d435i_{obs}_color_optical_frame"
        names = dict(zip(kin.arm_names(obs), self.q_start[obs], strict=True))
        tcp0 = kin.tcp_pose(obs, self.q_start[obs])
        tcp_to_cam = np.linalg.inv(tcp0) @ kin.fk(cam_link, "base_link", names)
        refs = [r for r in v16_grasps() if r["arm"] == obs]
        feasible = []
        for k in range(self.VIEW_DRAWS):
            yaw = np.radians(rng.uniform(-1.0, 1.0) * self.VIEW_YAW_DEG)
            d = np.array([np.cos(yaw) * n[0] - np.sin(yaw) * n[1], np.sin(yaw) * n[0] + np.cos(yaw) * n[1], 0.0])
            el = np.radians(rng.uniform(*self.VIEW_ELEV_DEG))
            dist = rng.uniform(*self.VIEW_DIST_M)
            pos = cap + dist * (np.cos(el) * d + np.array([0.0, 0.0, np.sin(el)]))
            z = (cap - pos) / np.linalg.norm(cap - pos)  # optical axis at the cap
            y = np.array([0.0, 0.0, -1.0]) - z * (-z[2])  # image down = world down
            y /= np.linalg.norm(y)
            cam = np.eye(4)
            cam[:3, :3], cam[:3, 3] = np.stack([np.cross(y, z), y, z], axis=1), pos
            tcp = cam @ np.linalg.inv(tcp_to_cam)
            # IK seeded with, and pulled towards, the real arm's configuration at the v16 grasp nearest to this pose
            # (as the grasping arm's plan): a natural reaching configuration, which the joint-space move from the start
            # reaches without sweeping the arm past the head and torso (the Cartesian path from the raised start did).
            ref = min(refs, key=lambda r: np.linalg.norm(np.array([r["x"], r["y"], r["z"]]) - tcp[:3, 3]))
            q = _ik(obs, tcp, np.array(ref["q_grasp"]))
            err = kin.pose_error(kin.tcp_pose(obs, q), tcp)
            tip = kin.tcp_pose(obs, q)[:3, 3]
            clear = float(np.linalg.norm(tip[:2] - cap[:2]))
            # (and the camera at most VIEW_CROSS_M past the robot's centre plane: reaching further across, the observer
            # runs into the grasping arm or its own joint limits on the way)
            cross = pos[1] if obs == "right" else -pos[1]
            ok = (
                np.linalg.norm(err[:3]) < 0.01
                and np.linalg.norm(err[3:]) < 0.1
                and clear >= self.VIEW_CLEAR_M
                and cross <= self.VIEW_CROSS_M
            )
            # Isaac blocked the observer on stretched poses (the point VIEW_ABOVE_M above not reachable) and near joint
            # limits: require both.
            margin = float(min(min(qi - lo, hi - qi) for qi, (lo, hi) in zip(q, _joint_limits(obs), strict=True)))
            above = tcp.copy()
            above[2, 3] += self.VIEW_ABOVE_M
            ok = ok and margin >= self.VIEW_JOINT_MARGIN
            if ok:
                q_ab = _ik(obs, above, q)
                ok = np.linalg.norm(kin.pose_error(kin.tcp_pose(obs, q_ab), above)[:3]) < 0.01
                ok = ok and self._table_clear(obs, [self.q_start[obs], q_ab, q])
                ok = ok and self._arms_clear(obs, [q_ab, q])
            if ok:
                dq = float(np.abs(q - self.q_start[obs]).max())
                feasible.append((q, pos, k, clear, dq, tcp, float(np.degrees(yaw)), float(dist), float(np.degrees(el))))
        if not feasible:
            self.obs_log = {"view": None}
            return
        w = np.array(
            [
                np.exp(
                    -0.5 * (f[6] / self.VIEW_PREFER_YAW_DEG) ** 2
                    - 0.5 * ((f[7] - self.VIEW_DIST_M[0]) / self.VIEW_PREFER_DIST_M) ** 2
                )
                for f in feasible
            ]
        )
        best = feasible[int(rng.choice(len(feasible), p=w / w.sum()))]
        self.q_view, self._view_tcp = best[0], best[5]
        # Two joint-space min-jerk segments: start -> VIEW_ABOVE_M above the view pose -> down to it. A direct move
        # from the raised start sweeps the hand over the bottle and onto its top (the view is beyond the bottle's
        # near side); from above, the descent is vertical and VIEW_CLEAR_M off the bottle's axis.
        above = self._view_tcp.copy()
        above[2, 3] += self.VIEW_ABOVE_M
        self.q_above = _ik(obs, above, self.q_view)
        if np.linalg.norm(kin.pose_error(kin.tcp_pose(obs, self.q_above), above)[:3]) > 0.01:
            self.q_above = self.q_view.copy()  # (above not reachable: direct move)
        d1 = float(np.abs(self.q_above - self.q_start[obs]).max())
        d2 = float(np.abs(self.q_view - self.q_above).max())
        speed = SEGMENT_STEP_LIMIT * CONTROL_HZ / MIN_JERK_PEAK  # the grasping arm's limit, so both arms match
        self.obs_move_s1 = max(float(rng.uniform(1.8, 2.6)), d1 / speed)
        self.obs_move_s2 = max(float(rng.uniform(1.0, 1.4)), d2 / speed)
        self.obs_move_s = self.obs_move_s1 + self.obs_move_s2
        # The retreat gets the same treatment. On a fixed VIEW_RETREAT_S the 0.25 m back and 0.15 m up asked for
        # about 0.45 m/s at the min-jerk peak, well past the joint rate limit: the command sat pinned at
        # SEGMENT_STEP_LIMIT for the whole segment, the arm fell up to 0.22 rad behind it (always joint 2, which
        # carries the gravity load) and then caught up in a single tick. Measured over the v7b smoke that corrupted
        # 3 of 30 episodes -- the recorded action no longer described the recorded state.
        back = self._view_tcp.copy()
        back[:3, 3] += -self.VIEW_RETREAT_M[0] * self._view_tcp[:3, 2] + np.array([0.0, 0.0, self.VIEW_RETREAT_M[1]])
        q_back = _ik(obs, back, self.q_view)
        d3 = float(np.abs(q_back - self.q_view).max())
        self.obs_retreat_s = max(self.VIEW_RETREAT_S, d3 / speed)
        self.obs_log = {
            "view": {
                "cam_m": np.round(best[1], 3).tolist(),
                "feasible": len(feasible),
                "pick": best[2],
                "yaw_deg": round(best[6], 1),
                "dist_m": round(best[7], 3),
                "elev_deg": round(best[8], 1),
                "tip_clear_mm": round(best[3] * 1000),
                "dq_max": round(best[4], 2),
                "move_s": round(self.obs_move_s, 2),
                "retreat_s": round(self.obs_retreat_s, 2),
            }
        }

    VIEW_ABOVE_M = 0.15
    VIEW_JOINT_MARGIN = 0.4  # rad from every joint limit at the view pose
    VIEW_TABLE_CLEAR_M = 0.08  # every observer link origin (link3 .. fingertip) over the table stays this far above it
    _TABLE_LINKS = ("link3", "link4", "link5", "link6", "link7", "link8", "hand_tcp")

    VIEW_ARM_CLEAR_M = 0.25  # observer link origins (link3 .. fingertip) vs the grasping arm's at pre-grasp / standoff
    # How far the observer's own links may reach across the robot's centre plane towards the grasping side.
    # VIEW_CROSS_M bounds the camera alone, which let an elbow or forearm swing across and sit in the grasping
    # arm's way while the camera itself stayed on its side.
    VIEW_CENTRE_CROSS_M = 0.05

    def _arms_clear(self, side: str, qs: list) -> bool:
        """Whether the observer at the joint configurations `qs` keeps VIEW_ARM_CLEAR_M from the grasping arm at its
        pre-grasp and standoff configurations and from the grasp fingertip, and keeps all of its own links on its
        own side of the centre plane (within VIEW_CENTRE_CROSS_M)."""
        g = self.side
        q_g = [self.plan.q_pregrasp] + ([self.plan.q_standoff] if self.plan.q_standoff is not None else [])

        def pts(s, q):
            names = dict(zip(kin.arm_names(s), q, strict=True))
            return [kin.fk(f"{s}_fr3_{link}", "base_link", names)[:3, 3] for link in self._TABLE_LINKS]

        own = [p for q in qs for p in pts(side, q)]
        # y > 0 is the left arm's side: the observer's links stay on its own signed side of y = 0.
        cross = max((p[1] if side == "right" else -p[1]) for p in own)
        if cross > self.VIEW_CENTRE_CROSS_M:
            return False
        other = [p for q in q_g for p in pts(g, q)] + [self.plan.grasp[:3, 3]]
        return all(np.linalg.norm(a - b) >= self.VIEW_ARM_CLEAR_M for a in own for b in other)

    def _table_clear(self, side: str, waypoints: list, n: int = 12) -> bool:
        """Whether the joint-space path through `waypoints` keeps the arm's links VIEW_TABLE_CLEAR_M above the table top
        wherever they are over it (3 cm margin at the near edge)."""
        sp = self._spec
        x0, x1 = sp.table_front_x - 0.03, sp.table_front_x + sp.table_depth
        y0, y1 = sp.table_center_y - sp.table_width / 2, sp.table_center_y + sp.table_width / 2
        for qa, qb in itertools.pairwise(waypoints):
            for s in np.linspace(0.0, 1.0, n):
                names = dict(zip(kin.arm_names(side), (1 - s) * qa + s * qb, strict=True))
                for link in self._TABLE_LINKS:
                    p = kin.fk(f"{side}_fr3_{link}", "base_link", names)[:3, 3]
                    if x0 <= p[0] <= x1 and y0 <= p[1] <= y1 and p[2] < sp.table_height + self.VIEW_TABLE_CLEAR_M:
                        return False
        return True

    # Back along the view axis, and up, once the fingers have the cap. Far enough that the lift has the space to
    # itself: at (0.12, 0.06) and starting only at the lift, the observer was still beside the bottle while it
    # came up, which is where the two arms crowded each other on the robot.
    VIEW_RETREAT_M = (0.25, 0.15)
    VIEW_RETREAT_S = 1.2

    def _sync_observer_arrival(self) -> None:
        """Time the observer so that both arms reach their working positions together: the observer arrives at the
        view pose exactly as the grasping arm arrives at the standoff, the start of the straight insertion.

        Until 2026-11-12 both arms simply started at tick 0, so the observer rushed to its post and waited there --
        measured over the v7b smoke, 1.9 to 10.3 s of it, median 5.6 s. Two arms moving on unrelated schedules is
        also what the pair looked like on the robot. Whichever arm has further to go now sets the pace: if the
        observer is the slower one it leads from tick 0 and the grasping arm's move to the pre-grasp is stretched
        to match, which is the "move the observer first" case; otherwise the observer waits at its start pose and
        sets off late. Either way neither arm is left standing at its destination while the other is still moving.
        """
        # The grasping arm is at the standoff when the curve segment ends; unstaged, the straight approach starts
        # right after the move to the pre-grasp.
        t_standoff = self.move_s + (self.plan.approach_s if self.staged else 0.0)
        if self.obs_move_s > t_standoff:
            self.move_s = max(self.obs_move_s - (self.plan.approach_s if self.staged else 0.0), 1.0)
            t_standoff = self.move_s + (self.plan.approach_s if self.staged else 0.0)
        self.obs_start_s = max(t_standoff - self.obs_move_s, 0.0)
        if self.obs_log.get("view"):
            self.obs_log["view"]["start_s"] = round(self.obs_start_s, 2)
            self.obs_log["view"]["standoff_s"] = round(t_standoff, 2)

    def _observer_step(self) -> None:
        """Joint-space min-jerk from the start to the view pose, timed by _sync_observer_arrival so that it lands
        as the grasping arm reaches the standoff; hold; when the grasping arm has the cap, back off along the view
        axis and up (Cartesian, still looking at the cap's start)."""
        if self.q_view is None:
            return
        obs = self.obs_side
        if self.tick == 0:
            self._obs_retreat_tick = None
        # From the close, not from the lift: the fingers have the cap by then, so the view has done its job and
        # the observer can be out of the way before the bottle starts coming up.
        if self.phase in ("close", "lift", "hold", "done") and self._obs_retreat_tick is None:
            self._obs_retreat_tick = self.tick
            self.obs_log["retreat_tick"] = self.tick
        if self._obs_retreat_tick is not None:
            s = min_jerk((self.tick - self._obs_retreat_tick) / (self.obs_retreat_s * CONTROL_HZ))
            back = self._view_tcp.copy()
            back[:3, 3] += -self.VIEW_RETREAT_M[0] * self._view_tcp[:3, 2] + np.array(
                [0.0, 0.0, self.VIEW_RETREAT_M[1]]
            )
            target = self._view_tcp.copy()
            target[:3, 3] = (1 - s) * self._view_tcp[:3, 3] + s * back[:3, 3]
            q = self.q_cmd[obs].copy()
            for _ in range(5):
                dq, _ = kin.ik_step(obs, q, target, self.q_view, null_gain=0.1)
                q = q + dq
        elif self.tick < self.obs_start_s * CONTROL_HZ:
            q = self.q_start[obs]  # the grasping arm has further to go: hold, and set off late enough to arrive with it
        elif self.tick < (self.obs_start_s + self.obs_move_s1) * CONTROL_HZ:
            s = min_jerk((self.tick - self.obs_start_s * CONTROL_HZ) / (self.obs_move_s1 * CONTROL_HZ))
            q = (1 - s) * self.q_start[obs] + s * self.q_above
        else:
            s = min_jerk(
                (self.tick - (self.obs_start_s + self.obs_move_s1) * CONTROL_HZ) / (self.obs_move_s2 * CONTROL_HZ)
            )
            q = (1 - s) * self.q_above + s * self.q_view
        step = q - self.q_cmd[obs]
        peak = np.abs(step).max()
        if peak > SEGMENT_STEP_LIMIT:  # the same ceiling as the grasping arm's segments
            step *= SEGMENT_STEP_LIMIT / peak
        self.q_cmd[obs] = self.q_cmd[obs] + step

    # --- DAgger takeover (v6a) -----------------------------------------------------------------------------------
    CORRIDOR_EXTRA_M = 0.04  # insertion corridor: up to standoff + this behind the cap ...
    CORRIDOR_LAT_M = 0.04  # ... and within this of the fingertip axis
    NEAR_LAT_M = 0.012  # closer than 3 cm to the cap, a larger sideways offset would hit the cap: back off first
    CORRIDOR_ROT_RAD = (
        0.52  # ... and the hand turned by < 30 deg from the planned grasp (the policy's own good grasps: ~25)
    )

    def takeover(self, cmd16: np.ndarray, q_meas: np.ndarray, bottle0: tuple, bottle: tuple, tick: int) -> str:
        """Take over from a policy at an arbitrary state (DAgger): continue from the policy's last command and finish the
        grasp the staged way. bottle0: the bottle's pose at the episode start (what the plan refers to); bottle: now.
        Returns the route taken:
          grasp   -- the policy closed the gripper on the cap: keep it closed, lift;
          reopen  -- the policy closed the gripper off the cap: open at once, back off along the axis, re-align, insert;
          insert  -- in the insertion corridor: insert from here, the sideways error corrected in the first 60%;
          backoff -- too close to the cap with a large sideways error, or past it: back off, re-align, insert;
          reach   -- elsewhere: straight to the standoff, align, insert."""
        assert self.staged, "takeover needs the staged approach (standoff_m > 0)"
        side, p = self.side, self.plan
        self.took_over = True
        self.q_cmd = {"left": np.asarray(cmd16[:7], float).copy(), "right": np.asarray(cmd16[7:14], float).copy()}
        self.finger = {"left": float(cmd16[14]), "right": float(cmd16[15])}
        self.tick, self.t_phase = tick, 0
        self.phase_ticks, self.events = {}, [(tick, "takeover")]
        self._bottle_plan, self._quat_plan = np.asarray(bottle0[0], float), np.asarray(bottle0[1], float)
        self._retarget(bottle)
        self.retries = 0
        self._q_ref, self._desired_prev, self.q_bias = None, None, np.zeros(7)
        self._seg_start, self._lift_from = None, None
        tm = kin.tcp_pose(side, q_meas)
        g = self.cur_grasp
        axis = g[:3, 2]
        rel = tm[:3, 3] - g[:3, 3]
        remaining = -float(rel @ axis)  # still to go along the axis (< 0: past the cap)
        lat = rel + remaining * axis
        latn = float(np.linalg.norm(lat))
        rot_err = float(np.linalg.norm(kin.pose_error(tm, g)[3:]))
        standoff_m = float(np.linalg.norm(p.grasp[:3, 3] - p.standoff[:3, 3]))
        closed = self.finger[side] < START_FINGER - 0.005
        in_corridor = (
            0.0 < remaining < standoff_m + self.CORRIDOR_EXTRA_M
            and latn < self.CORRIDOR_LAT_M
            and rot_err < self.CORRIDOR_ROT_RAD
        )
        self.log["takeover"] = {
            "tick": tick,
            "remaining_mm": round(remaining * 1000, 1),
            "lateral_mm": round(latn * 1000, 1),
            "rot_deg": round(float(np.degrees(rot_err)), 1),
            "closed": bool(closed),
        }
        if closed and abs(remaining) < 0.012 and latn < 0.015:
            route = "grasp"  # the policy closed on the cap: keep it closed and lift
        elif closed:
            route = "reopen"
            self.finger[side] = START_FINGER
        elif in_corridor and not (remaining < 0.03 and latn > self.NEAR_LAT_M):
            route = "insert"
        elif remaining < standoff_m and (remaining < 0.0 or latn < 0.08):
            route = "backoff"  # beside / past the cap: back off along the axis first, a straight line could sweep the bottle
        else:
            route = "reach"
        self.log["takeover"]["route"] = route
        if route == "grasp":
            self.finger[side] = FINGER_CLOSED
            self.phase_ticks["approach"] = tick  # (convert_to_lerobot.py: the close starts here)
            self.phase = "close"
        elif route == "insert":
            self.cur_standoff = g.copy()
            self.cur_standoff[:3, 3] = g[:3, 3] - axis * remaining
            self._lat0 = lat
            self._insert_dur = max(remaining / self.insert_speed, latn / 0.02, 0.8)
            self._sub = "line"
            self._bottle_insert = (np.asarray(bottle[0], float), np.asarray(bottle[1], float))
            self.phase = "approach"
        elif route in ("reopen", "backoff"):
            # back off to at least ~2 cm behind the standoff's distance along the axis (never less than retreat_m)
            self.retreat_m = float(np.clip(max(self.retreat_m, standoff_m - max(remaining, 0.0) + 0.02), 0.03, 0.15))
            self._retreat_from = tm
            self.phase = "retreat"
        else:
            self._start_reach()
            self.phase = "curve"
        self.events.append((tick, self.phase))
        return route

    def _start_reach(self) -> None:
        """Straight Cartesian move from the current command to the standoff (takeover), ~10 cm/s."""
        d = float(np.linalg.norm(kin.tcp_pose(self.side, self.q_cmd[self.side])[:3, 3] - self.cur_standoff[:3, 3]))
        self._curve = (max(d / 0.10, 1.0), None)

    @property
    def ik_ok(self) -> bool:
        """Pre-grasp reachable, and the straight approach/lift possible at a natural speed (near-singular paths
        would need > MAX_SEGMENT_S under the rate limit)."""
        err = kin.pose_error(kin.tcp_pose(self.side, self.plan.q_pregrasp), self.plan.pregrasp)
        reach = np.linalg.norm(err[:3]) < 0.01 and np.linalg.norm(err[3:]) < 0.1
        if self.plan.q_aim is not None:  # the wrong first aim must be a reachable pose too
            aim = self.plan.pregrasp.copy()
            aim[:3, 3] += self.plan.aim_error
            e = kin.pose_error(kin.tcp_pose(self.side, self.plan.q_aim), aim)
            reach = reach and np.linalg.norm(e[:3]) < 0.01 and np.linalg.norm(e[3:]) < 0.1
        if self.observer and self.q_view is None:  # no reachable side view for the observer arm
            return False
        if self.staged:
            e = kin.pose_error(kin.tcp_pose(self.side, self.plan.q_standoff), self.plan.standoff)
            reach = reach and np.linalg.norm(e[:3]) < 0.005 and np.linalg.norm(e[3:]) < 0.05
            reach = reach and self.plan.insert_need <= MAX_SEGMENT_S
        return bool(reach and self.plan.approach_s <= MAX_SEGMENT_S)

    def _advance(self, nxt: str) -> None:
        self.phase_ticks[self.phase] = self.tick
        self.phase, self.t_phase, self._seg_start = nxt, 0, None
        self.events.append((self.tick, nxt))

    # --- staged approach (v6) ------------------------------------------------------------------------------------
    def _servo(
        self, desired: np.ndarray, q_meas: np.ndarray | None, axis: np.ndarray | None = None, gain: float = 0.5
    ) -> np.ndarray | None:
        """Command towards `desired` (feed-forward IK) plus a joint offset integrated from the measured fingertip's
        error (all of it, or only its part across `axis` while moving along it). Returns the measured pose error
        against `desired` (6-vector) or None without a measurement."""
        side = self.side
        if self._q_ref is None:
            self._q_ref = self.q_cmd[side] - self.q_bias
        q_ref = self._q_ref
        for _ in range(5):
            dq, _ = kin.ik_step(side, q_ref, desired, self.plan.q_pregrasp, null_gain=0.02)
            q_ref = q_ref + dq
        self._q_ref = q_ref
        err = None
        if q_meas is not None:
            tm = kin.tcp_pose(side, q_meas)
            err = kin.pose_error(tm, desired)
            if self._desired_prev is not None:  # the measurement answers the previous tick's command
                e = kin.pose_error(tm, self._desired_prev)
                if axis is not None:
                    e[:3] -= (e[:3] @ axis) * axis
                tgt = tm.copy()
                tgt[:3, 3] += e[:3]
                tgt[:3, :3] = self._desired_prev[:3, :3]
                dq, _ = kin.ik_step(side, q_meas, tgt, q_meas, null_gain=0.0)
                self.q_bias = np.clip(self.q_bias + gain * dq, -0.1, 0.1)
        step = q_ref + self.q_bias - self.q_cmd[side]
        peak = np.abs(step).max()
        if peak > MAX_JOINT_STEP:
            step *= MAX_JOINT_STEP / peak
        self.q_cmd[side] = self.q_cmd[side] + step
        self._desired_prev = desired
        return err

    def _bottle_moved(self, bottle, ref: tuple) -> tuple[float, float]:
        """(horizontal displacement in m, rotation in deg) against ref = (position, quaternion)."""
        pos, quat = bottle
        d = float(np.linalg.norm(np.asarray(pos)[:2] - ref[0][:2]))
        dot = abs(float(np.dot(np.asarray(quat), ref[1])))
        return d, float(np.degrees(2 * np.arccos(min(dot, 1.0))))

    def _retarget(self, bottle) -> None:
        """After a recovery: grasp where the bottle now stands (no deliberate offset any more)."""
        shift = np.zeros(3) if bottle is None else np.asarray(bottle[0]) - self._bottle_plan
        self.cur_grasp = self.plan.grasp.copy()
        self.cur_grasp[:3, 3] += shift
        self.cur_standoff = self.plan.standoff.copy()
        self.cur_standoff[:3, 3] += shift
        self.offset = np.zeros(3)

    def _staged_step(self, q_meas, bottle) -> None:
        side, p = self.side, self.plan
        hz = CONTROL_HZ
        axis = p.grasp[:3, 2]  # fingertip axis = insertion direction
        if bottle is not None and self._bottle_plan is None:
            self._bottle_plan, self._quat_plan = np.asarray(bottle[0], float), np.asarray(bottle[1], float)
        if self.phase == "curve":
            if bottle is not None:
                d, _ = self._bottle_moved(bottle, (self._bottle_plan, self._quat_plan))
                self.log["disturbed_before_insert_mm"] = max(self.log["disturbed_before_insert_mm"], round(d * 1000, 1))
            dur, ctrl = self._curve if self._curve is not None else (p.approach_s, p.approach_ctrl)
            if self._cartesian(self.cur_standoff, dur, ctrl):
                self._q_ref, self._desired_prev, self.q_bias = None, None, np.zeros(7)
                self._curve = None
                self._advance("align")
        elif self.phase == "align":
            err = self._servo(self.cur_standoff, q_meas)
            ok = err is None or (np.linalg.norm(err[:3]) < 0.003 and np.linalg.norm(err[3:]) < 0.05)
            self._ok_ticks = self._ok_ticks + 1 if ok else 0
            timeout = self.t_phase >= 1.5 * hz
            if (self._ok_ticks >= 2 and self.t_phase >= 2) or timeout:
                self.log["align_s"].append(round((self.t_phase + 1) / hz, 2))
                self.log["align_timeouts"] += int(timeout and not ok)
                self._ok_ticks, self._sub = 0, "line"
                self._bottle_insert = (
                    None if bottle is None else (np.asarray(bottle[0], float), np.asarray(bottle[1], float))
                )
                self._advance("approach")
        elif self.phase == "approach":
            length = float(np.linalg.norm(self.cur_grasp[:3, 3] - self.cur_standoff[:3, 3]))
            dur = max(length / self.insert_speed, p.insert_need, 0.5)  # (average speed of the min-jerk profile)
            if self._insert_dur is not None:  # takeover: insertion from where the policy left the hand
                dur = self._insert_dur
            if self._sub == "line":
                s = min_jerk(self.t_phase / (dur * hz))
                desired = self.cur_grasp.copy()
                desired[:3, 3] = (1 - s) * self.cur_standoff[:3, 3] + s * self.cur_grasp[:3, 3]
                if np.any(self._lat0):  # takeover: the sideways error goes in the first 60% of the insertion
                    desired[:3, 3] += self._lat0 * (1.0 - min_jerk(self.t_phase / (0.6 * dur * hz)))
                err = self._servo(desired, q_meas, axis=axis)
                if s >= 1.0:
                    self._sub, self._t_settle = "settle", 0
            else:
                err = self._servo(self.cur_grasp, q_meas)
                self._t_settle += 1
            touched = False
            if bottle is not None and self._bottle_insert is not None:
                d, rot = self._bottle_moved(bottle, self._bottle_insert)
                touched = d > 0.005 or rot > 5.0
            missed = False
            if self._sub == "settle" and (err is None or np.linalg.norm(err[:3]) < 0.004 or self._t_settle >= 0.6 * hz):
                # Arrived. The expert knows where the cap is: a deliberate offset that touched nothing still misses it.
                missed = bool(np.any(self.offset))
                if not missed:
                    self._advance("close")
                    return
            if (touched or missed) and self.retries < self.max_retries:
                self.log["contacts"].append(
                    {
                        "tick": self.tick,
                        "touched": touched,
                        "missed": missed,
                        "shift_mm": None
                        if bottle is None
                        else round(1000 * self._bottle_moved(bottle, self._bottle_insert)[0], 1),
                    }
                )
                self.retries += 1
                start = kin.tcp_pose(side, q_meas if q_meas is not None else self.q_cmd[side])
                self._retreat_from = start
                self._advance("retreat")
            elif missed:  # out of retries: close anyway (the episode fails and is dropped)
                self._advance("close")
        elif self.phase == "retreat":
            dur = max(self.retreat_m / 0.06, 0.6)
            s = min_jerk(self.t_phase / (dur * hz))
            desired = self.cur_grasp.copy()
            desired[:3, 3] = self._retreat_from[:3, 3] - axis * self.retreat_m * s
            self._servo(desired, q_meas, axis=axis)
            if s >= 1.0:
                self._retarget(bottle)
                self._insert_dur, self._lat0 = None, np.zeros(3)
                far = (
                    self.took_over
                    and np.linalg.norm(kin.tcp_pose(side, self.q_cmd[side])[:3, 3] - self.cur_standoff[:3, 3]) > 0.025
                )
                if far:  # (takeover: the back-off can end well off the standoff; align only servos small errors)
                    self._start_reach()
                    self._advance("curve")
                else:
                    self._advance("align")
        elif self.phase == "close":
            self.finger[side] = FINGER_CLOSED
            if self.t_phase >= p.close_s * hz and self._observer_is_clear():
                self._advance("lift")

    # Fraction of its retreat the observer must have covered before the bottle starts coming up. The retreat runs
    # at the joint rate limit, so it takes 2.7-8.3 s where the close takes about 0.8: waiting for all of it would
    # add seconds to every episode for no gain, while the sideways part of the retreat -- the part that clears the
    # lift -- is nearly all done by two thirds of the way through a min-jerk profile.
    OBS_CLEAR_FRAC = 0.66

    def _observer_is_clear(self) -> bool:
        """Whether the observer has retreated far enough that the lift has the space to itself.

        The grasping arm waits on it rather than the other way round: the fingers already have the cap, so holding
        a moment costs nothing, whereas hurrying the observer is what drove it past its torque limit (the command
        saturated at the rate limit, joint 2 fell up to 0.22 rad behind, then caught up in one tick).
        """
        if not self.observer or self.q_view is None or self._obs_retreat_tick is None:
            return True
        return (self.tick - self._obs_retreat_tick) >= self.OBS_CLEAR_FRAC * self.obs_retreat_s * CONTROL_HZ

    def _cartesian(self, goal: np.ndarray, duration: float, ctrl: np.ndarray | None = None) -> bool:
        side = self.side
        if self._seg_start is None:
            self._seg_start = kin.tcp_pose(side, self.q_cmd[side])
        s = min_jerk(self.t_phase / (duration * CONTROL_HZ))
        target = np.eye(4)
        if ctrl is None:
            target[:3, 3] = (1 - s) * self._seg_start[:3, 3] + s * goal[:3, 3]
        else:  # quadratic Bezier through the control point
            target[:3, 3] = (1 - s) ** 2 * self._seg_start[:3, 3] + 2 * (1 - s) * s * ctrl + s**2 * goal[:3, 3]
        target[:3, :3] = slerp_rot(self._seg_start[:3, :3], goal[:3, :3], s)
        q = self.q_cmd[side].copy()
        for _ in range(5):
            dq, _ = kin.ik_step(side, q, target, self.plan.q_pregrasp, null_gain=0.05)
            q = q + dq
        step = q - self.q_cmd[side]
        peak = np.abs(step).max()
        if peak > MAX_JOINT_STEP:  # scale the whole step (keeps its direction) rather than clipping joints one by one
            step *= MAX_JOINT_STEP / peak
        self.q_cmd[side] = self.q_cmd[side] + step
        return self.t_phase >= duration * CONTROL_HZ

    def step(
        self, finger_measured: float | None = None, q_measured: np.ndarray | None = None, bottle: tuple | None = None
    ) -> np.ndarray:
        """q_measured: the grasping arm's 7 measured joints; bottle: (position, quaternion) -- both used by the staged
        approach only (without them it runs open loop and cannot detect a touched bottle)."""
        side, p = self.side, self.plan
        if self.staged and self.phase in ("curve", "align", "approach", "retreat", "close"):
            self._staged_step(q_measured, bottle)
        elif self.phase == "move_pregrasp":
            frac = self.t_phase / (self.move_s * CONTROL_HZ)
            s = min_jerk(frac)
            self.finger[side] = START_FINGER + s * (FINGER_WIDE - START_FINGER)  # (FINGER_WIDE = start: stays)
            q_pre = p.q_pregrasp
            if p.q_aim is not None:  # wrong first aim, corrected to the true pre-grasp during the move
                c = min_jerk((frac - p.correct_from) / p.correct_frac)
                q_pre = p.q_aim + c * (p.q_pregrasp - p.q_aim)
            q_goal = (1 - s) * self.q_start[side] + s * q_pre
            if self.dq_bow is not None:
                q_goal = q_goal + np.sin(np.pi * s**self.bow_p) * self.dq_bow
            self.q_cmd[side] = self.q_cmd[side] + np.clip(q_goal - self.q_cmd[side], -MAX_JOINT_STEP, MAX_JOINT_STEP)
            if s >= 1.0 and np.abs(self.q_cmd[side] - p.q_pregrasp).max() < 1e-3:
                self._advance("curve" if self.staged else "approach")
        elif self.phase == "approach":
            if self._cartesian(p.grasp, p.approach_s, p.approach_ctrl):
                self._advance("close")
        elif self.phase == "close":
            if self.close_ramp:
                r = min((self.t_phase + 1) / (self.close_ramp * CONTROL_HZ), 1.0)
                self.finger[side] = FINGER_WIDE + r * (FINGER_CLOSED - FINGER_WIDE)
            else:
                self.finger[side] = FINGER_CLOSED
            if self.t_phase >= p.close_s * CONTROL_HZ and self._observer_is_clear():
                self._advance("lift")
        elif self.phase == "lift":
            if self._lift_from is None:
                self._lift_from = self.q_cmd[side].copy()
            s = min_jerk(self.t_phase / (p.lift_s * CONTROL_HZ))
            q_goal = self._lift_from + s * p.dq_lift
            self.q_cmd[side] = self.q_cmd[side] + np.clip(q_goal - self.q_cmd[side], -MAX_JOINT_STEP, MAX_JOINT_STEP)
            if s >= 1.0 and np.abs(self.q_cmd[side] - (self._lift_from + p.dq_lift)).max() < 1e-3:
                self._advance("hold")
        elif self.phase == "hold" and self.t_phase >= p.hold_s * CONTROL_HZ:
            self._advance("done")
        self._observer_step()
        self.t_phase += 1
        self.tick += 1
        return np.concatenate([self.q_cmd["left"], self.q_cmd["right"], [self.finger["left"], self.finger["right"]]])

    @property
    def done(self) -> bool:
        return self.phase == "done"
