"""The GARMI robot client's policy execution, in simulated time (deterministic, no threads).

Mirrors examples/garmi/validate_realtime.py of the openpi repo (BlendRTCExecutor + the client's safety
post-processing in garmi_vla_client.run_episode), but latency is counted in control ticks instead of wall-clock
time, so a closed-loop evaluation does not depend on how fast the machine renders or simulates:

  * an inference started at tick t sees the observation of tick t and its chunk becomes usable at tick
    t + delay_steps, at index delay_steps (the steps executed meanwhile from the old chunk);
  * the prefetch starts when the current chunk has delay_steps + max(margin, blend_steps) steps left, and the
    server is told the not-yet-executed tail (`prev_actions`, `inference_delay`) as on the robot;
  * at a switch, the old chunk's next min(blend_steps, steps left) actions are cross-faded linearly into the new
    chunk (old 1 -> 0, new 0 -> 1) and the rest of the old chunk is dropped;
  * if a chunk runs out before the next one is usable (a stall), the last command is held.
With `execute_steps` (e.g. 8-12 instead of the ~24 above), each chunk is used for that many steps only: the next
inference starts execute_steps - delay_steps steps after the chunk took over, so the new chunk takes over after
exactly execute_steps executed steps of the current one (and the old -> new cross-fade is the same). The long unused
tail is then only partly a constraint: `prefix_horizon` (sent as `prefix_horizon`) limits the server's guidance
to that many leading steps (frozen for the delay, then decaying), so the rest of the new chunk is predicted freely.
The simulation is paused while the server computes, so the measured latency only sets delay_steps.
"""

from __future__ import annotations

import dataclasses

import numpy as np

N_ACT = 16  # what the robot executes: 14 joints + 2 grippers


class SimTimeBlendRTC:
    def __init__(self, infer, *, horizon: int, delay_steps: int, blend_steps: int = 5, margin_steps: int = 1,
                 execute_steps: int = 0, prefix_horizon: int = 0):
        self.infer, self.horizon = infer, horizon
        self.delay_steps, self.blend_steps, self.margin_steps = delay_steps, blend_steps, margin_steps
        self.execute_steps, self.prefix_horizon = execute_steps, prefix_horizon
        self._start = 0  # chunk index at which the current chunk took over
        self.chunk: np.ndarray | None = None
        self.idx = 0
        self._pending: tuple[np.ndarray, int, int] | None = None  # (chunk, ready tick, trigger tick)
        self._blend_old: np.ndarray | None = None
        self._blend_i = 0
        self._last: np.ndarray | None = None
        self.n_infer = self.n_switches = self.n_stall_ticks = 0
        self.overlaps: list[int] = []

    def _start_blend(self, old_chunk: np.ndarray, old_idx: int) -> None:
        left = old_chunk.shape[0] - old_idx
        self.overlaps.append(left)
        n = min(self.blend_steps, left, self.horizon - self.idx)
        self._blend_old = old_chunk[old_idx:old_idx + n] if n > 0 else None
        self._blend_w = np.linspace(0.0, 1.0, n) if n > 1 else np.ones(1)
        self._blend_i = 0

    def _call(self, obs: dict) -> np.ndarray:
        self.n_infer += 1
        return np.asarray(self.infer(obs), dtype=np.float64)

    def step(self, tick: int, make_obs) -> tuple[np.ndarray, bool]:
        """Action (horizon row incl. any extra columns) for this tick, and whether it is a held stall action."""
        if self.chunk is None:  # first chunk: the robot waits for it, as the real client does
            self.chunk, self.idx = self._call(make_obs()), 0

        if self._pending is not None and tick >= self._pending[1]:
            new_chunk, _, trigger = self._pending
            old_chunk, old_idx = self.chunk, self.idx
            self.chunk, self._pending = new_chunk, None
            self.idx = min(tick - trigger, self.horizon - 1)
            self._start = self.idx
            self.n_switches += 1
            self._start_blend(old_chunk, old_idx)

        if self.idx >= self.horizon:  # old chunk used up before the new one is usable: hold the last command
            if self._pending is None:  # (only without RTC prefetch, e.g. delay larger than the chunk)
                self._trigger(tick, make_obs)
            self.n_stall_ticks += 1
            return self._last.copy(), True

        action = self.chunk[self.idx].copy()
        if self._blend_old is not None and self._blend_i < len(self._blend_old):
            w = self._blend_w[self._blend_i]
            action = (1.0 - w) * self._blend_old[self._blend_i] + w * action
            self._blend_i += 1

        if self._pending is None:
            if self.execute_steps:  # replan after execute_steps steps of this chunk (switch = trigger + delay)
                due = self.idx - self._start >= max(self.execute_steps - self.delay_steps, 0)
            else:
                due = self.horizon - self.idx <= self.delay_steps + max(self.margin_steps, self.blend_steps)
            if due:
                self._trigger(tick, make_obs)
        self.idx += 1
        self._last = action
        return action, False

    def _trigger(self, tick: int, make_obs) -> None:
        obs = make_obs()
        tail = self.chunk[self.idx:, :N_ACT].copy() if self.idx < self.horizon else np.zeros((0, N_ACT))
        if len(tail):
            obs["prev_actions"] = tail.astype(np.float32)
            obs["inference_delay"] = int(min(self.delay_steps, len(tail)))
            if self.prefix_horizon:
                obs["prefix_horizon"] = int(min(max(self.prefix_horizon, self.delay_steps), len(tail)))
        self._pending = (self._call(obs), tick + max(self.delay_steps, 1), tick)


@dataclasses.dataclass
class ClientLimits:
    """garmi_vla_client.Args defaults (as validate_realtime.py)."""

    max_joint_step: float = 0.02  # rad per tick, against the last command
    max_target_jump: float = 0.6  # refuse targets this far from the measured arm (rad)
    max_lead: float = 0.15  # hold joints whose command leads the measured arm by more (rad)
    gripper_hz: float = 2.0
    gripper_deadband: float = 0.003  # m
    # Binary gripper (0 = off, the predicted finger position is sent as is): close fully (target 0, the force limit
    # sets the squeeze, like a Franka Grasp) as soon as the prediction drops below binary_close, open fully again only
    # above binary_open (hysteresis). A change is sent at once, not at gripper_hz.
    binary_close: float = 0.0  # m, e.g. 0.025
    binary_open: float = 0.030  # m


class ClientSafety:
    """The robot client's post-processing of the policy's 16 actions into the commands sent to the robot."""

    def __init__(self, q_start: np.ndarray, limits: ClientLimits, control_hz: float = 15.0):
        self.limits, self.period = limits, 1.0 / control_hz
        self.last_cmd = np.asarray(q_start[:14], dtype=np.float64).copy()
        self.grip = np.asarray(q_start[14:16], dtype=np.float64).copy()
        self.grip_open = self.grip.copy()
        self.closed = [False, False]
        self.last_grip_t = -1e9
        self.counts = {"jump_refused": 0, "rate_limited": 0, "lead_held": 0, "grip_cmds": 0}

    def __call__(self, tick: int, action: np.ndarray, q_measured: np.ndarray) -> np.ndarray:
        lim = self.limits
        target = np.asarray(action[:14], dtype=np.float64)
        q_arm = np.asarray(q_measured[:14], dtype=np.float64)
        if np.abs(target - q_arm).max() > lim.max_target_jump:
            target = self.last_cmd
            self.counts["jump_refused"] += 1
        if np.any(np.abs(target - self.last_cmd) > lim.max_joint_step):
            self.counts["rate_limited"] += 1
        cmd = self.last_cmd + np.clip(target - self.last_cmd, -lim.max_joint_step, lim.max_joint_step)
        lead = np.abs(cmd - q_arm) > lim.max_lead
        if lead.any():
            self.counts["lead_held"] += 1
        cmd = np.where(lead, self.last_cmd, cmd)
        self.last_cmd = cmd
        now = tick * self.period
        if lim.binary_close > 0:
            for side in range(2):
                g = float(action[14 + side])
                closed = g < lim.binary_close or (self.closed[side] and g <= lim.binary_open)
                if closed != self.closed[side]:
                    self.closed[side] = closed
                    self.grip[side] = 0.0 if closed else self.grip_open[side]
                    self.counts["grip_cmds"] += 1
        elif now - self.last_grip_t >= 1.0 / lim.gripper_hz:
            for side in range(2):
                g = float(action[14 + side])
                if abs(g - self.grip[side]) >= lim.gripper_deadband:
                    self.grip[side] = g
                    self.counts["grip_cmds"] += 1
            self.last_grip_t = now
        return np.concatenate([cmd, self.grip])
