"""执行器：力指令 -> 实际力。

包含：指令延迟队列（含采样/通信延迟）、力饱和（由保守额定转矩换算）、
速率限制、丢帧、看门狗失能、急停。F_m 的一阶响应在 plant 中（τ_F）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def force_limit_from_motor(cfg: Dict) -> float:
    """由保守额定转矩换算驱动力上限：F = eta*G*tau / r。"""
    mech, spec = cfg["mechanics"], cfg["motor_spec"]
    act = cfg["actuator"]
    override = act.get("force_limit_override_N")
    if override is not None:
        return float(override)
    if act.get("use_conservative_rated_torque_as_limit", True):
        tau = float(spec["rated_torque_Nm"])
    else:
        tau = float(spec["stall_torque_Nm"])
    return float(mech["efficiency_eta"] * mech["reduction_ratio_G"] * tau / mech["r_m"])


def torque_from_force(force: float, cfg: Dict) -> float:
    mech = cfg["mechanics"]
    return float(force * mech["r_m"] / (mech["efficiency_eta"] * mech["reduction_ratio_G"]))


def force_from_torque(torque: float, cfg: Dict) -> float:
    mech = cfg["mechanics"]
    return float(torque * mech["efficiency_eta"] * mech["reduction_ratio_G"] / mech["r_m"])


def iq_from_force(force: float, cfg: Dict) -> Optional[float]:
    """仅当 K_t 已标定（MEASURED）时才给出 i_q；否则返回 None。"""
    kt = cfg["motor_spec"].get("torque_constant_Nm_per_iq_A")
    if not kt:
        return None
    return torque_from_force(force, cfg) / float(kt)


class Actuator:
    """纯力接口 F_cmd -> F_m 的命令侧（延迟/限幅/丢帧/失能）。"""

    def __init__(self, cfg: Dict, rng: Optional[np.random.Generator] = None,
                 ideal: bool = False):
        act = cfg["actuator"]
        self.cfg = cfg
        self.ideal = ideal
        self.delay = 0.0 if ideal else float(act["delay_s"])
        self.slew_limit = np.inf if ideal else float(act["force_slew_limit_Nps"])
        self.F_max = force_limit_from_motor(cfg)
        self.p_loss = 0.0 if ideal else float(act.get("missing_packet_probability", 0.0))
        self.watchdog = float(act.get("watchdog_timeout_s", 0.1))
        self.rng = rng or np.random.default_rng(0)

        self._queue: List[Tuple[float, float]] = []   # [(t_apply, force)]
        self._active = 0.0
        self._last_cmd_t = -1e9
        self._prev_cmd = 0.0
        self.enabled = True
        self.emergency_stop = False
        self.lost_packets = 0
        self.saturated_steps = 0
        self.slew_limited_steps = 0

    # ------------------------------------------------------------------ #
    def command(self, f_raw: float, t_now: float, dt: float = 0.01) -> float:
        """接收控制器原始指令，做速率限制+饱和，并按延迟排队。返回入队值。"""
        if self.emergency_stop or not self.enabled:
            self._queue.append((t_now + self.delay, 0.0))
            self._prev_cmd = 0.0
            return 0.0

        if self.ideal:
            self._active = float(f_raw)
            self._prev_cmd = self._active
            self._last_cmd_t = t_now
            return self._active

        desired = float(f_raw)
        raw_delta = desired - self._prev_cmd
        bounded_delta = float(np.clip(raw_delta, -self.slew_limit * dt, self.slew_limit * dt))
        if abs(bounded_delta - raw_delta) > 1e-12:
            self.slew_limited_steps += 1
        cmd = self._prev_cmd + bounded_delta
        sat = float(np.clip(cmd, -self.F_max, self.F_max))
        if abs(sat - cmd) > 1e-12:
            self.saturated_steps += 1
        # 丢帧：本周期指令未送达 -> 保持上一条（不排队新值，等效 ZOH 延长）
        if self.p_loss > 0.0 and self.rng.random() < self.p_loss:
            self.lost_packets += 1
            self._prev_cmd = sat
            return sat
        self._queue.append((t_now + self.delay, sat))
        self._prev_cmd = sat
        self._last_cmd_t = t_now
        return sat

    def force_at(self, t: float) -> float:
        """取出当前生效指令（零阶保持），含看门狗超时归零。"""
        if self.ideal:
            return self._active
        while self._queue and self._queue[0][0] <= t:
            self._active = self._queue.pop(0)[1]
        if self.watchdog > 0 and (t - self._last_cmd_t) > self.watchdog:
            self._active = 0.0
        if self.emergency_stop:
            return 0.0
        return self._active

    def zero(self, t_now: float) -> None:
        self._prev_cmd = 0.0
        self._queue.append((t_now + self.delay, 0.0))

    def disable(self, t_now: float) -> None:
        self.enabled = False
        self.zero(t_now)

    def trip_emergency_stop(self, t_now: float) -> None:
        self.emergency_stop = True
        self.zero(t_now)

    def reset(self) -> None:
        self._queue.clear()
        self._active = 0.0
        self._prev_cmd = 0.0
        self._last_cmd_t = -1e9
        self.enabled = True
        self.emergency_stop = False
        self.lost_packets = 0
        self.saturated_steps = 0
        self.slew_limited_steps = 0
