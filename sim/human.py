"""人手输入模型：固定力波形 + 闭环虚拟用户（带反应延迟与用力速率限制）。

虚拟用户可以观测外部运动（x/v），但机器人助力控制器**禁止**读取 F_h 真值。
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# 固定力给定试验（任务书 §5.1）
# --------------------------------------------------------------------------- #
def _ramp(t: float, t0: float, t1: float, v0: float, v1: float) -> float:
    if t <= t0:
        return v0
    if t >= t1:
        return v1
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)


class FixedForceHuman:
    """给定 F_h(t) 波形。幅值为**单侧**力。"""

    PROFILES: Tuple[str, ...] = ("push_release", "pull_release", "weak_push",
                                 "rapid_reverse", "oscillatory", "none",
                                 "push_release_early", "step")

    def __init__(self, force_n: float = 5.0, profile: str = "push_release",
                 release_time_s: float = 2.5, rng: Optional[np.random.Generator] = None,
                 noise_std: float = 0.0):
        if profile not in self.PROFILES:
            raise ValueError(f"未知力波形 {profile}")
        self.profile = profile
        self.f0 = float(force_n)
        self.release_time_s = float(release_time_s)
        self.rng = rng or np.random.default_rng(0)
        self.noise_std = float(noise_std)

    def force(self, t: float, x: float = 0.0, v: float = 0.0) -> float:
        f = self.f0
        p = self.profile
        if p == "none":
            base = 0.0
        elif p == "step":
            base = f
        elif p == "push_release":
            base = _ramp(t, 0.0, 0.5, 0.0, f)
            base = _ramp(t, 2.5, 3.0, base, 0.0)
        elif p == "pull_release":
            base = _ramp(t, 0.0, 0.5, 0.0, -f)
            base = _ramp(t, 2.5, 3.0, base, 0.0)
        elif p == "weak_push":
            base = _ramp(t, 0.0, 1.0, 0.0, f)
            base = _ramp(t, 4.0, 5.0, base, 0.0)
        elif p == "rapid_reverse":
            base = f
            if t >= 2.0:
                base = _ramp(t, 2.0, 2.2, f, -f)
        elif p == "oscillatory":
            freq = 1.0
            base = f * np.sin(2.0 * np.pi * freq * t)
        elif p == "push_release_early":
            tr = self.release_time_s
            base = _ramp(t, 0.0, 0.5, 0.0, f)
            base = _ramp(t, tr, tr + 0.05, base, 0.0)
        else:  # pragma: no cover
            base = 0.0
        if self.noise_std > 0:
            base += self.rng.normal(0.0, self.noise_std)
        return float(base)


# --------------------------------------------------------------------------- #
# 闭环虚拟用户（任务书 §5.2）
# --------------------------------------------------------------------------- #
class CoupledUser:
    """F_h,des = Kp*(x_ref-x) + Kd*(v_ref-v)，再经过饱和/速率限制/一阶滞后/反应延迟。

    参考轨迹 x_ref(t) 由 [t_start, x_target, t_end] 段序列给出（线性插值）。
    观测使用反应延迟后的 x/v（人不能瞬时反应）。
    """

    def __init__(self, ref_sequence: Sequence[Sequence[float]], kp: float = 20.0,
                 kd: float = 7.0, force_max: float = 8.0, tau_h: float = 0.15,
                 reaction_delay: float = 0.10, slew_limit: float = 60.0,
                 rng: Optional[np.random.Generator] = None, noise_std: float = 0.0):
        self.seq = [(float(a), float(b), float(c)) for a, b, c in ref_sequence]
        self.kp, self.kd = float(kp), float(kd)
        self.force_max = float(force_max)
        self.tau_h = float(tau_h)
        self.reaction_delay = float(reaction_delay)
        self.slew_limit = float(slew_limit)
        self.rng = rng or np.random.default_rng(0)
        self.noise_std = float(noise_std)
        self.reset()

    def reset(self) -> None:
        self.F_h = 0.0
        self._buf: deque = deque(maxlen=6000)
        self._last_obs_t = -1e9

    # ---- 参考轨迹 ----
    def x_ref(self, t: float) -> Tuple[float, float]:
        prev_x = 0.0   # 起点固定在 x=0；每段 (t0, x_target, t1) 表示 t1 时刻到达 x_target
        for t0, x_t, t1 in self.seq:
            if t < t0:
                return prev_x, 0.0
            if t <= t1:
                span = max(t1 - t0, 1e-9)
                frac = (t - t0) / span
                v = (x_t - prev_x) / span
                return prev_x + (x_t - prev_x) * frac, v
            prev_x = x_t
        return prev_x, 0.0

    def force(self, t: float, x: float, v: float) -> float:
        # 反应延迟：观测 t - delay 时刻的运动
        self._buf.append((t, x, v))
        t_obs = t - self.reaction_delay
        obs = None
        for entry in reversed(self._buf):
            if entry[0] <= t_obs:
                obs = entry
                break
        if obs is None:
            obs = self._buf[0]
        x_o, v_o = obs[1], obs[2]
        x_r, v_r = self.x_ref(t)
        des = self.kp * (x_r - x_o) + self.kd * (v_r - v_o)
        des = float(np.clip(des, -self.force_max, self.force_max))
        # 一阶滞后 + 速率限制
        dt = max(t - self._last_obs_t, 1e-6) if self._last_obs_t > -1e8 else 0.001
        self._last_obs_t = t
        alpha = dt / (self.tau_h + dt)
        new = self.F_h + alpha * (des - self.F_h)
        delta = float(np.clip(new - self.F_h, -self.slew_limit * dt, self.slew_limit * dt))
        self.F_h = self.F_h + delta
        if self.noise_std > 0:
            return float(self.F_h + self.rng.normal(0.0, self.noise_std))
        return float(self.F_h)


def build_human(cfg: Dict, human_spec: Dict, task_spec: Optional[Dict] = None,
                variant: Optional[str] = None,
                rng: Optional[np.random.Generator] = None):
    """按配置构造人手模型。variant 取 {'soft','mid','stiff'} 之一。"""
    hcfg = dict(cfg["human"])
    if variant:
        hcfg.update(cfg["human"].get("variants", {}).get(variant, {}))
    htype = human_spec.get("type", "fixed")
    if htype == "fixed":
        return FixedForceHuman(
            force_n=human_spec.get("force_N", hcfg["fixed_force_N_per_side"]),
            profile=human_spec.get("profile", "push_release"),
            release_time_s=human_spec.get("release_time_s", 2.5),
            rng=rng, noise_std=hcfg.get("force_noise_std_N", 0.0))
    seq = (task_spec or {}).get("x_ref_sequence", [[0.0, 0.8, 3.0]])
    return CoupledUser(
        ref_sequence=seq,
        kp=hcfg["kp_N_per_m"], kd=hcfg["kd_Ns_per_m"],
        force_max=hcfg["force_max_N_per_side"], tau_h=hcfg["force_response_time_constant_s"],
        reaction_delay=hcfg["reaction_delay_s"], slew_limit=hcfg.get("force_slew_limit_Nps", 60.0),
        rng=rng, noise_std=hcfg.get("force_noise_std_N", 0.0))
