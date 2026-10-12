"""被控对象：单侧半车一维平动动力学 + 滚刷合力 + 静/动摩擦（平滑/粘滑）。

真实模型（任务书 §2.1）：

    ẋ = v
    m v̇ = F_h + F_m - D(v,t),      D = F_res(v) - F_br(t)
    τ_F Ḟ_m = sat(u, -F_max, F_max) - F_m

外力净阻力 D 不保证恒正：摩擦力有方向，滚刷也可助推。
本模块的真值只允许被离线评价器读取（见 tests/test_no_privileged_truth.py）。

性能说明：内层循环使用 math 标量函数（而非 numpy 标量），
在 1 核机器上把 7 s 仿真的耗时从 ~0.57 s 降到 ~0.15 s，使 300 组 Monte Carlo 可执行。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

FRICTION_MODES = ("ideal", "dry", "wet", "new_brush")


@dataclass
class PlantParams:
    """可被 Monte Carlo 独立改写的物理参数（与配置解耦，便于鲁棒性采样）。"""
    m: float = 4.0
    b: float = 2.0
    vs: float = 0.06
    v_eps: float = 0.015
    Fc: Dict[str, float] = field(default_factory=lambda: {"dry": 1.5, "wet": 3.5, "new_brush": 2.8})
    Fs: Dict[str, float] = field(default_factory=lambda: {"dry": 2.5, "wet": 5.0, "new_brush": 4.5})
    brush_bias: float = 0.0
    brush_amp: float = 0.5
    brush_freq: float = 6.0
    brush_noise_std: float = 0.0
    brush_noise_cutoff: float = 2.0
    tau_F: float = 0.04
    enable_sticking: bool = True
    stick_v_threshold: float = 0.004
    stick_dwell: float = 0.02
    F_max: float = 6.25
    integrator: str = "rk4"

    @classmethod
    def from_config(cls, cfg: Dict) -> "PlantParams":
        mech = cfg["mechanics"]
        fric = mech["friction"]
        brush = mech["brush"]
        return cls(
            m=float(mech["m_side_kg"]),
            b=float(fric["b_Ns_per_m"]),
            vs=float(fric["vs_mps"]),
            v_eps=float(fric["v_epsilon_mps"]),
            Fc={m: float(fric[m]["Fc_N"]) for m in ("dry", "wet", "new_brush")},
            Fs={m: float(fric[m]["Fs_N"]) for m in ("dry", "wet", "new_brush")},
            brush_bias=float(brush["bias_N"]),
            brush_amp=float(brush["ripple_amplitude_N"]),
            brush_freq=float(brush["ripple_frequency_Hz"]),
            brush_noise_std=float(brush.get("noise_std_N", 0.0)),
            brush_noise_cutoff=float(brush.get("noise_cutoff_Hz", 2.0)),
            tau_F=float(cfg["actuator"]["tau_force_s"]),
            enable_sticking=bool(mech.get("enable_sticking_friction", True)),
            stick_v_threshold=float(mech.get("stick_velocity_threshold_mps", 0.004)),
            stick_dwell=float(mech.get("stick_dwell_s", 0.02)),
            F_max=float(cfg["controllers"]["shared"]["max_force_N"]),
            integrator=str(cfg["simulation"].get("integrator", "rk4")),
        )


class FrictionBrushModel:
    """摩擦 + 滚刷合力的（平滑）模型；支持 mid-run 工况切换。"""

    def __init__(self, p: PlantParams, mode: str = "dry", seed: int = 0, ideal: bool = False):
        self.p = p
        self.mode = mode
        self.ideal = ideal
        self._rng = np.random.default_rng(seed)
        self._noise = 0.0
        self.brush_bias_override: Optional[float] = None
        self.refresh()

    def refresh(self) -> None:
        """缓存当前工况参数 / 刷头参数（避免内层循环反复查 dict）。"""
        self._b = 0.0 if self.ideal else self.p.b
        self._Fc = 0.0 if self.ideal else self.p.Fc[self.mode]
        self._Fs = 0.0 if self.ideal else self.p.Fs[self.mode]
        self._vs2 = self.p.vs ** 2
        self._veps = self.p.v_eps
        self._ideal_flag = self.ideal

    # ---- 工况 ----
    def set_mode(self, mode: str) -> None:
        if mode not in FRICTION_MODES:
            raise ValueError(f"未知摩擦工况 {mode}")
        if mode != self.mode:
            self.mode = mode
            self.refresh()

    @property
    def Fc(self) -> float:
        return 0.0 if self.ideal else self.p.Fc[self.mode]

    @property
    def Fs(self) -> float:
        return 0.0 if self.ideal else self.p.Fs[self.mode]

    @property
    def b(self) -> float:
        return 0.0 if self.ideal else self.p.b

    # ---- 滚刷 ----
    def set_brush_bias(self, bias: float) -> None:
        self.brush_bias_override = float(bias)

    @property
    def brush_bias(self) -> float:
        if self.brush_bias_override is not None:
            return self.brush_bias_override
        return self.p.brush_bias

    def update_brush_noise(self, dt: float) -> float:
        """一阶低通白噪声（有色噪声），按 plant 子步推进。"""
        if self.p.brush_noise_std <= 0.0:
            self._noise = 0.0
            return 0.0
        tau = 1.0 / (2.0 * math.pi * max(self.p.brush_noise_cutoff, 1e-3))
        alpha = dt / (tau + dt)
        self._noise += alpha * (self._rng.normal(0.0, self.p.brush_noise_std) - self._noise)
        return self._noise

    def brush_force(self, t: float) -> float:
        if self._ideal_flag:
            return 0.0
        return (self.brush_bias
                + self.p.brush_amp * math.sin(6.283185307179586 * self.p.brush_freq * t)
                + self._noise)

    # ---- 摩擦 ----
    def friction_force(self, v: float) -> float:
        """F_res(v)：粘滞 + Stribeck 静/动摩擦 + 方向平滑。"""
        if self._ideal_flag:
            return 0.0
        av = -v if v < 0.0 else v
        stribeck = self._Fc + (self._Fs - self._Fc) * math.exp(-(av / self.p.vs) ** 2)
        return self._b * v + stribeck * math.tanh(v / self._veps)

    def resistance(self, v: float, t: float) -> Tuple[float, float, float]:
        """返回 (D, F_res, F_br)。"""
        f_br = self.brush_force(t)
        f_res = self.friction_force(v)
        return f_res - f_br, f_res, f_br


class Plant:
    """单侧平动对象；真值仅供离线评价。"""

    def __init__(self, p: PlantParams, mode: str = "dry", seed: int = 0,
                 ideal: bool = False, instant_force: bool = False):
        self.p = p
        self.friction = FrictionBrushModel(p, mode=mode, seed=seed, ideal=ideal)
        self.ideal = ideal
        self.instant_force = instant_force
        self.record_log = True   # 运行器自行记录时间序列时可关闭以提速

        # 状态
        self.x = 0.0
        self.v = 0.0
        self.F_m = 0.0
        self.a = 0.0
        self.t = 0.0

        # 粘滑状态
        self.stuck = bool(p.enable_sticking) and not ideal
        self._last_stick_change = -1e9

        self.log: Dict[str, List[float]] = {k: [] for k in
                                            ("t", "x", "v", "a", "F_m", "F_h", "F_br", "F_res", "D")}

    # ------------------------------------------------------------------ #
    # 动力学
    # ------------------------------------------------------------------ #
    def _dvdt(self, v: float, f_h: float, f_br: float) -> float:
        f_res = self.friction.friction_force(v)
        return (f_h + self.F_m - f_res + f_br) / self.p.m

    def _dFmdt(self, u: float) -> float:
        if self.instant_force or self.p.tau_F <= 0.0:
            return (u - self.F_m) / 1e-6
        u_sat = -self.p.F_max if u < -self.p.F_max else (self.p.F_max if u > self.p.F_max else u)
        return (u_sat - self.F_m) / self.p.tau_F

    def step(self, f_h: float, u: float, dt: float) -> None:
        """推进一个 plant 子步；u 为执行器当前生效指令（零阶保持）。

        f_h 由人手模型在**本子步起点**给出，保持常数（零阶保持）。
        """
        p = self.p
        # ---- 粘滑：判定 + 零速锁定 ----
        if self.stuck:
            f_ext = f_h + self.F_m + self.friction.brush_force(self.t)
            if abs(f_ext) <= self.friction.Fs:
                # 保持静止：摩擦提供反力，v=0, a=0，仅 F_m 演化
                self.a = 0.0
                self.v = 0.0
                if self.instant_force or p.tau_F <= 0.0:
                    self.F_m = -p.F_max if u < -p.F_max else (p.F_max if u > p.F_max else u)
                else:
                    self.F_m += dt * self._dFmdt(u)
                if self.record_log:
                    self._record(f_h)
                self.t += dt
                self.friction.update_brush_noise(dt)
                return
            self.stuck = False
            self._last_stick_change = self.t

        # ---- 自由运动积分 ----
        f_br = self.friction.brush_force(self.t)
        if self.instant_force or p.tau_F <= 0.0 or p.integrator == "euler":
            # 半隐式 Euler（粘滑/理想瞬态力下确定性更强，也快 3~4 倍）
            a = self._dvdt(self.v, f_h, f_br)
            self.v += a * dt
            self.x += self.v * dt
            self.a = a
            if self.instant_force or p.tau_F <= 0.0:
                self.F_m = -p.F_max if u < -p.F_max else (p.F_max if u > p.F_max else u)
            else:
                self.F_m += dt * self._dFmdt(u)
        else:
            self._rk4(f_h, f_br, u, dt)

        # 反向运动时摩擦自然反号，无需特判；此处只做粘滑回落判定
        if p.enable_sticking and not self.ideal:
            f_ext = f_h + self.F_m + f_br
            if (abs(self.v) < p.stick_v_threshold and abs(f_ext) <= self.friction.Fs
                    and (self.t - self._last_stick_change) >= p.stick_dwell):
                self.stuck = True
                self.v = 0.0
                self.a = 0.0
                self._last_stick_change = self.t

        if self.record_log:
            self._record(f_h)
        self.t += dt
        self.friction.update_brush_noise(dt)

    def _rk4(self, f_h: float, f_br: float, u: float, dt: float) -> None:
        m, tau = self.p.m, self.p.tau_F

        def deriv(x: float, v: float, fm: float) -> Tuple[float, float, float]:
            self.F_m = fm  # 供 _dvdt 使用
            dv = (f_h + fm - self.friction.friction_force(v) + f_br) / m
            dfm = (float(np.clip(u, -self.p.F_max, self.p.F_max)) - fm) / tau
            return v, dv, dfm

        x0, v0, fm0 = self.x, self.v, self.F_m
        k1 = deriv(x0, v0, fm0)
        k2 = deriv(x0 + 0.5 * dt * k1[0], v0 + 0.5 * dt * k1[1], fm0 + 0.5 * dt * k1[2])
        k3 = deriv(x0 + 0.5 * dt * k2[0], v0 + 0.5 * dt * k2[1], fm0 + 0.5 * dt * k2[2])
        k4 = deriv(x0 + dt * k3[0], v0 + dt * k3[1], fm0 + dt * k3[2])
        self.x = x0 + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        self.v = v0 + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        self.F_m = fm0 + dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        self.a = (f_h + self.F_m - self.friction.friction_force(self.v) + f_br) / m

    # ------------------------------------------------------------------ #
    # 真值记录（仅供离线评价）
    # ------------------------------------------------------------------ #
    def _record(self, f_h: float) -> None:
        d, f_res, f_br = self.friction.resistance(self.v, self.t)
        self.log["t"].append(self.t)
        self.log["x"].append(self.x)
        self.log["v"].append(self.v)
        self.log["a"].append(self.a)
        self.log["F_m"].append(self.F_m)
        self.log["F_h"].append(f_h)
        self.log["F_br"].append(f_br)
        self.log["F_res"].append(f_res)
        self.log["D"].append(d)

    # 显式真值接口（命名带 *_true，便于静态检查与 grep）
    @property
    def x_true(self) -> float:
        return self.x

    @property
    def v_true(self) -> float:
        return self.v

    @property
    def a_true(self) -> float:
        return self.a

    @property
    def D_true(self) -> float:
        return self.friction.resistance(self.v, self.t)[0]

    def truth_snapshot(self) -> Dict[str, float]:
        return {"x_true": self.x, "v_true": self.v, "a_true": self.a,
                "F_m_true": self.F_m, "D_true": self.D_true}


def make_plant_from_config(cfg: Dict, mode: str = "dry", seed: int = 0,
                           ideal: bool = False, instant_force: bool = False,
                           params: Optional[PlantParams] = None) -> Plant:
    p = params or PlantParams.from_config(cfg)
    return Plant(p, mode=mode, seed=seed, ideal=ideal, instant_force=instant_force)
