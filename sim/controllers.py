"""控制算法组 C0–C5（任务书 §4）。

统一安全外壳（§4.1）：REST / START / MOVE / STOP(+REVERSE) 四阶段状态机，
速度 + 时间双滞回，死区 / 最大力 / 最大力变化率 / 超时归零统一在此实现。

本模块**禁止引用 plant 真值**（静态检查见 tests/test_no_privileged_truth.py）。
只有 `requires_oracle = True` 的 C4/C5_oracle 变体会通过显式传入的 `oracle`
字典读取真值，且被标为不可部署。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .actuator import force_limit_from_motor

STATE_REST = "REST"
STATE_START = "START"
STATE_MOVE = "MOVE"
STATE_STOP = "STOP"


class SafetyShell:
    """四阶段统一安全外壳；输出各反馈项的缩放门控。"""

    def __init__(self, cfg: Dict):
        s = cfg["controllers"]["shared"]
        self.deadband = float(s["speed_stop_deadband_mps"])
        self.start_hyst = float(s["start_hysteresis_s"])
        self.stop_hyst = float(s["stop_hysteresis_s"])
        self.a_intent = float(s.get("a_intent_threshold_mps2", 0.2))
        self.soft_start = bool(s.get("enable_soft_start", True))
        self.soft_start_time = float(s.get("soft_start_time_s", 0.15))
        self.timeout_zero = float(s.get("command_timeout_zero_s", 0.3))
        self.reverse_hold = float(s.get("reverse_hold_s", 0.3))
        self.stop_command_scale = float(
            cfg["controllers"]["C1_existing"].get("stop_command_scale", 0.25))
        self.reset()

    def reset(self) -> None:
        self.state = STATE_REST
        self.t_state = 0.0
        self.t_intent_seen = 0.0
        self._acc_sign = 0
        self._accel_intent_timer = 0.0
        self._stop_timer = 0.0
        self.now = 0.0

    def update(self, v_hat: float, a_hat: float, dt: float) -> Dict[str, float]:
        self.now += dt
        self.t_state += dt
        sgn = 1 if v_hat > 0 else (-1 if v_hat < 0 else 0)
        moving = abs(v_hat) > self.deadband
        accel_intent = abs(a_hat) > self.a_intent

        if self.state == STATE_REST:
            if moving or accel_intent:
                self._accel_intent_timer += dt if accel_intent else 0.0
                if moving or self._accel_intent_timer >= self.start_hyst:
                    self._enter(STATE_START)
                    self.t_intent_seen = self.now
        elif self.state == STATE_START:
            if not moving and not accel_intent:
                self._stop_timer += dt
                if self._stop_timer >= self.stop_hyst:
                    self._enter(STATE_STOP)
            else:
                self._stop_timer = 0.0
                if self.t_state >= self.soft_start_time or not self.soft_start:
                    self._enter(STATE_MOVE)
                    self._acc_sign = sgn
        elif self.state == STATE_MOVE:
            if not moving:
                self._stop_timer += dt
                if self._stop_timer >= self.stop_hyst:
                    self._enter(STATE_STOP)
            else:
                self._stop_timer = 0.0
                self.t_intent_seen = self.now
                if sgn != 0 and self._acc_sign != 0 and sgn != self._acc_sign:
                    self._enter(STATE_MOVE, reverse=True)
                    self._acc_sign = sgn
        elif self.state == STATE_STOP:
            if moving:
                self._enter(STATE_START)
                self._stop_timer = 0.0
            elif self.t_state >= self.stop_hyst:
                self._enter(STATE_REST)
            else:
                self._stop_timer += dt

        gates = self._gates()
        gates["state"] = self.state
        gates["intent_idle_s"] = self.now - self.t_intent_seen
        if self.state in (STATE_REST,) or (self.now - self.t_intent_seen) > self.timeout_zero \
                and self.state != STATE_MOVE:
            gates.update(a=0.0, v=0.0, d=0.0)
        return gates

    def _enter(self, state: str, reverse: bool = False) -> None:
        self.state = state
        self.t_state = 0.0
        if state == STATE_MOVE and reverse:
            self.t_state = -self.reverse_hold  # 反向过渡窗口

    def _gates(self) -> Dict[str, float]:
        if self.state == STATE_REST:
            return {"a": 0.0, "v": 0.0, "d": 0.0}
        if self.state == STATE_START:
            ramp = min(1.0, self.t_state / max(self.soft_start_time, 1e-6)) \
                if self.soft_start else 1.0
            return {"a": ramp, "v": ramp, "d": ramp}
        if self.state == STATE_MOVE:
            if self.t_state < 0:
                s = self.stop_command_scale
                return {"a": s, "v": 1.0, "d": s}
            return {"a": 1.0, "v": 1.0, "d": 1.0}
        # STOP
        return {"a": self.stop_command_scale, "v": 1.0, "d": self.stop_command_scale}


class BaseController:
    name = "BASE"
    requires_oracle = False

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.shared = cfg["controllers"]["shared"]
        self.shell = SafetyShell(cfg)
        self.force_limit = force_limit_from_motor(cfg)
        self.max_force = min(float(self.shared["max_force_N"]), self.force_limit)
        self.info: Dict[str, float] = {}
        self.reset()

    def reset(self) -> None:
        self.shell.reset()
        self.info = {"D_hat": 0.0, "state": STATE_REST, "a_gate": 0.0, "v_gate": 0.0,
                     "d_gate": 0.0, "F_cmd_raw": 0.0, "F_cmd_limited": 0.0}

    def update(self, est: Dict[str, float], dt: float, t: float,
               oracle: Optional[Dict[str, float]] = None) -> float:  # pragma: no cover
        raise NotImplementedError

    # --- 通用后处理：门控 + 硬限幅（执行器还会再做一次饱和/速率限制） ---
    def _finalize(self, f_cmd: float, gates: Dict[str, float]) -> float:
        limited = float(np.clip(f_cmd, -self.max_force, self.max_force))
        self.info.update(state=gates["state"], a_gate=gates["a"], v_gate=gates["v"],
                         d_gate=gates["d"], F_cmd_raw=f_cmd, F_cmd_limited=limited)
        return limited


class C0Controller(BaseController):
    """完全无助力（物理基准）。"""
    name = "C0"

    def update(self, est, dt, t, oracle=None):
        gates = self.shell.update(est["v_hat"], est["a_hat"], dt)
        return self._finalize(0.0, gates)


class C1Controller(BaseController):
    """现有方案：i = i_base + k*a（力等效抽象版 F_cmd = F_base(q,sgn v) + K_a,old*a_hat）。

    注意：缺少现行固件准确系数，本实现是**模型抽象**，不是实机效果复现。
    """
    name = "C1"

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        c = cfg["controllers"]["C1_existing"]
        self.f_base = float(c.get("force_base_N", 0.0))
        self.k_a_old = float(c.get("accel_gain_force_equivalent_kg", 0.3))
        self.tau_lpf = float(c.get("accel_lpf_tau_s", 0.02))
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._a_filt = 0.0

    def update(self, est, dt, t, oracle=None):
        gates = self.shell.update(est["v_hat"], est["a_hat"], dt)
        alpha = dt / (self.tau_lpf + dt)
        self._a_filt += alpha * (est["a_hat"] - self._a_filt)
        sign = 1.0 if est["v_hat"] >= 0 else -1.0
        phase_scale = 1.0 if gates["state"] in (STATE_MOVE, STATE_START) else \
            self.shell.stop_command_scale
        if gates["state"] == STATE_REST:
            phase_scale = 0.0
        f_cmd = phase_scale * (self.f_base * sign + self.k_a_old * self._a_filt)
        return self._finalize(f_cmd, gates)


class C2Controller(BaseController):
    """只用加速度+速度反馈的基础阻抗控制：F_cmd = K_a*a_hat - B_d*v_hat。"""
    name = "C2"

    def __init__(self, cfg: Dict, key: str = "C2_impedance"):
        super().__init__(cfg)
        c = cfg["controllers"][key]
        self.Ka = float(c["Ka_kg"])
        self.Bd = float(c["Bd_Ns_per_m"])

    def update(self, est, dt, t, oracle=None):
        gates = self.shell.update(est["v_hat"], est["a_hat"], dt)
        f_cmd = (gates["a"] * self.Ka * est["a_hat"]
                 - gates["v"] * self.Bd * est["v_hat"])
        return self._finalize(f_cmd, gates)


class C3Controller(C2Controller):
    """阻抗控制 + 名义阻力补偿：F_cmd = K_a*a_hat - B_d*v_hat + alpha_D*D_hat。

    D_hat 只使用独立辨识到的**名义**阻力参数与可获得的滤波速度；
    零速附近用死区 + 软切换衰减，避免 f_h=0 时刷头方向力/噪声诱发主动起步。
    """
    name = "C3"

    def __init__(self, cfg: Dict, key: str = "C3_impedance_comp"):
        BaseController.__init__(self, cfg)
        c = cfg["controllers"][key]
        self.Ka = float(c["Ka_kg"])
        self.Bd = float(c["Bd_Ns_per_m"])
        self.alpha_D = float(c["alpha_D"])
        self.Fc_est = float(c["estimated_Fc_N"])
        self.b_est = float(c["estimated_b_Ns_per_m"])
        self.v_deadband = float(c["velocity_compensation_deadband_mps"])
        self.v_ramp = float(c.get("comp_ramp_velocity_mps", 0.06))
        self.v_eps_est = float(cfg["mechanics"]["friction"]["v_epsilon_mps"])

    def D_hat(self, v: float) -> float:
        """名义净阻力估计（不含真实参数、不含滚刷真值）。"""
        return self.b_est * v + self.Fc_est * np.tanh(v / self.v_eps_est)

    def _comp_scale(self, v: float) -> float:
        av = abs(v)
        if av <= self.v_deadband:
            return 0.0
        return float(np.clip((av - self.v_deadband) / max(self.v_ramp, 1e-9), 0.0, 1.0))

    def update(self, est, dt, t, oracle=None):
        gates = self.shell.update(est["v_hat"], est["a_hat"], dt)
        v, a = est["v_hat"], est["a_hat"]
        d_hat = self.D_hat(v)
        comp = self.alpha_D * gates["d"] * self._comp_scale(v) * d_hat
        f_cmd = gates["a"] * self.Ka * a - gates["v"] * self.Bd * v + comp
        self.info["D_hat"] = d_hat
        self.info["comp_scale"] = self._comp_scale(v)
        return self._finalize(f_cmd, gates)


class C4aController(C3Controller):
    """上限对照：真实 a,v（仍用名义 D_hat）。不可部署。"""
    name = "C4a_perfect_state"
    requires_oracle = True

    def update(self, est, dt, t, oracle=None):
        est_perfect = dict(est)
        if oracle:
            est_perfect["a_hat"] = oracle["a_true"]
            est_perfect["v_hat"] = oracle["v_true"]
        return super().update(est_perfect, dt, t, oracle)


class C4bController(C3Controller):
    """上限对照：真实 D。仅离线最佳情况，严禁参与实机参数排名。"""
    name = "C4b_oracle_D"
    requires_oracle = True

    def update(self, est, dt, t, oracle=None):
        gates = self.shell.update(est["v_hat"], est["a_hat"], dt)
        v, a = est["v_hat"], est["a_hat"]
        d_true = float(oracle["D_true"]) if oracle else 0.0
        comp = self.alpha_D * gates["d"] * self._comp_scale(v) * d_true
        f_cmd = gates["a"] * self.Ka * a - gates["v"] * self.Bd * v + comp
        self.info["D_hat"] = d_true
        return self._finalize(f_cmd, gates)


class C5Controller(BaseController):
    """导纳对照（可选）：M_d*ṿ_ref + B_d*v_ref = F_h_hat；F_cmd = Kp*e + Ki∫e + F_ff。

    无独立力传感器时，F_h_hat 由动力学逆推（m_hat*a_hat - F_m_fb + D_hat），
    受 §3.1 的可识别性限制；use_true_force=True 时使用真实 F_h 作理想基准。
    """
    name = "C5_admittance"
    requires_oracle = False

    def __init__(self, cfg: Dict, key: str = "C5_admittance"):
        super().__init__(cfg)
        c = cfg["controllers"][key]
        self.Md = float(c["Md_adm_kg"])
        self.Bd = float(c["Bd_adm_Ns_per_m"])
        self.Kp = float(c["Kp"])
        self.Ki = float(c["Ki"])
        self.ff_gain = float(c.get("ff_gain", 1.0))
        self.i_limit = float(c.get("integral_limit_Ns", 5.0))
        self.use_true_force = bool(c.get("use_true_force", False))
        self.m_hat = float(cfg["mechanics"]["m_side_kg"])
        self.Fc_est = float(cfg["controllers"]["C3_impedance_comp"]["estimated_Fc_N"])
        self.b_est = float(cfg["controllers"]["C3_impedance_comp"]["estimated_b_Ns_per_m"])
        if self.use_true_force:
            self.name = "C5_oracle_force"
            self.requires_oracle = True
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.v_ref = 0.0
        self._integral = 0.0

    def update(self, est, dt, t, oracle=None):
        gates = self.shell.update(est["v_hat"], est["a_hat"], dt)
        a_hat, v_hat = est["a_hat"], est["v_hat"]
        if self.use_true_force and oracle is not None:
            f_h_hat = float(oracle["F_h_true"])
        else:
            d_hat = self.b_est * v_hat + self.Fc_est * np.tanh(v_hat / 0.015)
            f_h_hat = self.m_hat * a_hat - est.get("F_m_fb", 0.0) + d_hat
        # 导纳参考模型
        dv_ref = (f_h_hat - self.Bd * self.v_ref) / self.Md
        self.v_ref += dt * dv_ref
        err = self.v_ref - v_hat
        self._integral = float(np.clip(self._integral + err * dt, -self.i_limit, self.i_limit))
        f_cmd = gates["a"] * (self.Kp * err + self.Ki * self._integral) + self.ff_gain * f_h_hat
        return self._finalize(f_cmd, gates)


CONTROLLER_CLASSES = {
    "C0": C0Controller,
    "C1": C1Controller,
    "C2": C2Controller,
    "C3": C3Controller,
    "C4a": C4aController,
    "C4b": C4bController,
    "C5": C5Controller,
}

DEPLOYABLE = ("C0", "C1", "C2", "C3")
NON_DEPLOYABLE = ("C4a", "C4b", "C5")


def build_controller(name: str, cfg: Dict, param_overrides: Optional[Dict] = None):
    """构造控制器；param_overrides 可覆盖 Ka/Bd/alpha_D（扫参用）。"""
    cls = CONTROLLER_CLASSES[name]
    ctrl = cls(cfg)
    for key, val in (param_overrides or {}).items():
        if hasattr(ctrl, key):
            setattr(ctrl, key, float(val))
            if key == "Ka":
                ctrl.Ka = float(val)
    return ctrl
