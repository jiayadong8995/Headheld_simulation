"""状态估计器：一维互补滤波（默认）+ 可选线性 Kalman（第二阶段对照）。

    v^-_k = v_hat_{k-1} + Ts*(a_imu,k - b_a_hat)
    v_hat_k = (1-lambda)*v^-_k + lambda*v_enc,k

a_hat 经一阶低通（截止 f_a），防止加速度正反馈放大高频噪声。
**本模块禁止访问 plant 真值。**
"""
from __future__ import annotations

from typing import Dict

import numpy as np


class StateEstimator:
    def __init__(self, cfg: Dict, mode: str = "complementary"):
        fu = cfg["sensors"]["fusion"]
        self.cfg = cfg
        self.mode = mode if mode in ("complementary", "kalman") else "complementary"
        self.lam = float(fu["complementary_encoder_weight_lambda"])
        f_a = float(cfg.get("estimator", {}).get(
            "accel_lpf_cutoff_Hz", fu["accel_lpf_cutoff_Hz"]))
        self.tau_a = 1.0 / (2.0 * np.pi * max(f_a, 1e-3))
        self.use_bias_est = bool(fu.get("enable_bias_estimator", False))
        self.bias_gain = float(fu.get("bias_estimator_gain", 0.02))
        self.kf_q_v = float(fu.get("kalman_q_v", 1e-3))
        self.kf_q_b = float(fu.get("kalman_q_b", 1e-6))
        self.kf_r_v = float(fu.get("kalman_r_v", 1e-3))
        self.reset()

    def reset(self) -> None:
        self.v_hat = 0.0
        self.a_hat = 0.0
        self.a_filt = 0.0
        self.bias_hat = 0.0
        self._kf_P = np.eye(2) * 1e-3
        self._kf_x = np.zeros(2)
        self.F_m_fb = 0.0
        self.diagnostics: Dict[str, float] = {}

    def update(self, meas: Dict[str, float], dt: float) -> Dict[str, float]:
        a_imu = float(meas.get("a_imu", 0.0))
        v_enc = float(meas.get("v_enc", 0.0))
        self.F_m_fb = float(meas.get("F_m_fb", 0.0))

        # 加速度低通（带限）
        if self.tau_a > 0:
            alpha = dt / (self.tau_a + dt)
            self.a_filt += alpha * (a_imu - self.a_filt)
        else:
            self.a_filt = a_imu

        if self.mode == "kalman":
            self._kalman_step(a_imu, v_enc, dt)
        else:
            v_pred = self.v_hat + dt * (a_imu - self.bias_hat)
            self.v_hat = (1.0 - self.lam) * v_pred + self.lam * v_enc
            if self.use_bias_est:
                self.bias_hat += self.bias_gain * dt * (v_enc - v_pred)

        self.a_hat = self.a_filt
        return {"v_hat": self.v_hat, "a_hat": self.a_hat, "bias_hat": self.bias_hat,
                "F_m_fb": self.F_m_fb, "v_enc": v_enc, "a_imu": a_imu,
                "stale": float(meas.get("stale", 0.0))}

    def _kalman_step(self, a_imu: float, v_enc: float, dt: float) -> None:
        """状态 x=[v, b_a]，输入 a_imu：v' = v + dt*(a_imu - b)。"""
        F = np.array([[1.0, -dt], [0.0, 1.0]])
        B = np.array([dt, 0.0])
        Q = np.diag([self.kf_q_v * dt, self.kf_q_b * dt])
        x = F @ self._kf_x + B * a_imu
        P = F @ self._kf_P @ F.T + Q
        H = np.array([[1.0, 0.0]])
        R = np.array([[self.kf_r_v]])
        y = v_enc - (H @ x)[0]
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        self._kf_x = x + (K * y).ravel()
        self._kf_P = (np.eye(2) - K @ H) @ P
        self.v_hat = float(self._kf_x[0])
        self.bias_hat = float(self._kf_x[1])
