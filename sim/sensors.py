"""传感器链路：IMU 加速度、轮速编码器、量化、噪声、时间戳与通信延迟、失帧。

测量模型（任务书 §3）：

    a_imu = a + g*sin(theta) + b_a + n_a + e_vib
    v_enc = (1-s)*v + n_v + e_q

控制器只能通过 `read_delayed()` 拿到带噪声、带延迟的测量值。
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np

G = 9.80665


class SensorPipeline:
    """按采样周期采集，按总延迟读出。"""

    def __init__(self, cfg: Dict, rng: Optional[np.random.Generator] = None,
                 ideal: bool = False):
        s = cfg["sensors"]
        self.cfg = cfg
        self.ideal = ideal
        self.rng = rng or np.random.default_rng(0)
        self.delay = 0.0 if ideal else float(cfg["actuator"]["delay_s"])
        self.p_loss = 0.0 if ideal else float(cfg["actuator"].get("missing_packet_probability", 0.0))

        imu = s["imu"]
        enc = s["encoder"]
        self.sigma_a = 0.0 if ideal else float(imu["accel_noise_std_mps2"])
        self.bias_a = 0.0 if ideal else float(imu["accel_bias_mps2"])
        self.tilt_rad = 0.0 if ideal else float(np.deg2rad(imu.get("tilt_deg", 0.0)))
        self.vib_amp = 0.0 if ideal else float(imu.get("vibration_amplitude_mps2", 0.0))
        self.vib_freq = float(imu.get("vibration_frequency_Hz", 6.0))

        self.sigma_v = 0.0 if ideal else float(enc["speed_noise_std_mps"])
        self.slip = 0.0 if ideal else float(enc.get("slip_fraction", 0.0))
        self.quant = 0.0 if ideal else float(enc.get("quantization_mps", 0.0))

        self._buf: Deque[Tuple[float, float, float, float]] = deque(maxlen=4000)
        self._last_out: Dict[str, float] = {"a_imu": 0.0, "v_enc": 0.0, "F_m_fb": 0.0}
        self.packets_lost = 0
        self.samples = 0

    # ------------------------------------------------------------------ #
    def sample(self, t: float, a_true: float, v_true: float, f_m_true: float) -> None:
        """在传感器采样时刻采集一次（含噪声/偏置/倾斜投影/振动/量化/打滑）。"""
        if self.ideal:
            self._buf.append((t, a_true, v_true, f_m_true))
            self.samples += 1
            return
        vib = self.vib_amp * np.sin(2.0 * np.pi * self.vib_freq * t)
        a_meas = (a_true + G * np.sin(self.tilt_rad) + self.bias_a
                  + self.rng.normal(0.0, self.sigma_a) + vib)
        v_meas = (1.0 - self.slip) * v_true + self.rng.normal(0.0, self.sigma_v)
        if self.quant > 0:
            v_meas = self.quant * np.round(v_meas / self.quant)
        self._buf.append((t, float(a_meas), float(v_meas), float(f_m_true)))
        self.samples += 1

    def read_delayed(self, t_now: float) -> Dict[str, float]:
        """读出 t_now - delay 时刻（或更早最近一次）的测量；含丢帧冻结。"""
        if self.ideal:
            if self._buf:
                _, a_m, v_m, f_m = self._buf[-1]
                return {"a_imu": a_m, "v_enc": v_m, "F_m_fb": f_m, "stale": 0.0, "t_meas": t_now}
            return {"a_imu": 0.0, "v_enc": 0.0, "F_m_fb": 0.0, "stale": 1.0, "t_meas": t_now}

        t_target = t_now - self.delay
        chosen = None
        for entry in reversed(self._buf):
            if entry[0] <= t_target + 1e-12:
                chosen = entry
                break
        if chosen is None:
            chosen = self._buf[0] if self._buf else (t_now, 0.0, 0.0, 0.0)
        stale = float(t_now - chosen[0] > 3.0 * max(self.delay, 0.02))
        out = {"a_imu": chosen[1], "v_enc": chosen[2], "F_m_fb": chosen[3],
               "stale": stale, "t_meas": chosen[0]}
        if self.p_loss > 0.0 and self.rng.random() < self.p_loss:
            self.packets_lost += 1
            return dict(self._last_out, stale=1.0, t_meas=chosen[0])
        self._last_out = out
        return out

    def reset(self) -> None:
        self._buf.clear()
        self._last_out = {"a_imu": 0.0, "v_enc": 0.0, "F_m_fb": 0.0}
        self.packets_lost = 0
        self.samples = 0
