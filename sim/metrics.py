"""统一评价指标与计算口径（任务书 §8）。

全部指标给出 mean / P95 / worst（多工况汇总时由 sweep 层负责）。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _mask_motion(v: np.ndarray, f_h: np.ndarray, deadband: float) -> np.ndarray:
    return (np.abs(v) > deadband) | (np.abs(f_h) > 0.1)


def integrate_trapz(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 2:
        return 0.0
    return float(np.trapezoid(y, t))


def reference_velocity(t: np.ndarray, f_h: np.ndarray, md: float, bd: float,
                       v0: float = 0.0) -> np.ndarray:
    """离线理想目标阻抗响应：M_d*ṿ_ref + B_d*v_ref = F_h（只用于评价，控制器不可读）。"""
    out = np.zeros_like(t)
    v = v0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        dv = (f_h[i - 1] - bd * v) / max(md, 1e-6)
        v += dv * dt
        out[i] = v
    return out


def bandlimited_jerk(t: np.ndarray, a: np.ndarray, cutoff_hz: float = 10.0) -> np.ndarray:
    """对带限加速度求导，避免直接对高频噪声差分。"""
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 1e-3
    tau = 1.0 / (2.0 * np.pi * max(cutoff_hz, 1e-3))
    alpha = dt / (tau + dt)
    a_f = np.zeros_like(a)
    acc = a[0]
    for i, val in enumerate(a):
        acc += alpha * (val - acc)
        a_f[i] = acc
    return np.gradient(a_f, t)


def find_release_time(t: np.ndarray, f_h: np.ndarray) -> Optional[float]:
    """固定力试验：识别人手松手时刻（峰值后首次跌落到 5% 以下）。"""
    if len(f_h) == 0:
        return None
    peak = float(np.max(np.abs(f_h)))
    if peak < 1e-6:
        return None
    idx_peak = int(np.argmax(np.abs(f_h)))
    for i in range(idx_peak, len(f_h)):
        if abs(f_h[i]) < 0.05 * peak:
            return float(t[i])
    return None


def stop_time_after(t: np.ndarray, v: np.ndarray, t0: float, v_thresh: float = 0.03,
                    hold_s: float = 1.0) -> Optional[float]:
    """t0 之后速度首次持续 < v_thresh 的时刻。"""
    idx0 = int(np.searchsorted(t, t0))
    n_hold = max(int(hold_s / max(np.mean(np.diff(t)), 1e-6)), 1)
    for i in range(idx0, len(v) - n_hold):
        if np.all(np.abs(v[i:i + n_hold]) < v_thresh):
            return float(t[i + n_hold])
    return None


def count_direction_reversals(v: np.ndarray, thresh: float = 0.03) -> int:
    sign = np.sign(np.where(np.abs(v) > thresh, v, 0.0))
    sign = sign[sign != 0]
    if len(sign) < 2:
        return 0
    return int(np.sum(np.abs(np.diff(sign)) > 0))


def oscillation_metric(t: np.ndarray, v: np.ndarray, band: tuple = (2.0, 40.0)) -> float:
    """带内速度能量占比（持续振荡检测）。"""
    if len(v) < 8:
        return 0.0
    dt = float(np.mean(np.diff(t)))
    v_d = v - np.mean(v)
    spec = np.abs(np.fft.rfft(v_d)) ** 2
    freq = np.fft.rfftfreq(len(v_d), dt)
    lo, hi = band
    band_e = float(np.sum(spec[(freq >= lo) & (freq <= hi)]))
    total = float(np.sum(spec[1:])) or 1e-12
    return band_e / total


def compute_metrics(rec: Dict, cfg: Dict, Md: float, Bd: float,
                    is_coupled: bool = False, x_target: Optional[float] = None) -> Dict[str, float]:
    t = rec["t"]
    v = rec["v_true"]
    a = rec["a_true"]
    f_h = rec["F_h_true"]
    f_m = rec["F_m_true"]
    x = rec["x_true"]
    f_cmd = rec["F_cmd"]
    shared = cfg["controllers"]["shared"]
    deadband = float(shared["speed_stop_deadband_mps"])

    v_ref = reference_velocity(t, f_h, Md, Bd)
    mask = _mask_motion(v, f_h, deadband)
    if not np.any(mask):
        mask = np.ones_like(v, dtype=bool)

    m: Dict[str, float] = {}
    # ---- 控制效果 ----
    m["RMSE_v"] = float(np.sqrt(np.mean((v - v_ref) ** 2)))
    m["v_ref_peak"] = float(np.max(np.abs(v_ref))) if len(v_ref) else 0.0
    m["RMSE_v_rel"] = m["RMSE_v"] / max(m["v_ref_peak"], 1e-6)
    z_res = f_h - Md * a - Bd * v
    m["Z_residual_RMSE"] = float(np.sqrt(np.mean(z_res[mask] ** 2)))
    A = np.column_stack([a[mask], v[mask]])
    try:
        coef, res, *_ = np.linalg.lstsq(A, f_h[mask], rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((f_h[mask] - pred) ** 2))
        ss_tot = float(np.sum((f_h[mask] - np.mean(f_h[mask])) ** 2)) or 1e-12
        m["M_eff_fit"] = float(coef[0])
        m["B_eff_fit"] = float(coef[1])
        m["fit_R2"] = 1.0 - ss_res / ss_tot
    except np.linalg.LinAlgError:
        m["M_eff_fit"] = m["B_eff_fit"] = float("nan")
        m["fit_R2"] = float("nan")

    # ---- 手柄交互 ----
    m["F_h_rms"] = float(np.sqrt(np.mean(f_h ** 2)))
    m["F_h_peak"] = float(np.max(np.abs(f_h)))
    m["W_h_plus"] = integrate_trapz(t, np.maximum(f_h * v, 0.0))

    # ---- 手感与安全 ----
    m["a_peak"] = float(np.max(np.abs(a)))
    m["a_rms"] = float(np.sqrt(np.mean(a ** 2)))
    jerk = bandlimited_jerk(t, a)
    m["jerk_peak"] = float(np.max(np.abs(jerk)))
    m["jerk_rms"] = float(np.sqrt(np.mean(jerk ** 2)))
    m["v_peak"] = float(np.max(np.abs(v)))
    m["x_final"] = float(x[-1])
    m["x_max"] = float(np.max(x))
    m["x_min"] = float(np.min(x))
    m["dir_reversals"] = float(count_direction_reversals(v))
    m["osc_index"] = oscillation_metric(t, v)

    # ---- 电机负担 ----
    m["F_cmd_peak"] = float(np.max(np.abs(f_cmd)))
    m["F_cmd_rms"] = float(np.sqrt(np.mean(f_cmd ** 2)))
    m["F_m_peak"] = float(np.max(np.abs(f_m)))
    m["F_m_rms"] = float(np.sqrt(np.mean(f_m ** 2)))
    m["sat_ratio"] = float(np.mean(np.abs(rec["F_cmd_raw"]) > rec["meta"]["force_limit"] + 1e-9))
    m["W_m_plus"] = integrate_trapz(t, np.maximum(f_m * v, 0.0))
    m["F_rate_peak"] = float(np.max(np.abs(np.gradient(rec["F_cmd"], t))))

    # ---- 状态机 ----
    states = rec["state"]
    uniq, counts = np.unique(states, return_counts=True)
    for s, c in zip(uniq, counts):
        m[f"state_frac_{s}"] = float(c / len(states))
    m["state_switches"] = float(np.sum(states[1:] != states[:-1]))
    m["packets_lost"] = float(rec["meta"].get("packets_lost", 0))

    # ---- 松手行为 ----
    t_rel = find_release_time(t, f_h)
    m["release_time_s"] = float(t_rel) if t_rel is not None else float("nan")
    if t_rel is not None:
        i_rel = int(np.searchsorted(t, t_rel))
        t_stop = stop_time_after(t, v, t_rel, v_thresh=0.03, hold_s=1.0)
        if t_stop is not None:
            i_stop = int(np.searchsorted(t, t_stop))
            m["extra_travel_after_release_m"] = float(x[i_stop] - x[i_rel])
            m["stop_time_s"] = float(t_stop - t_rel)
        else:
            m["extra_travel_after_release_m"] = float(x[-1] - x[i_rel])
            m["stop_time_s"] = float("nan")
    else:
        m["extra_travel_after_release_m"] = float("nan")
        m["stop_time_s"] = float("nan")

    # ---- 静止不自走（只统计「首次真实运动之前」的窗口）----
    moving = (np.abs(v) > 0.05) | (np.abs(x) > 0.01)
    idx_motion = np.argmax(moving) if np.any(moving) else len(v)
    pre = np.arange(len(v)) < max(idx_motion, 1)
    if np.any(pre):
        m["standstill_drift_m"] = float(np.max(x[pre]) - np.min(x[pre]))
        m["standstill_vmax"] = float(np.max(np.abs(v[pre])))
        m["standstill_window_s"] = float(t[pre][-1] - t[pre][0]) if len(t[pre]) > 1 else 0.0
    else:
        m["standstill_drift_m"] = 0.0
        m["standstill_vmax"] = 0.0
        m["standstill_window_s"] = 0.0

    # ---- 耦合任务 ----
    if is_coupled and x_target is not None:
        ref = rec.get("x_ref")
        if ref is not None:
            err = np.abs(np.asarray(ref) - x)
            m["task_pos_err_final"] = float(err[-1])
            m["task_pos_err_max"] = float(np.max(err))
            m["task_overshoot"] = float(max(np.max(x - np.asarray(ref)), 0.0))
        m["completion_time_s"] = float(t[-1])
    return m


def summarize(rows) -> Dict[str, Dict[str, float]]:
    """把多工况指标列表汇总成 mean/P95/worst。"""
    keys = [k for k in rows[0].keys() if isinstance(rows[0][k], (int, float))]
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            out[k] = {"mean": float("nan"), "p95": float("nan"), "worst": float("nan")}
            continue
        out[k] = {"mean": float(np.mean(vals)), "p95": float(np.percentile(np.abs(vals), 95)),
                  "worst": float(np.max(np.abs(vals)))}
    return out
