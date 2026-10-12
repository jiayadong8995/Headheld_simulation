"""松手安全停止与静止不自走（任务书 §10-3、工程期望）。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim.config import load_base, load_scenarios, resolve_scenario  # noqa: E402
from sim.metrics import compute_metrics, find_release_time, stop_time_after  # noqa: E402
from sim.scenarios import run_case                                 # noqa: E402

DEPLOYABLE = ["C0", "C1", "C2", "C3"]


def test_release_then_stop_for_all_deployable():
    """S09：非零初速度松手后必须自然减速并在 4 s 内停稳（|v|<0.03 持续 1 s）。"""
    base = load_base()
    sc = load_scenarios()
    cfg = resolve_scenario(base, sc, "S09")
    for cname in DEPLOYABLE:
        res = run_case(cfg, cfg["_scenario_spec"], cname, seed=11)
        m = compute_metrics(res.rec, cfg, 2.6, 4.0)
        assert not np.isnan(m["release_time_s"])
        assert not np.isnan(m["stop_time_s"]), f"{cname} 未停稳"
        assert m["stop_time_s"] < 4.0, (cname, m["stop_time_s"])
        assert m["extra_travel_after_release_m"] < 1.0, (cname, m["extra_travel_after_release_m"])


def test_no_self_start_under_imu_bias_and_vibration():
    """S10：站立无推力（带 IMU 偏置/倾斜/振动）不得自走：10 s 位移 < 2 cm。"""
    base = load_base()
    sc = load_scenarios()
    cfg = resolve_scenario(base, sc, "S10")
    for cname in DEPLOYABLE:
        res = run_case(cfg, cfg["_scenario_spec"], cname, seed=13)
        drift = float(np.max(res.rec["x_true"]) - np.min(res.rec["x_true"]))
        vmax = float(np.max(np.abs(res.rec["v_true"])))
        assert drift < 0.02, f"{cname} 自走位移 {drift:.4f} m"
        assert vmax < 0.03, f"{cname} 自走速度 {vmax:.4f} m/s"


def test_no_creep_before_force_applied():
    """S07 小力起步：静摩擦阶段（t<0.6s，F_h<1.8N<F_s）不得出现无推力蠕动。"""
    base = load_base()
    sc = load_scenarios()
    cfg = resolve_scenario(base, sc, "S07")
    for cname in DEPLOYABLE:
        res = run_case(cfg, cfg["_scenario_spec"], cname, seed=17)
        t, f_h, v = res.rec["t"], res.rec["F_h_true"], res.rec["v_true"]
        early = t < 0.6
        assert np.max(np.abs(f_h[early])) < 2.0
        assert np.max(np.abs(v[early])) < 0.02, \
            f"{cname} 静摩擦阶段出现早期运动 {np.max(np.abs(v[early])):.4f} m/s"
        assert np.max(np.abs(res.rec["x_true"][early])) < 0.01


def test_stop_metrics_are_finite_for_push_release():
    base = load_base()
    sc = load_scenarios()
    cfg = resolve_scenario(base, sc, "S03")
    res = run_case(cfg, cfg["_scenario_spec"], "C3", seed=19)
    t_rel = find_release_time(res.rec["t"], res.rec["F_h_true"])
    assert t_rel is not None and 2.4 < t_rel < 3.6
    t_stop = stop_time_after(res.rec["t"], res.rec["v_true"], t_rel)
    assert t_stop is not None
