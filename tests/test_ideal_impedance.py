"""M1 核心硬性检查：理想线性无阻力下 C2 的理论等效惯量/阻尼（任务书 §10-1）。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim.config import load_base, load_scenarios, resolve_scenario  # noqa: E402
from sim.metrics import compute_metrics, reference_velocity       # noqa: E402
from sim.scenarios import run_case                                # noqa: E402


def _run(sid):
    base = load_base()
    sc = load_scenarios()
    cfg = resolve_scenario(base, sc, sid)
    return cfg, run_case(cfg, cfg["_scenario_spec"], "C2", seed=20261012)


def test_c2_matches_target_impedance_forward():
    cfg, res = _run("S01")
    m = float(cfg["mechanics"]["m_side_kg"])
    ka = float(cfg["controllers"]["C2_impedance"]["Ka_kg"])
    bd = float(cfg["controllers"]["C2_impedance"]["Bd_Ns_per_m"])
    met = compute_metrics(res.rec, cfg, m - ka, bd)
    # 最小二乘辨识出的等效惯量/阻尼应等于标称目标值
    assert abs(met["M_eff_fit"] - (m - ka)) < 0.05, met["M_eff_fit"]
    assert abs(met["B_eff_fit"] - bd) < 0.15, met["B_eff_fit"]
    assert met["fit_R2"] > 0.99
    # 数值误差目标：RMSE_v 相对参考峰值速度 < 2%
    assert met["RMSE_v_rel"] < 0.02, met["RMSE_v_rel"]


def test_c2_matches_target_impedance_backward():
    cfg, res = _run("S02")
    m = float(cfg["mechanics"]["m_side_kg"])
    ka = float(cfg["controllers"]["C2_impedance"]["Ka_kg"])
    bd = float(cfg["controllers"]["C2_impedance"]["Bd_Ns_per_m"])
    met = compute_metrics(res.rec, cfg, m - ka, bd)
    assert abs(met["M_eff_fit"] - (m - ka)) < 0.05
    assert met["RMSE_v_rel"] < 0.02


def test_c0_is_free_mass():
    base = load_base()
    sc = load_scenarios()
    cfg = resolve_scenario(base, sc, "S01")
    res = run_case(cfg, cfg["_scenario_spec"], "C0", seed=1)
    m = float(cfg["mechanics"]["m_side_kg"])
    met = compute_metrics(res.rec, cfg, m, 0.0)
    assert abs(met["M_eff_fit"] - m) < 0.02
    assert abs(met["B_eff_fit"]) < 0.05


def test_reference_velocity_converges_to_analytic_steady_state():
    """恒定 5 N、B_d=4、M_d=2.6 时，稳态速度应趋于 F_h/B_d = 1.25 m/s。"""
    t = np.arange(0, 20.0, 0.001)
    f_h = np.full_like(t, 5.0)
    v_ref = reference_velocity(t, f_h, md=2.6, bd=4.0)
    assert abs(v_ref[-1] - 1.25) < 0.01, v_ref[-1]
