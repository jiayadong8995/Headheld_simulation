"""被控对象符号与镜像正确性（任务书 §10 硬性检查 2）。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim.config import load_base, load_scenarios, resolve_scenario  # noqa: E402
from sim.plant import PlantParams, Plant                          # noqa: E402
from sim.scenarios import run_case                                # noqa: E402


def test_friction_opposes_motion():
    p = PlantParams.from_config(load_base())
    pl = Plant(p, mode="dry")
    for v in (0.1, 0.5, 1.0, -0.1, -0.5, -1.0):
        f_res = pl.friction.friction_force(v)
        assert np.sign(f_res) == np.sign(v), "摩擦必须与速度反号"


def test_brush_bias_can_be_negative():
    p = PlantParams.from_config(load_base())
    pl = Plant(p, mode="dry")
    pl.friction.set_brush_bias(-0.8)
    assert pl.friction.brush_force(0.0) < 0


def test_reverse_push_moves_backwards():
    base = load_base()
    sc = load_scenarios()
    cfg = resolve_scenario(base, sc, "S02")
    res = run_case(cfg, cfg["_scenario_spec"], "C2", seed=1)
    assert res.rec["x_true"][-1] < 0, "反向推力必须产生负位移"
    assert np.min(res.rec["v_true"]) < -0.1


def test_mirror_symmetry():
    """对称摩擦模型下，±F_h 应产生镜像运动。"""
    base = load_base()
    sc = load_scenarios()
    cfg1 = resolve_scenario(base, sc, "S01")
    r1 = run_case(cfg1, cfg1["_scenario_spec"], "C2", seed=7)
    cfg2 = resolve_scenario(base, sc, "S02")
    r2 = run_case(cfg2, cfg2["_scenario_spec"], "C2", seed=7)
    assert np.allclose(r1.rec["x_true"], -r2.rec["x_true"], atol=1e-9)
    assert np.allclose(r1.rec["v_true"], -r2.rec["v_true"], atol=1e-9)
    assert np.allclose(r1.rec["F_m_true"], -r2.rec["F_m_true"], atol=1e-6)


def test_sticking_holds_at_rest_below_static_friction():
    """合力未超过静摩擦时，对象必须保持静止（粘滑版本）。"""
    base = load_base()
    p = PlantParams.from_config(base)
    pl = Plant(p, mode="dry")
    pl.F_m = 0.5          # 小于 Fs=2.5 N
    for _ in range(500):
        pl.step(f_h=0.5, u=0.0, dt=0.001)
    assert abs(pl.v) < 1e-9
    assert abs(pl.x) < 1e-9


def test_breakaway_above_static_friction():
    base = load_base()
    p = PlantParams.from_config(base)
    pl = Plant(p, mode="dry")
    for _ in range(2000):
        pl.step(f_h=4.0, u=0.0, dt=0.001)   # > Fs
    assert pl.x > 0.005, "超过静摩擦后必须起步"
