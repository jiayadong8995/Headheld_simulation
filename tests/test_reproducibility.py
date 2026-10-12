"""可复现性：相同种子结果一致；不同种子存在统计差异（任务书 §10-5）。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim.config import load_base, load_scenarios, resolve_scenario  # noqa: E402
from sim.plant import PlantParams                                  # noqa: E402
from sim.sweep import sample_plant_params                          # noqa: E402
from sim.scenarios import run_case                                 # noqa: E402


def test_same_seed_identical():
    base = load_base()
    sc = load_scenarios()
    cfg = resolve_scenario(base, sc, "S03")
    a = run_case(cfg, cfg["_scenario_spec"], "C3", seed=42)
    b = run_case(cfg, cfg["_scenario_spec"], "C3", seed=42)
    assert np.array_equal(a.rec["x_true"], b.rec["x_true"])
    assert np.array_equal(a.rec["F_h_true"], b.rec["F_h_true"])
    assert np.array_equal(a.rec["v_hat"], b.rec["v_hat"])


def test_different_seed_differs():
    base = load_base()
    sc = load_scenarios()
    cfg = resolve_scenario(base, sc, "S11")
    a = run_case(cfg, cfg["_scenario_spec"], "C3", seed=1)
    b = run_case(cfg, cfg["_scenario_spec"], "C3", seed=2)
    assert not np.array_equal(a.rec["x_true"], b.rec["x_true"])


def test_mc_sampling_respects_physical_constraints():
    base = load_base()
    rng = np.random.default_rng(0)
    params = PlantParams.from_config(base)
    for _ in range(200):
        p = sample_plant_params(rng, params, base["monte_carlo"])
        assert p.m > 0 and p.b >= 0
        for mode in p.Fc:
            assert p.Fs[mode] >= p.Fc[mode], "禁止生成 Fs<Fc 的非法样本"
