"""执行器限制、饱和、丢帧与看门狗必须实际生效（任务书 §10-4）。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim.actuator import Actuator, force_limit_from_motor, force_from_torque, torque_from_force  # noqa: E402
from sim.config import load_base, load_scenarios, resolve_scenario  # noqa: E402
from sim.scenarios import run_case                                 # noqa: E402


def test_force_limit_from_rated_torque():
    cfg = load_base()
    f_lim = force_limit_from_motor(cfg)
    assert abs(f_lim - 0.25 / 0.04) < 1e-9
    assert f_lim < 0.85 / 0.04 - 1e-9, "不得把堵转转矩当作持续力上限"


def test_torque_force_roundtrip():
    cfg = load_base()
    for f in (-6.0, -1.0, 0.0, 2.0, 6.0):
        assert abs(force_from_torque(torque_from_force(f, cfg), cfg) - f) < 1e-9


def test_actuator_saturates_and_slew_limits():
    cfg = load_base()
    cfg["actuator"]["force_slew_limit_Nps"] = 1e6   # 先隔离速率限制，单测饱和
    act = Actuator(cfg, ideal=False)
    f_lim = act.F_max
    act.command(1000.0, 0.0, 0.01)
    active = act.force_at(0.05)   # 延迟 0.02s 后生效，且在看门狗 0.1s 之内
    assert abs(active) <= f_lim + 1e-9
    assert abs(active - f_lim) < 1e-9
    assert act.saturated_steps > 0

    cfg2 = load_base()
    cfg2["actuator"]["force_slew_limit_Nps"] = 1.0
    act2 = Actuator(cfg2)
    act2.command(10.0, 0.0, 0.01)
    assert abs(act2.force_at(1.0)) <= 1.0 * 0.01 + 1e-9, "速率限制必须生效"
    assert act2.slew_limited_steps > 0


def test_watchdog_zeroes_command():
    cfg = load_base()
    cfg["actuator"]["watchdog_timeout_s"] = 0.05
    act = Actuator(cfg)
    act.command(3.0, 0.0, 0.01)
    assert abs(act.force_at(1.0)) < 1e-12


def test_controller_cannot_bypass_force_limit():
    """整链路：真实 F_m 绝不超过保守力上限。"""
    base = load_base()
    sc = load_scenarios()
    cfg = resolve_scenario(base, sc, "S12")
    # 激进增益 + 短延迟，逼迫控制器给出大指令
    res = run_case(cfg, cfg["_scenario_spec"], "C3",
                   ctrl_params={"Ka": 3.0, "Bd": 8.0, "alpha_D": 1.0}, seed=3)
    f_lim = res.meta["force_limit"]
    assert np.max(np.abs(res.rec["F_m_true"])) <= f_lim + 1e-6
    assert np.max(np.abs(res.rec["F_cmd"])) <= f_lim + 1e-9


def test_packet_loss_is_counted():
    cfg = load_base()
    cfg["actuator"]["missing_packet_probability"] = 0.5
    act = Actuator(cfg, rng=np.random.default_rng(0))
    for k in range(200):
        act.command(1.0, k * 0.01, 0.01)
    assert act.lost_packets > 0
