"""场景定义与运行器（任务书 §6.3 采样顺序 / §7 场景矩阵）。

关键约束：
  * 控制器每步只能拿到**带延迟的测量**（sensors.read_delayed），
  * 真值只进入日志供离线评价器使用，
  * 每个 plant 子步的人手力由人手模型基于外部可观测运动给出。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .actuator import Actuator, force_limit_from_motor
from .config import load_scenarios, resolve_scenario
from .controllers import build_controller
from .estimator import StateEstimator
from .human import build_human
from .plant import PlantParams, Plant
from .sensors import SensorPipeline


@dataclass
class CaseResult:
    scenario: str
    controller: str
    rec: Dict
    meta: Dict

    def __getitem__(self, key):
        return self.rec[key]


def _schedule_value(schedule: Sequence[Sequence[float]], t: float, default: float = 0.0) -> float:
    val = default
    for entry in schedule:
        if t >= entry[0] - 1e-12:
            val = entry[1]
    return val


def _mode_at(schedule: Sequence[Sequence], t: float) -> str:
    mode = schedule[0][1] if schedule else "dry"
    for entry in schedule:
        if t >= entry[0] - 1e-12:
            mode = entry[1]
    return mode


def run_case(cfg: Dict, spec: Dict, controller_name: str, *,
             ctrl_params: Optional[Dict] = None,
             seed: int = 20261012,
             human_variant: Optional[str] = None,
             task_spec: Optional[Dict] = None,
             plant_params: Optional[PlantParams] = None,
             duration_s: Optional[float] = None,
             collect: bool = True,
             estimator_mode: Optional[str] = None) -> CaseResult:
    """运行单个 (场景, 控制器) 组合。"""
    sim = cfg["simulation"]
    dt = float(sim["dt_plant_s"])
    Tc = float(sim["T_control_s"])
    n_sub = max(int(round(Tc / dt)), 1)
    duration = float(duration_s if duration_s is not None else
                     spec.get("duration_s", sim["default_duration_s"]))
    n_ctrl = int(round(duration / Tc))

    rng = np.random.default_rng(seed)
    sched = spec.get("friction_schedule", [[0.0, "dry"]])
    mode0 = _mode_at(sched, 0.0)
    ideal = mode0 == "ideal"
    act_ideal = bool(spec.get("actuator", {}).get("ideal", False)) or ideal
    sens_ideal = bool(spec.get("sensors", {}).get("ideal", False)) or ideal

    if plant_params is None:
        plant_params = PlantParams.from_config(cfg)
    if ideal:
        # 理想线性无阻力：D=0，滚刷=0，无粘滑
        pp = copy.copy(plant_params)
        pp.brush_bias, pp.brush_amp, pp.brush_noise_std = 0.0, 0.0, 0.0
        plant_params = pp
    plant = Plant(plant_params, mode=mode0, seed=seed, ideal=ideal,
                  instant_force=act_ideal)
    plant.record_log = False   # 运行器自己记录时间序列（预分配数组，快得多）
    actuator = Actuator(cfg, rng=np.random.default_rng(seed + 1), ideal=act_ideal)
    sensors = SensorPipeline(cfg, rng=np.random.default_rng(seed + 2), ideal=sens_ideal)
    estimator = StateEstimator(cfg, mode=estimator_mode or
                               cfg["sensors"]["fusion"].get("mode", "complementary"))
    controller = build_controller(controller_name, cfg, ctrl_params)

    human_spec = dict(spec.get("human", {"type": "fixed", "profile": "push_release"}))
    human = build_human(cfg, human_spec, task_spec=task_spec, variant=human_variant,
                        rng=np.random.default_rng(seed + 3))

    brush_sched = (spec.get("brush", {}) or {}).get("bias_schedule")
    is_coupled = human_spec.get("type") == "coupled"
    x_ref_fn = getattr(human, "x_ref", None)
    human_force = human.force

    # 预分配日志缓冲（plant 步）
    num_keys = ("t", "x_true", "v_true", "a_true", "F_m_true", "F_h_true", "D_true",
                "F_br_true", "F_res_true", "F_cmd", "F_cmd_raw", "u_active", "a_imu",
                "v_enc", "a_hat", "v_hat", "D_hat", "x_ref")
    N = n_ctrl * n_sub
    buf = {k: np.zeros(N) for k in num_keys}
    buf_state = np.empty(N, dtype="U8")
    i = 0
    tc, F_cmd_ctrl, states_ctrl = [], [], []
    friction = plant.friction

    for k in range(n_ctrl):
        t_k = k * Tc
        # --- 工况 / 滚刷调度 ---
        if brush_sched:
            friction.set_brush_bias(_schedule_value(brush_sched, t_k))
        if k > 0:
            friction.set_mode(_mode_at(sched, t_k))

        # --- 传感器采样（t_k 的状态） ---
        sensors.sample(t_k, plant.a, plant.v, plant.F_m)

        # --- 测量 -> 估计 -> 控制 ---
        meas = sensors.read_delayed(t_k)
        est = estimator.update(meas, Tc)
        oracle = plant.truth_snapshot() if controller.requires_oracle else None
        f_raw = controller.update(est, Tc, t_k, oracle)
        f_sent = actuator.command(f_raw, t_k, Tc)

        state = controller.info.get("state", "REST")
        a_imu, v_enc = meas["a_imu"], meas["v_enc"]
        a_hat, v_hat = est["a_hat"], est["v_hat"]
        d_hat = controller.info.get("D_hat", 0.0)
        tc.append(t_k); F_cmd_ctrl.append(f_sent); states_ctrl.append(state)
        collect_here = collect

        for j in range(n_sub):
            t = t_k + j * dt
            f_h = human_force(t, plant.x, plant.v)
            u = actuator.force_at(t)
            plant.step(f_h, u, dt)
            if collect_here:
                d, f_res, f_br = friction.resistance(plant.v, t)
                xr = x_ref_fn(t)[0] if (is_coupled and x_ref_fn) else np.nan
                buf["t"][i] = t
                buf["x_true"][i] = plant.x
                buf["v_true"][i] = plant.v
                buf["a_true"][i] = plant.a
                buf["F_m_true"][i] = plant.F_m
                buf["F_h_true"][i] = f_h
                buf["D_true"][i] = d
                buf["F_br_true"][i] = f_br
                buf["F_res_true"][i] = f_res
                buf["F_cmd"][i] = f_sent
                buf["F_cmd_raw"][i] = f_raw
                buf["u_active"][i] = u
                buf["a_imu"][i] = a_imu
                buf["v_enc"][i] = v_enc
                buf["a_hat"][i] = a_hat
                buf["v_hat"][i] = v_hat
                buf["D_hat"][i] = d_hat
                buf["x_ref"][i] = xr
                buf_state[i] = state
                i += 1

    rec: Dict[str, np.ndarray] = {}
    if collect:
        for key in num_keys:
            rec[key] = buf[key][:i]
        rec["state"] = buf_state[:i]
    rec["tc"] = np.asarray(tc)
    rec["F_cmd_ctrl"] = np.asarray(F_cmd_ctrl)
    rec["state_ctrl"] = np.asarray(states_ctrl)
    rec["meta"] = {
        "scenario": spec.get("name", ""),
        "seed": seed,
        "controller": controller_name,
        "force_limit": force_limit_from_motor(cfg),
        "packets_lost": int(actuator.lost_packets),
        "sensor_packets_lost": int(sensors.packets_lost),
        "saturated_steps": int(actuator.saturated_steps),
        "slew_limited_steps": int(actuator.slew_limited_steps),
        "is_coupled": is_coupled,
        "human_variant": human_variant,
        "ctrl_params": dict(ctrl_params or {}),
        "ideal": ideal,
        "estimator_mode": estimator.mode,
    }
    return CaseResult(scenario=cfg.get("_scenario", "custom"), controller=controller_name,
                      rec=rec, meta=rec["meta"])


def run_scenario_set(base_cfg: Dict, scenario_id: str, controllers: Sequence[str], *,
                     scenarios_cfg: Optional[Dict] = None, seed: Optional[int] = None,
                     tasks: Optional[Sequence[str]] = None,
                     human_variants: Optional[Sequence[str]] = None,
                     collect: bool = True) -> List[CaseResult]:
    """运行一个场景下的多控制器（人手耦合任务可额外展开）。"""
    sc = scenarios_cfg or load_scenarios()
    cfg = resolve_scenario(base_cfg, sc, scenario_id)
    spec = cfg["_scenario_spec"]
    seed = int(seed if seed is not None else base_cfg["simulation"]["random_seed"])
    results: List[CaseResult] = []
    if not tasks:
        for cname in controllers:
            results.append(run_case(cfg, spec, cname, seed=seed, collect=collect))
    else:
        for task_id in tasks:
            task = sc["coupled_tasks"][task_id]
            for cname in controllers:
                variants = human_variants or [None]
                for var in variants:
                    spec_c = copy.deepcopy(spec)
                    spec_c["duration_s"] = task["duration_s"]
                    spec_c["human"] = {"type": "coupled"}
                    if "friction_switch" in task:
                        sw = task["friction_switch"]
                        t_sw = sw["at"] * task["duration_s"]
                        base_mode = spec.get("friction_schedule", [[0.0, "dry"]])[0][1]
                        spec_c["friction_schedule"] = [[0.0, base_mode], [t_sw, sw["mode"]]]
                    res = run_case(cfg, spec_c, cname, seed=seed, task_spec=task,
                                   human_variant=var, collect=collect)
                    res.rec["meta"]["task"] = task_id
                    res.rec["meta"]["human_variant"] = var
                    results.append(res)
    return results


def run_all_scenarios(base_cfg: Dict, controllers: Sequence[str], *,
                      scenarios_cfg: Optional[Dict] = None,
                      seed: Optional[int] = None,
                      collect: bool = True) -> List[CaseResult]:
    sc = scenarios_cfg or load_scenarios()
    out: List[CaseResult] = []
    for sid in sc["scenarios"]:
        out.extend(run_scenario_set(base_cfg, sid, controllers, scenarios_cfg=sc,
                                    seed=seed, collect=collect))
    return out
