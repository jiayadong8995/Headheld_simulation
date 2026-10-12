"""扫参、网格、Monte Carlo 鲁棒性评估与 Pareto 筛选（任务书 §6.1/§6.2）。"""
from __future__ import annotations

import copy
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import load_scenarios, resolve_scenario
from .metrics import compute_metrics, summarize
from .plant import PlantParams
from .scenarios import run_case

CAND_CONTROLLERS = ("C0", "C1", "C2", "C3")


# --------------------------------------------------------------------------- #
# 目标阻抗（各控制器对应的评价基准）
# --------------------------------------------------------------------------- #
def target_impedance(controller: str, cfg: Dict,
                     ctrl_params: Optional[Dict] = None) -> Tuple[float, float]:
    cp = ctrl_params or {}
    m_nom = float(cfg["mechanics"]["m_side_kg"])
    if controller == "C0":
        return m_nom, 0.0
    if controller == "C1":
        ka = float(cp.get("Ka", cfg["controllers"]["C1_existing"]["accel_gain_force_equivalent_kg"]))
        return m_nom - ka, 0.0
    if controller in ("C2", "C3", "C4a", "C4b"):
        key = "C3_impedance_comp" if controller in ("C3", "C4a", "C4b") else "C2_impedance"
        ka = float(cp.get("Ka", cfg["controllers"][key]["Ka_kg"]))
        bd = float(cp.get("Bd", cfg["controllers"][key]["Bd_Ns_per_m"]))
        return m_nom - ka, bd
    if controller in ("C5", "C5_oracle_force"):
        c = cfg["controllers"]["C5_admittance"]
        return float(c["Md_adm_kg"]), float(c["Bd_adm_per_m"] if "Bd_adm_per_m" in c else c["Bd_adm_Ns_per_m"])
    return m_nom, 0.0


def evaluate_case(cfg: Dict, spec: Dict, controller: str, *, ctrl_params=None,
                  seed: int = 0, task_spec=None, plant_params=None,
                  human_variant=None, duration_s=None) -> Dict[str, float]:
    res = run_case(cfg, spec, controller, ctrl_params=ctrl_params, seed=seed,
                   task_spec=task_spec, plant_params=plant_params,
                   human_variant=human_variant, duration_s=duration_s)
    Md, Bd = target_impedance(controller, cfg, ctrl_params)
    x_target = None
    if task_spec and task_spec.get("x_ref_sequence"):
        x_target = task_spec["x_ref_sequence"][-1][1]
    m = compute_metrics(res.rec, cfg, Md, Bd, is_coupled=bool(task_spec), x_target=x_target)
    m["controller"] = controller
    m["scenario"] = res.scenario
    if ctrl_params:
        for k, v in ctrl_params.items():
            m[f"param_{k}"] = float(v)
    return m


# --------------------------------------------------------------------------- #
# Monte Carlo 采样
# --------------------------------------------------------------------------- #
def sample_plant_params(rng: np.random.Generator, base: PlantParams, mc: Dict) -> PlantParams:
    p = copy.deepcopy(base)
    p.m = float(base.m * rng.uniform(*mc["m_ratio"]))
    p.b = float(base.b * rng.uniform(*mc["b_ratio"]))
    f_ratio = rng.uniform(*mc["Fc_ratio"])
    fs_ratio = rng.uniform(*mc["Fs_ratio"])
    p.Fc = {k: float(v * f_ratio) for k, v in base.Fc.items()}
    p.Fs = {k: float(max(v * fs_ratio, p.Fc[k])) for k, v in base.Fs.items()}  # 强制 Fs>=Fc
    p.brush_bias = float(rng.uniform(*mc["brush_bias_N"]))
    p.brush_amp = float(rng.uniform(*mc["ripple_amplitude_N"]))
    p.brush_freq = float(rng.uniform(*mc["ripple_frequency_Hz"]))
    p.tau_F = float(rng.uniform(*mc["tau_F_s"]))
    return p


def sample_env_overrides(rng: np.random.Generator, mc: Dict) -> Dict:
    loss = float(rng.choice(mc["packet_loss_set"]))
    return {
        "actuator": {"delay_s": float(rng.uniform(*mc["delay_s"])),
                     "missing_packet_probability": loss},
        "sensors": {"imu": {"accel_bias_mps2": float(rng.uniform(*mc["imubias_range"]
                                                                if "imubias_range" in mc
                                                                else mc["imu_bias_mps2"])),
                            "tilt_deg": float(rng.uniform(*mc["tilt_deg"]))},
                    "encoder": {"slip_fraction": float(rng.uniform(*mc["slip_fraction"]))}},
    }


def run_monte_carlo(base_cfg: Dict, n_runs: int, seed: int, outdir: Path,
                    scenarios_cfg: Optional[Dict] = None,
                    candidates: Optional[List[Dict]] = None,
                    progress=None) -> Tuple[List[Dict], List[Dict]]:
    """成对 Monte Carlo：同一随机物理/噪声样本评估所有候选。

    返回 (controller_comparison_rows, candidate_ranking_rows)。
    """
    sc = scenarios_cfg or load_scenarios()
    mc = base_cfg["monte_carlo"]
    rng = np.random.default_rng(seed)
    base_params = PlantParams.from_config(base_cfg)
    cmp_rows: List[Dict] = []
    rank_rows: List[Dict] = []

    scen_list = base_cfg.get("sweeps", {}).get("mc_scenarios", ["S03", "S08", "S09"])
    task = sc["coupled_tasks"]["T1"]
    if candidates is None:
        m = float(base_cfg["mechanics"]["m_side_kg"])
        # 基线（用于计算同一随机样本下的配对改善比）
        candidates = [{"controller": "C0"}, {"controller": "C1"},
                      {"controller": "C2", "Ka": m * 0.35, "Bd": 4.0},
                      {"controller": "C3", "Ka": m * 0.35, "Bd": 4.0, "alpha_D": 0.5}]
        candidates += [{"controller": "C3", "Ka": m * r, "Bd": bd, "alpha_D": ad}
                       for r, bd, ad in itertools.product(
                           [0.2, 0.35, 0.5], [2.0, 4.0], [0.5, 1.0])]

    for i in range(n_runs):
        s = int(rng.integers(0, 2 ** 31 - 1))
        p = sample_plant_params(rng, base_params, mc)
        env = sample_env_overrides(rng, mc)
        tau_h = float(rng.uniform(*mc["human_tau_h_s"]))

        for sid in scen_list:
            cfg = resolve_scenario(base_cfg, sc, sid)
            cfg = _deep_merge_local(cfg, env)
            spec = cfg["_scenario_spec"]
            for cname in CAND_CONTROLLERS:
                try:
                    m = evaluate_case(cfg, spec, cname, seed=s, plant_params=p)
                    m["sample"] = i
                    m["scenario"] = sid
                    cmp_rows.append(m)
                except Exception as exc:  # 记录失效样本
                    cmp_rows.append({"sample": i, "scenario": sid, "controller": cname,
                                     "error": str(exc)[:120]})

        # 候选参数排名：人机耦合 T1 任务
        for cand in candidates:
            cname = cand["controller"]
            ctrl_params = {k: v for k, v in cand.items() if k != "controller"}
            cfg = resolve_scenario(base_cfg, sc, "S03")
            cfg = _deep_merge_local(cfg, env)
            spec = copy.deepcopy(cfg["_scenario_spec"])
            spec["duration_s"] = task["duration_s"]
            spec["human"] = {"type": "coupled"}
            try:
                m = evaluate_case(cfg, spec, cname, ctrl_params=ctrl_params, seed=s,
                                  task_spec=task, plant_params=p)
                m["sample"] = i
                m["candidate"] = json.dumps(cand, sort_keys=True)
                m["tau_h"] = tau_h
                rank_rows.append(m)
            except Exception as exc:
                rank_rows.append({"sample": i, "candidate": json.dumps(cand, sort_keys=True),
                                  "error": str(exc)[:120]})
        if progress and (i + 1) % max(n_runs // 10, 1) == 0:
            progress(i + 1, n_runs)
    return cmp_rows, rank_rows


def mc_standstill_screen(base_cfg: Dict, n_runs: int, seed: int,
                         candidates: Optional[List[Dict]] = None,
                         scenarios_cfg: Optional[Dict] = None,
                         duration_s: float = 3.0,
                         progress=None) -> List[Dict]:
    """独立安全筛查：用同一 MC 样本的 IMU 偏置/倾斜/打滑/滚刷设置，
    跑「站立无推力」试验（S10），检查是否出现自走/蠕动。

    任务书 §10-3 要求该检查独立测量，不能只看有人推力的场景。
    """
    sc = scenarios_cfg or load_scenarios()
    mc = base_cfg["monte_carlo"]
    rng = np.random.default_rng(seed)
    base_params = PlantParams.from_config(base_cfg)
    if candidates is None:
        m = float(base_cfg["mechanics"]["m_side_kg"])
        candidates = [{"controller": "C0"}, {"controller": "C1"},
                      {"controller": "C2", "Ka": m * 0.35, "Bd": 4.0},
                      {"controller": "C3", "Ka": m * 0.35, "Bd": 4.0, "alpha_D": 0.5}]
        candidates += [{"controller": "C3", "Ka": m * r, "Bd": bd, "alpha_D": ad}
                       for r, bd, ad in itertools.product([0.2, 0.35, 0.5], [2.0, 4.0], [0.5, 1.0])]
    rows: List[Dict] = []
    for i in range(n_runs):
        s = int(rng.integers(0, 2 ** 31 - 1))
        p = sample_plant_params(rng, base_params, mc)
        env = sample_env_overrides(rng, mc)
        for cand in candidates:
            cname = cand.get("controller", "C3")
            ctrl_params = {k: v for k, v in cand.items() if k != "controller"}
            cfg = _deep_merge_local(resolve_scenario(base_cfg, sc, "S10"), env)
            spec = cfg["_scenario_spec"]
            try:
                m = evaluate_case(cfg, spec, cname, ctrl_params=ctrl_params or None,
                                  seed=s, plant_params=p, duration_s=duration_s)
                rows.append({"sample": i, "candidate": json.dumps(cand, sort_keys=True),
                             "standstill_drift_m": m["standstill_drift_m"],
                             "standstill_vmax": m["standstill_vmax"],
                             "osc_index": m["osc_index"]})
            except Exception as exc:
                rows.append({"sample": i, "candidate": json.dumps(cand, sort_keys=True),
                             "error": str(exc)[:120]})
        if progress and (i + 1) % max(n_runs // 10, 1) == 0:
            progress(i + 1, n_runs)
    return rows


def _deep_merge_local(base: Dict, override: Dict) -> Dict:
    from .config import deep_merge
    return deep_merge(base, override)


# --------------------------------------------------------------------------- #
# 安全筛选 + Pareto
# --------------------------------------------------------------------------- #
def safety_filter(rows: List[Dict], m_nom: float, m_min_ratio: float = 0.75,
                  sat_limit: float = 0.5, osc_limit: float = 0.6) -> Tuple[List[Dict], List[Dict]]:
    """先做安全筛选：K_a<m_min、无自启动、无持续振荡、饱和不超限。"""
    ok, bad = [], []
    for r in rows:
        if "error" in r:
            bad.append(r); continue
        reasons = []
        ka = r.get("param_Ka", 0.0)
        if ka >= m_nom * m_min_ratio:
            reasons.append("Ka>=m_min")
        if r.get("sat_ratio", 0.0) > sat_limit:
            reasons.append("saturation")
        if r.get("osc_index", 0.0) > osc_limit:
            reasons.append("oscillation")
        if r.get("standstill_drift_m", 0.0) > 0.02 and r.get("F_h_peak", 0.0) < 0.5:
            reasons.append("self_start")
        (bad if reasons else ok).append({**r, "fail_reasons": ",".join(reasons)})
    return ok, bad


def pareto_front(rows: List[Dict], objectives: Sequence[str]) -> List[Dict]:
    """非支配前沿（全部按最小化）。"""
    front = []
    for r in rows:
        dominated = False
        for q in rows:
            if q is r:
                continue
            le = all(q[o] <= r[o] + 1e-12 for o in objectives)
            lt = any(q[o] < r[o] - 1e-12 for o in objectives)
            if le and lt:
                dominated = True
                break
        if not dominated:
            front.append(r)
    return front
