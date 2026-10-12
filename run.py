#!/usr/bin/env python3
"""assist_sim 命令行入口（任务书 §9.2）。

    python run.py --scenario S03 --controllers C0,C1,C2,C3 --config config/base.yaml
    python run.py --scenario all --controllers C0,C1,C2,C3
    python run.py --scenario coupled --controllers C0,C1,C2,C3
    python run.py --sweep ka_bd_alpha
    python run.py --monte-carlo 300 --seed 20261012
    python run.py --milestone M1|M2|M3
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sim import plots as P                                    # noqa: E402
from sim.config import deep_merge, load_base, load_scenarios, provenance_map, resolve_scenario  # noqa: E402
from sim.metrics import compute_metrics, summarize            # noqa: E402
from sim.plant import PlantParams                             # noqa: E402
from sim.scenarios import run_case, run_scenario_set          # noqa: E402
from sim.sweep import (mc_standstill_screen, pareto_front, run_monte_carlo,  # noqa: E402
                       safety_filter, target_impedance)
from sim.actuator import force_limit_from_motor               # noqa: E402

ALL_SCENARIOS = [f"S{i:02d}" for i in range(1, 13)]
DEPLOYABLE = ["C0", "C1", "C2", "C3"]


# --------------------------------------------------------------------------- #
def make_run_dir(prefix: str = "run") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "outputs" / f"{stamp}_{prefix}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def env_fingerprint() -> Dict:
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "git": git_rev(), "time": datetime.now().isoformat(timespec="seconds")}


def dump_resolved(outdir: Path, cfg: Dict, scenarios_cfg: Dict) -> None:
    with open(outdir / "config_resolved.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump({"base": {k: v for k, v in cfg.items() if not k.startswith("_")},
                        "scenarios_file": "config/scenarios.yaml"}, fh,
                       allow_unicode=True, sort_keys=False)
    with open(outdir / "env_fingerprint.json", "w", encoding="utf-8") as fh:
        json.dump(env_fingerprint(), fh, ensure_ascii=False, indent=2)
    pm = provenance_map(cfg)
    rows = []
    for k, v in pm.items():
        rows.append({"parameter": k, "source": v})
    pd.DataFrame(rows).to_csv(outdir / "parameter_provenance.csv", index=False)


def md_bd_table(cfg: Dict, controllers: Sequence[str]) -> Dict[str, tuple]:
    out = {}
    for c in controllers:
        out[c] = target_impedance(c, cfg)
    return out


# --------------------------------------------------------------------------- #
# M1 / M2：场景矩阵
# --------------------------------------------------------------------------- #
def run_scenarios(outdir: Path, cfg: Dict, scenarios_cfg: Dict, scenario_ids: List[str],
                  controllers: Sequence[str], seed: int, detail_figs: bool = True,
                  write_ts: bool = True, tasks: Optional[List[str]] = None,
                  human_variants: Optional[List[str]] = None) -> pd.DataFrame:
    figdir = outdir / "figures"
    tsdir = outdir / "timeseries"
    tsdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []

    for sid in scenario_ids:
        t0 = time.time()
        results = run_scenario_set(cfg, sid, controllers, scenarios_cfg=scenarios_cfg,
                                  seed=seed, tasks=tasks, human_variants=human_variants)
        scfg = resolve_scenario(cfg, scenarios_cfg, sid)
        md_of = {c: target_impedance(c, scfg)[0] for c in controllers}
        bd_of = {c: target_impedance(c, scfg)[1] for c in controllers}
        if tasks:
            groups: Dict[tuple, List] = {}
            for res in results:
                groups.setdefault((res.rec["meta"].get("task"),
                                   res.rec["meta"].get("human_variant")), []).append(res)
            for (task_id, var), grp in groups.items():
                P.plot_scenario_panel(grp, figdir, scfg, md_of, bd_of,
                                      tag=f"{task_id}_{var}")
        else:
            P.plot_scenario_panel(results, figdir, scfg, md_of, bd_of)
        if detail_figs:
            for res in results:
                P.plot_single_controller(res, figdir, scfg, md_of[res.controller],
                                         bd_of[res.controller])
        for res in results:
            Md, Bd = md_of[res.controller], bd_of[res.controller]
            task_id = res.rec["meta"].get("task")
            is_coupled = bool(task_id)
            x_target = None
            if is_coupled:
                seq = scenarios_cfg["coupled_tasks"][task_id].get("x_ref_sequence", [])
                x_target = seq[-1][1] if seq else None
            m = compute_metrics(res.rec, scfg, Md, Bd, is_coupled=is_coupled,
                                x_target=x_target)
            m.update({"scenario": res.scenario, "controller": res.controller,
                      "seed": seed, "run_time_s": round(time.time() - t0, 2),
                      "packets_lost": res.meta["packets_lost"],
                      "saturated_steps": res.meta["saturated_steps"]})
            var = res.rec["meta"].get("human_variant")
            if task_id:
                m["task"] = task_id
            if var:
                m["human_variant"] = var
            rows.append(m)
            if write_ts:
                df = pd.DataFrame({k: v for k, v in res.rec.items()
                                   if isinstance(v, np.ndarray) and len(v) == len(res.rec["t"])})
                name = f"{res.scenario}_{res.controller}"
                if task_id:
                    name += f"_{task_id}"
                if var:
                    name += f"_{var}"
                df.to_csv(tsdir / f"{name}.csv", index=False, float_format="%.6g")
        print(f"[scenario] {sid}  {len(results)} runs  {time.time()-t0:.1f}s")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# M3a：单因素 / 两因素扫参
# --------------------------------------------------------------------------- #
def run_sweep(outdir: Path, cfg: Dict, scenarios_cfg: Dict, seed: int) -> pd.DataFrame:
    figdir = outdir / "figures"
    sw = cfg["sweeps"]
    m_nom = float(cfg["mechanics"]["m_side_kg"])
    rows: List[Dict] = []

    # 单因素：Ka/m、Bd、alpha_D（其余保持默认）—— 评价口径：S03 力给定 + T1 耦合 + S09 松手
    def eval_params(params: Dict, sid: str, task: bool = False) -> Dict:
        scfg = resolve_scenario(cfg, scenarios_cfg, sid)
        spec = copy.deepcopy(scfg["_scenario_spec"])
        task_spec = None
        if task:
            task_spec = scenarios_cfg["coupled_tasks"]["T1"]
            spec["duration_s"] = task_spec["duration_s"]
            spec["human"] = {"type": "coupled"}
        res = run_case(scfg, spec, "C3", ctrl_params=params, seed=seed, task_spec=task_spec)
        Md, Bd = target_impedance("C3", scfg, params)
        m = compute_metrics(res.rec, scfg, Md, Bd, is_coupled=task,
                            x_target=(task_spec or {}).get("x_ref_sequence", [[0, 0, 1]])[-1][1])
        m.update({"controller": "C3", "scenario": sid, **{f"param_{k}": v for k, v in params.items()}})
        return m

    for ratio in sw["Ka_over_m_nominal"]:
        for bd in [sw["Bd_Ns_per_m"][1]]:
            p = {"Ka": m_nom * ratio, "Bd": bd, "alpha_D": cfg["controllers"]["C3_impedance_comp"]["alpha_D"]}
            rows.append({**eval_params(p, "S03"), "sweep": "Ka", "level": ratio})
            rows.append({**eval_params(p, "S09"), "sweep": "Ka", "level": ratio})
            rows.append({**eval_params(p, "S03", task=True), "sweep": "Ka_coupled", "level": ratio})
            rows.append({**eval_params(p, "S08"), "sweep": "Ka", "level": ratio})

    for bd in sw["Bd_Ns_per_m"]:
        p = {"Ka": m_nom * 0.35, "Bd": bd, "alpha_D": cfg["controllers"]["C3_impedance_comp"]["alpha_D"]}
        rows.append({**eval_params(p, "S03"), "sweep": "Bd", "level": bd})
        rows.append({**eval_params(p, "S09"), "sweep": "Bd", "level": bd})
        rows.append({**eval_params(p, "S03", task=True), "sweep": "Bd_coupled", "level": bd})

    for ad in sw["alpha_D"]:
        p = {"Ka": m_nom * 0.35, "Bd": 4.0, "alpha_D": ad}
        rows.append({**eval_params(p, "S04"), "sweep": "alpha_D", "level": ad})
        rows.append({**eval_params(p, "S05"), "sweep": "alpha_D", "level": ad})
        rows.append({**eval_params(p, "S06"), "sweep": "alpha_D", "level": ad})

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "sweep_metrics.csv", index=False)

    # 两因素网格 Ka/m × Bd：三个互相制约的指标
    ka_list = sw["Ka_over_m_nominal"]
    bd_list = sw["Bd_Ns_per_m"]
    grids = {k: np.zeros((len(bd_list), len(ka_list))) for k in ("F_h_rms", "stop_time_s", "jerk_rms")}
    for i, bd in enumerate(bd_list):
        for j, ratio in enumerate(ka_list):
            p = {"Ka": m_nom * ratio, "Bd": bd, "alpha_D": 0.5}
            m_c = eval_params(p, "S03", task=True)
            grids["F_h_rms"][i, j] = m_c["F_h_rms"]
            m_s = eval_params(p, "S09")
            grids["stop_time_s"][i, j] = np.nan_to_num(m_s.get("stop_time_s", np.nan), nan=99.0)
            m_j = eval_params(p, "S08")
            grids["jerk_rms"][i, j] = m_j["jerk_rms"]
    for key, grid in grids.items():
        P.plot_heatmap(grid, ka_list, bd_list, figdir, f"grid_{key}.png",
                       f"K_a/m × B_d → {key}", key)
        pd.DataFrame(grid, index=bd_list, columns=ka_list).to_csv(outdir / f"grid_{key}.csv")
    return df


# --------------------------------------------------------------------------- #
# M3b：Monte Carlo
# --------------------------------------------------------------------------- #
def run_mc(outdir: Path, cfg: Dict, scenarios_cfg: Dict, n_runs: int, seed: int) -> None:
    def prog(i, n):
        print(f"[monte-carlo] {i}/{n}")

    cmp_rows, rank_rows = run_monte_carlo(cfg, n_runs, seed, outdir,
                                          scenarios_cfg=scenarios_cfg, progress=prog)
    pd.DataFrame(cmp_rows).to_csv(outdir / "mc_controller_comparison_raw.csv", index=False)

    m_nom = float(cfg["mechanics"]["m_side_kg"])
    # --- 控制器配对比较汇总 ---
    dfc = pd.DataFrame(cmp_rows)
    summary_rows = []
    if not dfc.empty and "error" not in dfc.columns:
        for (sid, cname), grp in dfc.groupby(["scenario", "controller"]):
            s = summarize(grp.to_dict("records"))
            row = {"scenario": sid, "controller": cname, "n": len(grp)}
            for k, v in s.items():
                row[f"{k}_mean"] = v["mean"]
                row[f"{k}_p95"] = v["p95"]
                row[f"{k}_worst"] = v["worst"]
            summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(outdir / "robustness_summary.csv", index=False)

    # --- 候选参数安全筛选 + Pareto ---
    dfr = pd.DataFrame([r for r in rank_rows if "error" not in r])
    # 配对基线：同一随机样本下的 C1 / C0（保证比较公平）
    if not dfr.empty:
        key_c1 = json.dumps({"controller": "C1"}, sort_keys=True)
        key_c0 = json.dumps({"controller": "C0"}, sort_keys=True)
        if key_c1 in set(dfr.candidate):
            b1 = dfr[dfr.candidate == key_c1][["sample", "F_h_rms", "W_h_plus"]].rename(
                columns={"F_h_rms": "F_h_rms_C1", "W_h_plus": "W_h_plus_C1"})
            dfr = dfr.merge(b1, on="sample", how="left")
            dfr["improve_vs_C1"] = 1.0 - dfr.F_h_rms / dfr.F_h_rms_C1
        if key_c0 in set(dfr.candidate):
            b0 = dfr[dfr.candidate == key_c0][["sample", "F_h_rms"]].rename(
                columns={"F_h_rms": "F_h_rms_C0"})
            dfr = dfr.merge(b0, on="sample", how="left")
            dfr["improve_vs_C0"] = 1.0 - dfr.F_h_rms / dfr.F_h_rms_C0
    ok_rows, bad_rows = safety_filter(dfr.to_dict("records"), m_nom)
    pd.DataFrame(ok_rows).to_csv(outdir / "mc_candidates_safe.csv", index=False)
    pd.DataFrame(bad_rows).to_csv(outdir / "mc_candidates_rejected.csv", index=False)

    # 按候选聚合（跨样本均值）后再做 Pareto
    agg = {}
    for r in ok_rows:
        agg.setdefault(r["candidate"], []).append(r)
    cands = []
    for cid, rs in agg.items():
        d = {"candidate": cid, "n_samples": len(rs)}
        for key in ("F_h_rms", "F_h_peak", "W_h_plus", "jerk_rms", "jerk_peak", "improve_vs_C1",
                    "improve_vs_C0", "osc_index", "sat_ratio", "F_m_rms", "task_pos_err_final",
                    "extra_travel_after_release_m", "stop_time_s", "state_switches"):
            vals = np.array([x.get(key, np.nan) for x in rs], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals):
                d[f"{key}_mean"] = float(np.mean(vals))
                d[f"{key}_p95"] = float(np.percentile(np.abs(vals), 95))
                d[f"{key}_worst"] = float(np.max(np.abs(vals)))
        cands.append(d)
    objectives = ["F_h_rms_mean", "jerk_rms_mean", "osc_index_mean", "sat_ratio_mean"]
    cands_ok = [c for c in cands if all(o in c for o in objectives)]
    front = pareto_front(cands_ok, objectives) if cands_ok else []
    with open(outdir / "recommended_params.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump({"pareto_candidates": sorted(front, key=lambda x: x["F_h_rms_mean"])[:5],
                        "all_candidates_mean": sorted(cands, key=lambda x: x.get("F_h_rms_mean", 9e9))},
                       fh, allow_unicode=True, sort_keys=False)
    pd.DataFrame(cands).to_csv(outdir / "mc_candidate_means.csv", index=False)
    print(f"[monte-carlo] 完成：安全候选 {len(ok_rows)}，被剔除 {len(bad_rows)}，"
          f"Pareto {len(front)} 组")


def run_standstill(outdir: Path, cfg: Dict, scenarios_cfg: Dict, n_runs: int, seed: int) -> None:
    """独立「站立不自走」安全筛查（任务书 §10-3）：MC 样本 × 候选参数。"""
    def prog(i, n):
        print(f"[standstill] {i}/{n}")

    rows = mc_standstill_screen(cfg, n_runs, seed, scenarios_cfg=scenarios_cfg, progress=prog)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "mc_standstill_raw.csv", index=False)
    agg = df.groupby("candidate", as_index=False).agg(
        n=("sample", "count"),
        drift_mean=("standstill_drift_m", "mean"),
        drift_p95=("standstill_drift_m", lambda s: float(np.nanpercentile(s, 95))),
        drift_worst=("standstill_drift_m", "max"),
        vmax_worst=("standstill_vmax", "max"),
        osc_p95=("osc_index", lambda s: float(np.nanpercentile(s, 95))))
    agg["selfstart_rate"] = df.assign(flag=df.standstill_drift_m > 0.02).groupby(
        "candidate").flag.mean().values
    agg = agg.sort_values("drift_worst", ascending=False)
    agg.to_csv(outdir / "mc_standstill_summary.csv", index=False)
    print(f"[standstill] 完成：{len(df)} 次站立试验，最差漂移 "
          f"{df.standstill_drift_m.max():.4f} m")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="双滚刷手持洗地机单侧助力控制仿真")
    ap.add_argument("--scenario", default=None, help="S01..S12 | all | coupled")
    ap.add_argument("--controllers", default="C0,C1,C2,C3")
    ap.add_argument("--config", default=str(ROOT / "config" / "base.yaml"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sweep", default=None, choices=["ka_bd_alpha"])
    ap.add_argument("--monte-carlo", type=int, default=0)
    ap.add_argument("--standstill-screen", type=int, default=0,
                    help="独立站立不自走安全筛查的样本数（§10-3）")
    ap.add_argument("--milestone", default=None, choices=["M1", "M2", "M3"])
    ap.add_argument("--no-detail-figs", action="store_true")
    ap.add_argument("--no-timeseries", action="store_true")
    args = ap.parse_args()

    cfg = load_base(args.config)
    sc = load_scenarios()
    seed = args.seed if args.seed is not None else int(cfg["simulation"]["random_seed"])

    controllers = [c.strip() for c in args.controllers.split(",") if c.strip()]
    tag = args.milestone or (args.scenario or args.sweep or "mc")
    outdir = Path(args.out) if args.out else make_run_dir(str(tag))
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"=== assist_sim  |  run dir: {outdir}")
    print(f"=== 力上限（保守额定包络）: {force_limit_from_motor(cfg):.3f} N")
    dump_resolved(outdir, cfg, sc)

    did_something = False
    if args.milestone == "M1":
        df = run_scenarios(outdir, cfg, sc, ["S01", "S02"], ["C0", "C2"], seed,
                           detail_figs=not args.no_detail_figs,
                           write_ts=not args.no_timeseries)
        df.to_csv(outdir / "scenarios_metrics.csv", index=False)
        did_something = True
    elif args.milestone == "M2":
        ids = ALL_SCENARIOS
        df = run_scenarios(outdir, cfg, sc, ids, DEPLOYABLE, seed,
                           detail_figs=not args.no_detail_figs,
                           write_ts=not args.no_timeseries)
        # 人机耦合轨迹任务：S03/S05/S08/S09 × T1/T2 × 3 组保守虚拟人（§5.2、§7）
        df_c = run_scenarios(outdir, cfg, sc, ["S03", "S05", "S08", "S09"], DEPLOYABLE, seed,
                            detail_figs=False, write_ts=not args.no_timeseries,
                            tasks=["T1", "T2"], human_variants=["soft", "mid", "stiff"])
        df.to_csv(outdir / "scenarios_metrics.csv", index=False)
        df_c.to_csv(outdir / "coupled_metrics.csv", index=False)
        did_something = True
    elif args.milestone == "M3":
        run_sweep(outdir, cfg, sc, seed)
        if args.monte_carlo > 0:
            run_mc(outdir, cfg, sc, args.monte_carlo, seed)
        did_something = True
    else:
        if args.scenario:
            ids = ALL_SCENARIOS if args.scenario == "all" else (
                ["S03", "S05", "S08", "S09"] if args.scenario == "coupled"
                else [args.scenario])
            df = run_scenarios(outdir, cfg, sc, ids, controllers, seed,
                               detail_figs=not args.no_detail_figs,
                               write_ts=not args.no_timeseries)
            df.to_csv(outdir / "scenarios_metrics.csv", index=False)
            did_something = True
        if args.sweep:
            run_sweep(outdir, cfg, sc, seed)
            did_something = True
        if args.monte_carlo > 0:
            run_mc(outdir, cfg, sc, args.monte_carlo, seed)
            did_something = True
        if args.standstill_screen > 0:
            run_standstill(outdir, cfg, sc, args.standstill_screen, seed)
            did_something = True
    if not did_something:
        ap.print_help()
    print(f"=== 输出目录: {outdir}")


if __name__ == "__main__":
    main()
