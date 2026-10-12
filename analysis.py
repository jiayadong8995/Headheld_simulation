#!/usr/bin/env python3
"""把某次运行的输出汇总成报告用的 Markdown 数据表（可复现）。

用法：
    python analysis.py outputs/M2_full outputs/M3_param_study
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(df: pd.DataFrame, cols=None, digits=3) -> str:
    d = df.copy()
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    num = d.select_dtypes(include=[np.number]).columns
    d[num] = d[num].round(digits)
    return d.to_markdown(index=False)


def _read(path: Path, **kw):
    """安全读取 CSV：空文件/无列时返回 None。"""
    try:
        df = pd.read_csv(path, **kw)
        return df if len(df.columns) else None
    except Exception:
        return None


def summarise_run(run_dir: Path) -> str:
    out = [f"## 运行目录 `{run_dir}`\n"]
    metrics = run_dir / "scenarios_metrics.csv"
    if metrics.exists() and _read(metrics) is not None:
        d = _read(metrics)
        out.append("### 1. 场景 × 控制器 关键指标（力给定试验）\n")
        cols = ["scenario", "controller", "F_h_rms", "F_h_peak", "W_h_plus", "RMSE_v", "RMSE_v_rel",
                "M_eff_fit", "B_eff_fit", "fit_R2", "v_peak", "a_peak", "jerk_rms",
                "extra_travel_after_release_m", "stop_time_s", "standstill_drift_m",
                "standstill_vmax", "dir_reversals", "sat_ratio", "F_m_peak", "F_m_rms",
                "W_m_plus", "state_switches", "packets_lost"]
        out.append(_fmt(d, cols))
        out.append("")
        # 释放行为对比
        rel = d[d.scenario.isin(["S03", "S04", "S09", "S12"])]
        out.append("### 2. 松手停止行为（同一力波形下比较）\n")
        out.append(_fmt(rel, ["scenario", "controller", "x_final", "v_peak",
                              "extra_travel_after_release_m", "stop_time_s", "jerk_peak"]))
        out.append("")
    coupled = run_dir / "coupled_metrics.csv"
    if coupled.exists() and _read(coupled) is not None:
        d = _read(coupled)
        if "task" in d.columns:
            d = d[d.task.notna()]
        out.append("### 3. 人机耦合轨迹任务：相同任务下的人手负担\n")
        base = d[d.controller == "C1"].set_index(["scenario", "task", "human_variant"])["F_h_rms"]
        d["vs_C1_F_h_rms"] = [
            float(1 - r.F_h_rms / base.get((r.scenario, r.task, r.human_variant), np.nan))
            for r in d.itertuples()]
        out.append(_fmt(d, ["scenario", "task", "human_variant", "controller", "F_h_rms",
                            "F_h_peak", "W_h_plus", "vs_C1_F_h_rms", "completion_time_s",
                            "task_pos_err_final", "jerk_rms", "F_m_rms", "state_switches"]))
        out.append("")
    sweep = run_dir / "sweep_metrics.csv"
    if sweep.exists() and _read(sweep) is not None:
        d = _read(sweep)
        out.append("### 4. 单因素扫参（按场景分组，勿跨场景平均）\n")
        agg = d.groupby(["sweep", "scenario", "level"], as_index=False).agg(
            F_h_rms=("F_h_rms", "mean"), v_peak=("v_peak", "mean"),
            extra_travel=("extra_travel_after_release_m", "mean"),
            stop_time=("stop_time_s", "mean"), jerk_rms=("jerk_rms", "mean"),
            F_m_rms=("F_m_rms", "mean"), osc=("osc_index", "mean"),
            sat=("sat_ratio", "mean"))
        out.append(_fmt(agg))
        out.append("")
    for name, title in (("grid_F_h_rms.csv", "K_a/m × B_d 网格：耦合任务 RMS 手力 [N]"),
                        ("grid_stop_time_s.csv", "K_a/m × B_d 网格：松手停稳时间 [s]"),
                        ("grid_jerk_rms.csv", "K_a/m × B_d 网格：带限 jerk RMS")):
        p = run_dir / name
        if p.exists() and _read(p) is not None:
            out.append(f"### 5. {title}\n")
            g = _read(p, index_col=0)
            out.append(g.round(3).to_markdown())
            out.append("")
    rob = run_dir / "robustness_summary.csv"
    if rob.exists() and _read(rob) is not None:
        d = _read(rob)
        key = [c for c in d.columns if c.endswith("_mean") or c.endswith("_p95") or c.endswith("_worst")]
        keep = ["scenario", "controller", "n"] + [c for c in key if any(
            k in c for k in ("F_h_rms", "F_h_peak", "stop_time", "extra_travel",
                             "osc_index", "sat_ratio", "standstill_drift", "F_m_rms",
                             "jerk_rms", "RMSE_v_rel", "state_switches"))]
        out.append("### 6. Monte Carlo 鲁棒性汇总（配对样本，mean/P95/worst）\n")
        out.append(_fmt(d, keep))
        out.append("")
    ab = run_dir / "ablation_metrics.csv"
    if ab.exists() and _read(ab) is not None:
        d = _read(ab)
        out.append("### 5b. 贡献分离（ablation）\n")
        out.append(_fmt(d, ["case", "variant", "F_h_rms", "F_h_peak", "W_h_plus", "v_peak",
                            "RMSE_v_rel", "extra_travel_after_release_m", "stop_time_s",
                            "jerk_rms", "F_m_rms", "osc_index", "task_pos_err_final"]))
        out.append("")
    cand_path = run_dir / "mc_candidate_means.csv"
    if cand_path.exists() and _read(cand_path) is not None:
        cand = _read(cand_path)
        out.append("### 7. 候选参数（300 组随机工况下的均值/P95）\n")
        out.append(_fmt(cand.sort_values("F_h_rms_mean"),
                        ["candidate", "n_samples", "F_h_rms_mean", "F_h_rms_p95",
                         "improve_vs_C1_mean", "improve_vs_C0_mean", "F_h_peak_mean",
                         "W_h_plus_mean", "jerk_rms_mean", "jerk_rms_p95",
                         "osc_index_mean", "sat_ratio_mean", "F_m_rms_mean",
                         "task_pos_err_final_mean", "extra_travel_after_release_m_mean",
                         "extra_travel_after_release_m_worst"]))
        out.append("")
    st = run_dir / "mc_standstill_summary.csv"
    if st.exists() and _read(st) is not None:
        out.append("### 8b. 独立站立不自走筛查（4800 次）\n")
        out.append(_fmt(_read(st)))
        out.append("")
    for name, title in (("mc_candidates_rejected.csv", "被安全筛选剔除的样本（原因分布）"),
                        ("mc_candidates_safe.csv", "通过安全筛选的样本数")):
        p = run_dir / name
        if p.exists() and _read(p) is not None:
            d = _read(p)
            out.append(f"### 8. {title}\n")
            if "fail_reasons" in d.columns and len(d):
                out.append(d.fail_reasons.value_counts().to_markdown())
            else:
                out.append(f"行数：{len(d)}")
            out.append("")
    rec = run_dir / "recommended_params.yaml"
    if rec.exists():
        out.append("### 9. Pareto 推荐参数（`recommended_params.yaml` 摘要）\n")
        out.append("```yaml")
        out.append("\n".join(rec.read_text(encoding="utf-8").splitlines()[:60]))
        out.append("```")
        out.append("")
    return "\n".join(out)


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]] or [Path("outputs")]
    parts = ["# assist_sim 数据汇总（自动生成）\n"]
    for d in dirs:
        if d.exists():
            parts.append(summarise_run(d))
    text = "\n".join(parts)
    target = dirs[0] / "report_data.md"
    target.write_text(text, encoding="utf-8")
    print(f"written {target}  ({len(text)} chars)")


if __name__ == "__main__":
    main()
