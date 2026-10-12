#!/usr/bin/env python3
"""贡献分离（ablation）：速度反馈 / 加速度反馈 / 阻力补偿 各自的效果与代价（任务书 §1.3-5、§10-Q2）。

固定同一被控对象、同一随机种子、同一人手模型，只改控制器结构：

    no_assist           C0
    existing_analog     C1（i_base + k*a 的力等效占位实现）
    accel_only          C2，Ka=1.4, Bd=0      （只有加速度反馈）
    damp_only           C2，Ka=0,   Bd=4      （只有速度阻尼）
    accel_damp          C2，Ka=1.4, Bd=4      （两项都有）
    accel_damp_comp0.5  C3，α_D=0.5
    accel_damp_comp1.0  C3，α_D=1.0

用法：
    python contributions.py [--out outputs/contributions] [--seed 20261012]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sim.config import load_base, load_scenarios, resolve_scenario   # noqa: E402
from sim.metrics import compute_metrics                              # noqa: E402
from sim import plots as P                                           # noqa: E402
from sim.scenarios import run_case                                   # noqa: E402
from sim.sweep import target_impedance                               # noqa: E402

ABLATION = [
    ("no_assist", "C0", {}),
    ("existing_analog", "C1", {}),
    ("accel_only", "C2", {"Ka": 1.4, "Bd": 0.0}),
    ("damp_only", "C2", {"Ka": 0.0, "Bd": 4.0}),
    ("accel_damp", "C2", {"Ka": 1.4, "Bd": 4.0}),
    ("accel_damp_comp05", "C3", {"Ka": 1.4, "Bd": 4.0, "alpha_D": 0.5}),
    ("accel_damp_comp10", "C3", {"Ka": 1.4, "Bd": 4.0, "alpha_D": 1.0}),
]
CASES = [("S03", None), ("S04", None), ("S09", None), ("S08", None), ("S03", "T1")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "contributions"))
    ap.add_argument("--seed", type=int, default=20261012)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)

    base = load_base()
    sc = load_scenarios()
    rows: List[Dict] = []

    for sid, task_id in CASES:
        cfg = resolve_scenario(base, sc, sid)
        spec = dict(cfg["_scenario_spec"])
        task_spec = None
        if task_id:
            task_spec = sc["coupled_tasks"][task_id]
            spec["duration_s"] = task_spec["duration_s"]
            spec["human"] = {"type": "coupled"}
        for label, cname, params in ABLATION:
            res = run_case(cfg, spec, cname, ctrl_params=params, seed=args.seed,
                           task_spec=task_spec, human_variant="mid" if task_id else None)
            Md, Bd = target_impedance(cname, cfg, params)
            m = compute_metrics(res.rec, cfg, Md, Bd, is_coupled=bool(task_id),
                                x_target=(task_spec or {}).get("x_ref_sequence", [[0, 1, 1]])[-1][1])
            m.update({"case": f"{sid}{'_' + task_id if task_id else ''}", "variant": label,
                      "controller": cname,
                      **{f"param_{k}": v for k, v in params.items()}})
            rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "ablation_metrics.csv", index=False, float_format="%.6g")

    cols = ["case", "variant", "F_h_rms", "F_h_peak", "W_h_plus", "v_peak", "RMSE_v_rel",
            "extra_travel_after_release_m", "stop_time_s", "jerk_rms", "F_m_rms", "F_m_peak",
            "osc_index", "sat_ratio", "task_pos_err_final", "W_m_plus"]
    print(df[cols].round(3).to_string(index=False))

    # 贡献分离柱状图（每个 case 一组）
    for case in df.case.unique():
        sub = df[df.case == case]
        for metric, fname in (("F_h_rms", f"{case.replace('/', '_')}_F_h_rms.png"),
                              ("extra_travel_after_release_m", f"{case.replace('/', '_')}_stop.png"),
                              ("jerk_rms", f"{case.replace('/', '_')}_jerk.png")):
            if sub[metric].notna().any():
                P.plot_metric_bars(sub.to_dict("records"), outdir / "figures",
                                   [metric], xlabel="variant", fname=fname,
                                   title=f"{case} — {metric}")
    print(f"\n输出: {outdir}/ablation_metrics.csv, figures/*.png")


if __name__ == "__main__":
    main()
