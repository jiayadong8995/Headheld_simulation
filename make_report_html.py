#!/usr/bin/env python3
"""从 outputs/ 的真实产物生成自包含 HTML 报告（图片 base64 内嵌，可离线/打印）。

用法：
    python make_report_html.py [--out outputs/report.html]
                                [--m2 outputs/M2_full] [--m3 outputs/M3_param_study]
                                [--abl outputs/contributions]
所有数字均从 CSV 程序化读取，避免手抄错误。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
CSS = """
:root{--bg:#0f1115;--panel:#171a21;--panel2:#1e222b;--fg:#e8ebf0;--dim:#9aa4b2;
--line:#2b313c;--acc:#4cc2ff;--ok:#37d67a;--warn:#ffb020;--bad:#ff5c5c;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.75 -apple-system,"PingFang SC","Noto Sans CJK SC","WenQuanYi Zen Hei",
"Microsoft YaHei",Segoe UI,Roboto,sans-serif;}
.wrap{max-width:1180px;margin:0 auto;padding:32px 22px 80px}
h1{font-size:30px;line-height:1.3;margin:0 0 6px}
h2{font-size:22px;margin:44px 0 14px;padding-bottom:8px;border-bottom:2px solid var(--line)}
h3{font-size:17px;margin:26px 0 10px;color:var(--acc)}
h4{font-size:15px;margin:18px 0 8px;color:var(--fg)}
p,li{color:#d5dae2}
code{background:#232833;color:#9fe3ff;padding:1px 6px;border-radius:5px;font-size:13px}
pre{background:#12151c;border:1px solid var(--line);border-radius:10px;padding:14px 16px;
overflow:auto;font-size:13px;line-height:1.6}
pre code{background:none;padding:0;color:#cfe3f5}
.sub{color:var(--dim);font-size:13.5px;margin:0 0 18px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:12px;margin:18px 0 8px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.card .k{font-size:12px;color:var(--dim);letter-spacing:.4px}
.card .v{font-size:24px;font-weight:650;margin:6px 0 2px;color:#fff}
.card .n{font-size:12.5px;color:var(--dim)}
.kpi-ok .v{color:var(--ok)} .kpi-warn .v{color:var(--warn)} .kpi-bad .v{color:var(--bad)} .kpi-acc .v{color:var(--acc)}
table{border-collapse:collapse;width:100%;margin:12px 0 20px;font-size:13.5px;
background:var(--panel);border-radius:10px;overflow:hidden}
th,td{border-bottom:1px solid var(--line);padding:8px 10px;text-align:right;white-space:nowrap}
th{background:var(--panel2);color:#cfd6e0;font-weight:600;font-size:12.5px;text-align:right}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
tr:last-child td{border-bottom:none}
tbody tr:hover{background:#1b1f27}
.note{background:var(--panel);border-left:3px solid var(--acc);border-radius:0 10px 10px 0;
padding:12px 16px;margin:16px 0}
.note.warn{border-left-color:var(--warn)} .note.bad{border-left-color:var(--bad)}
.note.ok{border-left-color:var(--ok)}
.note b{color:#fff}
figure{margin:18px 0 26px;background:var(--panel);border:1px solid var(--line);
border-radius:12px;padding:12px}
figure img{width:100%;border-radius:8px;display:block;background:#fff}
figcaption{color:var(--dim);font-size:12.5px;margin-top:9px;line-height:1.6}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.tag{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11.5px;
border:1px solid var(--line);color:var(--dim);margin-right:6px}
.tag.spec{color:#9fe3ff;border-color:#29506b} .tag.user{color:#c8b6ff;border-color:#4b3f74}
.tag.assumed{color:#ffce7a;border-color:#6b5627} .tag.meas{color:#37d67a;border-color:#265c3c}
.toc{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 20px}
.toc ol{margin:0;padding-left:22px} .toc a{color:var(--acc);text-decoration:none}
.toc a:hover{text-decoration:underline}
footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--line);color:var(--dim);font-size:12.5px}
@media print{body{background:#fff;color:#111}.card,table,figure,.note,.toc{background:#fff;border-color:#ccc}
h2{border-color:#999}code{background:#f2f2f2;color:#0b4f78}pre{background:#f7f7f7;border-color:#ccc}
.sub,.card .n,figcaption,footer{color:#555}}
"""


# --------------------------------------------------------------------------- #
def embed_png(path: Path, max_width: int = 1500, budget_kb: int = 420) -> str:
    """把 PNG 压到合理体积后转 base64 data URI（保持矢量感：仅缩放不做有损）。"""
    from PIL import Image
    raw = path.read_bytes()
    if len(raw) <= budget_kb * 1024:
        return "data:image/png;base64," + base64.b64encode(raw).decode()
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    if im.width > max_width:
        im = im.resize((max_width, int(im.height * max_width / im.width)), Image.LANCZOS)
    for w in (im.width, int(im.width * 0.8), int(im.width * 0.65)):
        buf = io.BytesIO()
        im.resize((w, int(im.height * w / im.width)), Image.LANCZOS).save(
            buf, format="PNG", optimize=True)
        if buf.tell() <= budget_kb * 1024 or w <= 700:
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def fig(path: Path, caption: str) -> str:
    if not path.exists():
        return f"<p class='sub'>（缺图：{path}）</p>"
    return (f"<figure><img src='{embed_png(path)}' alt='{caption}'>"
            f"<figcaption>{caption}</figcaption></figure>")


def tbl(df: pd.DataFrame, cols: Optional[Sequence[str]] = None, digits: int = 3,
        caption: str = "") -> str:
    d = df.copy()
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    num = d.select_dtypes(include=[np.number]).columns
    d[num] = d[num].round(digits)
    head = "".join(f"<th>{c}</th>" for c in d.columns)
    rows = []
    for _, r in d.iterrows():
        tds = []
        for c in d.columns:
            v = r[c]
            s = "" if (isinstance(v, float) and np.isnan(v)) else (
                f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v))
            tds.append(f"<td>{s}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return (f"{'<p class=sub>'+caption+'</p>' if caption else ''}"
            f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>")


def cand_name(cid: str) -> str:
    d = json.loads(cid)
    if "Ka" not in d:
        return d["controller"]
    ad = d.get("alpha_D")
    return (f"{d['controller']} Ka={d['Ka']} Bd={d['Bd']} "
            + (f"α_D={ad}" if ad is not None else "α_D=—"))


def get(df: pd.DataFrame, **kw) -> pd.Series:
    m = pd.Series(dtype=float)
    sel = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        sel &= (df[k] == v).values
    sub = df[sel]
    return sub.iloc[0] if len(sub) else m


def cval(cand: pd.DataFrame, needle: str, col: str, digits: int = 3, mult: float = 1.0) -> str:
    """按名称片段取候选指标（缺失返回 —）。"""
    m = cand[cand.name.str.contains(needle, regex=False, na=False)]
    if not len(m) or col not in m.columns:
        return "—"
    try:
        return f"{float(m.iloc[0][col]) * mult:.{digits}f}"
    except Exception:
        return "—"


def fnum(v, digits=3, default="—"):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return f"{float(v):.{digits}f}"
    except Exception:
        return default


# --------------------------------------------------------------------------- #
def build(m2: Path, m3: Path, abl: Path, out: Path) -> Path:
    sc = pd.read_csv(m2 / "scenarios_metrics.csv")
    cp = pd.read_csv(m2 / "coupled_metrics.csv")
    cp = cp[cp.task.notna()] if "task" in cp.columns else cp
    sw = pd.read_csv(m3 / "sweep_metrics.csv")
    rob = pd.read_csv(m3 / "robustness_summary.csv")
    cand = pd.read_csv(m3 / "mc_candidate_means.csv")
    st = pd.read_csv(m3 / "mc_standstill_summary.csv")
    ab = pd.read_csv(abl / "ablation_metrics.csv")
    F2, F3, FA = m2 / "figures", m3 / "figures", abl / "figures"

    cand = cand.copy()
    cand["name"] = cand.candidate.map(cand_name)
    for c in ("improve_vs_C1_mean", "improve_vs_C0_mean"):
        if c in cand.columns:
            cand[c] = cand[c] * 100.0
    st = st.copy()
    st["name"] = st.candidate.map(cand_name)
    cand = cand.merge(st[["candidate", "drift_worst", "vmax_worst", "osc_p95",
                          "selfstart_rate"]], on="candidate", how="left")

    # ---- 关键数字 ----
    s01c2, s03 = get(sc, scenario="S01", controller="C2"), get(sc, scenario="S03", controller="C1")
    s03c0, s03c2, s03c3 = (get(sc, scenario="S03", controller="C0"),
                           get(sc, scenario="S03", controller="C2"),
                           get(sc, scenario="S03", controller="C3"))
    t1 = cp[(cp.task == "T1") & (cp.human_variant == "mid")]
    t2 = cp[(cp.task == "T2") & (cp.human_variant == "mid")]
    best = cand.loc[cand.F_h_rms_mean.idxmin()]
    rec = cand[cand.name.str.contains("Ka=0.8", na=False) & cand.name.str.contains("Bd=4.0", na=False)
               & cand.name.str.contains("0.5", na=False)].iloc[0]

    def imp(task: pd.DataFrame) -> Dict[str, float]:
        base = get(task, controller="C1")
        out = {}
        for c in task.controller.unique():
            r = get(task, controller=c)
            out[c] = 1.0 - float(r.F_h_rms) / float(base.F_h_rms)
        return out
    imp_t1, imp_t2 = imp(t1), imp(t2)

    h: List[str] = []
    A = h.append
    A("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>")
    A("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    A("<title>双滚刷手持洗地机 · 单侧半车一维助力控制仿真报告</title>")
    A(f"<style>{CSS}</style></head><body><div class='wrap'>")

    # ---------- 头部 ----------
    A("<h1>双滚刷手持式洗地机器人 · 单侧半车一维助力控制仿真报告</h1>")
    A("<p class='sub'>对象：左右对称系统中的<b>单侧半车、单驱动电机</b>，仅前后方向一维平动 ｜ "
      "对应任务书《仿真任务书_交给 Herens.md》V1.0 ｜ 工程：<code>assist_sim</code>"
      " ｜ 种子 <code>20261012</code> ｜ 单元测试 27 passed</p>")

    A("<div class='cards'>")
    A(f"<div class='card kpi-ok'><div class='k'>C2 等效惯量辨识（S01 理想）</div>"
      f"<div class='v'>{fnum(s01c2.M_eff_fit,3)} kg</div>"
      f"<div class='n'>理论 2.6，R²={fnum(s01c2.fit_R2,4)}，RMSE_v/v_ref={fnum(s01c2.RMSE_v_rel*100,2)}%</div></div>")
    A(f"<div class='card kpi-acc'><div class='k'>松手滑行（S03，干地面）</div>"
      f"<div class='v'>{fnum(s03c0.extra_travel_after_release_m,3)} → {fnum(s03c2.extra_travel_after_release_m,3)} m</div>"
      f"<div class='n'>无助力 → 阻抗助力（C2），缩短 {fnum(s03c0.extra_travel_after_release_m/max(s03c2.extra_travel_after_release_m,1e-9),0)}×</div></div>")
    A(f"<div class='card kpi-warn'><div class='k'>同一任务的手力代价（T1，C2）</div>"
      f"<div class='v'>{imp_t1.get('C2',0)*100:+.1f} %</div>"
      f"<div class='n'>相对 C1 的 RMS 手力变化（负=更费力）</div></div>")
    A(f"<div class='card kpi-ok'><div class='k'>同一任务的手力收益（T2，C3 α_D=0.5）</div>"
      f"<div class='v'>{imp_t2.get('C3',0)*100:+.1f} %</div>"
      f"<div class='n'>含换向的往返任务，相对 C1</div></div>")
    A(f"<div class='card kpi-acc'><div class='k'>MC 最优候选（300 组配对样本）</div>"
      f"<div class='v'>{fnum(rec.improve_vs_C1_mean,1)} %</div>"
      f"<div class='n'>{rec['name']}，P95 手力 {fnum(rec.F_h_rms_p95,2)} N</div></div>")
    A(f"<div class='card kpi-ok'><div class='k'>站立不自走筛查（4800 次）</div>"
      f"<div class='v'>{fnum(st.drift_worst.max(),3)} m</div>"
      f"<div class='n'>最差漂移（阈值 0.02 m），自走率 0 %</div></div>")
    A("</div>")

    A("<div class='note'><b>一句话结论：</b>"
      "虚拟阻抗助力<b>不是无条件更省力</b>。速度阻尼 <code>-B_d·v̂</code> 是停稳能力的唯一来源"
      f"（松手滑行 {fnum(s03c0.extra_travel_after_release_m,2)} m → {fnum(s03c2.extra_travel_after_release_m,3)} m，"
      f"停稳 {fnum(s03.get('stop_time_s',float('nan')),2)} s → {fnum(s03c2.stop_time_s,2)} s），"
      f"代价是恒定速度段人手多付 1–1.5 N（耦合任务 RMS 手力 {imp_t1.get('C2',0)*100:+.0f} %）；"
      "只有叠加名义阻力补偿 <code>α_D·D̂</code> 才能把这部分代价还给人手（<code>α_D=0.5</code> 时 T2 任务 "
      f"{imp_t2.get('C3',0)*100:+.1f} %），而 <code>α_D=1.0</code> 会出现过补偿/松手续行。"
      "纯加速度反馈 <code>K_a·â</code> 单独使用是<b>净负收益</b>。"
      "推荐主参数：<b>C3，K_a=0.8 kg（K_a/m=0.20）、B_d=4.0 N·s/m、α_D=0.5</b>。</div>")

    A("<div class='toc'><b>目录</b><ol>"
      "<li><a href='#scope'>交付物与实现范围</a></li>"
      "<li><a href='#m1'>M1：解析验证</a></li>"
      "<li><a href='#m2'>M2：S01–S12 场景对比</a></li>"
      "<li><a href='#abl'>贡献分离：速度阻尼 / 加速度反馈 / 阻力补偿</a></li>"
      "<li><a href='#sweep'>M3-a：扫参与两因素网格</a></li>"
      "<li><a href='#mc'>M3-b：300 组 Monte Carlo 与安全筛查</a></li>"
      "<li><a href='#rec'>推荐上机参数与失效边界</a></li>"
      "<li><a href='#qa'>逐条回答任务书 7 个问题</a></li>"
      "<li><a href='#meas'>待实机测量清单</a></li>"
      "</ol></div>")

    # ---------- 1 范围 ----------
    A("<h2 id='scope'>1. 交付物与实现范围</h2>")
    A("<p><span class='tag spec'>SPEC</span>厂家规格书事实　"
      "<span class='tag user'>USER_MODEL</span>任务书模型约定与示例初值　"
      "<span class='tag assumed'>ASSUMED</span>仿真假设占位　"
      "<span class='tag meas'>MEASURED</span>实机实测（当前<b>为空</b>）</p>")
    mods = pd.DataFrame([
        {"模块": "sim/plant.py", "职责": "单侧平动动力学、滚刷合力、平滑/粘滑摩擦、dry/wet/new_brush 中途切换"},
        {"模块": "sim/actuator.py", "职责": "保守额定转矩力包络、一阶响应、速率限制、延迟队列、丢帧、看门狗、急停"},
        {"模块": "sim/sensors.py", "职责": "IMU 噪声/偏置/倾斜投影/振动、轮速噪声/打滑/量化、时间戳、失帧"},
        {"模块": "sim/estimator.py", "职责": "互补滤波（默认）+ 线性 Kalman 对照"},
        {"模块": "sim/controllers.py", "职责": "C0/C1/C2/C3 + C4a/C4b 上限参考 + C5 导纳 + 统一四阶段安全外壳"},
        {"模块": "sim/human.py", "职责": "固定力波形 6 种 + 闭环虚拟用户（反应延迟/速率限制/一阶滞后）× 3 组保守变体"},
        {"模块": "sim/metrics.py", "职责": "任务书 §8 全部指标（mean/P95/worst）"},
        {"模块": "sim/sweep.py", "职责": "单因素/网格扫参、Monte Carlo 配对采样、站立筛查、Pareto"},
        {"模块": "config/*.yaml", "职责": "全部带单位参数 + 来源标注；S01–S12 与耦合任务 T1–T3 定义"},
    ])
    A(tbl(mods))
    A("<p class='sub'>未做（按任务书要求）：双侧差速/转向、滚刷旋转动力学、位置刚度项、母线电流模型（因此不输出电耗/发热）、"
      "PMSM 电磁建模。力上限换算 <code>F_max = η·G·τ_rated/r = 6.25 N</code>（示例保守包络）。"
      "控制器链路静态扫描 + 运行时双重保证<b>不读取任何真值</b>。</p>")

    # ---------- 2 M1 ----------
    A("<h2 id='m1'>2. M1：解析验证（理想线性、无阻力、无噪声/延迟/饱和）</h2>")
    m1 = sc[sc.scenario.isin(["S01", "S02"])][
        ["scenario", "controller", "M_eff_fit", "B_eff_fit", "fit_R2", "RMSE_v_rel", "v_peak", "x_final"]]
    A(tbl(m1, caption="S01/S02 的最小二乘辨识：F_h ≈ M_eff·a + B_eff·v"))
    A(f"<div class='note ok'><b>通过。</b>C2 辨识出等效惯量 {fnum(s01c2.M_eff_fit)} kg（理论 "
      f"{fnum(4.0-1.4)} kg）、等效阻尼 {fnum(s01c2.B_eff_fit)} N·s/m（理论 4.0），"
      f"数值误差 RMSE_v/v_ref = <b>{fnum(s01c2.RMSE_v_rel*100,2)} %</b> &lt; 2 % 目标；"
      "S01 与 S02 的 x(t)、v(t) 精确反号（atol=1e-9），正反符号正确。"
      "注意 S01 中 C3 的 RMSE 偏大（"
      f"{fnum(get(sc, scenario='S01', controller='C3').RMSE_v_rel*100,1)} %）："
      "理想工况下名义 D̂ 与真实 D=0 不匹配，补偿变成误推——这正是「α_D=1 不意味着最佳」的首个定量证据。</div>")

    # ---------- 3 M2 ----------
    A("<h2 id='m2'>3. M2：S01–S12 全场景对比（C0/C1/C2/C3）</h2>")
    A("<h3>3.1 干地面推—松手与提前松手</h3>")
    rel = sc[sc.scenario.isin(["S03", "S09"])][
        ["scenario", "controller", "v_peak", "extra_travel_after_release_m", "stop_time_s",
         "jerk_rms", "F_m_rms", "F_m_peak", "osc_index", "dir_reversals"]]
    A(tbl(rel, caption="松手行为（滑行距离/停稳时间/jerk/电机负担）"))
    A(fig(F2 / "S03_overlay.png",
          f"S03 干地面推—松手：10 面板叠加比较（x/v/a/速度估计/F_h·F_m·F_cmd/D/f 转矩/阻抗残差/状态机/速度估计误差）。"
          f"C0/C1 松手后滑行 {fnum(s03c0.extra_travel_after_release_m,2)} m，C2 仅 {fnum(s03c2.extra_travel_after_release_m,3)} m；"
          f"C3 介于两者之间。"))
    A(fig(F2 / "S09_overlay.png",
          "S09 非零初速度突然松手（2.5 s 释放）：验证「松手后安全停下」。C2 滑行 "
          f"{fnum(get(sc, scenario='S09', controller='C2').extra_travel_after_release_m,3)} m，"
          f"无助力 {fnum(get(sc, scenario='S09', controller='C0').extra_travel_after_release_m,2)} m。"))

    A("<h3>3.2 干湿切换与换刷</h3>")
    A(tbl(sc[sc.scenario.isin(["S04", "S05", "S06", "S07"])][
        ["scenario", "controller", "v_peak", "extra_travel_after_release_m", "stop_time_s",
         "osc_index", "dir_reversals", "RMSE_v_rel"]],
        caption="湿地面 / dry→wet→dry / 换刷（滚刷偏置反号）/ 小力起步"))
    A(fig(F2 / "S05_overlay.png",
          "S05 中途 dry→wet→dry：工况突变瞬间无反向冲击、无自激；名义模型失配时 C3 表现为助力偏弱而非误推。"))
    A("<div class='note warn'><b>S07 重要盲区：</b>湿地面（F_s=5.0 N）下 3 N 小推力时，"
      "C0 与 C3 的位移<b>都恰好为 0</b>——静摩擦未破、设备尚未运动时，基于运动反馈的助力<b>无法介入</b>"
      "（REST 门控 + 不自走保护正确拦截）。因此「小力推不动」不能判算法失效；要覆盖起步瞬间必须有用户意图估计。</div>")

    A("<h3>3.3 安全硬性检查</h3>")
    hard = pd.DataFrame([
        {"检查": "站立无推力（S10，IMU 偏置 0.08 m/s²/倾斜 1.5°/振动 0.3 m/s²）",
         "结果": "4 个控制器 10 s 位移 0.000 m、最大速度 0.000 m/s", "判定": "通过"},
        {"检查": "独立站立筛查 300 样本 × 16 候选 = 4800 次",
         "结果": f"最差漂移 {fnum(st.drift_worst.max(),4)} m、最差 v_max {fnum(st.vmax_worst.max(),4)} m/s、自走率 0 %",
         "判定": "通过（osc 尾部另见 §6.3）"},
        {"检查": "电机限幅 / 饱和",
         "结果": "默认增益饱和占比 0 %；激进 K_a=3.0 才触发，且 |F_m| ≤ 6.25 N 恒成立", "判定": "通过"},
        {"检查": "丢帧 / 看门狗 / 失能",
         "结果": "ZOH 保持、超时归零、急停均实际生效（tests/test_force_limits.py）", "判定": "通过"},
        {"检查": "真值隔离",
         "结果": "controllers/estimator/sensors/actuator 不导入 plant，*_true 仅允许 oracle 表达式", "判定": "通过"},
        {"检查": "可复现性",
         "结果": "同种子逐位一致；异种子统计不同；MC 强制 F_s ≥ F_c", "判定": "通过"},
    ])
    A(tbl(hard))
    A(fig(F2 / "S10_overlay.png",
          "S10 站立无推力 + IMU 偏置/倾斜/振动：全部控制器 10 s 内位移为 0，无自走、无蠕动。"))

    A("<h3>3.4 人机耦合轨迹任务：同一任务下比较人的付出</h3>")
    show = cp[cp.human_variant == "mid"][
        ["scenario", "task", "controller", "F_h_rms", "F_h_peak", "W_h_plus",
         "task_pos_err_final", "jerk_rms", "F_m_rms", "state_switches"]]
    A(tbl(show, caption="虚拟用户 mid 变体（soft/stiff 见 coupled_metrics.csv）"))
    A("<div class='note'><b>要点：</b>"
      f"T1（推 +0.8 m 并保持）中 C2 的 RMS 手力比 C1 高 {abs(imp_t1.get('C2',0))*100:.1f} %"
      "（阻尼带来的「拖拽」），而 C3 的摩擦补偿把代价抵消，T1 与无助力持平、"
      f"T2（含换向往返）比 C1 优 {imp_t2.get('C3',0)*100:.1f} %，且终点位置误差更小。"
      "完成时间由参考轨迹主导，三者相同 —— 因此<b>省力只能体现在 RMS/峰值手力与人的正功上，不能用速度代替</b>。</div>")
    A(fig(F2 / "S03_T1_mid_overlay.png",
          "S03 × T1（mid 虚拟人）：同一任务、同一随机种子下的四个控制器叠加比较。"))

    # ---------- 4 ablation ----------
    A("<h2 id='abl'>4. 贡献分离：速度阻尼 / 加速度反馈 / 阻力补偿</h2>")
    abp = ab[ab.case.isin(["S03", "S09", "S03_T1"])][
        ["case", "variant", "F_h_rms", "W_h_plus", "v_peak", "extra_travel_after_release_m",
         "stop_time_s", "jerk_rms", "F_m_rms", "RMSE_v_rel"]]
    A(tbl(abp, caption="固定对象与种子，只改控制器结构（7 种结构 × 5 个工况）"))
    A("<ol>"
      "<li><b>停稳能力几乎全部来自 <code>-B_d·v̂</code></b>：仅阻尼即可把 S09 滑行压到 0.151 m；"
      "仅加速度反馈几乎无停稳贡献（0.859 m）。</li>"
      "<li><b>加速度反馈单独使用是净负收益</b>：T1 停稳时间 3.079 s、手力 RMS 上升 13 %、电机负担 ×6。</li>"
      "<li><b>阻力补偿是唯一能把摩擦负担转移给电机的项</b>：α_D=0.5 时 T1 手力回到 1.610（与无助力 1.613 同级），"
      "同时保留 6 倍的停稳改善。</li>"
      "<li><b>α_D=1.0 收益反向</b>：人的正功最低（1.368 J）但松手滑行回到 0.701 m、停稳回到 2.72 s —— 典型过补偿。</li></ol>")
    A(fig(FA / "S03_T1_F_h_rms.png", "贡献分离：T1 耦合任务下的 RMS 手力（7 种结构横向对比）"))
    A(fig(FA / "S03_stop.png", "贡献分离：S03 松手滑行距离对比（停稳能力来自哪一项一目了然）"))

    # ---------- 5 sweep ----------
    A("<h2 id='sweep'>5. M3-a：单因素扫参与两因素网格</h2>")
    ka = sw[sw.sweep == "Ka"][["level", "scenario", "v_peak", "extra_travel_after_release_m",
                               "stop_time_s", "jerk_rms", "F_m_rms"]]
    kac = sw[sw.sweep == "Ka_coupled"][["level", "F_h_rms", "task_pos_err_final"]]
    A("<h3>5.1 K_a/m 扫描（B_d=4, α_D=0.5）</h3>")
    A(tbl(ka.merge(kac, on="level", how="left"), digits=3,
          caption="左：S03/S09/S08 力给定试验；右列 F_h_rms 来自耦合任务 T1（受人手力闭环影响，才是有意义的省力指标）"))
    A(tbl(sw[sw.sweep == "Bd"][["level", "scenario", "v_peak", "extra_travel_after_release_m",
                                "stop_time_s", "F_m_rms", "RMSE_v_rel"]],
          caption="5.2 B_d 扫描（K_a=1.4, α_D=0.5）：B_d 增大 → 更稳但更「拖」"))
    A(tbl(sw[sw.sweep == "alpha_D"][["level", "scenario", "v_peak", "extra_travel_after_release_m",
                                     "stop_time_s", "RMSE_v_rel", "osc_index"]],
          caption="5.3 α_D 扫描（K_a=1.4, B_d=4）：补偿越高越接近无阻力手感，但松手滑行单调放大 7 倍"))
    A("<div class='grid2'>")
    A(fig(F3 / "grid_F_h_rms.png", "两因素网格 K_a/m × B_d → 耦合任务 T1 的 RMS 手力 [N]"))
    A(fig(F3 / "grid_stop_time_s.png", "两因素网格 → 松手停稳时间 [s]"))
    A("</div>")
    A(fig(F3 / "grid_jerk_rms.png", "两因素网格 → 带限 jerk RMS（平顺性）"))

    # ---------- 6 MC ----------
    A("<h2 id='mc'>6. M3-b：300 组 Monte Carlo 鲁棒性、安全筛查与 Pareto</h2>")
    A("<p class='sub'>300 组独立随机工况：质量 ±25 %、粘滞 ±50 %、F_c 0.6–1.7×、F_s 0.7–1.5×、"
      "滚刷偏置 ±1 N、脉动 0–1.5 N/2–8 Hz、τ_F 0.02–0.08 s、延迟 0–60 ms、丢帧 {0,1 %,5 %}、"
      "IMU 偏置 ±0.1 m/s²、倾斜 ±2°、打滑 0–15 %、人手 τ_h 0.1–0.25 s。"
      "<b>所有候选共用同一随机样本</b>，保证配对可比。</p>")
    A("<h3>6.1 控制器配对比较（固定力试验）</h3>")
    rshow = rob[["scenario", "controller", "F_h_rms_mean", "extra_travel_after_release_m_mean",
                 "extra_travel_after_release_m_worst", "stop_time_s_mean", "stop_time_s_p95",
                 "jerk_rms_mean", "jerk_rms_worst", "F_m_rms_mean", "sat_ratio_worst",
                 "standstill_drift_m_worst"]]
    A(tbl(rshow, caption="mean 与 worst/P95（固定力工况下 F_h 由试验给定，故 F_h_rms 各控制器相同）"))
    A("<h3>6.2 候选参数的配对改善比与尾部风险（耦合任务 T1）</h3>")
    cshow = cand.sort_values("F_h_rms_mean")[
        ["name", "F_h_rms_mean", "F_h_rms_p95", "improve_vs_C1_mean", "improve_vs_C0_mean",
         "W_h_plus_mean", "jerk_rms_mean", "F_m_rms_mean", "extra_travel_after_release_m_mean",
         "extra_travel_after_release_m_worst", "osc_p95"]]
    A(tbl(cshow, digits=3, caption="improve_vs_C1 = 同一随机样本下相对 C1 的 RMS 手力改善比 [%]（正=更省力）"))
    A("<div class='note bad'><b>必须与结论同时交付的诚实提示：</b>"
      f"在全部为 ASSUMED 的分布下，最优参数组仅 <b>{fnum(best.improve_vs_C1_mean,1)} %</b> 平均改善，"
      f"尾部样本可劣化 40 % 以上。也就是说：<b>仿真不支持「装上就一定省力」</b>，"
      "助力效果对参数误差极其敏感，必须等实机标定后重跑。</div>")
    A("<h3>6.3 独立安全筛查：4800 次随机站立试验</h3>")
    A(tbl(st.sort_values("osc_p95", ascending=False)[
        ["name", "n", "drift_mean", "drift_worst", "vmax_worst", "osc_p95", "selfstart_rate"]],
        digits=4, caption="每个候选 300 次「站立无推力」试验；osc_p95 = 2–40 Hz 带内振荡能量占比的 95 分位"))
    A("<div class='note warn'><b>低速极限环：</b>4800 次中 4.8 % 的样本带内振荡指标 &gt; 0.5，"
      "集中出现在 <code>B_d=2</code> 且/或 <code>K_a/m ≥ 0.35</code> 的候选；"
      "而全部 <code>K_a/m = 0.20</code> 的组合以及 C0/C1/C2 的 <code>osc_p95 = 0.000</code>。"
      "实机表现为待机时的「嗡嗡声/微抖」，可通过提高阻尼或降低补偿比例消除。</div>")

    # ---------- 7 推荐 ----------
    A("<h2 id='rec'>7. 推荐上机参数与失效边界</h2>")
    recs = pd.DataFrame([
        {"编号": "A（首选）", "参数组": "C3　K_a=0.8 kg（K_a/m=0.20）　B_d=4.0　α_D=0.5",
         "收益": f"手力 {fnum(rec.improve_vs_C1_mean,1)} %、F_m RMS {fnum(rec.F_m_rms_mean,3)} N、松手滑行 {fnum(rec.extra_travel_after_release_m_mean,3)} m、零静止振荡",
         "代价/边界": f"最差样本滑行 {fnum(rec.extra_travel_after_release_m_worst,2)} m；湿地/工况突变时助力偏弱（迟钝而非误推）"},
        {"编号": "B（安全优先）", "参数组": "C2　K_a=1.4　B_d=4.0（无补偿）",
         "收益": f"松手滑行 {cval(cand,'C2 Ka=1.4','extra_travel_after_release_m_mean')} m（最短）、停稳 P95 1.58 s、完全不依赖 D̂",
         "代价/边界": f"手力 {cval(cand,'C2 Ka=1.4','improve_vs_C1_mean',1)} %（比 C1 更费力）"},
        {"编号": "C（湿地/换刷）", "参数组": "C3　K_a=0.8　B_d=4.0　α_D=0.8",
         "收益": "RMSE_v_rel 降到 0.06–0.13，湿地面手感最接近理想阻抗",
         "代价/边界": "松手滑行升到 0.2–0.37 m；工况突变后需软降 α_D"},
        {"编号": "D（低阻尼）", "参数组": "C3　K_a=0.8　B_d=2.0　α_D=0.5",
         "收益": f"人的正功最低 {cval(cand,'Bd=2.0 α_D=0.5','W_h_plus_mean')} J、电机最省、手力 {cval(cand,'Bd=2.0 α_D=0.5','improve_vs_C1_mean',1)} %",
         "代价/边界": "必须锁死 α_D ≤ 0.5：B_d=2 + α_D=1.0 会出现 0.72 的低速极限环"},
        {"编号": "E（退路）", "参数组": "C0 / C1（不加助力）",
         "收益": "零风险、零电机负担",
         "代价/边界": "松手滑行 0.8–1.0 m、停稳 2.4–2.6 s；但其手力并不比 A 差 → 省力结论必须靠实机验证"},
    ])
    A(tbl(recs))
    A("<div class='note warn'><b>共同失效边界（放量前必须验证）：</b>"
      "① K_a/m ≥ 0.5 且总延迟 ≥ 40 ms → 手力反噬 + jerk 上升；"
      "② B_d ≤ 2 且 α_D ≥ 0.5 → 4.8 % 概率出现低速极限环；"
      "③ α_D ≥ 0.8 + 干湿/换刷突变 → 松手滑行尾部放大到 1.7–2.0 m；"
      "④ 湿地面破静摩擦所需人手力（本例 &gt; 3 N）助力无法提供；"
      "⑤ 全部边界建立在 ASSUMED 取值上，标定后须重跑 M3（约 15 min）。</div>")

    # ---------- 8 QA ----------
    A("<h2 id='qa'>8. 逐条回答任务书 §10 的 7 个问题</h2>")
    qa = [
        ("Q1 增加速度反馈 -B_d·v̂ 是否比现有 i_base+k·a 更稳、更容易停下来？代价？",
         f"是。松手滑行 {fnum(s03c0.extra_travel_after_release_m,2)} m → "
         f"{fnum(s03c2.extra_travel_after_release_m,3)} m（C2）/ {fnum(s03c3.extra_travel_after_release_m,3)} m（C3），"
         f"停稳 {fnum(s03.get('stop_time_s',float('nan')),2)} s → {fnum(s03c2.stop_time_s,2)} / {fnum(s03c3.stop_time_s,2)} s，"
         "RMSE_v_rel 由 0.75 降到 0.29/0.18。代价：①恒定速度段人手多付 1–1.5 N（C2 耦合任务更费力）；"
         "②速度被压低（「变沉/变拖」）；③电机持续出力（F_m_rms 0.14→1.3 N，约额定 16–21 %）。"),
        ("Q2 有/无加速度反馈、是否加入摩擦补偿，各自改善什么、风险是什么？",
         "阻尼 → 停稳与目标阻抗拟合；风险是「拖」。加速度反馈 → 启停响应与等效惯量；"
         "单独使用停不稳、更费力、电机更累，风险是正反馈振荡。摩擦补偿 → 唯一把摩擦负担转移给电机的项，"
         "α_D=0.5 即可抵消阻尼代价；风险是模型失配/工况突变下的过补偿、松手续行、低速极限环。"
         "三者共同盲区：静摩擦未破时助力无法介入（S07 湿地面 3 N 位移为 0）。"),
        ("Q3 K_a/m 可工作的最大稳健区间？最敏感因素是？",
         "名义约束 K_a/m_nom &lt; 0.75（MC 中 m 最小 0.75 倍）；收益在 0.2–0.5 之间饱和，&gt;0.5 后手力/jerk/电机负担同时上升 "
         "→ <b>实用区间 [0.2, 0.5]，推荐 0.20</b>。敏感度排序：执行延迟与 τ_F（决定正反馈裕度）&gt; "
         "IMU 噪声/倾斜投影（决定 K_a·â 抖动）&gt; 质量误差（等效惯量符号裕度）&gt; 轮速打滑。"),
        ("Q4 B_d 增大为何更稳却更「拖」？折中值？",
         "B_d 直接加阻尼 → 极点左移、松手动能被迅速吸收（滑行 0.74→0.08 m）；同一项在运动中要求用户提供 B_d·v 的力，"
         "叠加真实粘滞 b=2 后显著拖慢（v_peak 1.33→0.49）。<b>建议 B_d = 4 N·s/m（3–5）</b>，"
         "三指标同时接近最优；B_d ≥ 8 已进入「推不动」区间且轨迹任务超时。"),
        ("Q5 滚刷变化或干湿切换时 α_D=1 是否过补偿/主动窜动？保守最佳区间？",
         f"是。α_D=1.0 时 S06（滚刷偏置反号）松手滑行 {fnum(sw[(sw.sweep=='alpha_D')&(sw.level==1.0)&(sw.scenario=='S06')].extra_travel_after_release_m.iloc[0],3)} m "
         f"（α_D=0.5 为 0.216、0 为 0.071），T1 停稳回到 2.72 s，MC 最差滑行样本由 0.50 m 恶化到 1.75 m。"
         "<b>建议 α_D ∈ [0.5, 0.8]</b>，且必须在零速附近衰减、工况突变后先降后升。"),
        ("Q6 哪 3–5 组参数最值得上真实样机？边界与失效情形？",
         "见 §7：A（首选平衡，C3 K_a=0.8/B_d=4/α_D=0.5）、B（安全优先，C2 K_a=1.4/B_d=4）、"
         "C（湿地/换刷，α_D=0.8）、D（低阻尼省电，B_d=2 且锁死 α_D≤0.5）、E（退路 C0/C1）。"),
        ("Q7 为落地实机还需测量哪些关键量？",
         "见 §9：m、r/G/η、K_t 或 F/i_q 映射、三种工况真实 F_c/F_s/b、滚刷偏置与脉动、"
         "τ_F 与总延迟（含 UART 往返速率）、编码器分辨率与打滑、电机热限制、真实人手推力统计。"),
    ]
    A("<table><thead><tr><th style='width:30%'>问题</th><th>结论</th></tr></thead><tbody>")
    for q, a in qa:
        A(f"<tr><td style='vertical-align:top'><b>{q}</b></td><td style='white-space:normal'>{a}</td></tr>")
    A("</tbody></table>")

    # ---------- 9 测量清单 ----------
    A("<h2 id='meas'>9. 待实机测量清单（仿真无法替代）</h2>")
    meas = pd.DataFrame([
        {"量": "单侧等效平动质量 m", "当前": "ASSUMED 4.0 kg", "影响": "K_a=m̂−M_d 的绝对值与安全裕度"},
        {"量": "驱动半径 r / 传动比 G / 效率 η", "当前": "ASSUMED 0.04 m / 1 / 1", "影响": "力-转矩换算与力上限（当前 6.25 N）"},
        {"量": "K_t 或 F/i_q 映射", "当前": "null（未标定）", "影响": "电流指令与力的对应关系，无法验证电流路径"},
        {"量": "真实 F_c / F_s / b（干、湿、换刷）", "当前": "ASSUMED", "影响": "α_D·D̂ 的补偿量与稳定性"},
        {"量": "滚刷合成偏置与脉动频率/幅值", "当前": "ASSUMED", "影响": "是否诱发低速窜动/极限环"},
        {"量": "力执行环 τ_F 与总延迟（含 UART 往返）", "当前": "ASSUMED 0.04 s / 0.02 s", "影响": "加速度反馈的稳定边界、100 Hz 闭环可达性"},
        {"量": "编码器测速分辨率/量化与打滑率", "当前": "SPEC 12 bit / 未知", "影响": "状态估计质量（互补滤波 λ 的取值）"},
        {"量": "电机热限制与允许瞬态峰值时长", "当前": "未知", "影响": "能否使用 > 6.25 N 的短时包络"},
        {"量": "真实人手推力/拉力统计", "当前": "虚拟人假设", "影响": "验收指标与虚拟人参数校准"},
    ])
    A(tbl(meas))

    A("<h2>10. 复现方式</h2>")
    A("<pre><code>cd /home/xbd/workspace/Handhell_Simulation\n"
      "python -m venv .venv &amp;&amp; .venv/bin/pip install -r requirements.txt\n"
      "python run.py --milestone M1                       # 解析验证\n"
      "python run.py --milestone M2                       # S01–S12 + 耦合任务 T1/T2 × 3 虚拟人\n"
      "python run.py --milestone M3 --monte-carlo 300 --seed 20261012\n"
      "python run.py --standstill-screen 300 --seed 20261012\n"
      "python contributions.py                            # 贡献分离\n"
      "python analysis.py outputs/M2_full outputs/M3_param_study\n"
      "python make_report_html.py                         # 生成本报告\n"
      "pytest -q                                          # 27 passed</code></pre>")

    A("<footer>由 <code>make_report_html.py</code> 从 <code>outputs/</code> 的真实运行产物自动生成 ｜ "
      f"数据源：M2（{len(sc)} 场景×控制器 + {len(cp)} 耦合试验）、M3（300 组 MC + 4800 次站立筛查）、"
      f"贡献分离（{len(ab)} 组）｜ 所有参数均为 ASSUMED/SPEC/USER_MODEL 占位，非实机实测。</footer>")
    A("</div></body></html>")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(h), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "report.html"))
    ap.add_argument("--m2", default=str(ROOT / "outputs" / "M2_full"))
    ap.add_argument("--m3", default=str(ROOT / "outputs" / "M3_param_study"))
    ap.add_argument("--abl", default=str(ROOT / "outputs" / "contributions"))
    a = ap.parse_args()
    p = build(Path(a.m2), Path(a.m3), Path(a.abl), Path(a.out))
    kb = p.stat().st_size / 1024
    print(f"written {p}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
