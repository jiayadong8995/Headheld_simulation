"""绘图：单场景规定曲线（控制器叠加比较）、汇总柱状图、扫参热图。

每个场景输出 ≥8 张规定曲线（任务书 §9.1）：
  1) x(t)、v(t)      2) a_true/a_imu/a_hat      3) v_true/v_enc/v_hat
  4) F_h/F_m/F_cmd   5) D_true/D_hat/F_br       6) 电机转矩 + 饱和/速率限幅
  7) 目标阻抗残差     8) REST/START/MOVE/STOP 状态机 + 换向/松手标注
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体（WenQuanYi Zen Hei 已安装，避免图上出现方框）
matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans", "Liberation Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.max_open_warning"] = 0

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]
STATE_ORDER = {"REST": 0, "START": 1, "MOVE": 2, "STOP": 3}


def _style(idx: int, label: str) -> Dict:
    return {"color": COLORS[idx % len(COLORS)], "label": label, "lw": 1.2}


def _torque(rec: Dict, cfg: Dict) -> np.ndarray:
    mech = cfg["mechanics"]
    return rec["F_m_true"] * mech["r_m"] / (mech["efficiency_eta"] * mech["reduction_ratio_G"])


def _grid():
    fig, axes = plt.subplots(5, 2, figsize=(13.5, 16.5), sharex="col")
    return fig, axes.ravel()


def _release_and_reverse(t: np.ndarray, f_h: np.ndarray, v: np.ndarray) -> List[tuple]:
    marks = []
    peak = float(np.max(np.abs(f_h))) if len(f_h) else 0.0
    if peak > 1e-6:
        idx = int(np.argmax(np.abs(f_h)))
        for i in range(idx, len(f_h)):
            if abs(f_h[i]) < 0.05 * peak:
                marks.append((t[i], "release"))
                break
    sign = np.sign(np.where(np.abs(v) > 0.03, v, 0.0))
    idxs = np.where(np.abs(np.diff(sign)) > 0)[0]
    for i in idxs[:3]:
        marks.append((t[i], "reverse"))
    return marks


def plot_scenario_panel(results: Sequence, outdir: Path, cfg: Dict,
                        md_of: Dict[str, float], bd_of: Dict[str, float],
                        tag: str = "") -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = _grid()
    marks_drawn = False

    for i, res in enumerate(results):
        rec, meta = res.rec, res.meta
        t = rec["t"]
        lab = f"{res.controller}" + (f"/{meta.get('human_variant')}" if meta.get("human_variant") else "")
        st = _style(i, lab)
        Md, Bd = md_of.get(res.controller, 2.6), bd_of.get(res.controller, 4.0)

        ax[0].plot(t, rec["x_true"], **st)
        ax[1].plot(t, rec["v_true"], **st)
        ax[2].plot(t, rec["a_true"], **st)
        ax[2].plot(t, rec["a_imu"], ls=":", color=st["color"], alpha=0.5, lw=0.7)
        ax[2].plot(t, rec["a_hat"], ls="--", color=st["color"], alpha=0.75, lw=0.8)
        ax[3].plot(t, rec["v_true"], **st)
        ax[3].plot(t, rec["v_enc"], ls=":", color=st["color"], alpha=0.5, lw=0.7)
        ax[3].plot(t, rec["v_hat"], ls="--", color=st["color"], alpha=0.75, lw=0.8)
        ax[4].plot(t, rec["F_h_true"], **st)
        ax[4].plot(t, rec["F_m_true"], ls="--", color=st["color"], alpha=0.75, lw=0.8)
        ax[4].plot(t, rec["F_cmd"], ls=":", color=st["color"], alpha=0.6, lw=0.7)
        ax[5].plot(t, rec["D_true"], **st)
        ax[5].plot(t, rec["D_hat"], ls="--", color=st["color"], alpha=0.75, lw=0.8)
        ax[5].plot(t, rec["F_br_true"], ls=":", color=st["color"], alpha=0.6, lw=0.7)
        ax[6].plot(t, _torque(rec, cfg), **st)
        ax[7].plot(t, rec["F_h_true"] - (Md * rec["a_true"] + Bd * rec["v_true"]), **st)
        if "x_ref" in rec and not np.all(np.isnan(rec["x_ref"])):
            ax[0].plot(t, rec["x_ref"], ls="--", color=st["color"], alpha=0.4, lw=0.8)
        if not marks_drawn:
            marks_drawn = True
            for tm, kind in _release_and_reverse(t, rec["F_h_true"], rec["v_true"]):
                ax[0].axvline(tm, ls=":", color=kind if False else "#888888", lw=0.8, alpha=0.8)
                ax[3].axvline(tm, ls=":", color="#888888", lw=0.8, alpha=0.6)

    if results:
        y = np.array([STATE_ORDER.get(s, 0) for s in results[0].rec["state"]], dtype=float)
        ax[8].plot(results[0].rec["t"], y, color="#333", lw=1.0)
        ax[8].set_yticks([0, 1, 2, 3], ["REST", "START", "MOVE", "STOP"])

    # 第 10 面板：速度估计误差 + 指令饱和/限幅标记（§9.1 第 6 项）
    for i, res in enumerate(results):
        rec = res.rec
        st = _style(i, res.controller)
        ax[9].plot(rec["t"], rec["v_hat"] - rec["v_true"], lw=1.0, color=st["color"], alpha=0.9)
        lim = res.meta["force_limit"]
        sat = np.abs(rec["F_cmd_raw"]) > lim + 1e-9
        if np.any(sat):
            ax[9].plot(rec["t"][sat], (rec["v_hat"] - rec["v_true"])[sat], "x",
                       ms=2.5, color=st["color"])
    ax[9].axhline(0.0, color="k", lw=0.6)

    ax[6].axhline(cfg["motor_spec"]["rated_torque_Nm"], color="k", ls="--", lw=0.8,
                  label="额定 0.25 N·m")
    ax[6].axhline(-cfg["motor_spec"]["rated_torque_Nm"], color="k", ls="--", lw=0.8)
    ax[6].legend(fontsize=7)

    labels = ["x [m]", "v [m/s]", "a [m/s²]", "v [m/s]\n(实:v_true 点:v_enc 虚:v_hat)",
              "F [N]\n(实:F_h 虚:F_m 点:F_cmd)", "D [N]\n(实:D_true 虚:D_hat 点:F_br)",
              "τ_w [N·m]（实线）", "阻抗残差 F_h-(M_d a+B_d v) [N]", "状态机",
              "速度估计误差 v_hat-v_true [m/s]\n(× = 指令饱和时刻)"]
    for i, a in enumerate(ax):
        a.set_ylabel(labels[i], fontsize=8)
        a.grid(alpha=0.3, lw=0.5)
    ax[8].set_xlabel("t [s]")
    ax[9].set_xlabel("t [s]")
    handles = ax[0].get_legend_handles_labels()[0]
    labs = ax[0].get_legend_handles_labels()[1]
    if handles:
        fig.legend(handles, labs, loc="upper center", ncol=min(4, len(handles)), fontsize=9)
    fig.suptitle(f"场景 {results[0].scenario} — {results[0].rec['meta'].get('scenario','')} {tag}",
                 y=0.996, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    name = f"{results[0].scenario}{('_' + tag) if tag else ''}_overlay.png"
    path = outdir / name
    fig.savefig(path, dpi=105)
    plt.close(fig)
    return path


def plot_single_controller(res, outdir: Path, cfg: Dict, Md: float, Bd: float) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rec = res.rec
    t = rec["t"]
    fig, ax = _grid()
    c = "#1f77b4"

    ax[0].plot(t, rec["x_true"], color=c)
    ax[1].plot(t, rec["v_true"], color=c)
    ax[2].plot(t, rec["a_true"], color=c, label="a_true")
    ax[2].plot(t, rec["a_imu"], ls=":", color="#d62728", label="a_imu")
    ax[2].plot(t, rec["a_hat"], ls="--", color="#2ca02c", label="a_hat")
    ax[2].legend(fontsize=7)
    ax[3].plot(t, rec["v_true"], color=c, label="v_true")
    ax[3].plot(t, rec["v_enc"], ls=":", color="#d62728", label="v_enc")
    ax[3].plot(t, rec["v_hat"], ls="--", color="#2ca02c", label="v_hat")
    ax[3].legend(fontsize=7)
    ax[4].plot(t, rec["F_h_true"], color=c, label="F_h")
    ax[4].plot(t, rec["F_m_true"], color="#d62728", label="F_m")
    ax[4].plot(t, rec["F_cmd"], ls="--", color="#2ca02c", label="F_cmd")
    ax[4].legend(fontsize=7)
    ax[5].plot(t, rec["D_true"], color=c, label="D_true")
    ax[5].plot(t, rec["D_hat"], ls="--", color="#2ca02c", label="D_hat")
    ax[5].plot(t, rec["F_br_true"], ls=":", color="#ff7f0e", label="F_br")
    ax[5].legend(fontsize=7)
    ax[6].plot(t, _torque(rec, cfg), color=c)
    ax[6].axhline(cfg["motor_spec"]["rated_torque_Nm"], color="k", ls="--", lw=0.8)
    ax[6].axhline(-cfg["motor_spec"]["rated_torque_Nm"], color="k", ls="--", lw=0.8)
    ax[7].plot(t, rec["F_h_true"] - (Md * rec["a_true"] + Bd * rec["v_true"]), color=c)
    y = np.array([STATE_ORDER.get(s, 0) for s in rec["state"]], dtype=float)
    ax[8].plot(t, y, color="#333", lw=1.0)
    ax[8].set_yticks([0, 1, 2, 3], ["REST", "START", "MOVE", "STOP"])
    labels = ["x [m]", "v [m/s]", "a [m/s²]", "v [m/s]", "F [N]", "D [N]", "τ_w [N·m]",
              "阻抗残差 [N]", "状态机"]
    for i, a in enumerate(ax[:9]):
        a.set_ylabel(labels[i], fontsize=8)
        a.grid(alpha=0.3, lw=0.5)
    y = np.array([STATE_ORDER.get(s, 0) for s in rec["state"]], dtype=float)
    ax[9].plot(t, rec["v_hat"] - rec["v_true"], color="#8c564b", lw=1.0)
    ax[9].set_ylabel("速度估计误差 v_hat-v_true [m/s]", fontsize=8)
    ax[9].grid(alpha=0.3, lw=0.5)
    ax[8].set_xlabel("t [s]")
    fig.suptitle(f"{res.scenario} / {res.controller}", y=0.996, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path = outdir / f"{res.scenario}_{res.controller}_detail.png"
    fig.savefig(path, dpi=95)
    plt.close(fig)
    return path


def plot_metric_bars(rows: List[Dict], outdir: Path, metrics: Sequence[str],
                     xlabel: str = "controller", fname: str = "metrics_bars.png",
                     title: str = "") -> Path:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.2 * n))
    if n == 1:
        axes = [axes]
    labels = [str(r[xlabel]) for r in rows]
    for ax, metric in zip(axes, metrics):
        vals = [float(r.get(metric, np.nan)) for r in rows]
        ax.bar(labels, vals, color=[COLORS[i % len(COLORS)] for i in range(len(labels))])
        ax.set_ylabel(metric, fontsize=9)
        ax.grid(alpha=0.3, axis="y", lw=0.5)
        ax.tick_params(labelsize=8)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path = outdir / fname
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_heatmap(grid: np.ndarray, xlabels: Sequence, ylabels: Sequence,
                 outdir: Path, fname: str, title: str, cbar_label: str,
                 xlabel: str = "K_a/m_nominal", ylabel: str = "B_d [N·s/m]") -> Path:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(xlabels)), [str(x) for x in xlabels])
    ax.set_yticks(range(len(ylabels)), [str(y) for y in ylabels])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, label=cbar_label)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="w", fontsize=7)
    fig.tight_layout()
    path = outdir / fname
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
