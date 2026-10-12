#!/usr/bin/env python3
"""绘制 assist_sim 控制与验证架构框图（可直接复现）。

输出：docs/architecture.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "architecture.png"

C_IO = "#1f4e79"      # 外部输入
C_SENSE = "#1d6f42"   # 测量
C_CTRL = "#8a4b08"    # 控制
C_ACT = "#5b2c6f"     # 执行
C_PLANT = "#7b241c"   # 被控对象
C_EVAL = "#566573"    # 离线评价

fig, ax = plt.subplots(figsize=(15.5, 9.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, text, color, fs=11, fc=None, ls="-"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
        linewidth=1.6, edgecolor=color, facecolor=fc or "#ffffff", linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#111111", zorder=3, linespacing=1.5)


def arrow(p, q, text="", color="#222222", ls="-", rad=0.0, fs=9.5, dx=0.0, dy=1.4):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=15, linewidth=1.5, color=color,
        linestyle=ls, zorder=4, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, text, fontsize=fs,
                color=color, ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92))


# ---------------- 真值隔离边界 ----------------
ax.add_patch(FancyBboxPatch((3, 8), 74, 84, boxstyle="round,pad=1.2,rounding_size=2",
                            linewidth=1.4, edgecolor="#999999", facecolor="#fafafa",
                            linestyle=(0, (7, 4)), zorder=1))
ax.text(4.5, 90.5, "仿真世界（真值只在此内部流动；控制器 API 不暴露）", fontsize=10,
        color="#666666", zorder=3)
ax.add_patch(FancyBboxPatch((79, 8), 18, 84, boxstyle="round,pad=1.2,rounding_size=2",
                            linewidth=1.4, edgecolor=C_EVAL, facecolor="#f4f6f7", zorder=1))
ax.text(88, 90.5, "离线评价器\n（可读真值）", fontsize=10, color=C_EVAL,
        ha="center", zorder=3)

# ---------------- 主链路 ----------------
box(6, 74, 20, 11, "人手模型\n固定力波形 / 闭环虚拟用户\n（反应延迟 tau_h、用力速率限制）", C_IO, 9.5)
box(34, 74, 18, 11, "被控对象 Plant\nm*a = F_h + F_m - D(v,t)\nD = F_res(v) - F_br(t)", C_PLANT, 10)
box(60, 74, 16, 11, "执行器 Actuator\n饱和 |tau|<=0.25Nm -> 6.25 N\n速率限制 / 延迟 / 丢帧 / 看门狗", C_ACT, 9.5)
box(34, 55, 18, 10, "传感器 Sensors\nIMU: a+g*sin(theta)+b_a+n_a+e_vib\n轮速: (1-s)v+n_v+e_q", C_SENSE, 9.5)
box(34, 38, 18, 10, "状态估计 Estimator\na_hat 一阶低通(10 Hz)\nv_hat 互补融合 / Kalman", C_SENSE, 9.5)
box(6, 38, 20, 10, "统一安全外壳\nREST -> START -> MOVE -> STOP(+REVERSE)\n死区 / 最大力 / 速率 / 超时归零", C_CTRL, 9)
box(6, 20, 48, 12,
    "控制器组（同一执行器 / 延迟 / 随机种子 / 人手模型，仅更换控制结构）\n"
    "C0  F_cmd = 0\n"
    "C1  i = i_base + k * a  的力等效占位实现（系数未标定，非实机复现）\n"
    "C2  F_cmd = K_a * a_hat - B_d * v_hat\n"
    "C3  = C2 + alpha_D * D_hat(v_hat)   零速死区 + 软切换\n"
    "C4a/C4b  上限参考（真实状态 / 真实 D，不可部署）   C5 导纳对照（F_h_hat 逆推 / oracle）",
    C_CTRL, 9)

# ---------------- 连线 ----------------
arrow((16, 74), (34, 79.5), "F_h [N]", C_IO)
arrow((52, 74), (60, 79.5), "真值状态", C_PLANT)
arrow((68, 74), (43, 71), "F_m [N]", C_ACT, rad=-0.18, dy=-2.6)
arrow((43, 74), (43, 65.5), "", C_PLANT, rad=0)
arrow((43, 55), (43, 48), "带噪声/延迟\n测量", C_SENSE, dy=1.8)
arrow((34, 43), (26.5, 43), "v_hat, a_hat", C_SENSE, dx=0.5)
arrow((16, 48), (16, 55), "", C_CTRL)
arrow((16, 38), (16, 32), "门控系数", C_CTRL, dx=4.6)
arrow((54, 26), (74, 26), "F_cmd_raw", C_CTRL)
arrow((74, 26), (80, 79.5), "F_cmd (限幅后)", C_ACT, rad=-0.30, dx=6.0, dy=0.0)

# 评价器支路
arrow((52, 79.5), (79, 70), "真值 x,v,a,F_h,D,F_m", C_EVAL, ls=(0, (5, 3)), rad=0.10, dx=3.0)
arrow((26, 26), (79, 40), "F_cmd / 状态机日志", C_EVAL, ls=(0, (5, 3)), rad=-0.18, dx=2.0, dy=-3.0)
arrow((79, 30), (88, 24), "", C_EVAL, ls=(0, (5, 3)))
ax.text(88, 21, "指标 / 曲线 / 报告\n（含 oracle 上限对照）", fontsize=9, color=C_EVAL,
        ha="center", va="top", zorder=3)

# oracle 通道
ax.add_patch(FancyArrowPatch((52, 62), (18, 51.5), arrowstyle="-|>", mutation_scale=13,
                             linewidth=1.3, color="#b03a2e", linestyle=(0, (2, 2)), zorder=4,
                             connectionstyle="arc3,rad=0.12"))
ax.text(38, 58.5, "oracle 通道（仅 C4a/C4b/C5_oracle，显式传参、标注不可部署）",
        fontsize=8.6, color="#b03a2e", ha="center", zorder=5,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.95))

# ---------------- 图例 ----------------
ax.text(6, 12.5, "真值隔离：controllers.py / estimator.py / sensors.py / actuator.py 均不导入 plant，"
                 "且不得出现 [_true] 下标访问（tests/test_no_privileged_truth.py 静态扫描）。",
        fontsize=9.4, color="#444444", zorder=5)
ax.text(6, 9.4, "采样顺序：t_k 采样 → 延迟读出 → 估计 → 控制器 → 安全外壳+执行器限幅 → "
                "plant 以 dt=0.001 s 积分（控制零阶保持），真值仅写入日志。",
        fontsize=9.4, color="#444444", zorder=5)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=115, facecolor="white")
print(f"written {OUT}")
