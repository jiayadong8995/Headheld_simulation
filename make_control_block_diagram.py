#!/usr/bin/env python3
"""控制框图（Control Block Diagram）—— 单侧半车一维助力控制的闭环结构。

只画控制论语义：求和点、传递函数、扰动输入、反馈回路。
工程模块 / 文件映射见 make_simulation_architecture.py。

排版要点：数学式一律 mathtext（$...$），规避本机 CJK 字体缺字形（U+0302/U+0307/U+2212）。
布局要点：全部正交走线，回路走廊与方框分离，线路之间零交叉。

输出：docs/control_block_diagram.png（+ .svg）
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"

ROOT = Path(__file__).resolve().parent
OUT_PNG = ROOT / "docs" / "control_block_diagram.png"
OUT_SVG = ROOT / "docs" / "control_block_diagram.svg"

INK = "#1b1b1b"
C_HUMAN = "#1f4e79"
C_PLANT = "#7b241c"
C_SENSE = "#1d6f42"
C_CTRL = "#8a4b08"
C_ACT = "#5b2c6f"
GREY = "#8b8b8b"

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 100)
ax.set_ylim(-2, 104)
ax.axis("off")
ax.set_position([0, 0, 1, 1])


# --------------------------------------------------------------------------- #
def box(x, y, w, h, title, body="", color=INK, fs_t=11.6, fs_b=9.4, fc="#ffffff", lw=1.9):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.1",
        linewidth=lw, edgecolor=color, facecolor=fc, zorder=2))
    if body:
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top",
                fontsize=fs_t, color=color, zorder=3)
        ax.text(x + w / 2, y + h - 6.3, body, ha="center", va="top",
                fontsize=fs_b, color="#1a1a1a", zorder=3, linespacing=1.5)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs_t, color=color, zorder=3)


def poly(pts, color=INK, ls="-", lw=1.9, arrow=True):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, ls=ls, lw=lw, solid_capstyle="round", zorder=4)
    if arrow:
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=16,
                                     lw=lw, color=color, linestyle=ls, zorder=4,
                                     shrinkA=0, shrinkB=0))


def node(p, color=INK, r=0.85):
    ax.add_patch(plt.Circle(p, r, color=color, zorder=5))


def label(x, y, text, color="#333333", fs=9.3, ha="center", va="center"):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va, zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.94))


def sign(x, y, text, color=INK, fs=12):
    ax.text(x, y, text, fontsize=fs, color=color, ha="center", va="center",
            zorder=6, fontweight="normal")


# --------------------------------------------------------------------------- #
# 输入层：人手模型（扰动源） + 阻力 D(v,t)
# --------------------------------------------------------------------------- #
box(6, 88, 24, 10, "人手模型（扰动源）",
    "固定力波形（6 种）\n"
    r"闭环虚拟用户 $F_h = K_p(x_{ref}-x)+K_d(v_{ref}-v)$" "\n"
    r"反应延迟 $\tau_h$ / 用力速率限制",
    C_HUMAN, fs_b=9.0)

box(62, 88, 28, 10, r"阻力 $D(v,t)$（非恒正）",
    r"$D = F_{res}(v) - F_{br}(t)$" "\n"
    "粘滞 + Stribeck 静/动 + 方向平滑\n"
    r"滚刷合力 $F_{br}(t)$（外部时变输入）；$D$ 可助推",
    C_PLANT, fs_b=9.0)


# --------------------------------------------------------------------------- #
# 前向通道（y 72..84）
# --------------------------------------------------------------------------- #
box(3, 70, 21, 14, "执行器",
    r"一阶响应 $1/(\tau_F s+1)$，$\tau_F=0.04$ s" "\n"
    "延迟队列 / 速率限制 / 丢帧\n"
    r"饱和 $|F_m| \leq F_{max} = 6.25$ N" "\n"
    "看门狗归零 / 急停",
    C_ACT, fs_b=8.8)

# 求和点 Σ
ax.add_patch(FancyBboxPatch((27.5, 73), 11, 11, boxstyle="round,pad=0.0,rounding_size=1.0",
                            linewidth=2.0, edgecolor=INK, facecolor="#f2f2f2", zorder=2))
ax.text(33, 78.5, r"$\Sigma$", ha="center", va="center", fontsize=17, color=INK, zorder=3)

box(44, 70, 18, 14, "被控对象",
    r"$1/(m\,s)$　$m = 4.0$ kg" "\n"
    r"$m\dot{v} = F_{net}$" "\n"
    r"$F_{net} = F_h + F_m - D$" "\n"
    "单侧半车一维平动",
    C_PLANT, fs_b=9.0)

box(72, 70, 16, 14, "位置积分",
    r"$1/s$" "\n"
    r"$x = \int v\,dt$",
    GREY, fs_b=9.4)

# 前向连线
poly([(24, 77), (27.5, 77)], color=C_ACT)                 # 执行器 → Σ
poly([(38.5, 77), (44, 77)], color=C_PLANT)
label(41.2, 79.7, r"$F_{net}$", color=C_PLANT, fs=9.2)               # Σ → 被控对象
node((66, 77), INK)                                        # v 节点
poly([(62, 77), (66, 77)], color=C_PLANT)                 # 被控对象 → v
poly([(66, 77), (72, 77)], color=INK, arrow=True)         # v → 1/s
poly([(88, 77), (95, 77)], color=GREY)
node((95, 77), GREY, r=0.7)
poly([(95, 77), (95, 100.5), (2, 100.5), (2, 91), (6, 91)],
     color=C_HUMAN, ls=(0, (4, 2.5)), lw=1.5)
label(45, 102.3, r"闭环虚拟用户观测 $x, v$", color=C_HUMAN, fs=8.8)                    # 1/s → x

label(25.7, 74.4, r"$F_m$", color=C_ACT, fs=10)
sign(25.8, 78.9, r"$+$", C_ACT)
label(69, 79.6, r"$v$", color=INK, fs=11)
label(91.5, 79.4, r"$x$", color=GREY, fs=11)

# 扰动输入
poly([(18, 88), (18, 86.6), (29.5, 86.6), (29.5, 84)], color=C_HUMAN)
sign(27.8, 84.8, r"$+$", C_HUMAN)
label(23.0, 85.1, r"$F_h$", color=C_HUMAN, fs=10.4)

poly([(62, 92.5), (35.5, 92.5), (35.5, 84)], color=C_PLANT)
sign(37.4, 84.8, r"$-$", C_PLANT)
label(49, 94.0, r"$D(v,t)$", color=C_PLANT, fs=10.6)

# D 的输入取自 v
poly([(66, 77), (66, 87)], color=C_PLANT, ls=(0, (4, 2.5)), lw=1.6, arrow=True)
label(70.5, 82.0, r"$v$", color=C_PLANT, fs=9.6)


# --------------------------------------------------------------------------- #
# 反馈通道（度量 → 控制 → 执行）
# --------------------------------------------------------------------------- #
# 真值 → 传感器（走最右走廊）
poly([(66, 77), (66, 69), (97, 69), (97, 41), (74, 41)], color=GREY, lw=1.7)
label(81, 70.2, r"真值 $x, v, a$（$a=\dot{v}$）", color=GREY, fs=9.2)

box(50, 34, 24, 13, "传感器 Sensors",
    r"$a_{imu} = a + g\sin\theta + b_a + n_a + e_{vib}$" "\n"
    r"$v_{enc} = (1-s)v + n_v + e_q$" "\n"
    "量化 / 时间戳 / 延迟 / 丢帧",
    C_SENSE, fs_b=8.6)

poly([(62, 47), (62, 55)], color=C_SENSE)
label(66.6, 51, r"$v_{enc}, a_{imu}$", color=C_SENSE, fs=9.0)

box(50, 55, 24, 13, "状态估计 Estimator",
    r"$v^-_k = \hat{v}_{k-1} + T_s(a_{imu,k}-\hat{b}_a)$" "\n"
    r"$\hat{v}_k = (1-\lambda)v^-_k + \lambda v_{enc,k}$" "\n"
    r"$\hat{a}$：一阶低通（10 Hz）",
    C_SENSE, fs_b=8.6)

poly([(50, 61.5), (46, 61.5)], color=C_SENSE)
label(48, 64.0, r"$\hat{v}, \hat{a}$", color=C_SENSE, fs=9.0)

box(26, 55, 20, 13, r"控制器 $C(s)$",
    r"$F_{cmd} = K_a\hat{a} - B_d\hat{v} + \alpha_D\hat{D}(\hat{v})$" "\n"
    r"$\hat{D}(\hat{v}) = b_{est}\hat{v} + F_{c,est}\tanh(\hat{v}/v_\epsilon)$" "\n"
    r"$K_a = \hat{m}-M_d < m$：C2 $\alpha_D{=}0$，C3 $\alpha_D{=}0.5$",
    C_CTRL, fs_b=8.4)

poly([(26, 61.5), (18, 61.5)], color=C_CTRL)
poly([(18, 66.2), (26, 66.2)], color=C_CTRL, ls=(0, (4, 2.5)), lw=1.7)
label(22, 67.6, r"门控 $g_a, g_v, g_D$", color=C_CTRL, fs=8.6)
label(22, 63.2, r"$F_{cmd}$", color=C_CTRL, fs=9.2)

box(3, 55, 15, 13, "统一安全外壳",
    "状态机 REST→START→\nMOVE→STOP(+REVERSE)\n"
    r"门控 $g_a, g_v, g_D$ 送控制器" "\n"
    "死区 0.03 m/s / 超时归零",
    C_CTRL, fs_b=8.2)

poly([(11.5, 68), (11.5, 70)], color=C_CTRL)
label(15.5, 69.4, r"$u$", color=C_CTRL, fs=9.6)


# --------------------------------------------------------------------------- #
# 注释
# --------------------------------------------------------------------------- #
ax.text(3, 30.0, "设计目标：改变用户感受到的等效惯量与阻尼 —— 期望手感 "
                 r"$F_h = M_d a + B_d v$。控制律为前馈阻抗整形，不含参考跟踪项（无位置刚度）。",
        fontsize=9.6, color="#3d3d3d", va="top", zorder=5)
ax.text(3, 25.6, r"默认参数（C2/C3）：$K_a = 1.4$ kg，$B_d = 4.0$ N·s/m，$\alpha_D = 0.5$，"
                 r"$F_{max} = 6.25$ N，$m = 4.0$ kg，$\tau_F = 0.04$ s。",
        fontsize=9.6, color="#3d3d3d", va="top", zorder=5)
ax.text(3, 21.2, "真值隔离：控制器 / 估计器 / 传感器 / 执行器均不导入 plant，"
                 "真值只写入日志（见仿真架构图）。",
        fontsize=9.6, color="#3d3d3d", va="top", zorder=5)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, facecolor="white")
fig.savefig(OUT_SVG, facecolor="white")
print(f"written {OUT_PNG}")
print(f"written {OUT_SVG}")
