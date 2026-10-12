#!/usr/bin/env python3
"""绘制 assist_sim 控制与验证架构框图（清晰版：正交走线 / 单向回路 / 零交叉 / 零缺字）。

排版与布局要点：
  * 数学式一律用 mathtext（$...$）排版 —— 上下标、hat(^)、点号(dot)、希腊字母、
    真减号由 matplotlib 内置 DejaVu 数学字体提供，规避本机 CJK 字体缺字形导致的豆腐块。
  * 蛇形单向信号流：Row1 左→右（人机→Plant→Sensors→Estimator）
                      Row2 右→左（控制器→安全外壳→执行器）
                      执行器经左侧垂直走廊回到 Plant，构成唯一回路。
  * 所有连线只走水平/垂直，且走廊与方框严格分离 —— 无穿框、无斜穿、无交叉。

输出：docs/architecture.png（+ .svg 矢量版）
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# 中文用文泉驿正黑；数学部分走 mathtext（内置 DejaVu），二者互不干扰。
matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"

ROOT = Path(__file__).resolve().parent
OUT_PNG = ROOT / "docs" / "architecture.png"
OUT_SVG = ROOT / "docs" / "architecture.svg"

INK = "#1b1b1b"
C_HUMAN = "#1f4e79"
C_PLANT = "#7b241c"
C_SENSE = "#1d6f42"
C_CTRL = "#8a4b08"
C_ACT = "#5b2c6f"
C_EVAL = "#4d5656"
C_ORACLE = "#b03a2e"
GREY = "#9a9a9a"

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.set_position([0, 0, 1, 1])


# --------------------------------------------------------------------------- #
# 基本图元
# --------------------------------------------------------------------------- #
def box(x, y, w, h, title, body="", color=INK, fs_t=12.6, fs_b=9.8, fc="#ffffff"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.3",
        linewidth=1.9, edgecolor=color, facecolor=fc, zorder=2))
    if body:
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top",
                fontsize=fs_t, color=color, zorder=3)
        ax.text(x + w / 2, y + h - 7.4, body, ha="center", va="top",
                fontsize=fs_b, color="#1f1f1f", zorder=3, linespacing=1.55)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs_t, color=color, zorder=3)


def poly(pts, color=INK, ls="-", lw=1.9, arrow=True):
    """正交折线；最后一段带箭头。"""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, ls=ls, lw=lw,
                solid_capstyle="round", zorder=4)
    if arrow:
        ax.add_patch(FancyArrowPatch(
            pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=17,
            lw=lw, color=color, linestyle=ls, zorder=4, shrinkA=0, shrinkB=0))


def label(x, y, text, color="#333333", fs=9.6, ha="center", va="center"):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va,
            zorder=6, bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.95))


# --------------------------------------------------------------------------- #
# 真值隔离边界 + 离线评价器
# --------------------------------------------------------------------------- #
ax.add_patch(FancyBboxPatch(
    (1.5, 40), 97, 56, boxstyle="round,pad=0.0,rounding_size=2.0",
    linewidth=1.4, edgecolor="#b8b8b8", facecolor="#fbfbfb",
    linestyle=(0, (8, 4)), zorder=1))
ax.text(3.2, 94.0, "仿真世界 —— 真值只在此内部流动；控制器 API 不暴露",
        fontsize=10.4, color="#707070", va="top", zorder=3)

box(3, 14, 94, 20,
    "离线评价器　（可读真值，不参与实时控制）",
    "接入：真值状态　·　$F_{cmd}$（限幅前后）　·　状态机日志\n"
    "产出：指标 / 曲线 / 报告（含 oracle 上限对照）\n"
    "校验：硬性检查 6/6　·　27 单元测试　·　S01–S12 场景矩阵　·　Monte Carlo　·　4800 次站立筛查",
    C_EVAL, fs_t=13.0, fs_b=10.4)

poly([(50, 39.0), (50, 34.2)], color=GREY, ls=(0, (5, 3)))
label(50, 36.6, "真值 / $F_{cmd}$ / 状态机日志　（离线，不参与实时控制）", color="#6b6b6b", fs=9.4)


# --------------------------------------------------------------------------- #
# Row 1（左 → 右）：人机模型 → Plant → Sensors → Estimator
# --------------------------------------------------------------------------- #
Y1, H1 = 76.0, 15.0

box(3, Y1, 18, H1,
    "人机模型",
    "固定力波形（6 种）\n"
    "闭环虚拟用户：\n"
    r"$F_h = K_p(x_{ref}-x) + K_d(v_{ref}-v)$" "\n"
    r"反应延迟 $\tau_h$ / 用力速率限制" "\n"
    "手感档：soft / mid / stiff",
    C_HUMAN, fs_b=9.2)

box(24, Y1, 23, H1,
    "被控对象 Plant",
    r"$m\dot{v} = F_h + F_m - D(v,t)$" "\n"
    r"$D(v,t) = F_{res}(v) - F_{br}(t)$　（不保证恒正）" "\n"
    "摩擦：粘滞 + Stribeck 静/动 + 方向平滑\n"
    "独立粘滑版：零速锁定 + 滞回驻留\n"
    r"积分 $dt = 1$ ms（控制零阶保持）",
    C_PLANT, fs_b=9.2)

box(50, Y1, 22, H1,
    "传感器 Sensors",
    r"IMU：$a_{imu} = a + g\sin\theta + b_a + n_a + e_{vib}$" "\n"
    r"轮速：$v_{enc} = (1-s)v + n_v + e_q$" "\n"
    "量化 + 时间戳 + 延迟 + 失帧\n"
    r"打滑 $s$ / 湿地面粘滑",
    C_SENSE, fs_b=9.2)

box(75, Y1, 22, H1,
    "状态估计 Estimator",
    r"$\hat{a}$：一阶低通（10 Hz）" "\n"
    r"$\hat{v} = (1-\lambda)(\hat{v}+T_s(\hat{a}-\hat{b})) + \lambda v_{enc}$" "\n"
    "可选 Kalman 对照\n"
    "（仅用测量，不看真值）",
    C_SENSE, fs_b=9.2)

Y1a = Y1 + H1 / 2
poly([(21, Y1a), (24, Y1a)], color=C_HUMAN)
label(22.5, Y1a + 2.8, r"$F_h$ [N]", color=C_HUMAN)

poly([(47, Y1a), (50, Y1a)], color=C_PLANT)
label(48.5, Y1a + 2.8, r"真值 $x, v, a$", color=C_PLANT)

poly([(72, Y1a), (75, Y1a)], color=C_SENSE)
label(73.5, Y1a + 2.8, "测量\n（噪声/延迟）", color=C_SENSE, fs=8.8)


# --------------------------------------------------------------------------- #
# Row 2（右 → 左）：控制器组 → 安全外壳 → 执行器
# --------------------------------------------------------------------------- #
Y2, H2 = 46.0, 16.0
Y2c = Y2 + H2 / 2

box(3, Y2, 22, H2,
    "执行器 Actuator",
    r"$F_{max} = \eta G \tau_{rated} / r = 6.25$ N" "\n"
    "延迟队列 → 速率限制（30 N/s）→ 饱和\n"
    "丢帧 ZOH 延长 / 看门狗归零 / 急停",
    C_ACT, fs_b=9.4)

box(28, Y2, 22, H2,
    "统一安全外壳",
    "状态机　REST→START→MOVE→STOP(+REVERSE)\n"
    r"输入 $\hat{v}, \hat{a}$ 与 $F_{cmd}^{raw}$" "\n"
    "死区 0.03 m/s / 最大力 / 速率 / 超时归零",
    C_CTRL, fs_b=9.4)

box(53, 44, 44, 20,
    "控制器组　（同一执行器 / 延迟 / 随机种子 / 人手模型，仅更换控制结构）",
    r"目标：$F_h = M_d a + B_d v$　　$F_{cmd} = K_a\hat{a} - B_d\hat{v} + \alpha_D\hat{D}$"
    r"　（$K_a = \hat{m} - M_d < m$）" "\n"
    r"C0 = 0（基线）　　C1 = $i_{base} + k\,a$ 的力等效占位（系数未标定，非实机复现）" "\n"
    r"C2 = $K_a\hat{a} - B_d\hat{v}$（虚拟阻抗）　　C3 = C2 + $\alpha_D\hat{D}$（+ 零速死区 / 软切换）" "\n"
    r"C4a/C4b = 上限参考（真实状态 / 真实 $D$，不可部署）　　C5 = 导纳对照（$F_h$ 逆推 / oracle）",
    C_CTRL, fs_b=9.4)

poly([(53, Y2c + 0.5), (50, Y2c + 0.5)], color=C_CTRL)
label(51.5, Y2c + 3.6, r"$F_{cmd}^{raw}$", color=C_CTRL, fs=9.2)

poly([(28, Y2c + 0.5), (25, Y2c + 0.5)], color=C_CTRL)
label(26.5, Y2c + 3.6, "$F_{cmd}$\n（门控+限幅）", color=C_CTRL, fs=9.0)


# --------------------------------------------------------------------------- #
# 反馈列：执行器 → Plant（唯一回路，走左侧垂直走廊）
# --------------------------------------------------------------------------- #
poly([(13, Y2 + H2), (13, 70.5), (35.0, 70.5), (35.0, Y1)], color=C_ACT, lw=2.1)
label(24.0, 72.6, r"$F_m$ [N]", color=C_ACT, fs=9.8)


# --------------------------------------------------------------------------- #
# 前向列：Estimator → 控制器组（右侧垂直走廊）
# --------------------------------------------------------------------------- #
poly([(95, Y1), (95, 64.0)], color=C_SENSE, lw=1.9)
label(95, 71.0, r"$\hat{v}, \hat{a}$", color=C_SENSE, fs=9.6)


# --------------------------------------------------------------------------- #
# oracle 通道：真值 D / 真实状态 → C4a/C4b/C5_oracle（虚线，独立走廊）
# --------------------------------------------------------------------------- #
poly([(45, Y1), (45, 68.0), (84.0, 68.0), (84.0, 64.0)], color=C_ORACLE, ls=(0, (4, 2.5)), lw=1.5)
label(64.5, 69.7,
      r"oracle 通道：真实状态 / 真实 $D$ —— 仅 C4a/C4b/C5_oracle 显式启用，不可部署",
      color=C_ORACLE, fs=9.0)


# --------------------------------------------------------------------------- #
# 注释
# --------------------------------------------------------------------------- #
ax.text(3, 10.6, "真值隔离：controllers / estimator / sensors / actuator 均不 import plant，"
                 "且不得出现 [_true] 下标访问（tests/test_no_privileged_truth.py 静态扫描）。",
        fontsize=9.4, color="#4a4a4a", va="top", zorder=5)
ax.text(3, 7.0, "采样顺序：t_k 采样 → 延迟读出 → 估计 → 控制器 → 安全外壳 + 执行器限幅 → "
                "plant 以 dt = 1 ms 积分（控制零阶保持）；真值仅写入日志。",
        fontsize=9.4, color="#4a4a4a", va="top", zorder=5)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, facecolor="white")
fig.savefig(OUT_SVG, facecolor="white")
print(f"written {OUT_PNG}")
print(f"written {OUT_SVG}")
