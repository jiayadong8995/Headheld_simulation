#!/usr/bin/env python3
"""仿真架构图（Simulation Architecture）—— 模块划分、运行回路、验证栈与产物。

只画工程语义：代码模块 / 文件映射、单步时序回路、真值隔离边界、验证与产物。
控制论语义（求和点、传递函数、扰动回路）见 make_control_block_diagram.py。

输出：docs/simulation_architecture.png（+ .svg）
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
OUT_PNG = ROOT / "docs" / "simulation_architecture.png"
OUT_SVG = ROOT / "docs" / "simulation_architecture.svg"

INK = "#1b1b1b"
C_CFG = "#1f4e79"
C_RUN = "#2e6b4f"
C_KERNEL = "#7b241c"
C_VERIFY = "#8a4b08"
C_OUT = "#4d5656"
GREY = "#8b8b8b"

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.set_position([0, 0, 1, 1])


# --------------------------------------------------------------------------- #
def band(x, y, w, h, title, color, fc):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.6",
        linewidth=1.9, edgecolor=color, facecolor=fc, zorder=1))
    ax.text(x + 2.2, y + h - 1.6, title, ha="left", va="top",
            fontsize=11.0, color=color, zorder=3)


def box(x, y, w, h, title, body="", color=INK, fs_t=10.4, fs_b=9.0, fc="#ffffff", lw=1.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.0",
        linewidth=lw, edgecolor=color, facecolor=fc, zorder=2))
    if body:
        ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top",
                fontsize=fs_t, color=color, zorder=3)
        ax.text(x + w / 2, y + h - 5.4, body, ha="center", va="top",
                fontsize=fs_b, color="#1a1a1a", zorder=3, linespacing=1.5)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs_t, color=color, zorder=3)


def pipe(x, y, w, h, name, mod):
    """运行回路上的模块块。"""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.0",
        linewidth=1.7, edgecolor=C_KERNEL, facecolor="#ffffff", zorder=2))
    ax.text(x + w / 2, y + h - 2.0, name, ha="center", va="top",
            fontsize=9.6, color=C_KERNEL, zorder=3)
    ax.text(x + w / 2, y + 1.6, mod, ha="center", va="bottom",
            fontsize=8.4, color="#555555", zorder=3)


def poly(pts, color=INK, ls="-", lw=1.8, arrow=True):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, ls=ls, lw=lw, solid_capstyle="round", zorder=4)
    if arrow:
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=16,
                                     lw=lw, color=color, linestyle=ls, zorder=4,
                                     shrinkA=0, shrinkB=0))


def label(x, y, text, color="#333333", fs=9.0, ha="center", va="center"):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va, zorder=6,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.94))


# --------------------------------------------------------------------------- #
# 配置层
# --------------------------------------------------------------------------- #
band(3, 86, 94, 12, "① 配置层　config/", C_CFG, "#f2f7fb")
box(6, 88, 43, 8, "base.yaml　参数与来源标注",
    "全部带单位；provenance = SPEC / USER_MODEL / ASSUMED / MEASURED",
    C_CFG, fs_t=10.2, fs_b=8.8)
box(53, 88, 41, 8, "scenarios.yaml　场景定义",
    "S01–S12 工况矩阵 + T1–T3 耦合任务（人手 / 控制器配对）",
    C_CFG, fs_t=10.2, fs_b=8.8)


# --------------------------------------------------------------------------- #
# 运行入口
# --------------------------------------------------------------------------- #
band(3, 66, 94, 14, "② 运行入口　run.py（CLI）", C_RUN, "#f1f8f3")
box(6, 68, 44, 10, "run.py　仿真主入口",
    r"--scenario S03 ｜ --milestone M1/M2/M3 ｜ --sweep" "\n"
    r"--monte-carlo N ｜ --standstill-screen N ｜ --seed 20261012",
    C_RUN, fs_t=10.6, fs_b=9.0)
box(54, 68, 40, 10, "辅助脚本",
    "contributions.py　贡献分离（7 结构 × 5 工况）\n"
    "analysis.py　汇总为报告数据表\n"
    "make_report_html.py　自包含 HTML 报告",
    C_RUN, fs_t=10.6, fs_b=9.0)


# --------------------------------------------------------------------------- #
# 仿真内核（真值隔离边界）
# --------------------------------------------------------------------------- #
band(3, 29, 94, 31, "③ 仿真内核　（真值隔离边界：控制侧一律不 import plant）",
     C_KERNEL, "#fdf4f3")

# 单步时序回路
pipe(6, 44, 16, 11, "传感器", "sim/sensors.py")
pipe(26, 44, 16, 11, "状态估计", "sim/estimator.py")
pipe(46, 44, 18, 11, "控制器", "sim/controllers.py")
pipe(68, 44, 12, 11, "安全外壳", "状态机 / 门控")
pipe(84, 44, 12, 11, "执行器", "sim/actuator.py")

for x0, x1 in ((22, 26), (42, 46), (64, 68), (80, 84)):
    poly([(x0, 49.5), (x1, 49.5)], color=C_KERNEL, lw=1.7)

box(6, 33, 24, 8, "被控对象", r"sim/plant.py", C_KERNEL, fs_t=10.2, fs_b=8.8)
ax.text(18, 35.4, r"真值只写入日志", ha="center", va="bottom", fontsize=8.4,
        color="#666666", zorder=3)

box(36, 33, 32, 8, "人手模型", "sim/human.py　固定力波形 / 闭环虚拟用户",
    C_KERNEL, fs_t=10.2, fs_b=8.8)

# 回路连线
poly([(36, 37), (30, 37)], color=C_KERNEL, lw=1.7)                    # 人手 → 被控对象
poly([(90, 44), (90, 31.5), (24, 31.5), (24, 33)], color=C_KERNEL, lw=1.7)  # 执行器 → 被控对象
poly([(15, 41), (15, 44)], color=C_KERNEL, lw=1.9)                    # 被控对象 → 传感器
label(20.5, 42.6, r"真值 $x, v, a$", color=C_KERNEL, fs=8.8)
label(57, 29.9, "单步 $\Delta t$：采样 → 延迟读出 → 估计 → 控制器 → 安全外壳门控/限幅 → 执行器 "
                "→ plant 积分（dt = 1 ms，控制零阶保持）", color="#555555", fs=8.8)


# --------------------------------------------------------------------------- #
# 验证栈
# --------------------------------------------------------------------------- #
band(3, 15, 94, 12, "④ 验证栈　tests/ ｜ sim/metrics.py ｜ sim/scenarios.py ｜ sim/sweep.py",
     C_VERIFY, "#fdf8f1")
box(6, 17, 43, 8, "单元测试与硬性检查",
    "27 单元测试（pytest）\n"
    "真值隔离静态扫描 / 限幅 / 丢帧 / 失能 / 正反推力镜像 / 同种子可重现",
    C_VERIFY, fs_t=10.2, fs_b=8.8)
box(53, 17, 41, 8, "里程碑实验",
    "M1 解析辨识 · M2 场景矩阵 S01–S12 + T1–T3\n"
    "M3 扫参 + Monte Carlo 300 · 4800 次站立安全筛查",
    C_VERIFY, fs_t=10.2, fs_b=8.8)


# --------------------------------------------------------------------------- #
# 产物
# --------------------------------------------------------------------------- #
band(3, 2, 94, 11, "⑤ 产物", C_OUT, "#f6f7f8")
box(6, 4, 28, 7, "outputs/", "逐条 CSV + 逐场景图\ngitignore，可由脚本重建",
    C_OUT, fs_t=10.2, fs_b=8.8)
box(38, 4, 26, 7, "results/", "归档：指标 CSV\n+ 102 张图 + 代表时序",
    C_OUT, fs_t=10.2, fs_b=8.8)
box(68, 4, 26, 7, "docs/", "report.html（自包含）\n控制框图 / 仿真架构图",
    C_OUT, fs_t=10.2, fs_b=8.8)


# --------------------------------------------------------------------------- #
# 层间箭头
# --------------------------------------------------------------------------- #
for y0, y1 in ((86, 80), (66, 60), (29, 27), (15, 13)):
    poly([(50, y0), (50, y1)], color=GREY, lw=1.8)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, facecolor="white")
fig.savefig(OUT_SVG, facecolor="white")
print(f"written {OUT_PNG}")
print(f"written {OUT_SVG}")
