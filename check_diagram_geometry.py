#!/usr/bin/env python3
"""图件几何自检：验证两张框图"零正交交叉 / 无缺字 / 无越界"。

为什么需要它：靠肉眼（或视觉模型）判断"线条有没有交叉"不可靠 —— 实测中视觉模型
两次把没有交叉的地方判成交叉、把 0.6 单位间距判成压线。这里改成解析式判定：
复跑绘图脚本 → 抓取全部折线段与箭头段 → 对所有水平×竖直线段做严格相交测试。

用法：
    python check_diagram_geometry.py        # 退出码 0 = 全部通过
"""
from __future__ import annotations

import io
import os
import sys
import warnings
from contextlib import redirect_stdout

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp

DIAGRAMS = ("make_control_block_diagram.py", "make_simulation_architecture.py")
EPS = 1e-6


def _collect(path: str):
    """复跑脚本，返回 (线段列表, 轴对象, 缺字警告数)。"""
    ns = {"__name__": "__main__", "__file__": os.path.abspath(path)}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with redirect_stdout(io.StringIO()):
            exec(compile(open(path, encoding="utf-8").read(), path, "exec"), ns)
    missing = sum(1 for x in w if "missing from font" in str(x.message))
    ax = ns["ax"]
    segs = []
    for ln in ax.lines:
        xs, ys = list(ln.get_xdata()), list(ln.get_ydata())
        for i in range(len(xs) - 1):
            segs.append(((xs[i], ys[i]), (xs[i + 1], ys[i + 1])))
    for p in ax.patches:
        if isinstance(p, mp.FancyArrowPatch):
            a, b = p._posA_posB
            segs.append((tuple(a), tuple(b)))
    return segs, ax, missing


def _crossings(segs):
    """统计正交线段间的严格交叉（共享端点不算）。"""
    hits = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            (x1, y1), (x2, y2) = segs[i]
            (x3, y3), (x4, y4) = segs[j]
            h1, v1 = abs(y1 - y2) < EPS, abs(x1 - x2) < EPS
            h2, v2 = abs(y3 - y4) < EPS, abs(x3 - x4) < EPS
            if h1 and v2:
                if (min(x1, x2) + EPS < x3 < max(x1, x2) - EPS
                        and min(y3, y4) + EPS < y1 < max(y3, y4) - EPS):
                    hits.append((x3, y1))
            elif v1 and h2:
                if (min(x3, x4) + EPS < x1 < max(x3, x4) - EPS
                        and min(y1, y2) + EPS < y3 < max(y1, y2) - EPS):
                    hits.append((x1, y3))
    return hits


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    ok = True
    for name in DIAGRAMS:
        path = os.path.join(root, name)
        segs, ax, missing = _collect(path)
        pts = [q for s in segs for q in s]
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        # 越界检查（包围盒必须落在坐标轴范围内）
        oob = [p for p in pts if not (xlo <= p[0] <= xhi and ylo <= p[1] <= yhi)]
        cross = _crossings(segs)
        status = "PASS" if (not cross and missing == 0 and not oob) else "FAIL"
        ok &= status == "PASS"
        print(f"[{status}] {name}")
        print(f"        线段 {len(segs)} 段 | 正交交叉 {len(cross)} 处 | "
              f"缺字警告 {missing} 条 | 越界点 {len(oob)} 个")
        for pt in cross:
            print(f"        交叉点 ({pt[0]:.2f}, {pt[1]:.2f})")
        for pt in oob:
            print(f"        越界点 ({pt[0]:.2f}, {pt[1]:.2f})")
    print("\n结论：" + ("全部通过 ✓" if ok else "存在不合格项 ✗"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
