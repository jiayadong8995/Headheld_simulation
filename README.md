# assist_sim — 双滚刷手持洗地机·单侧半车一维助力控制仿真

> 研究对象：左右对称系统中的**单侧半车、单驱动电机**，只模拟前后方向一维平动。
> 目标：定量比较 **现有电流加速度助力算法 C1** 与 **基于速度/加速度反馈的虚拟阻抗助力算法 C2/C3**，
> 完成鲁棒性、参数整定与风险结论，交付可复现实验代码、曲线、参数推荐。

本仓库严格按《仿真任务书_交给 Herens.md》V1.0 实施，**不扩展到双侧差速/转向**，不建滚刷旋转动力学，
不引入位置刚度项；前后滚刷对平动的影响只以合成外力 `F_br` 表征。

---

## 1. 快速开始

```bash
cd /home/xbd/workspace/Handhell_Simulation
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # 首次

# M1 最小验证（理想对象解析检验 + 正反向）
.venv/bin/python run.py --milestone M1

# M2 完整对比（S01–S12 + 人机耦合任务，C0/C1/C2/C3）
.venv/bin/python run.py --milestone M2

# M3 参数推荐（单因素/网格扫参 + 300 次 Monte Carlo + 独立站立筛查）
.venv/bin/python run.py --milestone M3 --monte-carlo 300 --seed 20261012
.venv/bin/python run.py --standstill-screen 300 --seed 20261012 \
    --out outputs/M3_param_study        # 独立「站立不自走」安全筛查（§10-3）

# 贡献分离实验（速度反馈 / 加速度反馈 / 阻力补偿 各自的效果与代价）
.venv/bin/python contributions.py --out outputs/contributions

# 把输出汇总成报告用的 Markdown 数据表
.venv/bin/python analysis.py outputs/M2_full outputs/M3_param_study

# 单场景 / 全部场景
.venv/bin/python run.py --scenario S03 --controllers C0,C1,C2,C3 --config config/base.yaml
.venv/bin/python run.py --scenario all --controllers C0,C1,C2,C3
.venv/bin/python run.py --sweep ka_bd_alpha

# 单元测试（含真值隔离、符号镜像、停稳、不自走、可复现）
.venv/bin/python -m pytest -q
```

每次运行生成 `outputs/<时间戳_标签>/`，内含：

| 文件 | 说明 |
|---|---|
| `config_resolved.yaml` | 本次运行的完整解析配置 |
| `parameter_provenance.csv` | 每个参数的来源标注（SPEC / USER_MODEL / ASSUMED / MEASURED） |
| `env_fingerprint.json` | Python/NumPy/平台/git 版本，保证可追溯 |
| `scenarios_metrics.csv` | 每 (场景, 控制器) 一行的全指标 |
| `coupled_metrics.csv` | 人机耦合任务指标 |
| `timeseries/*.csv` | 每步真值/测量/指令时间序列 |
| `figures/*.png` | 规定 8 张曲线（叠加比较）+ 单控制器细节图 + 热图 |
| `sweep_metrics.csv`、`grid_*.csv` | 扫参与两因素网格 |
| `robustness_summary.csv`、`mc_*.csv` | Monte Carlo 汇总与配对比较 |
| `recommended_params.yaml` | Pareto 候选参数（3–5 组） |

---

## 2. 目录结构

```text
assist_sim/
  run.py                  # CLI 入口（--scenario/--sweep/--monte-carlo/--milestone）
  config/
    base.yaml             # 全部带单位参数 + 每个参数的来源标注（provenance）
    scenarios.yaml        # S01–S12 与耦合任务 T1–T3 定义
  sim/
    plant.py      单侧动力学、滚刷合力、平滑/粘滑摩擦（+真值接口，只给离线评价）
    actuator.py   力响应、饱和、速率限制、延迟队列、丢帧、看门狗、失能/急停
    sensors.py    IMU/轮速、噪声、偏置、倾斜投影、振动、量化、打滑、时间戳、失帧
    estimator.py  互补滤波（默认）/ 线性 Kalman（可选对照）
    controllers.py C0/C1/C2/C3 + C4a/C4b 上限对照 + C5 导纳对照 + 统一安全外壳
    human.py      固定 F_h 波形 + 闭环虚拟用户（反应延迟/速率限制/一阶滞后）
    metrics.py    统一评价口径（§8）
    scenarios.py  §6.3 采样顺序的运行器
    sweep.py      扫参/网格/Monte Carlo/Pareto
    plots.py      规定 8 图 + 叠加比较 + 热图
  tests/          6 组硬性检查
```

## 3. 控制律与仿真要素

| 项 | 实现 |
|---|---|
| 真实模型 | `ẋ=v, m·v̇=F_h+F_m-D(v,t), τ_F·Ḟ_m=sat(u)-F_m`，`D=F_res-F_br`（**不保证恒正**） |
| 目标阻抗 | `F_h=M_d·a+B_d·v`，`K_a=m̂-M_d` |
| C0 | `F_cmd=0`（物理基准） |
| C1 | `i=i_base+k·a` 的**力等效抽象版** `F_cmd=F_base(q,sgn v̂)+K_a,old·â`（占位系数，非实机复现） |
| C2 | `F_cmd=K_a·â-B_d·v̂` |
| C3 | C2 + `α_D·D̂(v̂)`，`D̂` 只用名义摩擦参数 + 滤波速度，零速附近死区+软切换 |
| C4a/C4b | 上限对照（真实 `a,v` / 真实 `D`），**不可部署**，不参与参数排名 |
| C5 | 导纳对照（`F̂_h` 逆推 / `use_true_force` 时标为 `C5_oracle_force`） |
| 安全外壳 | REST/START/MOVE/STOP(+REVERSE) 四阶段，速度+时间双滞回，死区/最大力/速率/超时归零/失能 |
| 执行器 | 保守额定转矩包络 `F_max=η·G·0.25 N·m / r = 6.25 N`（r=0.04, G=η=1），延迟、丢帧、看门狗真实生效 |
| 摩擦 | 平滑 Stribeck + 粘滑（零速锁定、滞回驻留、破静摩擦），支持 dry/wet/new_brush 中途切换 |
| 测量 | `a_imu=a+g·sinθ+b_a+n_a+e_vib`，`v_enc=(1-s)v+n_v+e_q`；时间戳 + 通信延迟 + 失帧 |
| 融合 | `v̂=(1-λ)(v̂+Ts(âu-b̂))+λ·v_enc`，`â` 经一阶低通（截止 10 Hz） |

**真值隔离**：`sim/controllers.py`、`estimator.py`、`sensors.py`、`actuator.py` 不得导入 `plant`，
也不得访问 `*_true` 字段；只有 `C4a/C4b/C5_oracle` 通过显式传入的 `oracle` 字典读真值，
由 `tests/test_no_privileged_truth.py` 静态扫描 + `sim/scenarios.py` 的显式传参双重保证。

## 4. 标准命令一览（§9.2）

```bash
python run.py --scenario S03 --controllers C0,C1,C2,C3 --config config/base.yaml
python run.py --scenario all --controllers C0,C1,C2,C3 --config config/base.yaml
python run.py --sweep ka_bd_alpha --config config/base.yaml
python run.py --monte-carlo 300 --config config/base.yaml --seed 20261012
pytest -q
```

## 5. 里程碑与结果索引

| 里程碑 | 内容 | 产物 |
|---|---|---|
| **M1** | S01/S02 + C0/C2，理想力对象解析验证 | `outputs/M1_*`（`figures/S01_overlay.png` 等） |
| **M2** | S01–S12 + C0/C1/C2/C3，真实摩擦/滚刷/限幅/融合/耦合任务 | `outputs/M2_full/`（`scenarios_metrics.csv`、`coupled_metrics.csv`） |
| **M3** | 扫参 + 网格 + 300 组 Monte Carlo + Pareto 推荐 | `outputs/M3_*/`（`recommended_params.yaml`、`robustness_summary.csv`） |

结论、图表解读与**待实机测量清单**见 `technical_report.md`。

## 6. 待实机测量清单（仿真无法替代）

| 量 | 当前状态 | 影响 |
|---|---|---|
| 单侧等效平动质量 `m` | ASSUMED 4.0 kg | `K_a=m̂-M_d` 的绝对值 |
| 驱动半径 `r`、传动比 `G`、效率 `η` | ASSUMED 0.04 m / 1 / 1 | 力-转矩换算与力上限 |
| `K_t` 或 `F/i_q` 映射 | 未标定（null） | 电流指令与力的关系 |
| 真实摩擦 `F_c/F_s/b`（干/湿/换刷） | ASSUMED | `α_D·D̂` 的补偿量与稳定性 |
| 滚刷合成偏置/脉动频率幅值 | ASSUMED | 是否会诱发窜动 |
| 力执行环时间常数 `τ_F` 与总延迟 | ASSUMED 0.04 s / 0.02 s | 加速度反馈的稳定边界 |
| 编码器速度分辨率/打滑、UART 实测 100 Hz 可达性 | SPEC 12 bit / 未知 | 状态估计质量与闭环调度 |
| 电机热限制与允许瞬态时长 | 未知 | 是否可短时用峰值转矩 |

## 7. 已知限制

- C1 的所有系数都是**占位默认值**，结论中的「现有方案」只是该抽象模型的代表，不是实机效果复现。
- ≥2 kHz 的采样只用于物理积分，控制环路为 100 Hz；真实 UART 一问一答能否稳定 100 Hz 需实测。
- 人手模型是假设，不是实测人手统计；结论依赖 §6.2 的探索分布。
- 未建模：俯仰动力学、滚刷旋转动力学、电池电压跌落、温度、车轮-地面非线性滑移。
