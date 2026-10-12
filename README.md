# assist_sim — 双滚刷手持洗地机·单侧半车一维助力控制仿真

[![tests](https://img.shields.io/badge/tests-27%20passed-brightgreen)](#)　[![python](https://img.shields.io/badge/python-3.10%2B-blue)](#)

> 研究对象：左右对称系统中的**单侧半车、单驱动电机**，只模拟前后方向一维平动。
> 目标：定量比较 **现有电流加速度助力算法 C1** 与 **基于速度/加速度反馈的虚拟阻抗助力算法 C2/C3**，
> 完成鲁棒性、参数整定与风险结论，交付可复现实验代码、曲线、参数推荐。
>
> 📄 **[算法与验证总结（单文档总览）](ALGORITHM_AND_VALIDATION.md)** ｜
> 📊 **[自包含 HTML 报告](docs/report.html)**（下载后用浏览器打开）｜
> 📐 `docs/architecture.png` 控制架构框图（矢量版 [`docs/architecture.svg`](docs/architecture.svg)）｜
> 📋 [详细技术报告（逐条回答任务书 7 个问题）](technical_report.md)

本仓库严格按《仿真任务书_交给 Herens.md》V1.0 实施（任务书原件归档于 `docs/task_book/`），
**不扩展到双侧差速/转向**，不建滚刷旋转动力学，不引入位置刚度项；前后滚刷对平动的影响只以合成外力 `F_br` 表征。

| 结论速览（基于 300 组 Monte Carlo + 4800 次安全筛查） |
|---|
| 速度阻尼 `-B_d·v̂` 是**停稳能力的唯一来源**：松手滑行 0.746 m → **0.036 m**，停稳 2.61 s → **1.23 s** |
| 加速度反馈 `K_a·â` **单独使用是净负收益**（停不稳 + 更费力 + 电机更累），必须与阻尼成对使用 |
| 阻力补偿 `α_D·D̂` 是**唯一能把摩擦负担转移给电机的项**：`α_D=0.5` 时 T2 任务手力比传统算法优 **9.2 %** |
| `α_D=1.0` 会**过补偿**：换刷后松手滑行放大 7 倍，MC 最差样本 1.74 m |
| **推荐主参数**：`C3，K_a=0.8 kg（K_a/m=0.20）、B_d=4.0 N·s/m、α_D=0.5`（手力 +1.2 %、电机 0.267 N RMS、零低速振荡） |
| **诚实提示**：300 组随机工况下最优组仅 **63 % 样本优于传统算法**（均值 +1.2 %，尾部 −44 %）→ 仿真不支持"装上就一定省力"，须先实机标定 |


---

## 1. 快速开始

```bash
cd /home/xbd/workspace/Headheld_simulation
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

# 生成自包含 HTML 报告（图片 base64 内嵌，单文件可离线打开/打印）
.venv/bin/python make_report_html.py            # → outputs/report.html

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
Headheld_simulation/
  ALGORITHM_AND_VALIDATION.md   # ⭐ 算法与验证总结（单文档总览：模型/控制律/安全壳/全部结果）
  technical_report.md           # 详细技术报告（逐条回答任务书 7 个问题 + 风险边界）
  run.py                        # CLI：--scenario / --milestone M1|M2|M3 / --sweep / --monte-carlo / --standstill-screen
  analysis.py                   # 输出 → 报告用 Markdown 数据表
  contributions.py              # 贡献分离实验（速度阻尼 / 加速度反馈 / 阻力补偿）
  make_report_html.py           # 生成自包含 HTML 报告（图片 base64 内嵌）
  make_architecture_diagram.py  # 生成控制架构框图 docs/architecture.png + .svg
  config/
    base.yaml                   # 全部带单位参数 + 来源标注（provenance）
    scenarios.yaml              # S01–S12 与耦合任务 T1–T3 定义
  sim/
    plant.py       单侧动力学、滚刷合力、平滑/粘滑摩擦（真值接口只给离线评价）
    actuator.py    力包络、一阶响应、速率限制、延迟队列、丢帧、看门狗、失能/急停
    sensors.py     IMU/轮速：噪声、偏置、倾斜投影、振动、量化、打滑、时间戳、失帧
    estimator.py   互补滤波（默认）/ 线性 Kalman（可选对照）
    controllers.py C0/C1/C2/C3 + C4a/C4b 上限对照 + C5 导纳对照 + 统一安全外壳
    human.py       固定 F_h 波形 + 闭环虚拟用户（反应延迟/速率限制/一阶滞后）
    metrics.py     统一评价口径（任务书 §8）
    scenarios.py   §6.3 采样顺序的运行器
    sweep.py       扫参/网格/Monte Carlo/Pareto/站立安全筛查
    plots.py       规定 ≥8 图 + 控制器叠加比较 + 热图
  tests/                        # 6 组硬性检查（27 个用例）
  docs/
    architecture.png / .svg     # 控制与验证架构框图（位图 / 矢量）
    report.html                 # 自包含 HTML 报告
    task_book/                  # 任务书原件归档（任务书 / 初始配置 / 执行指令）
  results/                      # 归档的仿真产物（指标 CSV + 102 张图 + 代表时间序列）
  outputs/                      # 每次运行的原始工作目录（gitignore，可重建）
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
| **M1** | S01/S02 + C0/C2，理想力对象解析验证（`M_eff=2.591` vs 理论 2.6、`B_eff=3.978` vs 4.0、`RMSE_v/v_ref=0.73 % < 2 %`、正反精确镜像） | `pytest -q`（27 passed） |
| **M2** | S01–S12 + C0/C1/C2/C3，真实摩擦/滚刷/限幅/融合 + 耦合任务 T1/T2 × 3 组虚拟人（96 次耦合试验） | `outputs/M2_full/`（`scenarios_metrics.csv`、`coupled_metrics.csv`、84 张图） |
| **M3** | 单因素/两因素扫参 + **300 组 Monte Carlo**（配对样本）+ **4800 次站立安全筛查** + Pareto 推荐 | `outputs/M3_param_study/`（`recommended_params.yaml`、`robustness_summary.csv`、`mc_standstill_summary.csv`） |
| 附加 | 贡献分离实验（速度阻尼 / 加速度反馈 / 阻力补偿的净效果） | `outputs/contributions/ablation_metrics.csv` |

**结论速览**（完整推导与逐条回答 7 个问题见 `technical_report.md`）：

- 速度阻尼 `-B_d·v̂` 是**停稳能力的唯一来源**（松手滑行 0.75–0.89 m → 0.036–0.12 m，停稳 2.6 s → 1.2–1.4 s），
  代价是恒定速度段**人手多付 1–1.5 N**（耦合任务 RMS 手力 +20～28 %）；
- 加速度反馈 `K_a·â` **单独使用净负收益**（停不稳 + 更费力 + 电机更累），必须与阻尼成对使用；
- 阻力补偿 `α_D·D̂` 是**唯一能把这部分代价还给人手**的项：`α_D=0.5` 时 T2 任务手力比 C1 优 9 %，
  但 `α_D=1.0` 会过补偿（松手滑行放大到最差 1.7–2.0 m、T1 停稳回到 2.72 s）；
- **推荐主参数**：`C3，K_a=0.8 kg（K_a/m=0.20），B_d=4.0 N·s/m，α_D=0.5`（首选平衡）；
  `C2，K_a=1.4/B_d=4.0`（安全优先/降级模式，停稳最强但手力 −20 %）；
- **诚实结论**：在全部为 ASSUMED 参数、300 组随机工况下，最优参数组仅 **63 % 的样本优于 C1（均值 +1.2 %，尾部 −44 %）**
  —— 仿真**不支持"装上就一定省力"**，上机前必须按第 7 节清单完成标定并重跑 M3（一条命令，约 15 min）。

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
