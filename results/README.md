# results/ — 归档的仿真产物

本目录是 `outputs/` 的**归档快照**：跑完 M1/M2/M3 后从工作目录拷贝而来（原始逐条时间序列有 200+ MB，
只保留 6 个代表性样本的 `.csv.gz`，其余可用一条命令重跑生成）。

| 子目录 | 内容 | 生成命令 |
|---|---|---|
| `metrics/` | 全部指标表（场景/耦合/扫参/网格/MC/安全筛查/ablation）与 `recommended_params.yaml` | `run.py --milestone M2` / `--milestone M3 --monte-carlo 300` / `--standstill-screen 300` / `contributions.py` |
| `figures/` | 102 张曲线与热图，文件名前缀标示来源：`M2__` / `M3__` / `ablation__` | 同上 |
| `timeseries_samples/` | 6 个代表性逐步时间序列（gzip，含真值/测量/指令/状态机） | 上面的命令会写全量到 `outputs/*/timeseries/` |

**运行条件**：`seed = 20261012`，`dt_plant = 0.001 s`，`T_control = 0.01 s`，
代码提交见 `git log`（产物由该提交的代码生成，`outputs/*/env_fingerprint.json` 记录 Python/NumPy/平台版本）。

**重要**：所有物理参数（质量、半径、摩擦、`τ_F`、延迟等）均为 **ASSUMED 占位值**，不是样机实测；
力上限 6.25 N 由规格书额定转矩 0.25 N·m 与假设的 r=0.04 m 换算得到。上机前请先完成
`ALGORITHM_AND_VALIDATION.md` §10 的实测清单，再重跑一遍（约 15 min）。

指标表列名含义见 `sim/metrics.py` 的 `compute_metrics()`；其中 `improve_vs_C1` 为同一随机样本下
相对传统算法（C1 占位实现）的 RMS 手力改善比（正 = 更省力）。
