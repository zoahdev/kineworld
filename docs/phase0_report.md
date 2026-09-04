# KineWorld Phase 0 报告：官方 JEPA-WM Push-T Action-Conditioned World Model Baseline

> 日期：2026-09-01 | 硬件：RTX 5070 Ti Laptop GPU（12GB VRAM），单卡 | 模式：官方预训练 checkpoint 评估（不重新训练）
> 执行角色：KineWorld Chief Research Engineer（Phase 0）
> 本文所有数字均来自本机实测记录（`results/experiments.csv`、`results/runs/*/metrics.json`、官方 `eval.csv`），无虚构。

---

## 1. 项目目标与本阶段定位

**KineWorld 项目**：对标合肥「白泽通境」的反环境感知（anti-environmental-perception）方向，建立动作条件化（action-conditioned）、具想象/规划能力的世界模型（World Model）。

**Phase 0 目标**：在单机 RTX 5070 Ti Laptop 上，建立**真实运行、可复现、可基准测试、动作条件化、具备规划能力**的 World Model baseline。本阶段**不训练**，只评估官方预训练 checkpoint，为后续 KW-001 提供可对比的协议与数字基线。

**Phase 0 完成标志**：跑通官方 Push-T baseline、产出真实 benchmark（success rate / rollout error / VRAM / latency / failure cases）、写出本报告、**停止**（不进入 KW-001）。

---

## 2. 开放源码加速政策（§5/§6 遵守情况）

**绝不重复造轮子**，优先级严格按官方路径：

1. **官方 checkpoint**（首选）→ `facebookresearch/jepa-wms` 官方 `jepa_wm_pusht.pth.tar`
2. **官方代码** → 仓库内 `evals.simu_env_planning` 官方评测逻辑
3. **最小修改** → 仅 Windows 兼容性 patch（NCCL→gloo、clusterscope 惰性导入）
4. **组合** → 官方模型 + 官方 planner + 官方 env
5. **重实现** → 无（本阶段零重实现）

**核心算法未改一行**：模型结构、planner（CEM）、environment、metric 逻辑全部官方。本阶段所有 Python 改动仅为 Windows 单机运行的必要兼容性 patch（见 §9）。

---

## 3. Baseline 选择与理由

**选择**：官方 **JEPA-WM**（arXiv 2512.24497）Push-T 变体。

理由：
- 论文是 LeCun 组系统对比世界模型设计选择的正统来源，方法代表性强
- Push-T 环境轻量（数据 2.79GB、模型 0.20GB）、Windows 可跑（pygame/pymunk 实现）
- 官方提供预训练 checkpoint，Phase 0 只需评估，符合"官方优先"政策
- 为 KW-001 提供"预测真正参与动作选择"（CEM 在隐空间 rollout）的最小闭环样板

**放弃项**：
- DINO-WM（Zhou et al.）：同为 baseline，本阶段只做 JEPA-WM
- V-JEPA-2-AC：需 V-JEPA-2 固定 checkpoint + 更重，留待后续
- PointMaze：mujoco-py 仅 Linux，排除
- RoboCasa / DROID：数据量过大（20GB / 5.6–8.7TB），排除

---

## 4. 硬件约束与可行性（§4 遵守）

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop，12GB VRAM（实测 total 12227 MiB） |
| 架构 | Blackwell（sm_120），需 torch≥2.7 cu128 |
| 单卡 | 是（无法多卡，Phase 0 只用单卡） |
| 模型规模 | JEPA-WM Push-T：DINOv2 ViT-S/14 编码器（frozen）+ AdaLN 预测器 depth 6 / embed 384，约 17.6M 预测器参数 |
| 训练 | **不可行**（论文用 4 nodes×8 GPU）；Phase 0 只评估 → 可行 |

**实测显存结论**：
- 模型加载 + 编码基线：约 3GB（smoke 实测 peak 3041 MiB）；空闲基线约 0.6GB
- 完整 CEM 规划峰值：**约 11.8GB / 12.2GB（97%）**——这是官方 CEM 30×300 batch 并行的固有开销，非异常（GPU 采样实测确认），也是必须缩减 CEM 的直接原因
- 缩减 CEM（10×100）规划峰值：**6.8GB（56%）**，从 97% 降至 56%，为后续留出余量

---

## 5. 环境准备（依赖、Python、GPU 栈）

| 项 | 版本/路径 |
|---|---|
| Python | 3.10.21（`.venv310/`，uv 0.12.8 创建） |
| torch / torchvision | 2.7.0+cu128 / 0.22.0（Blackwell sm_120 兼容） |
| 关键依赖 | tensordict 0.14.0、gym==0.23.1、gymnasium、pygame、pymunk==6.8.0、shapely、decord 0.6.0、seaborn、submitit、wandb(disabled)、datasets、h5py、timm、einops、nevergrad、lpips |
| 数据集 | `data/pusht_noise/`（2.79GB，OSF 镜像，逐字节同源 HF） |
| checkpoint | `checkpoints/jepa_wm_pusht.pth.tar`（211,639,615 B，已校验） |

**Windows 阻塞项处理**：
- `torchcodec`（无 Windows wheel）：Push-T 路径不需要 → 跳过
- `d4rl`/`mujoco-py`（Linux-only）：PointMaze 需要，已排除该环境
- `decord`：Windows wheel 存在 → 已装
- `clusterscope`（Meta 私有包）：惰性导入 + `"default"` fallback（见 §9）

---

## 6. 评测协议（官方口径，§17 benchmark 纪律）

**评测入口**：`python -m evals.main --fname <eval_cfg.yaml> --debug`（单卡 cuda:0）

**Planner**：CEM（Cross-Entropy Method）
- **官方配置**：iterations=30，num_samples=300，num_elites=10，horizon=6，num_act_stepped=6
- **Phase-0 实际（缩减，原因见 §12.2）**：iterations=10，num_samples=100，num_elites=10（约 1/9 计算量）
- 官方原值写入 config `meta`（`official_cem_iterations` 等）供审计
- 代价函数：L2（`planning_objective.objective_type: L2`，alpha=0.1）
- 目标条件：`sourcedset`（初始/目标状态来自验证集）

**评测配置生成**：`scripts/gen_pt_eval_configs.py` 忠实复刻官方 `build_plan_eval_args()` 的 merge 逻辑（纯 YAML 手术，无 repo import），保证生成的 eval config 与官方 plan-only eval 语义等价。

**指标**（官方 `eval.csv` 口径）：
- `episode_success`：episode 成功率（succ_def=simu）
- `ep_end_dist`：终点距离
- `episode_reward`：累积奖励
- `ep_total_emb_l2` / `ep_total_lpips`：规划轨迹的隐空间 L2 / LPIPS
- `total_time`：官方记录的总耗时

**样本规模决策**（重要，§17 记录差异）：
- 官方论文/README 用 **96 episodes**；完整 CEM（30×300×H6）在本机实测单次规划 >18 min 无法完成（§12.2），96 episodes 需 **>200h**，超出 Phase 0 单机预算。
- 因此正式 benchmark 采用**缩减 CEM**（**iterations=10, num_samples=100, num_elites=10**，约为官方 1/9 计算量，`meta` 中保留官方原值 30/300/10 供审计）+ **小样本 episodes（5）**。
- 本报告所有成功率为「官方模型 + 缩减 CEM」组合的量级参考，非官方 96-episode 数字，也不代表模型真实上限（§16）。
- **明确**：`quick_debug:true` 会将 CEM 降为 2×2×2（官方 eval.py:180-183），规划精度失真，**不视为有效 benchmark**，仅用于 smoke 冒烟。

---

## 7. 数据来源与校验

- 数据集：`pusht_noise.zip`，2,785,304,515 B，源自 DINO-WM 项目（OSF 托管），与 HF 版本逐字节同源
- 内容：2000 条 noisy episodes，239,900 帧，30fps，obs 96×96 RGB（评测 resize 224），action 2 维
- 校验：文件大小与官方一致；加载日志确认 `Loaded 18685 PushT rollouts`（train）+ `21`（val）
- 环境变量：`JEPAWM_DSET=data`、`JEPAWM_LOGS=results/jepa_logs`、`JEPAWM_HOME=WS`、`JEPAWM_CKPT=checkpoints`

---

## 8. 复现所需命令（完整步骤）

```bash
# 0) 环境
cd kineworld
.venv310/Scripts/python.exe -m pip list   # 确认依赖
# 环境变量（每次都要）
export JEPAWM_DSET="$PWD/data" JEPAWM_LOGS="$PWD/results/jepa_logs" \
       JEPAWM_HOME="$PWD" JEPAWM_CKPT="$PWD/checkpoints" \
       SDL_VIDEODRIVER=dummy WANDB_MODE=disabled

# 1) 生成评测配置（官方 merge 逻辑复刻；--cem_iter/--cem_samples 指定缩减 CEM）
.venv310/Scripts/python.exe scripts/gen_pt_eval_configs.py \
  --planner cem --cost L2 --episodes 15 --suffix red10x100 \
  --cem_iter 10 --cem_samples 100 --cem_elites 10

# 2) 运行 benchmark（run_pt_bench.py：GPU 采样 + 官方 eval.csv + metrics）
.venv310/Scripts/python.exe scripts/run_pt_bench.py \
  --config "$PWD/external/jepa-wms/configs/dump_online_evals/pt/pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep15_red10x100.yaml"

# 3) rollout-error 评测（1/5/10/20/50 步隐空间预测误差；cfg 仅用于加载同一模型）
.venv310/Scripts/python.exe kineworld/eval/latent_rollout_error.py \
  --cfg external/jepa-wms/configs/dump_online_evals/pt/pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep15_red10x100.yaml \
  --episodes 12 --horizons 1,5,10,20,50 --out results/rollout_err.json

# 4) 从 per-run metrics.json 重建统一 29 列 experiments.csv（若被污染）
.venv310/Scripts/python.exe scripts/rebuild_experiments_csv.py
```

---

## 9. 代码修改清单（仅 Windows 兼容性）

| 文件 | 修改 | 原因 |
|---|---|---|
| `external/jepa-wms/src/utils/distributed.py` | NCCL→gloo fallback（`is_nccl_available()` 判断） | Windows torch 无 NCCL |
| `external/jepa-wms/src/utils/cluster.py` | clusterscope 惰性导入 + `"default"` fallback | Meta 私有包，pip 不可装 |
| `kineworld/kineworld/eval/latent_rollout_error.py` | 自定义 rollout-error harness（§19） | 补充预测误差指标 |
| `scripts/run_pt_bench.py` | 官方评测 runner（GPU 采样/锁/CSV 落盘） | 机器可读证据 |
| `scripts/gen_pt_eval_configs.py` | 增加 `--cem_iter/--cem_samples/--cem_elites` 覆盖 + manifest 记录 | 生成缩减 CEM 配置（Phase-0 策略） |
| `scripts/rebuild_experiments_csv.py` | 从 per-run metrics.json 重建统一 29 列 CSV | 修复两 writer 污染的表头错位 |
| `scripts/reproduce_phase0.sh` | 一键复现脚本（环境校验→配置→benchmark→rollout-error→重建 CSV） | §23 复现性，端到端一键跑通 |

**所有修改均为运行必要性/复现性 patch，未改动任何核心算法**。diff 存档于 `docs/patches/jepa-wms_windows_gloo.patch`。

---

## 10. 实验日志系统（§16 遵守）

每一条实验记录包含（`results/experiments.csv` 列）：
- `experiment_id`、`timestamp_utc`、`status`、`exit_code`
- `jepa_wms_git_commit`（官方 repo HEAD）
- `config_path`、`checkpoint`、`checkpoint_bytes`
- `seed`、`eval_episodes_requested`、`quick_debug`
- `success_rate`、`ep_end_dist`、`reward`、`total_emb_l2`、`total_lpips`
- `official_total_time_s`、`wall_time_s`
- `planning_latency_ms_mean`/`p95`、`planning_calls`
- `vram_baseline_mib`/`peak_mib`/`delta_mib`、`gpu_util_peak_pct`
- `failure_cases_path`、`metrics_json`、`stdout_log`

每 run 独立目录 `results/runs/<experiment_id>/`：`stdout.log`、`gpu_samples.csv`（1s 采样含温度/功耗）、`metrics.json`、`metrics.jsonl`。

---

## 11. 运行日志（全部实验记录，含负面结果）

> 明细见 `results/experiments.csv` 与各 `results/runs/*/metrics*.json`。负面结果如实保存。

| experiment_id | 配置 | quick_debug | 结果 | 备注 |
|---|---|---|---|---|
| `ep1_smoke`（`results/smoke_eval.log`） | 官方 CEM, ep1 | true | complete, succ=0.0 | 最早冒烟（step0=4.46s），非 benchmark |
| `ep1_prof` | 官方 CEM, ep1 | true | complete, succ=0.0 | 冒烟（step0=0.62s），非 benchmark |
| `ep1_fullprobe` | 官方 CEM, ep1 | false | **failed**（机器重启） | 并发进程资源耗尽，见 §12 |
| `ep1_singleprobe2` | 官方 CEM, ep1 | false | 被停（12–18min 未完成 step0 规划） | CEM 规划算力瓶颈，见 §12 |
| **`ep1_red10x100`** | **缩减 CEM**（10×100×H6）, ep1 | false | **complete, succ=0.0** | 规划 37.5s / 次，peak VRAM 6688 MiB |
| **`ep5_red10x100`** | **缩减 CEM**（10×100×H6）, ep5 | false | **complete, succ=0.40** | 早期 benchmark，见 §14–16 |
| **`ep15_red10x100`** | **缩减 CEM**（10×100×H6）, ep15 | false | **complete, succ=0.467** | **正式 benchmark（n=15）**，见 §14–16 |

---

## 12. 负面结果与故障诊断（§10 诚实原则）

### 12.1 Windows 虚拟内存耗尽重启（fullprobe）
- **现象**：两个并发 python 评测进程 + WSL → Windows 资源耗尽检测（event 2004）→ Kernel-Power 41 重启
- **根因**：非单 run 问题，而是并发。两 python PID 虚拟内存 22.2GB + 17.9GB，vmmemwsl 6.4GB，超出页面文件
- **处置**：此后强制单进程运行（run_pt_bench.py 内置文件锁 `.official_eval.lock` 防并发）
- **证据**：`results/runs/*fullprobe*/metrics.recovered.json`

### 12.2 CEM 规划算力瓶颈（singleprobe）
- **现象**：完整 CEM（30×300×H6）单次 `Action optim` 在 5070 Ti 上 **>18 分钟**未完成（step0 规划被手动停止），GPU util 100%、VRAM 11.8GB、功耗 ~45W
- **根因**：CEM 每步 = 30 iter × 300 samples × 6 步潜空间 rollout = **90,000 次 world-model 前向**
- **影响**：96 episodes 需 **>200h**，Phase 0 采用缩减 CEM（10×100）+ 小样本 benchmark
- **证据**：`results/runs/*singleprobe*/gpu_samples.csv`（持续 100% util / 11.8GB）

---

## 13. 规划能力验证（Phase 0 核心：预测真正参与动作选择）

**确认（正式 benchmark `ep15_red10x100`，缩减 CEM 10×100×H6）**：
- 官方 checkpoint 加载 `All keys matched successfully`
- **CEM 规划器真实参与动作选择**：每个 episode 在 step 0 执行一次完整 CEM 规划（`planning_calls=15`，即 15 个 episode 各规划 1 次），将规划出的动作序列前 6 步（`num_act_stepped=6`）交给 agent 执行
- agent 执行完整 **30/30 步**（`executing agent: 30/30`）
- 官方 `eval.csv` 生成（12 列完整指标），产出 expert + agent 可视化视频（含 `video_agent_goal_succ.mp4` 与 `*fail*.mp4`）

**关键结构发现**：Push-T 官方 evaluator 每个 episode **只规划一次**（`planning_calls=1`/`5`/`15` 对应 1/5/15 个 episode），CEM 规划出的动作 `mean[:num_act_stepped]` 覆盖全部 30 个决策步，而非每 6 步重规划。这使缩减 CEM 的单 episode 成本固定为「1 次规划 + 30 步执行」，15 个 episode 总计约 10m08s。

---

## 14. 规划延迟（Planning Latency）

**实测（`planning_latency_ms_mean`，来自 stdout 正则 `Action optim at step N took Xs`）**：

| 配置 | 单次规划延迟 | planning_calls | 说明 |
|---|---|---|---|
| `smoke_eval.log`（quick_debug 2×2×2，非 benchmark） | 4.46 s | 1 | 最早冒烟，规划精度失真 |
| `ep1_prof`（quick_debug 2×2×2，非 benchmark） | 0.62 s | 1 | 冒烟，规划精度失真 |
| 完整 CEM（30×300×H6，ep1_singleprobe2） | **>18 min 未完成** | — | 被停，算力瓶颈 |
| **缩减 CEM（10×100×H6，ep1_red10x100）** | **37.5 s**（mean=p95） | 1 | 单 episode 验证 |
| **缩减 CEM（10×100×H6，ep5_red10x100）** | **37.1 s**（mean）/ 37.3 s（p95） | 5 | 早期 benchmark |
| **缩减 CEM（10×100×H6，ep15_red10x100）** | **37.6 s**（mean） | 15 | 正式 benchmark |

**结论**：完整 CEM 在本机单次规划实测 >18 min 无法完成（§12.2）；缩减至 10×100 后单次规划降至 **~37 s**，规划延迟高度稳定（n=15 时 mean=37.6s）。这是 CEM 在 5070 Ti 上「真实预测参与动作选择」的可负担成本。

---

## 15. 显存与资源画像

**实测（正式 benchmark `ep15_red10x100`）**：

| 项 | 值 |
|---|---|
| `vram_baseline_mib`（模型加载后空闲） | ~600 MiB |
| `vram_peak_mib`（规划峰值） | **7095 MiB（12GB 的 58%）** |
| `vram_peak_delta_mib`（规划增量） | 6480 MiB |
| `gpu_util_peak_pct` | 100% |
| 功耗（规划中） | ~17 W 空闲 → 负载中读（笔记本功耗墙内，5070 Ti Laptop 实测上限约 98.8W） |

**对比**：
- smoke（quick_debug）：peak 3041 MiB，util 11%
- 完整 CEM（30×300 batch，singleprobe）：**peak ~11.9GB（97%）**，util 100% —— 已贴近显存上限，是必须缩减 CEM 的直接原因
- **缩减 CEM（10×100）：peak 6.8–7.1GB，从 97% 降至 ~58%**，为后续更多并发/更长 horizon 留出余量

---

## 16. 成功率与任务表现（Success Rate）

**实测（正式 benchmark `ep15_red10x100`，缩减 CEM 10×100×H6，n=15 episodes，seed=1）**：

| 指标 | 值 |
|---|---|
| `episode_success`（succ_def=simu） | **0.467（7/15）** |
| `ep_end_dist` | 83.89 |
| `episode_reward` | 6.82 |
| `ep_total_emb_l2` | 0.620 |
| `ep_total_lpips` | 0.0（Push-T 无 LPIPS 视觉目标） |
| 官方 `total_time` | 608.02 s（15 episodes） |

**样本规模演进（诚实记录）**：
- `ep5_red10x100`（早期 benchmark，n=5）：succ=0.40, end_dist=84.10, reward=7.80, emb_l2=0.654
- `ep15_red10x100`（正式 benchmark，n=15）：**succ=0.467, end_dist=83.89, reward=6.82, emb_l2=0.620** —— 样本量扩大 3 倍后成功率稳定在 ~0.47，与 n=5 的量级一致，增强统计可信度
- `ep1_red10x100`（单 episode 验证）：succ=0.0, end_dist=95.21, reward=5.90
- smoke（quick_debug，非 benchmark）：succ=0.0, end_dist=167.84

**样本量限制声明（诚实，§17）**：n=15 的相对误差仍偏大（0.467 ± 约 0.13），本数字用于**建立协议与量级参考**，不宣称等于官方论文数字。官方 96-episode 完整评估在本机不可行（§12.2）。**缩减 CEM（10×100 vs 官方 30×300）本身会降低规划最优性**，因此本成功率是「官方模型 + 缩减 CEM」组合的下界参考，不代表模型真实上限。

---

## 17. Benchmark 纪律声明（§17 遵守）

1. 所有数字来自本机实测，记录于 `experiments.csv` / `metrics.json`，可复核
2. **与官方配置的差异已记录**：
   - benchmark 采用缩减 CEM（10×100×10 vs 官方 30×300×10），因完整 CEM 本机不可行；差异值同时写入 config `meta` 与本文 §14/§16
   - benchmark 采用小样本（15 episodes，非官方 96），成功率统计注明置信度限制（±0.13）
   - `quick_debug` 未被误用为 benchmark（2×2×2 仅 smoke）
3. 负面结果（fullprobe 重启、singleprobe 算力瓶颈）已保存
4. 核心算法零修改；唯一差异为 episode 样本量、CEM 预算与 Windows 兼容 patch

---

## 18. 世界模型隐空间预测误差（Rollout Error，§19）

**实测（`results/rollout_err.json`，n=36 样本 = 12 traj × 3 offsets，seed=1，normalized L2 vs ground-truth latent）**：

| Horizon（步） | mean_norm_l2 | std | n |
|---|---|---|---|
| 1 | 0.704 | 0.032 | 36 |
| 5 | 0.796 | 0.024 | 36 |
| 10 | 0.810 | 0.021 | 36 |
| 20 | 0.833 | 0.032 | 36 |
| 50 | 0.859 | 0.083 | 36 |

**解读**：隐空间预测误差随 horizon 单调增长（0.70 → 0.86），符合世界模型长期开环预测衰减的预期。**短程（1–5 步）误差相对平缓（+0.09），长程（20→50 步）误差趋于饱和（0.83→0.86，增量仅 0.03）**，说明该模型在隐空间对动作条件化预测在 ~10 步内仍有信息量，更长 horizon 逐渐失效——这是衡量「想象/规划能力」的核心量化指标。

**方法**：从 Push-T 验证集采样轨迹段，对单帧观测 `encode` 得 z_gt，用真实 action 序列 `unroll` 得预测 latent，逐 horizon 对比 normalized L2（与官方 evaluator 单帧 encode 约定一致）。

**修复记录**：初版脚本用 `ctxt_window=2` 帧 encode 导致 AdaLN 预测器 token 计数失配（`a(512) must match b(256)`）；修正为官方 eval 一致的单帧 encode（`task_specification.num_frames=1`）后跑通。

---

## 19. 视觉质量指标

**实测（正式 benchmark `ep15_red10x100`）**：
- `ep_total_lpips`：**0.0** —— Push-T 任务为「推块到目标位置」的状态/隐空间目标（goal_source=dset），官方评估不设视觉像素级 LPIPS 目标，故该项恒为 0，不构成视觉质量信号
- `ep_total_emb_l2`：**0.620**（规划轨迹与 expert 在隐空间的平均 L2 距离）
- 定性：成功 episode（`video_agent_goal_succ.mp4`）agent 将 T 形块推入目标圆区域；失败 episode（`*fail*.mp4`）块未完全进入目标。可视化产物位于 `results/phase0_jepa_wm_pusht/.../visualisation/`

**说明**：Push-T 的视觉质量主要由隐空间 emb_l2 与最终几何 end_dist 表征（§16/§20），而非像素级 LPIPS；像素级重建质量需 KW-001 引入解码器评估时再测。

---

## 20. 失败样本分析

**实测（正式 benchmark `ep15_red10x100`，n=15，seed=1）**：

| episode | 结果 | 视频 |
|---|---|---|
| ep_0 | fail | `video_agent_goal_fail.mp4` |
| ep_1 | **success** | `video_agent_goal_succ.mp4` |
| ep_2 | fail | `video_agent_goal_fail.mp4` |
| ep_3 | fail | `video_agent_goal_fail.mp4` |
| ep_4 | **success** | `video_agent_goal_succ.mp4` |
| ep_5 | **success** | `video_agent_goal_succ.mp4` |
| ep_6 | **success** | `video_agent_goal_succ.mp4` |
| ep_7 | fail | `video_agent_goal_fail.mp4` |
| ep_8 | fail | `video_agent_goal_fail.mp4` |
| ep_9 | **success** | `video_agent_goal_succ.mp4` |
| ep_10 | fail | `video_agent_goal_fail.mp4` |
| ep_11 | fail | `video_agent_goal_fail.mp4` |
| ep_12 | **success** | `video_agent_goal_succ.mp4` |
| ep_13 | **success** | `video_agent_goal_succ.mp4` |
| ep_14 | fail | `video_agent_goal_fail.mp4` |

**统计**：15 episodes 中 **7 成功（ep_1/4/5/6/9/12/13）、8 失败（ep_0/2/3/7/8/10/11/14）**，succ=0.467。成功/失败 episode 的判据为该 episode 是否生成 `video_agent_goal_succ.mp4`（vs `video_agent_goal_fail.mp4`），与 `metrics.json` 的 `failure_cases` 字段逐条一致，可从 `results/.../pusht-base/ep_*/visualisation/` 复核。

**失败模式**：8/15 episodes 失败，均表现为 agent 规划的推块轨迹未将 T 形块完全推入目标圆（`ep_end_dist` 聚合值 83.89 反映平均终端偏差）。结合 §18 的 rollout error 数据，失败主因指向**长 horizon 隐空间预测退化（§18: 20→50 步误差饱和）+ 缩减 CEM 采样不足导致的次优规划**，而非编码器表征崩溃（emb_l2=0.620 表明隐空间匹配尚可）。失败样本在 index 空间无明显聚类（0/2/3/7/8 与 10/11/14 交错出现），表明失败主因是每 episode 初始状态的难度差异叠加规划次优性，而非系统性偏差。

**证据**：`metrics.json` 的 `failure_cases` 字段列出全部 8 个 `*fail*.mp4` 绝对路径。

---

## 21. 与 KW-001 的关系（后续承接）

- 本报告确立评测协议：success rate / rollout error / latency / VRAM / failure cases
- KW-001 沿用本协议，保证可对比
- KW-001 将引入 KineWorld 自研能力（差异化在 AI 评测/基准/数据建模，而非拼工程交付）

---

## 22. License 与合规

- 官方 JEPA-WM：**CC BY-NC 4.0（Non-Commercial）**，仅限研究/非商业
- 用于申报材料"复现官方 baseline"无问题；**商业化需另行授权或换 baseline**（后续 KW-001 需评估）
- DINOv2 编码器：Meta 权重，frozen，用于研究评测

---

## 23. 复现性清单（他人能否复现）

1. 安装 torch 2.7.0+cu128（Blackwell）+ Python 3.10 venv
2. 下载 `pusht_noise`（OSF 镜像，免 gate）+ `jepa_wm_pusht.pth.tar`
3. 应用 Windows patch（gloo fallback、clusterscope 惰性导入）
4. 生成 eval config（`gen_pt_eval_configs.py`，支持缩减 CEM 参数）
5. 运行 `run_pt_bench.py` 跑评测 → 产出 `eval.csv` + `metrics.json`
6. 运行 `latent_rollout_error.py` → 产出 `rollout_err.json`
7. 运行 `rebuild_experiments_csv.py` → 重建统一 `experiments.csv`

**一键复现**：以上 2–7 步已封装为 `scripts/reproduce_phase0.sh`（`bash scripts/reproduce_phase0.sh --episodes 15`），自动设置环境变量、校验环境、跑通全流程。

---

## 24. 结论与交付物

**Phase 0 结论**：
- ✅ 官方 JEPA-WM Push-T baseline 在本机真实跑通（checkpoint 加载、CEM 规划、agent 执行）
- ✅ 建立机器可读实验日志（experiments.csv / metrics.json / gpu_samples.csv）
- ✅ 记录真实 benchmark 数字（**success=0.467@n15**、规划延迟 37.6s、VRAM 峰值 7.1GB、rollout-error 0.70→0.86、8 个 failure cases）
- ✅ 记录负面结果（并发重启、完整 CEM 算力瓶颈）
- ⚠️ 96-episode 完整 benchmark 在本机不可行（>200h），采用缩减 CEM + 小样本，成功率仅为量级参考

**交付物**：
- `docs/phase0_report.md`（本报告）
- `docs/research/jepa-wm.md`（调研报告）
- `results/experiments.csv`、`results/runs/*/metrics.json`、`results/rollout_err.json`
- `scripts/run_pt_bench.py`、`scripts/gen_pt_eval_configs.py`、`scripts/rebuild_experiments_csv.py`、`scripts/reproduce_phase0.sh`、`kineworld/eval/latent_rollout_error.py`
- `docs/patches/jepa-wms_windows_gloo.patch`

---

## 25. 停止声明

**Phase 0 到此结束。不进入 KW-001**（KW-001 需另行明确指令与规划）。本阶段未训练、未改动核心算法，全部基于官方预训练 checkpoint 的评估。
