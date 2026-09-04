# KW-LEWM-0001 — LeWM vs JEPA-WM Push-T 公平比较协议（预注册）

- 日期：2026-09-02
- 类型：跨方法对照复现（**无训练**）
- 状态：**协议预注册**（执行前写定，结果后只回填数据不改判据）
- 脚本：待补（`verification/scripts/kw_lewm_0001_compare.py`）

> 本文件在下载权重 / 跑批 / 看结果**之前**写定公平比较协议。数字只有在协议可复现且透明披露预算时才具备可比性。

---

## 1. 目的与决策门

KW-EXP-0006 收官后，frozen JEPA-WM 上的不确定性诊断已到终点（四个候选无一个可压缩保形上界）。按已冻结决策门，下一轮 ONE highest-ROI 是 **LeWM（LeWorldModel）Push-T 官方权重同台复现**，目标：

1. 产出 LeWM 在 Push-T 上的**独立可复现**基准数字（不写论文 48× 等第三方成绩为 KineWorld 自有）；
2. 与 KineWorld 已有的 JEPA-WM 96-episode 基准（`success_rate=0.4583`，KW-EXP-0006）做**同台对比**；
3. 去风险商业许可（MIT），并验证 stable-worldmodel 作为统一 harness 是否值得 KineWorld 长期采用。

## 2. 前置事实核验（2026-09-02）

- **stable-worldmodel 0.1.1**（MIT，PyPI，Python≥3.10）：统一 env/data/CEM/MPC/eval，含 LeWM reference（`scripts/train/lewm.py`）与 Push-T 16 FoV。Push-T 专家数据集 `pusht_expert_train` 812MB + `pusht_expert_val` 81MB（可下载）。
- **LeWM 官方权重**：Hugging Face 官方 LeWM collection，约 15M 参数、72.3 MB、MIT。经 `hongqin/leworldmodel` 官方仓库与 HF 核验。
- **CEMSolver API（stable-worldmodel）**：`CEMSolver(cost=..., num_samples=300, n_steps=30, topk=30, device='cuda', seed=1234)`；`PlanConfig(horizon=..., receding_horizon=..., action_block=...)`；`world.evaluate(episodes=..., seed=...)` 返回 `success_rate`。
- **JEPA-WM 基准**（KineWorld 已有）：96 episodes，CEM 缩减 `10×100×10`（iter/samples/elites），nas=6 全开环，`success_rate=0.4583`，`actual_state_dist_mean=112.73`。

## 3. 公平比较的四个对齐维度

跨方法对比的关键是**逐项透明披露**，无法完全对齐的维度显式标注，不做隐式排名。

| 维度 | JEPA-WM（已有基准） | LeWM（待跑） | 对齐策略 |
|---|---|---|---|
| 任务/环境 | Push-T（jepa-wms 内置） | Push-T（`swm/PushT-v1`） | 同任务；环境实现可能不同，标注为"同任务、不同环境实现" |
| Episode 集/seed | seed=1，官方 96 episode | `evaluate(episodes=96, seed=1)` | **对齐 96 episodes + seed=1** |
| 规划器 | CEM 缩减 `10×100×10` | CEMSolver，**等预算对齐** | 见 §4 预算对齐 |
| 成功定义 | `pos_diff<20 且 angle_diff<π/9` | Push-T 标准（同判定） | 对齐 |
| 评估模式 | 全开环（nas=6，每 episode 1 次规划） | `world.evaluate()` 默认模式 | 标注模式差异 |
| 指标 | `success_rate`, `actual_state_dist_mean` | `success_rate`, end_dist | 对齐 success_rate 为主指标 |

## 4. CEM 预算对齐（核心难点）

JEPA-WM 用 `10×100×10`（10 次迭代 × 每次 100 采样 × 保留 10 精英）。stable-worldmodel 的 CEMSolver 参数为 `n_steps=30, num_samples=300, topk=30`。

**公平原则：对齐"一次规划的计算成本"（迭代次数 × 采样数），而非对齐参数名。**

- **主对照（等预算，规模较小，本机可行）**：JEPA 用 `10×100×10`（=1000 采样×10 迭代），LeWM 用 `CEMSolver(n_steps=10, num_samples=100, topk=10)` —— 两者单次规划 cost 相同量级（10 迭代 × 100 采样）。
- **若 LeWM 在 100 采样下不稳定**：报告预算披露后，可加跑 `n_steps=30, num_samples=300, topk=30`（官方默认），但**不作为与 JEPA 对齐的主数字**，而是"LeWM 官方默认配置"单独披露。

**判定规则**：
- 若在等预算（10×100）下 LeWM success_rate 显著高于 JEPA 0.4583 → LeWM 在本任务上确有更强动力学/规划收益（E1，单 seed）。
- 若等预算下 LeWM ≤ JEPA → 在缩减 CEM 预算下 LeWM 无优势（不能据此否定其全预算能力，需另行披露官方默认配置）。
- 无论结果如何，**不写"LeWM 论文的 48×/单卡数小时"为 KineWorld 自有成绩**。

## 5. 预注册结果报告内容

1. **LeWM 独立基准**：success_rate、end_dist，96 episodes + seed=1，附可复现脚本与 hash。
2. **预算披露**：CEM 配置（n_steps/num_samples/topk）、每次规划耗时、单次 eval wall-time、VRAM 峰值。
3. **同台对比表**：LeWM vs JEPA-WM 的 success_rate/end_dist，逐项标注对齐/差异维度。
4. **协议偏差声明**：环境实现、CEM 对齐、评估模式等任何无法完全对齐之处显式列出。
5. **结论分级**：E1（单 seed、单任务、单 checkpoint 集）。

## 6. Kill Criteria / 降级门

- **安装成本门**：stable-worldmodel 安装 + 环境适配超过 **1 天** → 降级为"仅分别复现，不做同台排名"，并在 ROI 记录。
- **权重获取失败**：HF 72.3MB 权重无法下载或格式不兼容 → 停止复现，改记为"官方权重不可独立获取"，不强行自训 LeWM（训练数据 13.1GB 未经 Founder 同意不下载）。
- **协议无法对齐**：若 LeWM 与 JEPA 在 CEM/环境上无法对齐到可判优劣 → 只做分别复现，不做排名。

## 7. 证据等级与限制

- E1。单 seed（seed=1）、单任务 Push-T、单 checkpoint 集。
- LeWM 权重来自官方（MIT），但"官方实现待独立复现"——本实验的 LeWM 数字是 KineWorld 独立跑出的，不背书论文数字。
- 不下载 13.1GB 训练数据；不进行任何训练。

## 8. 产物预期

- `verification/experiments/KW-LEWM-0001.md`（本文件，执行后回填）
- `verification/scripts/kw_lewm_0001_compare.py`（可复现，固定 seed）
- `verification/manifests/KW-LEWM-0001_manifest.json`
- LeWM eval 原始结果 + 与 JEPA-WM 的对比表
- 台账 `company/EVIDENCE_LEDGER.md` 追加对比行
