# CAPABILITY_AUDIT.md — KineWorld 能力审计

> 日期：2026-09-02。原则：严格区分 THIRD-PARTY CAPABILITY（复用的巨人能力）与 ACTUAL KINEWORLD CONTRIBUTION（我们真正新增的）。
> 诚实口径：任何对外材料不得把第三方能力写成 KineWorld 能力。

## 1. THIRD-PARTY CAPABILITY（全部来自 facebookresearch/jepa-wms，CC BY-NC 4.0）

| 能力 | 来源 | 证据 |
|---|---|---|
| DINOv2 ViT-S/14 视觉编码（frozen） | Meta 官方权重 | Phase 0 加载日志 |
| AdaLN 动作条件化预测器（17.6M，已训练） | jepa-wms 官方 checkpoint `jepa_wm_pusht.pth.tar`（epoch 50） | `All keys matched successfully` |
| CEM 规划器 | jepa-wms 官方代码 | 官方 eval 配置 |
| Push-T/MW/Wall 评测环境与协议 | jepa-wms 官方 evals | 官方 eval.csv |
| 闭环重规划机制（num_act_stepped<horizon 的 while-loop） | jepa-wms 官方 `plan_evaluator.py` | 代码审计（本审计轮确认） |
| success/end_dist/emb_l2 指标定义 | jepa-wms 官方 | eval.csv 12 列 |

**第三方能力结论**：Phase 0 的 succ=0.467 是"官方模型+官方代码+缩减 CEM"的成绩，**不是 KineWorld 的算法贡献**。

## 2. ACTUAL KINEWORLD CONTRIBUTION（我们真正新增的）

| 贡献 | 类型 | 证据 | 能力等级 |
|---|---|---|---|
| Windows 单机复现工程（gloo fallback、clusterscope 惰性导入、JEPAWM_* 环境变量协议、文件锁防并发） | 工程集成 | `docs/patches/jepa-wms_windows_gloo.patch` | 基础设施（非算法） |
| 实验日志系统（29 列 experiments.csv、metrics.json/jsonl、GPU 采样、重建脚本） | 评测基础设施 | `results/experiments.csv`、`scripts/rebuild_experiments_csv.py` | E1 |
| rollout-error 测量 harness（1/5/10/20/50 步 latent 预测误差，单帧 encode 约定修复） | **测量工具（新增）** | `kineworld/eval/latent_rollout_error.py`、`results/rollout_err.json`（n=36） | E1 |
| 完整 CEM 不可行性的量化证据（>18min/plan、96ep>200h、缩减 CEM 37.6s@58%VRAM） | **实证发现（新增）** | `results/runs/*singleprobe*/gpu_samples.csv`、phase0_report §12.2 | E1 |
| Push-T 官方 evaluator 每 episode 只规划一次（开环）的结构发现 | **实证发现（新增）** | planning_calls=15@15ep，plan_evaluator.py 代码审计 | E1 |
| 一键复现脚本 | 复现性基础设施 | `scripts/reproduce_phase0.sh` | E1 |
| KW-EXP-0001 闭环重规划实验（设计、执行与负面结果登记） | **受控实证 / 负面结果（新增）** | `verification/experiments/KW-EXP-0001.md`、`verification/manifests/KW-EXP-0001_manifest.json` | E1 | |

**KineWorld 贡献结论**：截至今日，KineWorld 的**算法/科学贡献仍为零**，全部贡献是：复现工程 + 测量工具 + 实证发现 + 实验基础设施 + 首个受控负面结果。KW-EXP-0001 证明了在当前操作点提高重规划频率不能提升 Push-T 成功率，但不是新算法。对外表述必须保持此口径。

## 3. 证据等级现状（E0-E4）

- 全部现有成果 = **E1（Internal Experiment）**
- 无 E2（Public Reproducible Protocol）——reproduce_phase0.sh 已具雏形，但未对外发布
- 无 E3/E4

## 4. 闭源边界现状

- 当前无任何需要闭源的资产（无自研权重/数据/运行时）
- 开源候选：评测 harness、rollout-error 工具、复现脚本（需先查 license 依赖，JEPA-WM CC BY-NC 会传染衍生 benchmark 的商业使用）
