# GIANTS.md — 世界前沿台账

> 最近全网扫描：2026-09-02。来源置信度分级：✅=官方/一手（论文、官方 repo、官方公告）；🔶=二手综述/媒体（需回溯一手核实）。
> 每条记录：最新重要工作 / 日期 / 论文 / 代码 / checkpoint / 数据 / 规模 / license / 已解决 / 部分解决 / 开放问题 / 可复用组件 / KineWorld 不应重造的部分。

---

## 1. JEPA 生态（本家主线）

### Meta FAIR — V-JEPA 2 / V-JEPA 2-AC ✅
- 日期：2025-06
- 工作：1B 参数编码器，1M+ 小时互联网视频预训练；V-JEPA 2-AC 用 <62h 无标注 DROID 机器人视频后训练
- 结果：Franka 零样本 pick-and-place 65–80% 成功（vs Octo ~15%），规划 ~16s/step（vs Cosmos 基线 ~4min）
- 可复用：冻结编码器作为 representation source、规划 baseline
- 不重造：视频 SSL 预训练本身（算力不可及，也无必要）

### facebookresearch/jepa-wms（= 本仓库 baseline，arXiv 2512.24497）✅
- 日期：2025-12（论文）；本地 commit `13cf1d9`
- 工作：消融研究，隔离 JEPA-WM 哪些设计选择驱动物理规划；报告超过 DINO-WM 与 V-JEPA 2-AC
- Push-T 配方：DINOv2 ViT-S/14（frozen）+ AdaLN 预测器（depth 6, 17.6M）+ ctxt2 + fsk5 + CEM 30×300×H6
- License：**CC BY-NC 4.0（非商业）** — 研究/申报可用，商业 Core 禁入
- 可复用：官方 checkpoint、官方评测 harness、CEM 规划器、多环境（pt/mw/wall/mz/robocasa/droid）
- KineWorld 已复现：✅（Phase 0，succ=0.467@n15 缩减 CEM）

### galilai-group/stable-worldmodel 0.1.1 ✅
- 日期：2026-06（PyPI 0.1.1）；官方 repo 当前 673 commits；MIT
- 工作：统一 world-model 数据采集、训练、环境、CEM/MPC 规划与评测接口；官方列出 LeWM 与 C-JEPA 使用案例，并直接提供 LeWM / PreJEPA 参考训练实现
- 环境：Push-T、Two-Room、OGBench、DMControl、Gymnasium Robotics、Craftax、Atari 等；Push-T 暴露 16 个 factors of variation，可直接做 OOD / hidden-dynamics protocol
- 直接依赖：base 为 torch/torchvision/numpy/gymnasium/einops/hydra 等；`train` 引入 stable-pretraining，`env` 引入 MuJoCo/OGBench/Craftax 等；完整传递许可仍需 lock 后审计
- 可复用：**Research Harness、CEM solver、环境注册、FoV/OOD、数据格式与 LeWM reference implementation**
- KineWorld 不应重造：通用 world-model harness、基础环境抽象、通用 CEM/MPC、数据格式转换
- KineWorld 真正可做的 gap：预注册证据层、校准/风险诊断、failure taxonomy、hidden/blind protocol、跨模型可比性与商业交付

### AMI Labs（LeCun 2025-11 离开 Meta 联合创立，CEO Alexandre LeBrun）🔶→✅
- 融资：$1.03B 种子轮 @ $3.5B pre-money（2026-03，NVIDIA/Temasek/Bezos Expeditions/Eric Schmidt）
- 方向：JEPA 系世界模型。ICML 2026 宣讲（冯雁）：WorldPrediction benchmark（长时程过程规划，VLM ≤50%、人类近完美 → Grounding Gap 论点）
- 相关工作：
  - **LeWorldModel / LeWM**（AMI Labs + NYU）✅：arXiv:2603.19312 v3 与官方 repo 已核实。端到端从原始像素训练，仅 next-embedding prediction + Gaussian latent regularizer 两项 loss；约 15M 参数，论文报告单卡数小时训练、规划最高比 foundation-WM 快 48×。代码与 Push-T 权重标注 MIT，官方 Push-T 权重 72.3 MB；官方 Push-T 数据 13.1 GB。→ **REPRODUCE（先权重同台评测，暂不下载训练数据）**
  - **VL-JEPA**🔶：语言空间世界模型，CVPR 2026 EgoVis 挑战赛冠军
  - **Latent Action Model**🔶：从无标注原始视频发现动作空间，可扩展 in-the-wild 视频 → WATCH（若开源，对"action 从哪来"是重大免费能力）
  - **V-JEPA 2.1**（Mur-Labadia et al. 2026）🔶：dense features 视频 SSL，JEPA-WAM 的表征基础

### WIMLE（Apex Lab / SFU）✅
- 论文：Aghabozorgi et al., ICLR 2026；arXiv:2602.14351；官方 repo：`github.com/mehranagh20/wimle`
- 方法：IMLE 条件随机世界模型 + ensemble/latent sampling 估计预测不确定性；用 inverse-variance weighting 降低不可靠 synthetic transitions 的训练影响。
- 论文声称：覆盖 40 个 continuous-control tasks；HumanoidBench 解出 8/14；Humanoid-run sample efficiency 相对最强竞品提升超过 50%。这些是论文声称，尚未在本机独立复现。
- 公开状态：代码 ✅；README 提供 DMC/HumanoidBench/MyoSuite 的独立 requirements、训练 launcher 和 results 数据包；未发现可直接下载的 frozen JEPA checkpoint。
- 工程边界：JAX/SAC/MBRL 完整训练系统，依赖 MuJoCo/DMC/HumanoidBench/MyoSuite 与 CUDA/JAX 配套；不是可直接插入当前 PyTorch frozen JEPA predictor 的 uncertainty head。
- 可复用：inverse-variance weighting 思路、ensemble×latent uncertainty 分解、calibration 评测设计。
- KineWorld 决策：**REUSE IDEAS / 不直接移植代码**；先做 paper-to-protocol 复刻，不启动完整 WIMLE 训练。

### HAUWM / Horizon-Calibrated Uncertainty ✅（论文一手；代码状态未确认）
- 论文：Wan, Gan, Zhan；ICLR 2026；OpenReview id `pZuZWRuPyi`，PDF 可访问。
- 方法：probabilistic ensemble，随机未来 horizon 预测；Horizon-Calibrated Uncertainty (HCU) loss 约束预测 variance 随 horizon 增长；下游需 fine-tuning。
- 论文原文摘要声称：在 MetaWorld、DMC、RoboDesk 等控制基准 fine-tune 后优于多个 SOTA；具体结果需以论文表格为准。
- 公开状态：一手 PDF/摘要已核实；公开页面标记代码“待确认”，未确认可下载 repo/checkpoint，不能视为免费可复用能力。
- 工程边界：方法需要 ensemble + variable-horizon pretraining，明显超出当前 frozen single-predictor 零训练约束。
- KineWorld 决策：**MUST READ / PROTOCOL REFERENCE**；不直接实现 HCU，不把二手页面的结果写成已复现事实。

- 日期：2025-07 "Critiques of World Models" + 2025-11 PAN
- 立场：直接攻击 JEPA 路线——latent-only 目标脆弱、易 collapse；提出 Generative Latent Prediction（潜空间预测 + 生成式重建 grounding）
- 实现：Qwen2.5-VL-7B backbone + Wan2.1-T2V-14B 视频扩散 decoder
- 状态：自评超过 Cosmos/V-JEPA 2（VLM-judge），**截至 2026-07 未公开权重/代码** → WATCH，不可复用
- 规模：27B 级，超出本机可行性，仅作思想参照

---

## 2. 生成式世界模型阵营

### NVIDIA Cosmos ✅
- 开放世界基础模型，号称 20M 小时真实交互/环境/驾驶数据
- 定位：physical AI 骨干；像素空间预测，规划延迟高（~4min/step 级别）
- 可复用：作为对比基线 / 数据引擎；不重造视频扩散

### Google DeepMind — Genie 3 ✅（官方公告核实）
- 实时可导航交互 3D 世界生成，官方报告 24fps、720p、可保持数分钟一致性
- 未开放权重；作为"世界生成"方向参照 → WATCH

### Wayve GAIA-3 🔶
- 15B 参数，可控多机位驾驶场景生成（训练+评测两用）→ 驾驶域 WATCH

### 1X world model 🔶
- 14B 视频骨干；单次 5s rollout 想象 ~11s（机器人需"停下来想"）——延迟问题的公开承认，佐证 Adaptive Compute 研究价值
- 自曝弱点：单目视频 3D grounding 弱、部分任务 ~0% 成功

---

## 3. Model-Based RL 生态

### Dreamer 系列（danijar/dreamerv3，JAX，MIT）✅
- DreamerV3：150+ 任务，单一超参集；RSSM 离散随机隐变量（原生不确定性）
- **Dreamer 4**（DeepMind）🔶：Minecraft 挖钻石（20,000 动作任务）纯离线数据、零在线交互 → 重大信号：offline world model 上限持续抬升
- 可复用：MIT、单卡可跑（200 episode 6-24h/RTX3090 级别）；KineWorld 可作 harness 内对照方法
- 训练成本对我们偏高 → 优先 frozen/pretrained 策略下暂不重训

### TD-MPC2（nicklashansen/tdmpc2，PyTorch，MIT）✅
- 连续动作、在线 MPPI 规划、部署时可换 reward；多任务版本
- 单卡 2-12h 级可复现 → REPRODUCE 候选（作为 CEM 规划对照组）
- 后续生态：HaM-World🔶（Soft-Hamiltonian 几何隐空间，无 actor 纯 CEM；Avg AUC 超 TD-MPC2 9.5%，长程 rollout MSE 仅其 45%）→ WATCH/研读

### ICLR 2026 批次（🔶 经二手索引，需回溯一手）
- **R2-Dreamer**：decoder-free MBRL（Barlow Twins 冗余约减），1.59× 快于 DreamerV3，DMC-Subtle 小目标增益大；代码 github.com/NM512/r2dreamer ✅
- **WIMLE**：IMLE 不确定性感知世界模型，长程 rollout 逆方差加权，HumanoidBench 8/14；state-based
- **Learning Massively Multitask World Models for Continuous Control**
- **Learning to Be Uncertainty: Pre-training World Models with Horizon-Calibrated Uncertainty** → 与 KineWorld 不确定性方向直接相关，必读
- **From Observations to Events: Event-Aware World Models** → 事件抽象方向
- **Object-Centric World Models from Few-Shot Annotations** → 对象中心方向

---

## 4. VLA / 机器人基础模型（相邻阵营，融合中）

- **π0.5（Physical Intelligence）**🔶：开放世界泛化 VLA；JEPA-WAM 显示 π0.5+JEPA 目标 → LIBERO-Plus 86.3%、真机双臂 90.3% ID —— "VLA 借世界模型组件"的融合证据
- **Octo**：开源 VLA baseline（被 V-JEPA 2-AC 65-80% vs 15% 对比的那个）
- **Generalist AI GEN-1**🔶：号称 99% 成功、50 万小时数据、99% 参数从头训
- **LeRobot（Hugging Face）**✅：数据/策略生态，MIT 类许可，可作数据管线
- 行业判断（🔶 NVIDIA 技术分析引述）：赢家不是纯 VLA 也不是纯 WM，而是混合——两谱系正在融合

---

## 5. 跨领域套利池（Frontier Arbitrage，§12）

| 领域 | 成熟思想 | 被 learned WM 吸收程度 | 优先检查 |
|---|---|---|---|
| Control Theory | MPC 闭环重规划、receding horizon | **部分**（jepa-wms 官方支持 num_act_stepped<horizon 但 Push-T 默认全开环） | ✅ 实验 KW-EXP-0001（进行中） |
| Bayesian Filtering | belief 递推、协方差传播 | 低（JEPA 无原生不确定性） | WATCH |
| POMDP | belief-space planning | 低 | WATCH |
| System Identification | 在线参数辨识、适应性控制 | 低 | WATCH |
| Active Learning / Bayesian Experimental Design | 信息增益选动作 | 低 | WATCH（1X 自曝延迟问题 = adaptive compute 需求证据） |
| Causal Discovery | 干预/反事实分级 | 极低 | UNKNOWN |
| Change-point detection | 动力学切换检测 | 低 | UNKNOWN |

---

## 6. 开放问题汇总（多来源交叉验证）

1. JEPA collapse 避免仍是启发式（EMA/stop-grad/VICReg），非原理性
2. LeCun 层级多时间尺度预测在规模上未实现
3. **长时程潜空间规划误差复合**（我们 Phase 0 实测：rollout L2 0.70→0.86@50步，与文献一致）
4. 分布偏移下脆弱性（2026 理论+benchmark 工作已实证）
5. JEPA 无原生不确定性量化
6. 像素生成路线的"physics slop"（不看时几何可选）与延迟
7. WorldPrediction：VLM 长时程过程规划 ≤50% vs 人类近完美

---

## 维护规则
- 每周随 WEEKLY_FRONTIER.md 更新；重大发布即时追加 FRONTIER_LOG.md
- 任何条目进入实验前必须回溯一手来源（🔶 → ✅）
