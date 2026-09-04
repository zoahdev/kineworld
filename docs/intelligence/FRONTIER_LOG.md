# FRONTIER_LOG.md — 前沿决策日志

> 记录重要工作 + 发布日期 + 决策：IGNORE / WATCH / REPRODUCE / INTEGRATE / PIVOT。
> 格式：| 日期 | 工作 | 来源置信度 | 决策 | 理由 | 下次复查 |

## 2026-09-02 首次全量扫描

| 日期 | 工作 | 置信度 | 决策 | 理由 | 复查 |
|---|---|---|---|---|---|
| 2025-06 | V-JEPA 2 / V-JEPA 2-AC (Meta) | ✅ | INTEGRATE（作表征/对照） | 1B 编码器不可重训，但 frozen 可用；规划对照基线 | — |
| 2025-12 | jepa-wms (arXiv 2512.24497) | ✅ | **INTEGRATED**（Phase 0 已复现） | 官方 checkpoint+harness 全复用 | — |
| 2026-02 | WIMLE (Aghabozorgi et al., ICLR 2026) | ✅（arXiv+官方 repo） | **REUSE IDEAS / 不直接移植** | IMLE 随机模型 + ensemble×latent uncertainty + inverse-variance weighting；代码是 JAX/SAC 完整训练系统，无 frozen JEPA checkpoint，无法直接接入当前 PyTorch evaluator | 读论文方法细节后 |
| 2026-02 | HAUWM / Horizon-Calibrated Uncertainty (Wan et al., ICLR 2026) | ✅（一手 PDF） | **MUST READ / PROTOCOL REFERENCE** | HCU 将 variance 与 horizon 绑定；一手摘要已核实，但代码/checkpoint 未确认，且需要 ensemble + pretraining，暂不移植 | 本周 |
| 2026-03 | AMI Labs $1.03B 种子轮 | 🔶 | WATCH | LeCun 路线商业化信号；跟踪其开源动作 | 月度 |
| 2026-03 (v3: 2026-06) | LeWorldModel / LeWM (AMI+NYU) | ✅（arXiv+官方 repo+HF） | **REPRODUCE NEXT（KW-EXP-0006 后）** | 约 15M 参数；代码/Push-T 权重 MIT；权重 72.3 MB；同为 Push-T，可低成本与当前 JEPA-WM 同台评测。训练数据 13.1 GB，未经 Founder 同意不下载 | 权重评测前冻结协议 |
| 2026 | Latent Action Model (AMI) | 🔶 | WATCH | 无标注视频发现动作空间；若开源 = 免费重大能力 | 月度 |
| 2026 | V-JEPA 2.1 (dense features) | 🔶 | WATCH | JEPA-WAM 的表征基础；等权重 | 月度 |
| 2026 | JEPA-WAM (LIBERO-Plus 79.2%, π0.5+JEPA 86.3%) | 🔶 | WATCH | WM 组件增强 VLA 的融合证据；非我们的直接赛道 | 季度 |
| 2026 | Dreamer 4 (DeepMind, Minecraft 离线) | 🔶 | WATCH | offline WM 上限信号；权重未出 | 季度 |
| 2025-2026 | Genie 3 / Cosmos / GAIA-3 | 🔶 | IGNORE（直接复用）/ WATCH（思想） | 像素生成路线，算力不可及；Cosmos 可作数据引擎长期选项 | 季度 |
| ICLR 2026 | R2-Dreamer | ✅（proceedings+repo） | REPRODUCE 候选（低优先） | decoder-free MBRL 对照；MIT；单卡可行 | 本月 |
| ICLR 2026 | WIMLE | 🔶 | **必读**（不确定性方向先决） | IMLE 不确定性 + 长程 rollout 加权；state-based 限制 | 本周读一手 |
| ICLR 2026 | Horizon-Calibrated Uncertainty 预训练 | 🔶 | **必读** | 与我们 N3b 节点直接对应 | 本周读一手 |
| ICLR 2026 | Event-Aware WM / Object-Centric WM / Massively Multitask WM | 🔶 | WATCH | 候选方向参照 | 月度 |
| 2026 | HaM-World | 🔶 | 研读 | 无 actor 纯 CEM + 几何隐空间；长程 rollout 误差压低思路 | 本月 |
| 2026-09-02 | KW-EXP-0001 闭环重规划 | ✅（本地实测） | **PIVOT（研究优先级）** | nas=2 未提升成功率（0.467→0.400），recovery=0、regression=1；不再做 naive replanning sweep，转向 diagnostics + uncertainty/dynamics | 下一轮实验前 |
| 2026-09-02 | LeWM 一手复核 | ✅ | **PROMOTE：WATCH→REPRODUCE** | 官方代码/权重/数据齐备，MIT，单卡量级明确；但 repo 很新（6 commits），须独立复现，论文数字不可当作 KineWorld 成绩 | KW-EXP-0006 完成后 |
| 2026-06 / 核验于 2026-09-02 | stable-worldmodel 0.1.1 | ✅（官方 repo+PyPI） | **INTEGRATE / STOP REBUILDING HARNESS** | MIT；673 commits；统一 env/data/CEM/MPC/eval，含 LeWM reference 与 Push-T 16 FoV。KineWorld 通用 harness 工作降级为薄证据 envelope | LeWM 复现时 |
| 2026 | 1X WM 延迟自曝（11s/5s rollout） | 🔶 | 证据归档 | adaptive compute 需求的一手行业证据 | — |

## 决策规则备忘
- REPRODUCE = 进 RESEARCH_ROI.md 评分排队
- 任何 🔶 条目进入实验前必须回溯一手（论文/repo/权重）
- PIVOT = 触发宪法 §2 mutable architecture 替换流程，需 Founder 知悉
