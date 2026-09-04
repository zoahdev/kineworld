# KineWorld Constitution

> 长期研究宪法。默认不得随意修改。修改需 Founder 明确决策并记录日期与理由。

- 版本：v1.0
- 生效日期：2026-09-02
- 状态：ACTIVE

---

## 0. 使命

**KineWorld exists to discover the smallest useful internal reality sufficient for prediction, planning, investigation, adaptation and action.**

We do not assume a final architecture.
Architectures are hypotheses.
Benchmarks are instruments.
Evidence decides.

---

## 1. 核心原则（Immutable Principles）

1. **Reality over pixel reconstruction** — 世界模型的目的不是重建现实的全部像素，而是学习对行动有用的现实。
2. **Action matters** — 必须能回答："What happens if I do this?"
3. **Partial observability is fundamental** — 机器维护的是 Belief，不是完美真实世界状态。
4. **Intelligence must be evaluated through action** — 最终看 planning / control / adaptation / recovery，不是只看 prediction loss。
5. **Uncertainty matters** — 模型必须区分"知道"和"不知道"。
6. **Learning continues after deployment** — prediction error 必须能变成新的学习信号。
7. **Compute is a resource** — 系统应该研究什么时候值得多想、什么时候应该直接行动。
8. **Representation is instrumental** — representation 服务于 prediction / planning / action，不是为了形式漂亮。
9. **Every claim requires evidence** — 任何能力声明必须有证据。
10. **Existing solved capabilities should be absorbed, not rebuilt** — 公开世界已经解决的，优先吸收。

---

## 2. Immutable Principles vs Mutable Architecture

严格区分：

- **A. Immutable Principles**：本文件 §1 的十条原则。长期不变。
- **B. Mutable Architecture**：当前最优假设。包括但不限于：JEPA、RSSM、Transformer、MPC、CEM、TD-MPC、Object-centric slots、Hierarchical world models、Memory architecture、Planner architecture、Encoder、Latent dimension。

以下全部只能被视为 **CURRENT BEST ARCHITECTURAL HYPOTHESIS**，而不是 FINAL ARCHITECTURE。如果未来实验表明其他方案更强：**立即替换**。禁止因为已经写了很多代码就维护错误路线。

---

## 3. 终极科学问题

**"机器应该怎样在自己的内部，构造一个足够好的现实？"**

核心研究方向：**Learn the smallest sufficient causal belief state for action.**

学习一个对于行动而言：尽可能小、但足够充分、具有不确定性、能够预测干预结果、能够支持规划和适应的内部世界信念。

核心数学直觉：

```
B_t = P( World_t | Observations_≤t, Actions_<t )

Belief Update:    B_{t+1} = U( B_t, O_{t+1}, A_t, Memory, Goal )

Causal Imagination:  P( B_{t+τ} | B_t, do(A), Goal )

Utility = Goal Progress + Information Gain - Risk - Energy Cost - Thinking Cost
```

---

## 4. 最终行为循环

```
Observe → Believe → Imagine → Investigate → Act → Learn → Observe → ...
```

- **Observe**：RGB / Depth / Robot State / Proprioception / Touch / Audio / 其他传感器
- **Believe**：维护 Persistent Probabilistic World Belief
- **Imagine**：模拟多个可能未来
- **Investigate**：不确定时主动设计观察或实验
- **Act**：根据内部未来进行行动
- **Learn**：比较 Predicted Future vs Actual Future，然后 adapt

---

## 5. 长期能力清单（Candidate Research Directions，非承诺）

A. Persistent Probabilistic Belief  B. Action-Conditioned Dynamics  C. Counterfactual Imagination  D. Calibrated Uncertainty  E. Object Permanence  F. Partial Observability  G. Active Investigation  H. Active Experiment Design  I. Hierarchical Temporal Abstraction  J. Adaptive Temporal Resolution  K. Adaptive Imagination Compute  L. Continual Adaptation  M. Working Memory  N. Episodic Memory  O. Semantic Memory  P. Procedural Memory  Q. Memory Consolidation  R. Causal Intervention  S. Counterfactual Generalization  T. Hybrid Reality Representation  U. Self Model  V. Other-Agent Model  W. Failure Recovery  X. Model-of-the-model  Y. Planner Routing  Z. Mixture of World Models

**不要一次实现全部。每一项都只是 Candidate Research Direction。**

---

## 6. 硬件约束

当前主要硬件：RTX 5070 Ti Laptop GPU（约 12GB VRAM）。

优先：Single GPU / Small Trainable Module / Frozen Encoder / Pretrained Encoder / Precomputed Embeddings / BF16-FP16 / Gradient Accumulation / Small Batch / Gradient Checkpointing if needed / Efficient simulation / Efficient experiment。

禁止把当前起点设成：自研 1B+ foundation model、自研 7B/20B、从零训练大型视觉模型、从零训练大型 VLA、4K future video、大型 video diffusion、大规模通用预训练。

当前自研训练规模优先：1M → 10M → 50M → 100M。大型模型尽量：pretrained + frozen + teacher + benchmark + representation source。

---

## 7. 最高战略：STAND ON GIANTS

任何问题先问："世界当前最强公开成果已经做到什么程度？" 然后问："KineWorld 还需要做什么？"

默认流程：SEARCH → FIND BEST EXISTING WORK → READ ORIGINAL SOURCE → FIND OFFICIAL CODE → FIND CHECKPOINT → FIND DATASET → REPRODUCE → BENCHMARK → IDENTIFY GAP → ONLY BUILD THE GAP。

禁止为了"原创代码比例高"重新实现成熟算法。KineWorld 的价值 = Scientific Contribution + Integration + New Capability + Evidence + Speed。

---

## 8. 最终运营原则

Do not rebuild the giants. Stand on them. Then climb where they have not reached.

Architectures are hypotheses. Evidence decides. Don't trust the claim. Verify it.

三个长期核心优势：
1. **Frontier Assimilation Velocity**（公开突破 → KineWorld 可用能力的时间）
2. **Evidence Velocity**（Idea → Real Result → Artifact → Report 的时间）
3. **Private Compounding Knowledge / Data**

目标不是设计一个永远不变的"最终 World Model"，而是设计一个**能够比单个团队更快发现什么才是正确 World Model 的系统**——Self-Correcting Research Company。
