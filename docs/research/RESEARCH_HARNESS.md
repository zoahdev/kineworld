# KineWorld Research Harness — Adapter Contract v0

状态：**FROZEN THIN EVIDENCE ENVELOPE**。接口契约与测试已实现；发现官方 `stable-worldmodel` 后停止扩写通用执行 harness。JEPA-WM / LeWM adapter 尚未实现；接口存在不等于能力已验证。

## 目的

把“模型是什么”和“如何公平评测”分开。官方模型代码保持原样。**环境、CEM/MPC、数据与基础 evaluator 直接复用 MIT 的 `stable-worldmodel`**；KineWorld 仅保留 capability declaration、license/source/hash、预算与证据元数据 envelope。

核心入口：`kineworld/harness/contracts.py`。

## 最小接口

- `encode(observation)`
- `update_belief(previous, observation, previous_action)`
- `predict(belief, action_sequence)`
- `estimate_uncertainty(prediction)`（可选，必须显式声明）
- `plan(belief, goal, budget)`
- `adapt(transition)`（可选，必须显式声明）

## 防止假能力

1. 每个 adapter 必须声明 source、commit、model SHA-256、license 和 capabilities。
2. 未声明的 uncertainty / adaptation 会抛出 `CapabilityNotSupported`，不能返回伪造默认值。
3. 规划必须接收 `PlanBudget`，结果必须报告 model evaluations 与 latency。
4. tensor / simulator payload 保持 opaque，避免为了统一接口重写成熟实现。
5. adapter 通过单元测试只证明接口符合；模型能力仍需独立实验 Claim ID。

## 接入顺序

1. KW-EXP-0006 完成前不接入 GPU 模型。
2. 不再自建环境/solver/model lifecycle；先验证 stable-worldmodel 能否读取官方 LeWM 权重并跑原生协议。
3. jepa-wms 作为 RED 研究对照，通过结果导入/adapter 隔离，不污染商业 runtime。
4. 产品不是通用 Evaluation SDK，而是建立在 SWM 等公开 harness 上的风险、校准与失败诊断层。

## 测试

```text
.venv310/Scripts/python.exe -m unittest discover -s tests -v
```

当前测试只用 Python 标准库，不导入 PyTorch，不占 GPU。
