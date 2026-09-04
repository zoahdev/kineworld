# KineWorld Blind Evaluation Protocol v0

状态：**DRAFT / NOT EXECUTED**。任何公开材料不得把本协议存在写成第三方验证已经发生。

## 角色分离

第三方：保存 hidden task/environment、随机种子、答案与评分器。  
KineWorld：只接收标准 observation，返回 action / prediction / uncertainty。  
第三方：执行环境并计算最终指标。

## 测试前

1. 双方冻结公开 API schema、预算与失败处理规则。
2. KineWorld 发布 `verification/commitments/` 格式的模型承诺。
3. 第三方发布 benchmark package hash，但不公开内容。
4. 明确网络、重试、超时、人工干预与日志保留规则。

## 最小任务集

- in-distribution control。
- unseen visual factor。
- unseen physical factor（摩擦/质量/几何至少一种）。
- partial observability / occlusion。
- failure injection 后 recovery。
- 模型应表达不知道的 OOD case。

## 指标

- task success + Wilson 95% CI。
- calibration：coverage / interval width / Brier 或 NLL（按输出类型）。
- selective risk：abstention-coverage curve。
- recovery success、steps、extra interaction cost。
- latency P50/P95、peak memory、model evaluations。
- invalid/timeout/crash rate。

## 防泄漏

- hidden seeds/task parameters 不进入 KineWorld 日志。
- 单次提交；若允许重试，重试次数和原因公开。
- 任何模型、planner、阈值或代码变化必须生成新 commitment_id，不能覆盖旧结果。
- 结果 manifest 包含请求/响应 trace hash，不要求公开敏感原始观测。

## 证据升级

- 内部模拟 hidden split：仍为 E1。
- 公开协议、第三方可重跑：E2。
- 独立团队在其环境重跑：E3。
- 第三方控制隐藏任务并先验承诺：E4。

## 当前下一门槛

只有当至少一个 KineWorld capability 达到稳定 E2，且不存在关键许可证阻塞时，才邀请第三方执行本协议。
