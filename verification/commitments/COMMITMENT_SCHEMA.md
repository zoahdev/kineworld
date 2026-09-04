# KineWorld Model Commitment v0

用途：在收到 hidden/blind task 前冻结被测对象，防止看题后换模型或改策略。

## 必填字段

- `commitment_id`
- `created_at_utc`
- `model_version`
- `model_sha256`（闭源权重只发布 hash）
- `runtime_image_or_lock_sha256`
- `source_commit`
- `dirty_patch_sha256`（无修改则为 null）
- `adapter_sha256`
- `planner_name`、`planner_config_sha256`
- `allowed_compute`（GPU、VRAM、wall time、model evaluations）
- `capabilities_declared`
- `license_classification`
- `signer`
- `signature_or_timestamp_receipt`

## 顺序

1. Freeze model/runtime/planner。
2. 计算 hash，生成 commitment JSON。
3. 将 commitment 交给第三方或可信时间戳服务。
4. 之后才接收 hidden task。
5. 执行时验证所有 hash；任一不一致即结果无效。
6. 发布 commitment、benchmark hash、结果 manifest 与 trace hash；核心权重不公开。

当前状态：仅协议；尚无 E3/E4 测试、签名或外部承诺。
