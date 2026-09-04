# KW-EXP-0006 Evidence Consistency Audit

审计时间：2026-09-02  
状态：**PASS_AFTER_CORRECTION**  
范围：只核对原始汇总、96 个 diagnostics、冻结配置与报告文本；未重跑统计实验。

## 一致项

- `metrics.json` status=`complete`、exit_code=0。
- 请求与完成 episode 数：96。
- 96 个 `ep_*/diagnostics.json` 全部存在。
- 官方 `eval.csv` 与 runner `metrics.json` 的 `episode_success` 均为 `0.4583333333333333`。
- diagnostics 中 `actual_success=True` 为 44，False 为 52，与官方 44/96 一致。
- 物理分析输入为 576 点、96 episodes、18 spikes；审计抽取与报告一致。

## 不一致项

1. `verification/experiments/KW-EXP-0006.md §7.1` 写 `0.4375 (42/96)`，但两份官方汇总和逐 episode diagnostics 均为 **`0.4583333 (44/96)`**。
2. `§7.7` 同一句先写“18 尖峰中 6 成功（33%）”，随后括号写“9/18 尖峰成功”。从 576 行分析输入重算为 **6/18 = 33.33%**；`9/18` 错误。

## 影响

- Q2 的 `obj_disp` Fisher 检验、Spearman 相关和 Q3/Q4 校准结论不依赖手工 success 汇总，当前未发现受影响证据。
- 任何 success rate、尖峰与任务成功关系的对外 claim 在报告更正并重新 hash 前均为 **WITHHELD**。
- 本审计不把相关性升级为因果结论，也不证明 obj_disp 可部署；报告已显示条件带平均效率约为 0 增益。

## 证据指针与审计时 hash

- frozen config：`external/jepa-wms/configs/dump_online_evals/pt/pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep96_physint.yaml`  
  SHA-256 `357d4dfe74e573358d90b791469bd034028cb3aa6843c26391eb7c7bbdc83cfb`
- runner metrics：`results/runs/pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep96_physint_20260902T014727Z/metrics.json`  
  SHA-256 `84085b8a44bd468d8d470fa5df5cae8d9061290786c2cb715aa85fa00ef4d95a`
- analysis script：`verification/scripts/kw_exp_0006_physical_error.py`  
  SHA-256 `2bc41cfe25c252b374a8a5e3f05b806f961263283ffab0b942311b47c5b7d43d`
- report at audit time：`verification/experiments/KW-EXP-0006.md`  
  SHA-256 `ea1efb9dc225b44a974fca10f1bf19c4b59a3db8d595f3e14f64c4bb0e539ade`

## 关闭记录

两处计数已更正为 44/96 与 6/18；manifest 已加入 config、metrics、analysis、script、checkpoint 与更正后 report hash。更正后 report SHA-256：`964376135b7efa90b7810bd0fb08011486e0973f4acb3fe2d5427450b60cbc34`。原错误与原因保留在本审计轨迹中。
