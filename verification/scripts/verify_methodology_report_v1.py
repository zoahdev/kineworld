"""KW methodology report verification — METHODOLOGY_CONFOUNDS_v1.

目的
----
`docs/research/METHODOLOGY_CONFOUNDS_v1.md` 是一篇对外方法论报告，里面每一个
数字都必须能从已归档的结果 JSON 回溯。本脚本做两件事：

  A. 数值校验：从各实验的结果 JSON 读取精确值，与报告声称的数字比对。
  B. 文本校验：确认报告正文里确实出现了这些数字（防止报告与数据脱节）。

任一失败即 exit 1。这是 R0「判据自校验门禁」精神的延伸——报告不是写完就算，
必须有机读的一致性证明。

用法
----
    python verification/scripts/verify_methodology_report_v1.py

纯标准库、无 GPU、无网络。约 1 秒。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs" / "research" / "METHODOLOGY_CONFOUNDS_v1.md"


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def dig(d, *keys):
    for k in keys:
        d = d[k]
    return d


# ---------------------------------------------------------------- 检查项定义
# 每项: (标签, 结果文件, JSON 取值路径, 期望值, 容差, 报告中应出现的字符串或 None)
CHECKS = [
    # ---- 混淆变量一：任务可达性 ------------------------------------------
    ("0002 scaler · 成功率", "results/kw_lewm_0002_scaler_ablation.json",
     ("conditions", "scaler", "summary", "n_success"), 0, 0, "0/8"),
    ("0002 scaler · 越界率", "results/kw_lewm_0002_scaler_ablation.json",
     ("conditions", "scaler", "summary", "out_of_play_rate"), 0.25, 1e-9, "0.25"),
    ("0002 scaler · block 位移中位数", "results/kw_lewm_0002_scaler_ablation.json",
     ("conditions", "scaler", "summary", "block_displacement_median"), 0.0, 1e-9, "**0.0**"),

    ("0003 · 成功率", "results/kw_lewm_0003_dataset_tasks.json",
     ("summary", "n_success"), 3, 0, "3/8"),
    ("0003 · 越界率", "results/kw_lewm_0003_dataset_tasks.json",
     ("summary", "out_of_play_rate"), 0.0, 1e-9, "0.00"),
    ("0003 · block 位移中位数", "results/kw_lewm_0003_dataset_tasks.json",
     ("summary", "block_displacement_median"), 27.91, 0.01, "27.91"),
    ("0003 · block 交互率", "results/kw_lewm_0003_dataset_tasks.json",
     ("summary", "block_interaction_rate"), 0.75, 1e-9, "0.75"),

    ("0004 · 成功率", "results/kw_lewm_0004_dataset_tasks_p50.json",
     ("summary", "n_success"), 18, 0, "18/50"),
    ("0004 · block 位移中位数", "results/kw_lewm_0004_dataset_tasks_p50.json",
     ("summary", "block_displacement_median"), 31.54, 0.01, "31.54"),
    ("0004 · block 交互率", "results/kw_lewm_0004_dataset_tasks_p50.json",
     ("summary", "block_interaction_rate"), 0.88, 1e-9, "0.88"),

    # ---- 弱动作条件 · 证据线 1（latent 机动性 / 动作杠杆）----------------
    ("§8 · 机动性比 (max_mob/gap)", "results/kw_lewm_0005_rootcause.json",
     ("mean_ratio_max_mob_over_gap",), 2.97, 0.01, "2.97"),
    ("§8 · context_domination_ratio", "results/kw_lewm_0005_rootcause.json",
     ("mean_context_domination_ratio",), 1.0003, 1e-4, "1.0003"),
    ("§8 · action_leverage", "results/kw_lewm_0005_rootcause.json",
     ("mean_action_leverage",), -1.15, 0.01, "−1.15"),

    # ---- 弱动作条件 · 证据线 2（闭环相关性）------------------------------
    ("§8.6 · block 位移 vs 目标距离 ρ", "results/kw_lewm_0005b_k4_corroboration.json",
     ("correlations", "block_displacement_vs_goal_distance", "spearman"), -0.39, 1e-9, "−0.39"),
    ("§8.6 · success vs 目标距离 ρ", "results/kw_lewm_0005b_k4_corroboration.json",
     ("correlations", "success_vs_goal_distance", "spearman"), 0.09, 1e-9, "+0.09"),
    ("§8.6 · success vs 位移量 ρ", "results/kw_lewm_0005b_k4_corroboration.json",
     ("correlations", "success_vs_block_displacement", "spearman"), 0.035, 1e-9, "+0.035"),
    ("§8.6 · in-play 位移>20 的任务数", "results/kw_lewm_0005b_k4_corroboration.json",
     ("near_but_failed", "inplay_disp_gt_20"), 27, 0, "27/27"),
    ("§8.6 · 其中失败数", "results/kw_lewm_0005b_k4_corroboration.json",
     ("near_but_failed", "inplay_disp_gt_20_but_failed"), 27, 0, "27/27"),
    ("§8.6 · 越界/in-play 目标距离比", "results/kw_lewm_0005b_k4_corroboration.json",
     ("out_of_play", "goal_distance_ratio_oop_over_inplay_mean"), 1.089, 1e-3, "1.089"),

    # ---- 弱动作条件 · 证据线 3（配对行为学）------------------------------
    ("0005c · agent_disp LeWM", "results/kw_lewm_0005c_paired_noop_baseline.json",
     ("paired_progress_needs_work_stratum", "agent_disp", "lewm", "median"),
     131.28, 0.01, "131.28"),
    ("0005c · agent_disp JEPA", "results/kw_lewm_0005c_paired_noop_baseline.json",
     ("paired_progress_needs_work_stratum", "agent_disp", "jepa", "median"),
     132.40, 0.01, "132.40"),
    # 精确值 75.8468 → 四舍五入为 75.8（不是 75.9）。此条目曾因手写成 75.9 被本脚本抓出。
    ("0005c · agent_progress LeWM", "results/kw_lewm_0005c_paired_noop_baseline.json",
     ("paired_progress_needs_work_stratum", "agent_progress", "lewm", "median"),
     75.85, 0.01, "+75.8"),
    ("0005c · agent_progress JEPA", "results/kw_lewm_0005c_paired_noop_baseline.json",
     ("paired_progress_needs_work_stratum", "agent_progress", "jepa", "median"),
     -36.2, 0.05, "−36.2"),
    ("0005c · agent_progress 配对差", "results/kw_lewm_0005c_paired_noop_baseline.json",
     ("paired_progress_needs_work_stratum", "agent_progress",
      "paired_diff_lewm_minus_jepa", "mean"), 132.9, 0.05, "+132.9"),
    ("0005c · agent_disp 配对差（含 0）", "results/kw_lewm_0005c_paired_noop_baseline.json",
     ("paired_progress_needs_work_stratum", "agent_disp",
      "paired_diff_lewm_minus_jepa", "mean"), -1.7, 0.05, "−1.7"),
    ("0005c · JEPA pos_progress 均值", "results/kw_lewm_0005c_paired_noop_baseline.json",
     ("paired_progress_needs_work_stratum", "pos_progress", "jepa", "mean"),
     -45.4, 0.05, "−45.4"),
    ("0005c · JEPA 失败归因 agent 位置项", "results/kw_lewm_0005c_paired_noop_baseline.json",
     ("failure_term_attribution", "jepa", "agent_position"), 0.857, 0.001, "85.7%"),
    ("0005c · block-only no-op", "results/kw_lewm_0005c_paired_noop_baseline.json",
     ("counterfactual_block_only_criterion", "noop", "k"), 16, 0, "16/50"),
    ("0005c · block-only JEPA", "results/kw_lewm_0005c_paired_noop_baseline.json",
     ("counterfactual_block_only_criterion", "jepa", "k"), 8, 0, "8/50"),
    ("0005c · JEPA vs no-op McNemar p", "results/kw_lewm_0005c_paired_noop_baseline.json",
     ("counterfactual_block_only_criterion", "mcnemar_jepa_vs_noop",
      "p_exact_two_sided"), 0.0078, 1e-4, "0.0078"),
    ("0005c · t=0 已解任务数", "results/kw_lewm_0005c_paired_noop_baseline.json",
     ("task_difficulty_audit", "n_solved_at_start"), 0, 0, "0/50"),

    # ---- 弱动作条件 · 证据线 4（无模型地板）------------------------------
    ("0005d · 地板界 U", "results/kw_lewm_0005d_modelfree_reference.json",
     ("model_free_floor", "U"), 0.0452, 1e-4, "0.0452"),
    ("0005d · CEM 先验成功数", "results/kw_lewm_0005d_modelfree_reference.json",
     ("model_free_floor", "per_arm_pooled_wilson", "cem_prior_random", "k"), 197, 0, "3.94%"),
    ("0005d · CEM 先验 rollout 数", "results/kw_lewm_0005d_modelfree_reference.json",
     ("model_free_floor", "per_arm_pooled_wilson", "cem_prior_random", "n"), 5000, 0, "5000"),
    ("0005d · uniform 成功数", "results/kw_lewm_0005d_modelfree_reference.json",
     ("model_free_floor", "per_arm_pooled_wilson", "uniform_random", "k"), 158, 0, "3.16%"),
    ("0005d · JEPA 成功数", "results/kw_lewm_0005d_modelfree_reference.json",
     ("model_free_floor", "jepa_wilson", "k"), 1, 0, "1/50"),
    ("0005d · JEPA Wilson 上界", "results/kw_lewm_0005d_modelfree_reference.json",
     ("model_free_floor", "jepa_wilson", "hi"), 0.105, 0.001, "0.105"),
    ("0005d · LeWM 成功数", "results/kw_lewm_0005d_modelfree_reference.json",
     ("model_free_floor", "lewm_wilson", "k"), 18, 0, "18/50"),
    ("0005d · LeWM Wilson 下界", "results/kw_lewm_0005d_modelfree_reference.json",
     ("model_free_floor", "lewm_wilson", "lo"), 0.241, 0.001, "0.241"),
    ("0005d · JEPA vs CEM 先验 p", "results/kw_lewm_0005d_modelfree_reference.json",
     ("paired_mcnemar_seed0", "jepa_vs_cem_prior_random_seed0",
      "p_exact_two_sided"), 1.0, 1e-9, "1.0"),
    ("0005d · LeWM vs CEM 先验 p", "results/kw_lewm_0005d_modelfree_reference.json",
     ("paired_mcnemar_seed0", "lewm_vs_cem_prior_random_seed0",
      "p_exact_two_sided"), 1.45e-4, 1e-6, "1.45e-4"),
    ("0005d · 等位移匹配 JEPA 配对差", "results/kw_lewm_0005d_modelfree_reference.json",
     ("matched_effort_control_POST_HOC", "jepa", "agent_progress_delta", "mean"),
     4.21, 0.01, "+4.21"),
    ("0005d · 等位移匹配 JEPA CI 下界", "results/kw_lewm_0005d_modelfree_reference.json",
     ("matched_effort_control_POST_HOC", "jepa", "agent_progress_delta", "lo"),
     -11.58, 0.01, "−11.58"),
    ("0005d · 等位移匹配 JEPA CI 上界", "results/kw_lewm_0005d_modelfree_reference.json",
     ("matched_effort_control_POST_HOC", "jepa", "agent_progress_delta", "hi"),
     20.46, 0.01, "+20.46"),
    ("0005d · 等位移匹配 LeWM 配对差", "results/kw_lewm_0005d_modelfree_reference.json",
     ("matched_effort_control_POST_HOC", "lewm", "agent_progress_delta", "mean"),
     124.20, 0.01, "+124.20"),

    # ---- 弱动作条件 · 证据线 5（干预性：加大推理期预算）------------------
    # 注意 variants 是 list，顺序固定为 baseline(0) / V1(1) / V2(2) / V3(3)。
    ("0006 · baseline 成功数", "results/kw_lewm_0006_inference_ablation.json",
     ("variants", 0, "n_success"), 0, 0, "0/25"),
    ("0006 · baseline block 位移中位", "results/kw_lewm_0006_inference_ablation.json",
     ("variants", 0, "block_displacement_median"), 35.10, 0.01, "35.10"),
    ("0006 · baseline 墙钟秒", "results/kw_lewm_0006_inference_ablation.json",
     ("variants", 0, "total_wall_seconds"), 1844.3, 1.0, "1844"),
    ("0006 · V1 成功数", "results/kw_lewm_0006_inference_ablation.json",
     ("variants", 1, "n_success"), 1, 0, "1/25"),
    ("0006 · V1 越界率", "results/kw_lewm_0006_inference_ablation.json",
     ("variants", 1, "out_of_play_rate"), 0.36, 1e-9, "0.36"),
    ("0006 · V1 墙钟秒", "results/kw_lewm_0006_inference_ablation.json",
     ("variants", 1, "total_wall_seconds"), 5456.5, 1.0, "5457"),
    ("0006 · V2 block 位移中位", "results/kw_lewm_0006_inference_ablation.json",
     ("variants", 2, "block_displacement_median"), 17.21, 0.01, "17.21"),
    ("0006 · V3 越界率", "results/kw_lewm_0006_inference_ablation.json",
     ("variants", 3, "out_of_play_rate"), 0.16, 1e-9, "0.16"),

    # ---- 0006b 事后补充（越界率配对；含 Bonferroni 校正）----------------
    ("0006b · V1 越界配对 McNemar p", "results/kw_lewm_0006b_oop_analysis.json",
     ("variants", "V1_plus_iters", "mcnemar_oop", "p_exact_two_sided"),
     0.0215, 1e-4, "0.0215"),
    ("0006b · V1 Bonferroni 校正 p", "results/kw_lewm_0006b_oop_analysis.json",
     ("variants", "V1_plus_iters", "mcnemar_oop_p_bonferroni"), 0.064, 1e-3, "0.064"),
    ("0006b · V1 越界数（变体）", "results/kw_lewm_0006b_oop_analysis.json",
     ("variants", "V1_plus_iters", "oop_n_variant"), 9, 0, "9/25"),
]


def main() -> int:
    if not REPORT.exists():
        print(f"FAIL: report not found: {REPORT}")
        return 1

    text = REPORT.read_text(encoding="utf-8")
    cache: dict[str, dict] = {}

    n_ok = 0
    n_fail = 0
    failures = []

    for label, rel, path, expected, tol, needle in CHECKS:
        try:
            if rel not in cache:
                cache[rel] = load(rel)
            actual = dig(cache[rel], *path)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            failures.append(f"{label}: 读取失败 {e}")
            continue

        ok_val = abs(float(actual) - float(expected)) <= tol

        ok_txt = True
        if needle is not None:
            ok_txt = needle in text

        if ok_val and ok_txt:
            n_ok += 1
            print(f"  ok   {label}: {actual!r}")
        else:
            n_fail += 1
            if not ok_val:
                failures.append(
                    f"{label}: 数值不符 actual={actual!r} expected={expected!r} tol={tol}")
            if not ok_txt:
                failures.append(f"{label}: 报告正文缺少字符串 {needle!r}")

    print()
    print(f"METHODOLOGY_CONFOUNDS_v1 verification: "
          f"{n_ok} ok / {n_fail} fail")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        print("\n结论：报告与已归档证据不一致，禁止对外分发。")
        return 1

    print("结论：报告中的每一项数字均可回溯至已归档结果 JSON。")
    print("注意：这只证明内部一致性，**不构成**第三方验证或独立复现。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
