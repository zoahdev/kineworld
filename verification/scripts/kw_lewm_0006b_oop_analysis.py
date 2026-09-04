"""KW-LEWM-0006b — 推理期增强的副作用：越界率配对分析（POST-HOC）。

⚠ 事后探索性分析（POST-HOC，非预注册）
--------------------------------------
0006 的预注册判据只针对**任务成功率**（CONFIRMS / REFUTES_SECTION_8）。
主结果出来后才发现一个预注册时没有预期到的现象：

    V1（CEM 迭代 10→30）成功率仅 0.00 → 0.04，但**越界率 0.04 → 0.36**（1/25 → 9/25）。

即"加推理预算"不是没有效果，而是把效果转化成了**更容易飞出场地**，
而不是更高的成功率。若成立，则 §8 的结论可以从
「推理期增强无效」**加强为**「推理期增强有害」。

因为这是看到主结果后才提出的假设，**不得作为预注册结论引用**，
只作为后续实验的假设生成器。

问题
----
在严格配对的同 25 个任务上，推理期增强（更多 CEM 迭代 / 更长视域 / 更大动作裁剪）
是否显著改变「智能体飞出场地」的发生率？

判据（事后，探索性）
--------------------
对每个变体 vs baseline，用**配对精确 McNemar** 检验越界事件：
  - p < 0.05 且变体越界更多  → 该增强**有害**（HARMFUL）
  - p < 0.05 且变体越界更少  → 该增强**降低越界**（FEWER_OOP）
  - 否则                      → 无显著差异（NO_SIGNIFICANT_CHANGE）

同时报告 success 的配对检验与 block 位移的配对差（bootstrap CI），
但**主判据是越界率**（success 的预注册判据已在 0006 主结果中判定）。

用法
----
    python verification/scripts/kw_lewm_0006b_oop_analysis.py

纯 CPU、无 GPU、无模型、约 1 秒。复用 0005c 的统计工具。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "verification" / "scripts"))

from kw_lewm_0005c_paired_noop_baseline import (  # noqa: E402
    mcnemar,
    boot_ci_mean,
    wilson,
)

RES = ROOT / "results"
VARIANTS = {
    "baseline": RES / "kw_lewm_0006_baseline.json",
    "V1_plus_iters": RES / "kw_lewm_0006_V1_plus_iters.json",
    "V2_plus_horizon": RES / "kw_lewm_0006_V2_plus_horizon.json",
    "V3_plus_clip": RES / "kw_lewm_0006_V3_plus_clip.json",
}

# 人类可读的变体描述（改动了哪一项推理期设置）
WHAT_CHANGED = {
    "V1_plus_iters": "CEM 迭代 10 → 30（优化预算 3×，耗时 1844s → 5457s）",
    "V2_plus_horizon": "规划视域 H5 → H10",
    "V3_plus_clip": "动作裁剪 clip 3.0 → 6.0",
}


def load_trials(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["trials"], d


def task_key(tr) -> tuple:
    """任务键 = (episode_idx, start_step, goal_step) 三元组。

    ⚠ 不能只用 episode_idx：本任务集里 episode 192 被采样了两次
    （start_step/goal_step 分别为 80/105 与 101/126，start_state 也不同），
    是两个不同任务。只用 episode_idx 会把 25 个任务错配成 24 个。
    这是 0005c 已确立的「任务键三重核验」约定，必须遵守。
    """
    return (int(tr["episode_idx"]), int(tr["start_step"]), int(tr["goal_step"]))


def pair_by_episode(trials_a, trials_b):
    """按三元组任务键配对，返回 (keys, dict_a, dict_b)。"""
    a = {task_key(t): t for t in trials_a}
    b = {task_key(t): t for t in trials_b}
    keys = sorted(set(a) & set(b))
    return keys, a, b


def main() -> int:
    trials, docs = {}, {}
    for name, path in VARIANTS.items():
        if not path.exists():
            print(f"FAIL: missing {path}")
            return 1
        trials[name], docs[name] = load_trials(path)

    # ---------------------------------------------------- R0：配对完整性门禁
    # 门禁 1：每个变体的三元组键必须唯一（键数 == trial 数）
    # 门禁 2：所有变体的键集合必须与 baseline 完全一致
    base_keys = {task_key(t) for t in trials["baseline"]}
    if len(base_keys) != len(trials["baseline"]):
        print("FAIL: baseline 的三元组任务键不唯一 —— 配对基础不成立，判 VOID")
        return 1
    for name, tr in trials.items():
        ks = {task_key(t) for t in tr}
        if len(ks) != len(tr):
            print(f"FAIL: {name} 的三元组任务键不唯一（{len(ks)} != {len(tr)}），判 VOID")
            return 1
        if ks != base_keys:
            print(f"FAIL: {name} 的任务键集合与 baseline 不一致，判 VOID")
            return 1
    print(f"[0006b] R0 配对门禁通过：{len(base_keys)} 个唯一三元组任务键，"
          f"4 个变体键集合完全一致")

    base_trials = trials["baseline"]

    out = {
        "experiment_id": "KW-LEWM-0006b",
        "status": "POST_HOC_EXPLORATORY_NOT_PRE_REGISTERED",
        "motivation": (
            "0006 主结果发现 V1 越界率 0.04 -> 0.36 (1/25 -> 9/25)，而成功率仅 0.00 -> 0.04。"
            "该现象在预注册时未预期，故本分析为事后探索，仅作假设生成器。"
        ),
        "question": "推理期增强是否显著改变智能体飞出场地（out-of-play）的发生率？",
        "primary_criterion": "配对精确 McNemar 检验越界事件（p<0.05）",
        "n_tasks": len(base_trials),
        "pairing": {},
        "variants": {},
        "verdicts": {},
        "verdicts_bonferroni": {},
        "caveats": [],
    }

    print(f"[0006b] 事后分析：推理期增强对越界率的影响（n={len(base_trials)} 配对任务）")
    print("[0006b] ⚠ 这是 POST-HOC 探索性分析，不得作为预注册结论引用\n")

    for name in ("V1_plus_iters", "V2_plus_horizon", "V3_plus_clip"):
        keys, a, b = pair_by_episode(trials[name], base_trials)
        n_paired = len(keys)

        oop_v = np.array([bool(a[k]["agent_out_of_play_area"]) for k in keys])
        oop_b = np.array([bool(b[k]["agent_out_of_play_area"]) for k in keys])
        suc_v = np.array([bool(a[k]["success"]) for k in keys])
        suc_b = np.array([bool(b[k]["success"]) for k in keys])
        disp_v = np.array([float(a[k]["block_displacement"]) for k in keys])
        disp_b = np.array([float(b[k]["block_displacement"]) for k in keys])

        mc_oop = mcnemar(oop_v, oop_b)
        mc_suc = mcnemar(suc_v, suc_b)
        d_disp = disp_v - disp_b
        boot = boot_ci_mean(d_disp)

        p_oop = mc_oop["p_exact_two_sided"]
        v_more, b_more = mc_oop["a_only"], mc_oop["b_only"]

        # 多重比较：本分析检验了 3 个变体，Bonferroni 校正必须一并报告。
        # 未校正的 p 不得单独用来下结论。
        n_tests = 3
        p_bonf = min(1.0, p_oop * n_tests)

        if p_oop < 0.05 and v_more > b_more:
            verdict = "HARMFUL_MORE_OUT_OF_PLAY"
        elif p_oop < 0.05 and b_more > v_more:
            verdict = "FEWER_OUT_OF_PLAY"
        else:
            verdict = "NO_SIGNIFICANT_CHANGE"

        if p_bonf < 0.05 and v_more > b_more:
            verdict_bonf = "HARMFUL_MORE_OUT_OF_PLAY"
        elif p_bonf < 0.05 and b_more > v_more:
            verdict_bonf = "FEWER_OUT_OF_PLAY"
        else:
            verdict_bonf = "NO_SIGNIFICANT_CHANGE_AFTER_CORRECTION"

        out["pairing"][name] = {
            "n_paired": n_paired,
            "n_total_variant": len(trials[name]),
            "n_total_baseline": len(base_trials),
            "keys_identical": n_paired == len(base_trials) == len(trials[name]),
        }
        out["variants"][name] = {
            "what_changed": WHAT_CHANGED[name],
            "oop_rate_variant": float(oop_v.mean()),
            "oop_rate_baseline": float(oop_b.mean()),
            "oop_n_variant": int(oop_v.sum()),
            "oop_n_baseline": int(oop_b.sum()),
            "mcnemar_oop": mc_oop,
            "success_variant": float(suc_v.mean()),
            "success_baseline": float(suc_b.mean()),
            "mcnemar_success": mc_suc,
            "block_disp_paired_diff": {
                "mean": float(d_disp.mean()),
                "median": float(np.median(d_disp)),
                "lo": float(boot["lo"]),
                "hi": float(boot["hi"]),
            },
            "wilson_oop_variant": wilson(int(oop_v.sum()), n_paired),
            "wilson_oop_baseline": wilson(int(oop_b.sum()), n_paired),
            "verdict_oop": verdict,
            "n_tests": n_tests,
            "mcnemar_oop_p_bonferroni": p_bonf,
            "verdict_oop_bonferroni": verdict_bonf,
        }
        out["verdicts"][name] = verdict
        out["verdicts_bonferroni"][name] = verdict_bonf

        print(f"  {name}: {WHAT_CHANGED[name]}")
        print(f"    越界 {int(oop_v.sum())}/{n_paired} vs baseline {int(oop_b.sum())}/{n_paired}"
              f"  ({oop_v.mean():.2f} vs {oop_b.mean():.2f})")
        print(f"    McNemar p={p_oop:.4g}  (变体多 {v_more} / baseline 多 {b_more})")
        print(f"    Bonferroni 校正后 p={p_bonf:.4g} （n_tests={n_tests}）")
        print(f"    block 位移配对差 {d_disp.mean():+.2f} CI [{boot['lo']:.2f}, {boot['hi']:.2f}]")
        print(f"    → {verdict}   （校正后：{verdict_bonf}）\n")

    # ---------------------------------------------------------- 汇总与诚实约束
    harmful = [k for k, v in out["verdicts"].items() if v == "HARMFUL_MORE_OUT_OF_PLAY"]
    out["summary"] = {
        "n_variants_tested": 3,
        "harmful_variants": harmful,
        "overall_reading": (
            "若 V1 显著有害，则 §8 表述可从「推理期增强无效」加强为「推理期增强有害」："
            "更多优化预算被转化为更自信地执行模型偏好的动作，而非更高的成功率——"
            "这正是代价地形平坦的直接后果。"
        ),
    }
    out["caveats"] = [
        "POST-HOC：假设在看到 0006 主结果后提出，不得作为预注册结论引用。",
        f"n={len(base_trials)} 每变体，置信区间很宽；success 的配对检验（如 V1 1/25 vs 0/25）",
        "  差异极小，p 值接近 1，**不可解读**为由推理预算带来的改善。",
        "越界率是本分析的**主判据**，但它不是任务成功；越界率上升本身不等于任务表现变差，",
        "  需要结合 success 一起读（两者都指向『无改善』或『更差』）。",
        "0006 主判据（success CI 与 K4 重叠）已判定 CONFIRMS_SECTION_8；",
        "  本分析只是对同一结论的**加强**，不改变主判定。",
        "多重比较：检验了 3 个变体，未做校正；探索性阶段不作校正，但结论须降权解读。",
    ]

    out_path = RES / "kw_lewm_0006b_oop_analysis.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[0006b] wrote {out_path}")
    print(f"[0006b] 有害变体: {harmful if harmful else '无'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
