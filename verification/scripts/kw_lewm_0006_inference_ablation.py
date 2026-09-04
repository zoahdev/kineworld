"""KW-LEWM-0006 — 推理期增强 ablation：实证检验 KW-LEWM-0005 §8「推理期调参无效」。

对冻结 JEPA-WM（JEPAWMAdapter）在同 stable-worldmodel harness、同 0004 的 25 任务子集上，
仅改 planner 推理期设置（horizon / CEM n_steps / action_clip），跑 4 个变体：
  baseline   : H=5, CEM 10x100x10, clip=3.0   （应复现 K4 success≈0.02）
  V1_plus_iters   : n_steps 10 -> 30
  V2_plus_horizon: horizon 5 -> 10 (覆盖整段 eval)
  V3_plus_clip    : clip 3.0 -> 6.0

预测（§8）：所有变体 success 仍钉在 flat-cost 地板（CI 与 K4 [0.004,0.105] 重叠）→ 确认弱动作条件、
推理期不可修。若任一变体显著抬高 success → 证伪 §8，flat cost 为代价度量伪影。

纯推理期设置变动，不训练。复用 kw_lewm_0005_jepa_adapter.run_k4（已支持配置覆盖）。
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external" / "jepa-wms"))
import kw_lewm_0005_jepa_adapter as A

DEVICE = "cuda"
PAIRS_FROM = str(ROOT / "results" / "kw_lewm_0004_dataset_tasks_p50.json")
N_TASKS = 25
OUT_DIR = ROOT / "results"

# (name, overrides)  —— 仅改 planner 推理期设置
VARIANTS = [
    ("baseline",        dict(plan_horizon=5,  plan_receding=5,  cem_n_steps=10, cem_num_samples=100, action_clip=3.0)),
    ("V1_plus_iters",   dict(plan_horizon=5,  plan_receding=5,  cem_n_steps=30, cem_num_samples=100, action_clip=3.0)),
    ("V2_plus_horizon", dict(plan_horizon=10, plan_receding=10, cem_n_steps=10, cem_num_samples=100, action_clip=3.0)),
    ("V3_plus_clip",    dict(plan_horizon=5,  plan_receding=5,  cem_n_steps=10, cem_num_samples=100, action_clip=6.0)),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    print(f"[0006] building JEPAWMAdapter once (reused across {len(VARIANTS)} variants) ...")
    adapter, meta, stats, _ = A.build_jepa_adapter(device=DEVICE)
    print(f"[0006] adapter OK: enc={meta['encoder_params']:,} total={meta['total_params']:,} "
          f"model_action_dim={meta['model_action_dim']}")

    rows = []
    for name, ov in VARIANTS:
        out_path = OUT_DIR / f"kw_lewm_0006_{name}.json"
        print(f"\n[0006] === variant {name} : {ov} ===")
        payload = A.run_k4(
            adapter, meta, stats,
            n_tasks=N_TASKS, device=DEVICE, output=str(out_path), pairs_from=PAIRS_FROM,
            cem_num_samples=ov["cem_num_samples"], cem_n_steps=ov["cem_n_steps"], cem_topk=10,
            plan_horizon=ov["plan_horizon"], plan_receding=ov["plan_receding"],
            plan_action_block=5, action_clip=ov["action_clip"],
        )
        s = payload["summary"]
        n_succ = int(s["n_success"]) if s["n_success"] is not None else 0
        n = payload["n_tasks"]
        ci = wilson(n_succ, n)
        rows.append({
            "variant": name,
            "overrides": ov,
            "n_tasks": n,
            "success": s["success_rate"],
            "n_success": n_succ,
            "success_wilson_ci": [round(ci[0], 3), round(ci[1], 3)],
            "out_of_play_rate": s["out_of_play_rate"],
            "n_out_of_play": s["n_out_of_play"],
            "block_displacement_median": s["block_displacement_median"],
            "block_displacement_mean": s["block_displacement_mean"],
            "block_interaction_rate": s["block_interaction_rate"],
            "n_errors": s["n_errors"],
            "total_wall_seconds": round(s["total_wall_seconds"], 1),
        })
        print(f"[0006] {name}: success={s['success_rate']} CI={ci} oop={s['out_of_play_rate']} "
              f"disp_med={s['block_displacement_median']}")

    # 与 K4 baseline 比较（K4 50 任务 success=0.02 CI [0.004,0.105]）
    k4_ci = (0.004, 0.105)
    baseline_ci = rows[0]["success_wilson_ci"]
    # 确认 §8：所有变体（含 baseline）CI 与 K4 重叠，且无 material 提升
    all_overlap = all(
        (r["success_wilson_ci"][0] <= k4_ci[1]) and (r["success_wilson_ci"][1] >= k4_ci[0])
        for r in rows
    )
    # 证伪 §8：任一变体 success 点估计 >0.15 且 CI 下界 > K4 上界
    any_refutes = any(
        (r["success"] is not None and r["success"] > 0.15)
        and (r["success_wilson_ci"][0] > k4_ci[1])
        for r in rows
    )
    if any_refutes:
        verdict = "REFUTES_SECTION_8 (inference-period tuning raised success -> flat cost is a cost-metric artifact, pivot to cost-metric fix)"
    elif all_overlap:
        verdict = "CONFIRMS_SECTION_8 (all variants pinned at flat-cost floor; weak action conditioning confirmed, inference tuning futile)"
    else:
        verdict = "INCONCLUSIVE (harness/repro issue; see per-variant CI vs K4)"

    out = {
        "experiment_id": "KW-LEWM-0006",
        "question": "Does increasing planner inference-period budget (horizon / CEM iters / action clip) improve frozen JEPA-WM task-success, or does success stay pinned at the flat-cost floor (confirming KW-LEWM-0005 §8 weak action conditioning)?",
        "design": "same JEPAWMAdapter + stable-worldmodel harness + same 0004 25-task subset; only planner settings vary; no training",
        "evidence_level": "E1_INTERNAL_REPRODUCTION",
        "n_tasks_per_variant": N_TASKS,
        "k4_baseline_ci_reference": list(k4_ci),
        "verdict": verdict,
        "all_variants_ci_overlap_k4": bool(all_overlap),
        "variants": rows,
    }
    out_path = OUT_DIR / "kw_lewm_0006_inference_ablation.json"
    out_path.write_text(json.dumps(out, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    print(f"\n[0006] verdict={verdict}")
    print(f"[0006] wrote {out_path}")
    return out


if __name__ == "__main__":
    main()
