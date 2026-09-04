"""KW-LEWM-0005b — 用已收集的 K4 50 任务闭环数据，独立佐证 §8「弱动作条件」结论。

§8 是 latent 空间诊断（动作对预测落点无杠杆）。本脚本从**真实闭环控制**结果
（results/kw_lewm_0005_jepa_same_platform.json，50 任务）提出独立佐证：
  - 若弱动作条件成立，CEM 只能「大致推」、无法按 goal 距离精确收敛 →
    block_displacement 应与 start_goal_distance 弱相关（推多远不由目标距离决定）。
  - 成功应近似随机（与 goal 距离、与位移量均弱相关）。
  - 越界(5 个 ep)应非「目标更远更难」，而是平坦代价 + 漂移的边界失效。

纯 CPU、不训练、E1（同 50 任务单 checkpoint/seed 集）。
复用项目约定：纯 numpy 实现 Wilson + Pearson + Spearman（不引 scipy）。
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "results" / "kw_lewm_0005_jepa_same_platform.json"


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    return pearson(rx, ry)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    trials = d["trials"]
    dist = np.array([t["start_goal_distance"] for t in trials], float)
    disp = np.array([t["block_displacement"] for t in trials], float)
    succ = np.array([1.0 if (t["success"] is not None and t["success"] > 0) else 0.0 for t in trials], float)
    oop = np.array([1.0 if t["agent_out_of_play_area"] else 0.0 for t in trials], float)
    ep = np.array([t["episode_idx"] for t in trials], int)

    # 1) block_displacement vs goal_distance：弱相关 => 推多远不由目标决定（粗推、不精确收敛）
    r_disp_dist_p = pearson(disp, dist)
    r_disp_dist_s = spearman(disp, dist)

    # 2) success vs goal_distance / vs displacement：弱相关 => 成功近似随机
    r_succ_dist_p = pearson(succ, dist)
    r_succ_dist_s = spearman(succ, dist)
    r_succ_disp_p = pearson(succ, disp)
    r_succ_disp_s = spearman(succ, disp)

    # 3) 越界 episode 的目标距离 vs 在场内 episode：是否「更远更难」？
    oop_mask = oop == 1
    inplay_mask = oop == 0
    oop_dist = dist[oop_mask]
    inplay_dist = dist[inplay_mask]
    oop_disp = disp[oop_mask]
    inplay_disp = disp[inplay_mask]

    # 4) 在场内任务中：disp>20（接近 success 位置阈值）但失败的比例
    inplay = ~oop_mask
    inplay_disp = disp[inplay]
    inplay_succ = succ[inplay]
    near_but_fail = int(np.sum((inplay_disp > 20) & (inplay_succ == 0)))
    near_total = int(np.sum(inplay_disp > 20))

    # 5) 成功 episode 特征
    succ_mask = succ == 1
    succ_eps = ep[succ_mask].tolist()
    succ_dist = dist[succ_mask].tolist()
    succ_disp = disp[succ_mask].tolist()

    out = {
        "experiment_id": "KW-LEWM-0005b",
        "purpose": "Independent closed-loop corroboration of KW-LEWM-0005 §8 (weak action conditioning) using already-collected K4 50-task data",
        "n": len(trials),
        "correlations": {
            "block_displacement_vs_goal_distance": {"pearson": round(r_disp_dist_p, 3), "spearman": round(r_disp_dist_s, 3)},
            "success_vs_goal_distance": {"pearson": round(r_succ_dist_p, 3), "spearman": round(r_succ_dist_s, 3)},
            "success_vs_block_displacement": {"pearson": round(r_succ_disp_p, 3), "spearman": round(r_succ_disp_s, 3)},
        },
        "out_of_play": {
            "n": int(oop.sum()),
            "episodes": ep[oop_mask].tolist(),
            "goal_distance_mean": round(float(oop_dist.mean()), 1) if len(oop_dist) else None,
            "goal_distance_median": round(float(np.median(oop_dist)), 1) if len(oop_dist) else None,
            "inplay_goal_distance_mean": round(float(inplay_dist.mean()), 1) if len(inplay_dist) else None,
            "inplay_goal_distance_median": round(float(np.median(inplay_dist)), 1) if len(inplay_dist) else None,
            "goal_distance_ratio_oop_over_inplay_mean": round(float(oop_dist.mean() / inplay_dist.mean()), 3) if len(oop_dist) and len(inplay_dist) else None,
        },
        "near_but_failed": {
            "inplay_disp_gt_20": near_total,
            "inplay_disp_gt_20_but_failed": near_but_fail,
            "fraction": round(near_but_fail / near_total, 3) if near_total else None,
        },
        "success_episodes": {"episodes": succ_eps, "goal_distance": [round(x, 1) for x in succ_dist], "block_displacement": [round(x, 1) for x in succ_disp]},
        "interpretation_hypothesis": (
            "If block_displacement ~ uncorrelated with goal_distance AND success ~ uncorrelated with both, "
            "the closed-loop data independently corroborates §8: CEM pushes 'roughly' (gross control works) but "
            "cannot converge to the precise goal pose (weak action conditioning). OOP episodes having goal_distance "
            "comparable to in-play would further show OOP is flat-cost drift, not 'harder goal'."
        ),
    }
    out_path = Path(__file__).resolve().parents[2] / "results" / "kw_lewm_0005b_k4_corroboration.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
