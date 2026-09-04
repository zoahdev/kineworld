"""KW-EXP-0005: Multi-horizon rollout error + horizon-conditioned calibration baseline.

Pre-registered analysis plan (written BEFORE results were inspected):

Context (from KW-EXP-0004):
    Every existing segment was scored at a single fixed prediction horizon
    (num_act_stepped = 2 latent steps = 10 env steps), so no horizon-error curve
    existed and horizon-conditioned calibration was not evaluable. This run uses
    num_act_stepped = 6 (official open-loop setting) so that ground truth exists
    at latent steps 1..6 (env frame index k*frameskip-1).

Q1 Error-vs-horizon curve:
    Mean predicted-vs-actual latent MSE per latent step k = 1..6, with bootstrap
    95% CI. Monotonic growth is expected; the SHAPE and the DISPERSION are the
    useful outputs, not the sign of the slope.

Q2 Plan optimism (objective mismatch) vs horizon:
    Per horizon, compare predicted_goal_latent_mse against actual_goal_latent_mse.
    optimism(k) = actual_goal_mse(k) - predicted_goal_mse(k).
    Positive and growing optimism => the planner believes it is approaching the
    goal while the realised trajectory is not. This is a direct, horizon-resolved
    test of the "planner-objective vs physical grounding mismatch" hypothesis.

Q3 Horizon-conditioned conformal calibration (now evaluable):
    Compare two upper prediction bands via leave-one-episode-out (15 folds):
      pooled   : one constant band fitted on all horizons
      per_h    : one constant band per latent step k
    Report marginal coverage and mean bound for both. Horizon is a USEFUL
    conditioning variable only if per_h reaches target coverage with a
    materially lower mean bound than pooled. This is the same test that
    rejected elite_loss_std as a conditioning variable in KW-EXP-0004.

Q4 Conditional coverage:
    Coverage split by episode success for both band types.

Kill criteria (pre-registered):
    If per_h does not beat pooled on mean bound at equal coverage, horizon is
    recorded as NOT a useful conditioning variable and we stop this direction.
"""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
ALPHAS = [0.10, 0.20]
SEED = 20260902


def load(tag_dir):
    rows = []
    base = ROOT / "results/phase0_jepa_wm_pusht/simu_env_planning/phase0" / tag_dir / "pusht-base"
    for p in sorted(base.glob("ep_*/diagnostics.json")):
        ep = p.parent.name
        for i, r in enumerate(json.load(open(p, encoding="utf-8"))):
            for h in r.get("per_horizon_latent_mse", []):
                rows.append(
                    {
                        "episode": ep,
                        "seg": i,
                        "k": int(h["latent_step"]),
                        "e": float(h["predicted_actual_latent_mse"]),
                        "pred_goal": float(h["predicted_goal_latent_mse"]),
                        "act_goal": float(h["actual_goal_latent_mse"]),
                        "success": bool(r["actual_success"]),
                        "end_dist": float(r["actual_state_dist"]),
                    }
                )
    return rows


def boot_ci(vals, rng, n_boot=4000):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return (float(vals.mean()), float(vals.mean()))
    b = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)])
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def qhat(vals, alpha):
    n = len(vals)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(np.asarray(vals, dtype=float), level))


def bands(rows, alpha):
    """Leave-one-episode-out: pooled constant band vs per-horizon constant bands."""
    eps = sorted({r["episode"] for r in rows})
    pooled = {"cov": [], "bound": []}
    per_h = {"cov": [], "bound": []}
    for held in eps:
        cal = [r for r in rows if r["episode"] != held]
        test = [r for r in rows if r["episode"] == held]
        q_pool = qhat([r["e"] for r in cal], alpha)
        q_map = {}
        for k in sorted({r["k"] for r in cal}):
            sub = [r["e"] for r in cal if r["k"] == k]
            if sub:
                q_map[k] = qhat(sub, alpha)
        for r in test:
            pooled["bound"].append(q_pool)
            pooled["cov"].append(r["e"] <= q_pool)
            b = q_map.get(r["k"], q_pool)
            per_h["bound"].append(b)
            per_h["cov"].append(r["e"] <= b)
    out = {}
    for name, d in (("pooled", pooled), ("per_horizon", per_h)):
        out[name] = {
            "coverage": round(float(np.mean(d["cov"])), 4),
            "mean_bound": round(float(np.mean(d["bound"])), 6),
        }
    out["mean_bound_reduction_pct"] = round(
        100.0 * (1 - out["per_horizon"]["mean_bound"] / out["pooled"]["mean_bound"]), 2
    )
    return out


def conditional(rows, alpha):
    """Coverage by success group for the per-horizon band."""
    eps = sorted({r["episode"] for r in rows})
    g = {"success": [], "failure": []}
    for held in eps:
        cal = [r for r in rows if r["episode"] != held]
        test = [r for r in rows if r["episode"] == held]
        q_map = {
            k: qhat([r["e"] for r in cal if r["k"] == k], alpha)
            for k in sorted({r["k"] for r in cal})
        }
        for r in test:
            b = q_map.get(r["k"], qhat([x["e"] for x in cal], alpha))
            g["success" if r["success"] else "failure"].append(r["e"] <= b)
    return {
        "coverage_success": round(float(np.mean(g["success"])), 4) if g["success"] else None,
        "coverage_failure": round(float(np.mean(g["failure"])), 4) if g["failure"] else None,
        "n_success": len(g["success"]),
        "n_failure": len(g["failure"]),
    }


def main():
    tag_dir = sys.argv[1]
    rows = load(tag_dir)
    if not rows:
        raise SystemExit(f"no per_horizon_latent_mse rows under {tag_dir}")
    rng = np.random.default_rng(SEED)

    curve, optimism = {}, {}
    for k in sorted({r["k"] for r in rows}):
        sub = [r for r in rows if r["k"] == k]
        e = [r["e"] for r in sub]
        lo, hi = boot_ci(e, rng)
        og = [r["act_goal"] - r["pred_goal"] for r in sub]
        olo, ohi = boot_ci(og, rng)
        curve[f"k={k}"] = {
            "n": len(sub),
            "mean_mse": round(float(np.mean(e)), 6),
            "median_mse": round(float(np.median(e)), 6),
            "std_mse": round(float(np.std(e, ddof=1)), 6) if len(e) > 1 else None,
            "min_mse": round(float(np.min(e)), 6),
            "max_mse": round(float(np.max(e)), 6),
            "mean_over_median": round(float(np.mean(e) / np.median(e)), 3),
            "ci95": [round(lo, 6), round(hi, 6)],
        }
        optimism[f"k={k}"] = {
            "mean_predicted_goal_mse": round(float(np.mean([r["pred_goal"] for r in sub])), 6),
            "mean_actual_goal_mse": round(float(np.mean([r["act_goal"] for r in sub])), 6),
            "mean_optimism": round(float(np.mean(og)), 6),
            "ci95": [round(olo, 6), round(ohi, 6)],
        }

    # Outlier attribution: which episodes drive the heavy tail of the error curve.
    by_ep = {}
    for r in rows:
        by_ep.setdefault(r["episode"], []).append(r)
    ep_rank = sorted(
        (
            {
                "episode": ep,
                "mean_mse": round(float(np.mean([x["e"] for x in v])), 6),
                "median_mse": round(float(np.median([x["e"] for x in v])), 6),
                "max_mse": round(float(np.max([x["e"] for x in v])), 6),
                "k1_mse": round(float([x["e"] for x in v if x["k"] == 1][0]), 6),
                "success": bool(v[-1]["success"]),
                "end_dist": round(float(v[-1]["end_dist"]), 3),
            }
            for ep, v in by_ep.items()
        ),
        key=lambda d: -d["max_mse"],
    )
    top_offenders = {}
    for k in sorted({r["k"] for r in rows}):
        s = sorted((r for r in rows if r["k"] == k), key=lambda r: -r["e"])[:3]
        top_offenders[f"k={k}"] = [
            {"episode": r["episode"], "mse": round(r["e"], 6)} for r in s
        ]

    payload = {
        "experiment": "KW-EXP-0005",
        "tag_dir": tag_dir,
        "n_points": len(rows),
        "n_episodes": len({r["episode"] for r in rows}),
        "horizons": sorted({r["k"] for r in rows}),
        "q1_error_curve": curve,
        "q1b_outlier_attribution": {
            "top_offenders_by_horizon": top_offenders,
            "episodes_ranked_by_max_mse": ep_rank,
        },
        "q2_plan_optimism": optimism,
        "q3_horizon_conditioned_bands": {f"alpha_{a}": bands(rows, a) for a in ALPHAS},
        "q4_conditional_coverage": {f"alpha_{a}": conditional(rows, a) for a in ALPHAS},
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
