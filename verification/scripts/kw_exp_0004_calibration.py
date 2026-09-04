"""KW-EXP-0004: Horizon-conditioned uncertainty calibration baseline (no training).

Pre-registered analysis plan (written before results were inspected):

Q1 Capability audit (code-level):
    Does the frozen JEPA world model emit any per-sample uncertainty?
    -> Inspect unroll path for dropout / sampling / noise / ensemble.

Q2 Calibration baseline:
    Target variable: e = predicted_actual_latent_mse (latent MSE between the
    planned terminal latent and the actually realised latent).
    Because the model has no variance head, the only honest predictor is a
    MARGINAL (constant) band. Build split-conformal upper prediction bounds
    with leave-one-episode-out folds (15 folds, 3 held-out segments each).
    Score variants (all produce an upper bound on e):
      const   : s = e            -> bound = qhat
      spread  : s = e / spread   -> bound = qhat * spread
      cemloss : s = e / cem_best_loss -> bound = qhat * cem_best_loss
    Metrics: marginal coverage, mean bound (efficiency), conditional coverage
    split by episode success.

Q3 Selective risk / abstention:
    Abstain on the k segments with the largest spread (k = 20%, 40%),
    measure mean retained error; compare against random abstention baseline
    (bootstrap over the random policy).

Q4 Horizon proxy:
    Error stratified by replan position (segment index 0/1/2) with bootstrap
    95% CI. CAVEAT: this is EXECUTION POSITION inside the episode, not
    prediction horizon. All 45 segments share the same prediction horizon
    (num_act_stepped=2 latent steps == 10 env steps), so this is NOT a
    horizon-conditioned curve.
"""

import json
import pathlib
import statistics

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN = (
    ROOT
    / "results/phase0_jepa_wm_pusht/simu_env_planning/phase0"
    / "pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep15_d1objfull/pusht-base"
)

ALPHAS = [0.10, 0.20]
SCORES = {
    "const": lambda r: r["e"],
    "spread": lambda r: r["e"] / r["spread"],
    "cemloss": lambda r: r["e"] / r["cem_best_loss"],
}


def load():
    rows = []
    for p in sorted(RUN.glob("ep_*/diagnostics.json")):
        ep = p.parent.name
        for i, r in enumerate(json.load(open(p, encoding="utf-8"))):
            if r.get("predicted_actual_latent_mse") is None:
                continue
            rows.append(
                {
                    "episode": ep,
                    "seg": i,
                    "e": float(r["predicted_actual_latent_mse"]),
                    "spread": float(r["cem_elite_loss_std"]),
                    "cem_best_loss": float(r["cem_best_loss"]),
                    "success": bool(r["actual_success"]),
                }
            )
    return rows


def conformal_quantile(vals, alpha):
    """Standard conformal quantile correction, upper bound, 1-alpha coverage."""
    n = len(vals)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(np.asarray(vals, dtype=float), level))


def run_conformal(rows, alpha):
    """Leave-one-episode-out split conformal. Returns per-score metrics."""
    episodes = sorted({r["episode"] for r in rows})
    out = {}
    for name, fn in SCORES.items():
        covered, bounds, groups = [], [], {"success": [], "failure": []}
        for held in episodes:
            cal = [r for r in rows if r["episode"] != held]
            test = [r for r in rows if r["episode"] == held]
            q = conformal_quantile([fn(r) for r in cal], alpha)
            for r in test:
                b = q * (1.0 if name == "const" else (r["spread"] if name == "spread" else r["cem_best_loss"]))
                bounds.append(b)
                hit = r["e"] <= b
                covered.append(hit)
                groups["success" if r["success"] else "failure"].append(hit)
        out[name] = {
            "coverage": round(float(np.mean(covered)), 4),
            "mean_bound": round(float(np.mean(bounds)), 6),
            "coverage_success": round(float(np.mean(groups["success"])), 4),
            "coverage_failure": round(float(np.mean(groups["failure"])), 4),
            "n_success": len(groups["success"]),
            "n_failure": len(groups["failure"]),
        }
    return out


def selective_risk(rows, rng, n_boot=4000):
    """Abstain on top-k by spread vs random. Lower retained error is better."""
    errs = np.array([r["e"] for r in rows])
    spreads = np.array([r["spread"] for r in rows])
    n = len(errs)
    res = {}
    for frac in (0.2, 0.4):
        k = int(round(frac * n))
        keep = np.argsort(spreads)[: n - k]  # abstain on largest spread
        spread_risk = float(errs[keep].mean())
        rand = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.permutation(n)[: n - k]
            rand[b] = errs[idx].mean()
        res[f"abstain_{int(frac*100)}pct"] = {
            "k": k,
            "spread_policy_risk": round(spread_risk, 6),
            "random_policy_risk_mean": round(float(rand.mean()), 6),
            "random_policy_p_value": round(float((rand <= spread_risk).mean()), 4),
        }
    return res


def horizon_proxy(rows, rng, n_boot=4000):
    """Error by replan position. NOT prediction horizon - see module docstring."""
    out = {}
    for seg in sorted({r["seg"] for r in rows}):
        vals = np.array([r["e"] for r in rows if r["seg"] == seg])
        boots = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)])
        out[f"segment_{seg}"] = {
            "n": int(len(vals)),
            "mean": round(float(vals.mean()), 6),
            "std": round(float(vals.std(ddof=1)), 6),
            "ci95_low": round(float(np.percentile(boots, 2.5)), 6),
            "ci95_high": round(float(np.percentile(boots, 97.5)), 6),
        }
    return out


def main():
    rows = load()
    rng = np.random.default_rng(20260902)
    payload = {
        "experiment": "KW-EXP-0004",
        "n_segments": len(rows),
        "n_episodes": len({r["episode"] for r in rows}),
        "prediction_horizon_latent_steps": 2,
        "prediction_horizon_env_steps": 10,
        "horizon_is_constant_across_segments": True,
        "target_error_pooled": {
            "mean": round(statistics.mean(r["e"] for r in rows), 6),
            "std": round(statistics.stdev([r["e"] for r in rows]), 6),
            "min": round(min(r["e"] for r in rows), 6),
            "max": round(max(r["e"] for r in rows), 6),
        },
        "conformal": {f"alpha_{a}": run_conformal(rows, a) for a in ALPHAS},
        "selective_risk": selective_risk(rows, rng),
        "replan_position_stratified": horizon_proxy(rows, rng),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
