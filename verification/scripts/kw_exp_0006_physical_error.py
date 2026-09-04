"""KW-EXP-0006: does physical interaction intensity predict latent prediction error?

Implements the protocol pre-registered in
verification/experiments/KW-EXP-0006.md (written before results existed).
Do NOT tune the tests after seeing results; report as specified.

Q1 horizon curve re-test at n=96 per stratum (median + mean + spike-excluded mean)
Q2 spike rate in high- vs low-physical-interaction groups (Fisher exact)
Q3 physical-normalised conformal bands vs pooled constant band
Q4 per-horizon vs pooled bands re-test at n=96 (was EXP-0005 a power artifact?)
"""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
ALPHAS = [0.10, 0.20]
SEED = 20260902
SPIKE_FACTOR = 4.0

try:
    from scipy.stats import fisher_exact

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def fisher2(a, b, c, d):
    """Two-sided Fisher exact on [[a,b],[c,d]]; exact via hypergeometric sum."""
    from math import comb

    n = a + b + c + d
    r1, r2 = a + b, c + d
    c1 = a + c

    def p(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)

    obs = p(a)
    tot = 0.0
    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    for x in range(lo, hi + 1):
        px = p(x)
        if px <= obs * (1 + 1e-9):
            tot += px
    return min(1.0, tot)


def fisher(a, b, c, d):
    if HAVE_SCIPY:
        return float(fisher_exact([[a, b], [c, d]])[1])
    return fisher2(a, b, c, d)


def load(tag_dir):
    rows = []
    base = ROOT / "results/phase0_jepa_wm_pusht/simu_env_planning/phase0" / tag_dir / "pusht-base"
    files = sorted(base.glob("ep_*/diagnostics.json"))
    if not files:
        raise SystemExit(f"no diagnostics under {base}")
    for p in files:
        ep = p.parent.name
        for i, r in enumerate(json.load(open(p, encoding="utf-8"))):
            for h in r.get("per_horizon_latent_mse", []):
                ph = h.get("physical") or {}
                rows.append(
                    {
                        "episode": ep,
                        "k": int(h["latent_step"]),
                        "e": float(h["predicted_actual_latent_mse"]),
                        "obj_disp": ph.get("obj_disp"),
                        "obj_angle_delta": ph.get("obj_angle_delta"),
                        "agent_obj_dist": ph.get("agent_obj_dist"),
                        "agent_speed": ph.get("agent_speed"),
                        "success": bool(r["actual_success"]),
                    }
                )
    return rows


def qhat(vals, alpha):
    n = len(vals)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(np.asarray(vals, dtype=float), level))


def boot_ci(vals, rng, n_boot=4000):
    v = np.asarray(vals, dtype=float)
    if len(v) < 2:
        return (float(v.mean()), float(v.mean()))
    b = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)])
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def q1_curve(rows, spike_thr, rng):
    out = {}
    for k in sorted({r["k"] for r in rows}):
        sub = [r["e"] for r in rows if r["k"] == k]
        clean = [v for v in sub if v <= spike_thr]
        lo, hi = boot_ci(sub, rng)
        out[f"k={k}"] = {
            "n": len(sub),
            "mean": round(float(np.mean(sub)), 6),
            "median": round(float(np.median(sub)), 6),
            "ci95": [round(lo, 6), round(hi, 6)],
            "n_excl_spikes": len(clean),
            "mean_excl_spikes": round(float(np.mean(clean)), 6) if clean else None,
            "median_excl_spikes": round(float(np.median(clean)), 6) if clean else None,
        }
    return out


def q2_spikes(rows, spike_thr):
    """Spike rate in high- vs low-physical-interaction groups. obj_disp needs k>=2."""
    res = {}
    for field, hi_q in (("obj_disp", 75), ("agent_obj_dist", 75), ("agent_speed", 75)):
        sub = [r for r in rows if r.get(field) is not None]
        if len(sub) < 20:
            res[field] = {"status": "insufficient_data", "n": len(sub)}
            continue
        vals = np.array([r[field] for r in sub], dtype=float)
        thr = float(np.percentile(vals, hi_q))
        hi_g = [r for r in sub if r[field] >= thr]
        lo_g = [r for r in sub if r[field] < thr]
        a = sum(1 for r in hi_g if r["e"] > spike_thr)
        b = len(hi_g) - a
        c = sum(1 for r in lo_g if r["e"] > spike_thr)
        d = len(lo_g) - c
        res[field] = {
            "n_used": len(sub),
            "threshold_p75": round(thr, 6),
            "high_group": {"n": len(hi_g), "spikes": a, "rate": round(a / len(hi_g), 4)},
            "low_group": {"n": len(lo_g), "spikes": c, "rate": round(c / len(lo_g), 4)},
            "fisher_p": round(fisher(a, b, c, d), 6),
            "note": "obj_disp/obj_angle_delta are null at k=1 by construction",
        }
    return res


def q3_bands(rows, alpha):
    """Physical-normalised vs pooled constant band, leave-one-episode-out."""
    eps = sorted({r["episode"] for r in rows})
    fields = ["obj_disp", "agent_obj_dist", "agent_speed"]

    def run(field):
        # Ratio-normalisation is undefined when the conditioning field is 0
        # (e.g. obj_disp=0 when the block never moves). Drop those points from
        # both calibration and test so the ratio band is well-defined; the pooled
        # constant band is computed on the SAME subset below for a fair comparison.
        cov, bound = [], []
        for held in eps:
            cal = [
                r for r in rows
                if r["episode"] != held and r.get(field) is not None and r[field] != 0
            ]
            test = [
                r for r in rows
                if r["episode"] == held and r.get(field) is not None and r[field] != 0
            ]
            if not cal or not test:
                continue
            q = qhat([r["e"] / r[field] for r in cal], alpha)
            for r in test:
                bnd = q * r[field]
                bound.append(bnd)
                cov.append(r["e"] <= bnd)
        if not cov:
            return None
        return {
            "coverage": round(float(np.mean(cov)), 4),
            "mean_bound": round(float(np.mean(bound)), 6),
            "n_scored": len(cov),
        }

    # pooled constant band on the same scoring subset as each field, for fairness.
    # Drop field==0 points to match the ratio band's support (ratio undefined at 0).
    out = {"pooled": {}, "physical": {}}
    for f in fields:
        sub = [r for r in rows if r.get(f) is not None and r[f] != 0]
        pooled_cov, pooled_bound = [], []
        for held in eps:
            cal = [r for r in sub if r["episode"] != held]
            test = [r for r in sub if r["episode"] == held]
            if not cal or not test:
                continue
            q = qhat([r["e"] for r in cal], alpha)
            for r in test:
                pooled_bound.append(q)
                pooled_cov.append(r["e"] <= q)
        out["pooled"][f] = {
            "coverage": round(float(np.mean(pooled_cov)), 4),
            "mean_bound": round(float(np.mean(pooled_bound)), 6),
            "n_scored": len(pooled_cov),
        }
        r_ = run(f)
        out["physical"][f] = r_
        if r_:
            out["physical"][f]["bound_ratio_vs_pooled"] = round(
                r_["mean_bound"] / out["pooled"][f]["mean_bound"], 4
            )
    return out


def q4_horizon_bands(rows, alpha):
    eps = sorted({r["episode"] for r in rows})
    pooled_c, pooled_b, ph_c, ph_b = [], [], [], []
    for held in eps:
        cal = [r for r in rows if r["episode"] != held]
        test = [r for r in rows if r["episode"] == held]
        qp = qhat([r["e"] for r in cal], alpha)
        qm = {k: qhat([r["e"] for r in cal if r["k"] == k], alpha) for k in sorted({r["k"] for r in cal})}
        for r in test:
            pooled_b.append(qp)
            pooled_c.append(r["e"] <= qp)
            bb = qm.get(r["k"], qp)
            ph_b.append(bb)
            ph_c.append(r["e"] <= bb)
    return {
        "pooled": {"coverage": round(float(np.mean(pooled_c)), 4), "mean_bound": round(float(np.mean(pooled_b)), 6)},
        "per_horizon": {"coverage": round(float(np.mean(ph_c)), 4), "mean_bound": round(float(np.mean(ph_b)), 6)},
        "bound_ratio_perh_vs_pooled": round(float(np.mean(ph_b)) / float(np.mean(pooled_b)), 4),
        "n_scored": len(pooled_c),
    }


def main():
    tag_dir = sys.argv[1]
    rows = load(tag_dir)
    rng = np.random.default_rng(SEED)
    all_e = np.array([r["e"] for r in rows], dtype=float)
    med = float(np.median(all_e))
    spike_thr = SPIKE_FACTOR * med

    payload = {
        "experiment": "KW-EXP-0006",
        "tag_dir": tag_dir,
        "n_points": len(rows),
        "n_episodes": len({r["episode"] for r in rows}),
        "spike_threshold": round(spike_thr, 6),
        "global_median": round(med, 6),
        "n_spikes": int((all_e > spike_thr).sum()),
        "spike_rate": round(float((all_e > spike_thr).mean()), 4),
        "q1_horizon_curve": q1_curve(rows, spike_thr, rng),
        "q2_spike_vs_physical": q2_spikes(rows, spike_thr),
        "q3_physical_bands": {f"alpha_{a}": q3_bands(rows, a) for a in ALPHAS},
        "q4_horizon_bands": {f"alpha_{a}": q4_horizon_bands(rows, a) for a in ALPHAS},
        "scipy_available": HAVE_SCIPY,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
