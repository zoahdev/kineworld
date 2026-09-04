#!/usr/bin/env python
"""KW-LEWM-0005c — Paired re-analysis: no-op baseline, failure-term attribution,
and agent-navigation vs object-manipulation decomposition.

STATUS: POST-HOC re-analysis of existing logs (NOT a pre-registered data
collection). No new rollouts, GPU-free. Evidence level E1.
Hypotheses and decision rules below were fixed BEFORE any statistic was
computed, so the verdict cannot be rationalised after the fact.

GROUND-TRUTH SUCCESS CRITERION (read from source, not memory)
------------------------------------------------------------
stable_worldmodel/envs/pusht/env.py::eval_state (verbatim):

    pos_diff   = np.linalg.norm(goal_state[:4] - cur_state[:4])
    angle_diff = np.abs(goal_state[4] - cur_state[4])
    angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
    success    = pos_diff < 20 and angle_diff < np.pi/9

CRITICAL: state[:4] = [agent_x, agent_y, block_x, block_y].  The AGENT
position and the BLOCK position share a single 20-px joint budget.  The task
therefore also requires the agent to arrive at the expert's agent position at
goal_step.  An earlier version of this script used a block-only pos_diff and
reproduced only 81% of logged successes; that version was VOID and this one
supersedes it.  The correction was made by reading env.py, not by tuning to
the data.

MOTIVATION
----------
KW-LEWM-0004 (LeWM) and KW-LEWM-0005 K4 (JEPA-WM) ran on the *same 50 tasks*
(pairing verified below).  Block-displacement magnitudes are nearly identical
(mean 51.6 vs 52.1) yet success is 18/50 vs 1/50.  So the gap is about WHERE
things end up, not HOW MUCH they move.  Because the agent DOF is directly
actuated, its terminal error is the cleanest behavioural readout of action
conditioning strength — the quantity §8 diagnosed in latent space.

PRE-DECLARED HYPOTHESES
-----------------------
H1 (baseline inflation): a no-op policy (never move) succeeds on exactly the
   tasks already inside tolerance at t=0.  If Wilson95(LeWM) overlaps
   Wilson95(no-op), then KW-LEWM-0004's headline 0.36 must be re-framed.
H2 (JEPA harmful): JEPA's success is significantly BELOW the no-op baseline
   (exact McNemar p<0.05 and lower point estimate) — i.e. it destroys poses
   that were already satisfied.
H3 (agent-navigation dominance): the agent-position term dominates JEPA's
   failures, and paired agent-gap progress is significantly larger for LeWM
   than for JEPA.
H4 (object manipulation): under a counterfactual BLOCK-ONLY criterion
   (drop the agent term), JEPA's success stays low.  If it stays low, the
   failure is not merely agent navigation; if it jumps to LeWM's level, the
   failure is agent navigation only.

DECISION RULES (fixed before computation)
-----------------------------------------
R0 the reconstructed predicate must reproduce logged `success` for >= 98% of
   the 100 trials; otherwise VOID.
R1 H1 supported iff Wilson95(LeWM) and Wilson95(no-op) overlap.
R2 H2 supported iff McNemar p<0.05 AND jepa_rate < noop_rate.
R3 H3 supported iff (a) the agent term is the single largest contributor to
   pos_diff_final in > 50% of JEPA failures, AND (b) the paired bootstrap 95%
   CI of (agent_gap_progress_LeWM - agent_gap_progress_JEPA) excludes 0.
R4 H4 supported iff JEPA block-only success Wilson95 upper bound < 0.30.

Outputs: results/kw_lewm_0005c_paired_noop_baseline.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

JEPA_JSON = os.path.join(ROOT, "results", "kw_lewm_0005_jepa_same_platform.json")
LEWM_JSON = os.path.join(ROOT, "results", "kw_lewm_0004_dataset_tasks_p50.json")
OUT_JSON = os.path.join(ROOT, "results", "kw_lewm_0005c_paired_noop_baseline.json")

POS_TOL = 20.0
ANG_TOL = math.pi / 9.0


# ----------------------------------------------------------------------------- stats
def wilson(k: int, n: int, z: float = 1.959963985) -> Dict[str, float]:
    if n == 0:
        return {"p": float("nan"), "lo": float("nan"), "hi": float("nan"), "k": k, "n": n}
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"p": p, "lo": max(0.0, c - h), "hi": min(1.0, c + h), "k": k, "n": n}


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(n + 1)]
    tol = probs[k] * (1 + 1e-9)
    return float(min(1.0, sum(pr for pr in probs if pr <= tol)))


def mcnemar(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = a.astype(bool)
    b = b.astype(bool)
    b_cnt = int(np.sum(a & ~b))
    c_cnt = int(np.sum(~a & b))
    return {
        "a_only": b_cnt,
        "b_only": c_cnt,
        "n_discordant": b_cnt + c_cnt,
        "p_exact_two_sided": binom_two_sided(b_cnt, b_cnt + c_cnt),
    }


def boot_ci_mean(x: np.ndarray, n_boot: int = 20000, seed: int = 20260903) -> Dict[str, float]:
    if x.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "n": int(x.size),
    }


# ----------------------------------------------------------------------------- geometry
def ang_gap(a: float, b: float) -> float:
    d = abs(a - b)
    return float(min(d, 2 * math.pi - d))


def trial_geometry(tr: Dict) -> Dict[str, float]:
    s = np.asarray(tr["start_state"], dtype=float)
    f = np.asarray(tr["final_state"], dtype=float)
    g = np.asarray(tr["goal_state"], dtype=float)

    pos_start = float(np.linalg.norm(g[:4] - s[:4]))
    pos_final = float(np.linalg.norm(g[:4] - f[:4]))
    agent_start = float(np.linalg.norm(g[:2] - s[:2]))
    agent_final = float(np.linalg.norm(g[:2] - f[:2]))
    block_start = float(np.linalg.norm(g[2:4] - s[2:4]))
    block_final = float(np.linalg.norm(g[2:4] - f[2:4]))
    a_start = ang_gap(float(s[4]), float(g[4]))
    a_final = ang_gap(float(f[4]), float(g[4]))

    return {
        "pos_gap_start": pos_start,
        "pos_gap_final": pos_final,
        "pos_progress": pos_start - pos_final,
        "agent_gap_start": agent_start,
        "agent_gap_final": agent_final,
        "agent_progress": agent_start - agent_final,
        "block_gap_start": block_start,
        "block_gap_final": block_final,
        "block_progress": block_start - block_final,
        "ang_gap_start": a_start,
        "ang_gap_final": a_final,
        "ang_progress": a_start - a_final,
        "block_disp": float(np.linalg.norm(f[2:4] - s[2:4])),
        "agent_disp": float(np.linalg.norm(f[:2] - s[:2])),
        "recon_success": float(pos_final < POS_TOL and a_final < ANG_TOL),
        "solved_at_start": float(pos_start < POS_TOL and a_start < ANG_TOL),
        "blockonly_success": float(block_final < POS_TOL and a_final < ANG_TOL),
        # dominant violated term among {agent, block, angle} for failures
        "dom_term_agent_frac": agent_final / max(1e-9, agent_final + block_final),
    }


def dominant_failure_term(g: Dict[str, float]) -> str:
    """Which term breaks the criterion hardest (normalised by its own tolerance)."""
    # agent & block share the 20px joint budget -> compare their squared shares
    agent_sq = g["agent_gap_final"] ** 2
    block_sq = g["block_gap_final"] ** 2
    pos_viol = math.sqrt(agent_sq + block_sq) / POS_TOL
    ang_viol = g["ang_gap_final"] / ANG_TOL
    if pos_viol >= ang_viol:
        return "agent_position" if agent_sq >= block_sq else "block_position"
    return "block_angle"


# ----------------------------------------------------------------------------- main
def main(output_path: str | None = None) -> Dict:
    jep = json.load(open(JEPA_JSON, "r", encoding="utf-8"))
    lew = json.load(open(LEWM_JSON, "r", encoding="utf-8"))
    J, L = jep["trials"], lew["trials"]

    key = lambda t: (t["episode_idx"], t["start_step"], t["goal_step"])
    paired = [key(t) for t in J] == [key(t) for t in L]
    same_start = all(np.allclose(a["start_state"], b["start_state"]) for a, b in zip(J, L))
    same_goal = all(np.allclose(a["goal_state"], b["goal_state"]) for a, b in zip(J, L))
    if not (paired and same_start and same_goal):
        raise SystemExit("pairing verification FAILED — refuse to compare")

    gj = [trial_geometry(t) for t in J]
    gl = [trial_geometry(t) for t in L]
    n = len(J)

    # --- R0 criterion validation ---------------------------------------------
    logged = np.array([t["success"] for t in J] + [t["success"] for t in L], dtype=float)
    recon = np.array([g["recon_success"] for g in gj] + [g["recon_success"] for g in gl], dtype=float)
    agree = float((logged == recon).mean())
    criterion_valid = agree >= 0.98
    mism = [
        {"model": m, "i": i, "logged": float(t["success"]), "recon": g["recon_success"],
         "pos_gap_final": g["pos_gap_final"], "ang_gap_final": g["ang_gap_final"]}
        for m, trials, geos in (("jepa", J, gj), ("lewm", L, gl))
        for i, (t, g) in enumerate(zip(trials, geos))
        if float(t["success"]) != g["recon_success"]
    ]

    sj = np.array([t["success"] for t in J], dtype=float).astype(bool)
    sl = np.array([t["success"] for t in L], dtype=float).astype(bool)
    solved0 = np.array([g["solved_at_start"] for g in gl], dtype=bool)  # task property
    noop = solved0.copy()

    # --- success rates & strata ----------------------------------------------
    def rates(mask: np.ndarray) -> Dict:
        m = int(mask.sum())
        return {
            "n": m,
            "lewm": wilson(int(sl[mask].sum()), m),
            "jepa": wilson(int(sj[mask].sum()), m),
            "noop": wilson(int(noop[mask].sum()), m),
        }

    overall = rates(np.ones(n, dtype=bool))
    strat_solved = rates(solved0)
    strat_work = rates(~solved0)

    tests = {
        "lewm_vs_noop": mcnemar(sl, noop),
        "jepa_vs_noop": mcnemar(sj, noop),
        "lewm_vs_jepa": mcnemar(sl, sj),
    }

    # --- harm on already-satisfied tasks -------------------------------------
    harm = {
        "n_solved_at_start": int(solved0.sum()),
        "lewm_broke": int(np.sum(solved0 & ~sl)),
        "jepa_broke": int(np.sum(solved0 & ~sj)),
    }
    if solved0.any():
        harm["lewm_break_rate"] = float(np.mean(~sl[solved0]))
        harm["jepa_break_rate"] = float(np.mean(~sj[solved0]))
        harm["lewm_agent_disp_median_on_solved"] = float(np.median([gl[i]["agent_disp"] for i in range(n) if solved0[i]]))
        harm["jepa_agent_disp_median_on_solved"] = float(np.median([gj[i]["agent_disp"] for i in range(n) if solved0[i]]))

    # --- failure-term attribution --------------------------------------------
    def attribution(succ: np.ndarray, geos) -> Dict[str, float]:
        fails = [geos[i] for i in range(n) if not succ[i]]
        if not fails:
            return {"n_fail": 0}
        terms = [dominant_failure_term(g) for g in fails]
        out = {"n_fail": len(fails)}
        for t in ("agent_position", "block_position", "block_angle"):
            out[t] = float(terms.count(t) / len(fails))
        out["agent_gap_final_median"] = float(np.median([g["agent_gap_final"] for g in fails]))
        out["block_gap_final_median"] = float(np.median([g["block_gap_final"] for g in fails]))
        out["ang_gap_final_median"] = float(np.median([g["ang_gap_final"] for g in fails]))
        return out

    attrib = {"jepa": attribution(sj, gj), "lewm": attribution(sl, gl)}

    # --- paired progress on the needs-work stratum ---------------------------
    w = ~solved0
    pick = lambda geos, k: np.array([geos[i][k] for i in range(n) if w[i]])
    prog = {}
    for k in ("agent_progress", "block_progress", "ang_progress", "pos_progress", "block_disp", "agent_disp"):
        pl, pjj = pick(gl, k), pick(gj, k)
        prog[k] = {
            "lewm": boot_ci_mean(pl),
            "jepa": boot_ci_mean(pjj),
            "paired_diff_lewm_minus_jepa": boot_ci_mean(pl - pjj),
        }

    # --- counterfactual block-only criterion ---------------------------------
    bo_j = np.array([g["blockonly_success"] for g in gj], dtype=float).astype(bool)
    bo_l = np.array([g["blockonly_success"] for g in gl], dtype=float).astype(bool)
    bo_noop = np.array([(gl[i]["block_gap_start"] < POS_TOL and gl[i]["ang_gap_start"] < ANG_TOL) for i in range(n)])
    blockonly = {
        "criterion": "block position <20 AND angle <pi/9 (agent term dropped)",
        "jepa": wilson(int(bo_j.sum()), n),
        "lewm": wilson(int(bo_l.sum()), n),
        "noop": wilson(int(bo_noop.sum()), n),
        "mcnemar_lewm_vs_jepa": mcnemar(bo_l, bo_j),
        "mcnemar_jepa_vs_noop": mcnemar(bo_j, bo_noop),
    }

    # --- verdicts -------------------------------------------------------------
    ovl = lambda a, b: not (a["hi"] < b["lo"] or b["hi"] < a["lo"])
    h1 = bool(ovl(overall["lewm"], overall["noop"]))
    h2 = bool(tests["jepa_vs_noop"]["p_exact_two_sided"] < 0.05 and overall["jepa"]["p"] < overall["noop"]["p"])
    d = prog["agent_progress"]["paired_diff_lewm_minus_jepa"]
    h3 = bool(attrib["jepa"].get("agent_position", 0.0) > 0.5 and (d["lo"] > 0 or d["hi"] < 0))
    h4 = bool(blockonly["jepa"]["hi"] < 0.30)

    if not criterion_valid:
        verdict = f"VOID (criterion agreement {agree:.2f} < 0.98)"
    else:
        verdict = "; ".join([
            "H1_BASELINE_INFLATION_" + ("SUPPORTED" if h1 else "REJECTED"),
            "H2_JEPA_WORSE_THAN_NOOP_" + ("SUPPORTED" if h2 else "REJECTED"),
            "H3_AGENT_NAV_DOMINATES_" + ("SUPPORTED" if h3 else "REJECTED"),
            "H4_OBJECT_MANIP_ALSO_BROKEN_" + ("SUPPORTED" if h4 else "REJECTED"),
        ])

    payload = {
        "experiment_id": "KW-LEWM-0005c-PAIRED-NOOP-BASELINE",
        "analysis_type": "post_hoc_reanalysis_of_existing_logs",
        "pre_registered": False,
        "hypotheses_fixed_before_computation": True,
        "evidence_level": "E1",
        "gpu_used": False,
        "sources": {"jepa": os.path.relpath(JEPA_JSON, ROOT), "lewm": os.path.relpath(LEWM_JSON, ROOT)},
        "pairing": {
            "identical_task_keys": paired,
            "identical_start": same_start,
            "identical_goal": same_goal,
            "n_tasks": n,
        },
        "criterion_validation": {
            "source": "stable_worldmodel/envs/pusht/env.py::eval_state",
            "predicate": "||goal[:4]-cur[:4]|| < 20 AND min(|dtheta|,2pi-|dtheta|) < pi/9",
            "note": "state[:4] = [agent_x, agent_y, block_x, block_y] -> agent and block share one 20px budget",
            "agreement_with_logged_success": agree,
            "n_trials_checked": int(logged.size),
            "n_mismatch": len(mism),
            "mismatches": mism[:10],
            "valid": criterion_valid,
            "superseded_earlier_blockonly_version_agreement": 0.81,
        },
        "task_difficulty_audit": {
            "n_solved_at_start": int(solved0.sum()),
            "frac_solved_at_start": float(solved0.mean()),
            "pos_gap_start_quartiles": [float(np.percentile([g["pos_gap_start"] for g in gl], q)) for q in (0, 25, 50, 75, 100)],
            "agent_gap_start_quartiles": [float(np.percentile([g["agent_gap_start"] for g in gl], q)) for q in (0, 25, 50, 75, 100)],
            "block_gap_start_quartiles": [float(np.percentile([g["block_gap_start"] for g in gl], q)) for q in (0, 25, 50, 75, 100)],
            "ang_gap_start_quartiles": [float(np.percentile([g["ang_gap_start"] for g in gl], q)) for q in (0, 25, 50, 75, 100)],
        },
        "success_rates": {"overall": overall, "stratum_solved_at_start": strat_solved, "stratum_needs_work": strat_work},
        "paired_tests_exact_mcnemar": tests,
        "harm_analysis": harm,
        "failure_term_attribution": attrib,
        "paired_progress_needs_work_stratum": prog,
        "counterfactual_block_only_criterion": blockonly,
        "verdict": verdict,
        "caveats": [
            "No-op baseline is ANALYTIC (final state == start state); it was not executed. Valid because Push-T is quasi-static and a zero-action agent has negligible drift, but residual agent velocity at t=0 is ignored.",
            "Post-hoc re-analysis of data collected for a different question; hypotheses were fixed in the header before computation. Evidence stays E1.",
            "Strata use the model-independent task property solved_at_start, so both models see identical strata.",
            "The block-only counterfactual is a diagnostic decomposition, NOT an alternative benchmark; it is not comparable to any published number.",
        ],
    }

    out = output_path or OUT_JSON
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[0005c] wrote {out}", file=sys.stderr)
    print(f"[0005c] criterion agreement={agree:.3f} valid={criterion_valid}", file=sys.stderr)
    print(f"[0005c] verdict: {verdict}", file=sys.stderr)
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=None)
    a = ap.parse_args()
    main(a.output)
