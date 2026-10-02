#!/usr/bin/env python3
"""Exact, dependency-free checks for a finite confounded world-model family.

All probabilities use fractions.Fraction. No network, training, randomness,
credentials, private datasets, or external packages are used.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction as F
from itertools import product
from functools import lru_cache
from math import comb
import json
from pathlib import Path

if not __debug__:
    raise RuntimeError("Run without -O or -OO: verification requires assertions.")


def reward2(m: int, world: int, hidden: int, action: int) -> int:
    """Twice the reward, making enumeration integer-only."""
    assert m >= 2 and all(0 <= x < m for x in (world, hidden, action))
    if action == hidden:
        return 1
    return 2 if action == world else 0


def logged_law(m: int, world: int) -> dict:
    counts = Counter((u, reward2(m, world, u, u)) for u in range(m))
    return {key: F(count, m) for key, count in counts.items()}


def intervention_value(m: int, world: int, action: int) -> F:
    return F(sum(reward2(m, world, u, action) for u in range(m)), 2 * m)


def closed_risk(m: int, n: int) -> F:
    q = F(1, m**n)
    gap = F(m - 1, m)
    return gap * (q - (1 - (1 - q) ** m) / m)


def closed_ambiguity(m: int, n: int) -> F:
    q = F(1, m**n)
    return q * (1 - (1 - q) ** (m - 1))


def closed_adaptive_risk(m: int, budget: int, candidates: int | None = None) -> F:
    k = m if candidates is None else candidates
    gap, erasure = F(m - 1, m), F(1, m)
    expectation = sum((F(max(k - 1 - b, 0) * comb(budget, b)) * gap**b * erasure**(budget-b)
                       for b in range(budget + 1)), F(0))
    return gap * expectation / k


def check_adaptive_designs() -> dict:
    """Exact Bellman optimization over every available action and posterior support."""
    checked = 0
    for m in range(2, 9):
        gap = F(m - 1, m)

        @lru_cache(None)
        def branches(mask: int, action: int):
            worlds = [j for j in range(m) if mask & (1 << j)]
            counts = Counter((reward2(m, j, u, action), j) for j in worlds for u in range(m))
            result = []
            for reward in sorted({r for r, j in counts}):
                feasible = [j for j in worlds if counts[(reward, j)]]
                assert len({counts[(reward, j)] for j in feasible}) == 1
                new_mask = sum(1 << j for j in feasible)
                probability = F(sum(counts[(reward, j)] for j in worlds), m * len(worlds))
                result.append((probability, new_mask))
            assert sum((p for p, _ in result), F(0)) == 1
            return tuple(result)

        @lru_cache(None)
        def optimal(mask: int, budget: int) -> F:
            k = mask.bit_count()
            if budget == 0:
                return gap * F(k - 1, k)
            action_risks = [sum((probability * optimal(new_mask, budget - 1)
                                 for probability, new_mask in branches(mask, a)), F(0))
                            for a in range(m)]
            return min(action_risks)

        for mask in range(1, 1 << m):
            for budget in range(2 * m + 1):
                assert optimal(mask, budget) == closed_adaptive_risk(m, budget, mask.bit_count())
                checked += 1
        assert optimal((1 << m) - 1, m) == closed_risk(m, 1)
    return {"support_budget_pairs_checked": checked, "action_counts": "2 through 8 inclusive", "budgets": "0 through twice the action count inclusive", "method": "exact Bellman minimization over all actions with reward-table-derived posterior transitions"}


def compatible_worlds(m: int, actions: tuple, observed: tuple) -> tuple:
    """Support-based elimination, with no access to hidden variables."""
    positive = {a for a, r in zip(actions, observed) if r == 2}
    if positive:
        assert len(positive) == 1
        return tuple(positive)
    eliminated = {a for a, r in zip(actions, observed) if r == 0}
    return tuple(a for a in range(m) if a not in eliminated)


def latent_transcript_counts(m: int, n: int, world: int) -> Counter:
    """Direct enumeration of all m**(m*n) independently reset hidden sequences."""
    actions = tuple(a for a in range(m) for _ in range(n))
    result = Counter()
    for hidden in product(range(m), repeat=len(actions)):
        transcript = tuple(reward2(m, world, u, a) for u, a in zip(hidden, actions))
        result[transcript] += 1
    return result


def factorized_transcript_counts(m: int, n: int, world: int) -> Counter:
    """Independent cross-check using per-action outcome multiplicities."""
    actions = tuple(a for a in range(m) for _ in range(n))
    supports = [Counter(reward2(m, world, u, a) for u in range(m)) for a in actions]
    result = Counter()
    for transcript in product(*(tuple(s) for s in supports)):
        multiplicity = 1
        for support, r in zip(supports, transcript):
            multiplicity *= support[r]
        result[transcript] = multiplicity
    return result


def check_audit(m: int, n: int, direct: bool) -> dict:
    actions = tuple(a for a in range(m) for _ in range(n))
    total = m ** len(actions)
    counts = [factorized_transcript_counts(m, n, j) for j in range(m)]
    if direct:
        for j in range(m):
            assert counts[j] == latent_transcript_counts(m, n, j)
    for count in counts:
        assert sum(count.values()) == total
    all_transcripts = set().union(*(set(c) for c in counts))
    for transcript in all_transcripts:
        feasible = tuple(j for j in range(m) if counts[j][transcript])
        assert feasible == compatible_worlds(m, actions, transcript)
        # Uniform prior has an exactly uniform posterior on feasible worlds.
        assert len({counts[j][transcript] for j in feasible}) == 1

    gap = F(m - 1, m)
    risks, ambiguities = [], []
    for world in range(m):
        risk, ambiguity = F(0), F(0)
        for transcript, count in counts[world].items():
            feasible = compatible_worlds(m, actions, transcript)
            assert world in feasible
            probability = F(count, total)
            # Compute regret from the actual reward table, not the closed formula.
            chosen_value = sum((intervention_value(m, world, a) for a in feasible), F(0)) / len(feasible)
            best_value = max(intervention_value(m, world, a) for a in range(m))
            risk += probability * (best_value - chosen_value)
            ambiguity += probability * (len(feasible) > 1)
        assert risk == closed_risk(m, n)
        assert ambiguity == closed_ambiguity(m, n)
        assert risk <= gap * F(m - 1, 2) * F(1, m ** (2 * n))
        risks.append(risk)
        ambiguities.append(ambiguity)
    assert len(set(risks)) == len(set(ambiguities)) == 1
    return {
        "actions": m,
        "queries_per_action": n,
        "audit_queries": m * n,
        "direct_hidden_enumeration": direct,
        "hidden_sequences_per_world": total,
        "distinct_observation_transcripts": len(all_transcripts),
        "exact_expected_regret": str(risks[0]),
        "exact_ambiguity_probability": str(ambiguities[0]),
    }


def compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for remaining in compositions(total - first, parts - 1):
                yield (first,) + remaining


def run_checks() -> dict:
    value_entries = 0
    for m in range(2, 33):
        reference = logged_law(m, 0)
        assert set(reference) == {(a, 1) for a in range(m)}
        assert set(reference.values()) == {F(1, m)}
        for world in range(m):
            assert logged_law(m, world) == reference
            for action in range(m):
                expected = 1 - F(1, 2 * m) if action == world else F(1, 2 * m)
                assert intervention_value(m, world, action) == expected
                value_entries += 1
        assert closed_risk(m, 0) == F(m - 1, m) ** 2
        assert closed_risk(m, 1) == F(m - 1, m) ** (m + 1) / m

    policy_grid_cases = 0
    for m in range(2, 6):
        denominator = 2 * m
        for weights in compositions(denominator, m):
            p = [F(w, denominator) for w in weights]
            worst_regret = max(F(m - 1, m) * (1 - p[j]) for j in range(m))
            assert worst_regret >= F(m - 1, m) ** 2
            policy_grid_cases += 1

    direct_cases = [(m, n) for m, max_n in [(2, 4), (3, 3), (4, 2), (5, 1)] for n in range(max_n + 1)]
    factor_cases = [(2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (6, 2)]
    audit_cases = [check_audit(m, n, True) for m, n in direct_cases]
    audit_cases += [check_audit(m, n, False) for m, n in factor_cases]
    adaptive_checks = check_adaptive_designs()

    # Negative control: relabeling logged actions as "interventions" does not
    # change their confounded reward law and reveals no world index.
    for m in range(2, 9):
        assert all(logged_law(m, j) == logged_law(m, 0) for j in range(m))
        assert closed_risk(m, 0) > closed_risk(m, 1)

    return {
        "schema_version": 1,
        "status": "PASS",
        "arithmetic": "exact integers and fractions.Fraction",
        "scope": "synthetic finite one-step family; no trained world model or real-world experiment",
        "logged_law_and_value_action_sizes": "2 through 32 inclusive",
        "value_table_entries_checked": value_entries,
        "policy_grid_cases": policy_grid_cases,
        "audit_configurations_checked": len(audit_cases),
        "direct_hidden_sequences_checked_across_worlds": sum(r["actions"] * r["hidden_sequences_per_world"] for r in audit_cases if r["direct_hidden_enumeration"]),
        "audit_cases": audit_cases,
        "adaptive_design_checks": adaptive_checks,
        "interpretation": "Finite checks corroborate the written proofs; they do not prove the all-m, all-n, or all-budget theorems or establish novelty.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional deterministic JSON results path")
    parser.add_argument("--check-report", type=Path, help="Check the marked generated numeric table in the report")
    args = parser.parse_args()
    result = run_checks()
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.check_report:
        report = args.check_report.read_text(encoding="utf-8")
        start = "<!-- BEGIN GENERATED RESULTS -->\n"
        end = "\n<!-- END GENERATED RESULTS -->"
        assert report.count(start) == report.count(end) == 1
        actual = report.split(start, 1)[1].split(end, 1)[0]
        assert actual == results_table(result), "Report table differs from recomputed exact results"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


def results_table(result: dict) -> str:
    selected = {(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1)}
    rows = [
        "| Actions m | Queries per action n | Total audit queries | Balanced design regret | Optimal adaptive regret at same budget | Balanced ambiguity probability |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["audit_cases"]:
        if (row["actions"], row["queries_per_action"]) in selected:
            values = [row["actions"], row["queries_per_action"], row["audit_queries"], row["exact_expected_regret"],
                      str(closed_adaptive_risk(row["actions"], row["audit_queries"])), row["exact_ambiguity_probability"]]
            rows.append("| " + " | ".join(str(v) for v in values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
