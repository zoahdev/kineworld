"""Evidence gate for an unpaired comparison of two binary success rates.

This module deliberately uses only the Python standard library so the claim
audit does not depend on the experiment environment. It is not a substitute
for a paired test when both methods execute the same frozen episodes.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Interval:
    low: float
    high: float


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Interval:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    _validate_count(successes, total)
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    radius = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denom
    return Interval(max(0.0, center - radius), min(1.0, center + radius))


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for [[a, b], [c, d]]."""
    if min(a, b, c, d) < 0:
        raise ValueError("contingency-table counts must be non-negative")
    row1, row2 = a + b, c + d
    col1 = a + c
    total = row1 + row2
    if total == 0:
        raise ValueError("contingency table must contain observations")

    lo = max(0, col1 - row2)
    hi = min(row1, col1)

    def probability(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(total, col1)

    observed = probability(a)
    tolerance = observed * 1e-12 + 1e-15
    return min(1.0, sum(probability(x) for x in range(lo, hi + 1) if probability(x) <= observed + tolerance))


def compare_unpaired(
    candidate_successes: int,
    candidate_total: int,
    baseline_successes: int,
    baseline_total: int,
    alpha: float = 0.05,
) -> dict:
    """Return auditable statistics and a conservative superiority gate."""
    _validate_count(candidate_successes, candidate_total)
    _validate_count(baseline_successes, baseline_total)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")

    candidate_rate = candidate_successes / candidate_total
    baseline_rate = baseline_successes / baseline_total
    candidate_ci = wilson_interval(candidate_successes, candidate_total)
    baseline_ci = wilson_interval(baseline_successes, baseline_total)
    p_value = fisher_exact_two_sided(
        candidate_successes,
        candidate_total - candidate_successes,
        baseline_successes,
        baseline_total - baseline_successes,
    )

    # Conservative Newcombe-style interval from the two Wilson intervals.
    effect_ci = Interval(
        candidate_ci.low - baseline_ci.high,
        candidate_ci.high - baseline_ci.low,
    )
    direction_positive = candidate_rate > baseline_rate
    statistical_gate = p_value < alpha and effect_ci.low > 0.0

    return {
        "comparison_type": "unpaired_cross_implementation",
        "candidate": {
            "successes": candidate_successes,
            "total": candidate_total,
            "rate_fraction": candidate_rate,
            "rate_percent": candidate_rate * 100.0,
            "wilson_95": asdict(candidate_ci),
        },
        "baseline": {
            "successes": baseline_successes,
            "total": baseline_total,
            "rate_fraction": baseline_rate,
            "rate_percent": baseline_rate * 100.0,
            "wilson_95": asdict(baseline_ci),
        },
        "effect_candidate_minus_baseline": {
            "fraction": candidate_rate - baseline_rate,
            "percentage_points": (candidate_rate - baseline_rate) * 100.0,
            "conservative_95": asdict(effect_ci),
        },
        "fisher_exact_two_sided_p": p_value,
        "alpha": alpha,
        "direction_positive": direction_positive,
        "statistical_gate_passed": statistical_gate,
        "claim_allowed": False,
        "claim_reason": (
            "Statistical gate passed, but a cross-implementation result is not a fair head-to-head claim."
            if statistical_gate
            else "Statistical gate did not pass; superiority is not supported."
        ),
    }


def _validate_count(successes: int, total: int) -> None:
    if not isinstance(successes, int) or not isinstance(total, int):
        raise TypeError("successes and total must be integers")
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("require 0 <= successes <= total and total > 0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-successes", type=int, required=True)
    parser.add_argument("--candidate-total", type=int, required=True)
    parser.add_argument("--baseline-successes", type=int, required=True)
    parser.add_argument("--baseline-total", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    print(json.dumps(compare_unpaired(
        args.candidate_successes,
        args.candidate_total,
        args.baseline_successes,
        args.baseline_total,
        args.alpha,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
