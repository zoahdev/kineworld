# Verification plan for the finite causal audit note

## Scope and chronology

This is deterministic verification of explicit finite mathematics. It is not a trained model experiment, and empirical evidence levels for checkpoints do not apply. The construction, formulas, and an initial exact checker run existed before this plan. The release verification is a rerun against these fixed criteria; this document is not presented as preregistration of the discovery.

Only synthetic rewards, standard library code, and cited public literature are inputs. There are no private datasets, prior repository experiment logs, credentials, model weights, paid compute services, or network dependencies. No pseudorandom generator is used, so a random seed is inapplicable.

## Frozen acceptance criteria

1. Reconstruct and compare the complete logged action and reward laws for every world with action count 2 through 32 inclusive.
2. Compute every interventional action value in those families directly from the reward table and match the proposed formulas exactly.
3. For each action count 2 through 5, enumerate all probability vectors with denominator twice the action count and confirm the claimed minimax lower bound on that finite grid. The written proof, not the grid, establishes the claim for all action distributions.
4. Directly enumerate every independent hidden state sequence for configurations (m,n): m=2 and n=0..4; m=3 and n=0..3; m=4 and n=0..2; m=5 and n=0..1. Independently construct transcript counts from product outcome multiplicities and require equality.
5. Additionally use product outcome multiplicities for (2,5), (3,4), (4,3), (5,2), (6,1), and (6,2). Do not describe these additional cases as direct hidden-sequence enumeration.
6. For every audit transcript in every checked case, validate the support elimination rule, equal likelihoods among feasible worlds, world-independent risk of the proposed rule, and exact equality to both the regret and ambiguity formulas.
7. Require the report's marked numerical table to match the regenerated exact table byte for byte.
8. A failed assertion blocks a PASS result. Disclose and fix a mathematical or implementation error; do not silently change the construction or test scope to fit a failed output.
9. For the adaptive extension, derive posterior transitions directly from the reward table, then perform exact Bellman minimization over every available action for every nonempty posterior support with m=2..8 and budgets 0..2m. Compare every value with the proposed binomial formula. These adaptive cases were fixed in this plan before the first Bellman verification run; the earlier balanced-design exploratory run remains disclosed above.

## Command and artifacts

```sh
python verify_causal_audit.py --output exact_results.json --check-report report.md
```

The public package consists of `report.md`, `verify_causal_audit.py`, `exact_results.json`, `verification_plan.md`, and `manifest.json`. The manifest records hashes and the machine-check scope. Any PDF is only a rendered view of the same reviewed report, not a new evidence source.

## Interpretation boundary

Passing finite checks corroborates, but does not formally verify, the all-m, all-n, and all-budget proofs. No benchmark replication, learned world-model superiority, statistical significance, external expert review, or novelty is implied. The balanced-design theorem optimizes final decision rules for a specified design. The adaptive extension optimizes reset-query designs only within the same known finite family, not general causal environments.
