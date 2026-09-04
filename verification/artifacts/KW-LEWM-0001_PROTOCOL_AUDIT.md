# KW-LEWM-0001 protocol audit

- Audit date: 2026-09-02
- Audit status: **CONDITIONAL / COMPARATIVE CLAIMS BLOCKED**
- Audited document: `verification/experiments/KW-LEWM-0001.md`
- Scope: protocol and claim validity only; this audit does not alter or interrupt the active WorkBuddy run.

## Executive decision

The proposed LeWM run is valid as an **independent reproduction on `swm/PushT-v1`**. The current protocol is not yet sufficient for a causal claim that LeWM or KineWorld is stronger than the existing JEPA-WM run. Equal episode count, seed value, and nominal CEM parameters do not establish a matched comparison when environment implementations, task instances, replanning schedules, horizons, and total model evaluations may differ.

Until the blockers below are resolved, the only defensible comparison language is:

> KineWorld independently reproduced LeWM and JEPA-WM on two disclosed Push-T implementations. The numbers are shown side by side for context and are not a head-to-head ranking.

## Claim blockers

### B0 — The cited “official repository” is a fork

The active protocol names `hongqin/leworldmodel` as the official repository. GitHub identifies it as a fork of `lucas-maes/le-wm`; the upstream repository is owned by the paper's first author and labels itself the official code base. Freeze the upstream commit from `lucas-maes/le-wm`. A fork may be used only if its divergence is audited and explicitly disclosed.

### B1 — Episode seed is not an episode identity

`seed=1` in two different environment implementations does not prove identical initial states, goal states, physics, observation preprocessing, action scaling, or episode termination. A 96-episode result from each implementation is therefore not paired data.

Required evidence for a head-to-head claim:

1. Export the 96 initial and goal states from both evaluators.
2. Compare their state definitions, units, ranges, action scaling, physics parameters, and termination rules.
3. Prefer replaying a single frozen episode manifest in both evaluators. If this cannot be done, classify the result as an unpaired cross-implementation reproduction.

### B2 — Planning budget is not aligned by `n_steps × num_samples` alone

Matching `10 × 100` per planning call does not align total inference compute. The existing JEPA-WM result uses one open-loop plan per episode. `WorldModelPolicy` can use model-predictive/receding-horizon planning, which may invoke the solver multiple times per episode. Horizon, action block, history length, model architecture, latent dimensions, and forward-pass cost also differ.

Required reporting:

- planning calls per episode;
- CEM iterations, samples, elites, horizon, action block, and receding horizon;
- total candidate trajectories and total predicted transitions per episode;
- total world-model forward calls, wall time, peak VRAM, and model parameter count;
- whether warm starts, early termination, or vectorized environments are used.

Two valid comparison views may then be reported separately:

1. **Protocol-native:** each method under its documented evaluation setup;
2. **Budget-constrained:** equalized total predicted transitions or measured inference budget per episode.

Neither view should be labelled fully fair unless the task-instance blocker is also resolved.

### B3 — Success definition needs code-level verification

The stable-worldmodel documentation defines success using the L2 error over the combined agent and block positions below 20 pixels plus block angle error below `pi/9`. This is stricter and more specific than saying only “Push-T standard.” The exact JEPA-WM evaluation code and the installed stable-worldmodel version must both be cited and frozen.

Required evidence:

- source file, function, repository/package version, and hash for both evaluators;
- raw terminal state for every episode so the success flag can be recomputed independently;
- a small invariant test proving the evaluator reproduces the stored success count.

### B4 — Statistical decision rule is underspecified

“Significantly higher” currently has no pre-registered test, confidence level, or minimum effect. With independent 96-episode samples, a point estimate above `0.4583` is not evidence of superiority.

Required before looking at results:

- report Wilson 95% confidence intervals for both success proportions;
- if episodes are genuinely paired, pre-register McNemar's exact test and paired effect size;
- if unpaired, pre-register a two-sided Fisher exact test or two-proportion exact/permutation test;
- set `alpha=0.05` and report the absolute success-rate difference with a confidence interval;
- do not claim superiority from non-overlapping point estimates alone, and do not treat failure to reject as equivalence.

Recommended minimum claim gate: superiority requires the pre-registered test to pass and the lower bound of the effect interval to exceed zero. A practical-effect threshold should be added if this result will guide product or financing claims.

### B5 — Distance metrics are not yet comparable

`actual_state_dist_mean` and `end_dist` must not be put in one ranking column until their state dimensions, normalization, units, time point, and aggregation are proven identical. Success rate is the only candidate primary metric at present, and even it remains cross-implementation until B1 is resolved.

### B6 — Dataset and goal-source provenance is unresolved

The official stable-worldmodel repository reports sharply different local storage sizes by format (video, LanceDB, HDF5). The protocol also references approximately 893 MB of Push-T assets while the official LeWM Hugging Face training dataset is about 13.1 GB. These may be different formats or artifacts, but they must not be described as the same dataset.

Required evidence:

- exact dataset/checkpoint repository IDs, filenames, revisions, sizes, hashes, and licenses;
- whether evaluation samples goals from an environment or from a dataset;
- exact bytes downloaded and whether any training data is required at test time;
- Founder approval before any large training-dataset download under the project mandate.

### B7 — Dependency license status remains provisional

Top-level MIT licenses do not prove the complete runtime is commercially clean. After installation, freeze the exact package lock and produce a transitive license inventory. Mark unknown, restrictive, model-specific, dataset-specific, or non-code terms as unresolved rather than green.

## Allowed result labels

| Evidence state | Allowed label | Prohibited label |
|---|---|---|
| LeWM runs successfully on stable-worldmodel | Independent LeWM reproduction | Fair head-to-head win |
| Both methods run, but B1/B2 unresolved | Cross-implementation contextual comparison | LeWM/KineWorld is stronger |
| Same frozen task manifest and compute accounting, statistical gate passes | Controlled Push-T result under the disclosed protocol | General world-model superiority |
| Multiple seeds/tasks/checkpoints plus external rerun | Broader empirical evidence with stated scope | “World's best” without benchmark-wide proof |

## Required manifest additions

The final `KW-LEWM-0001_manifest.json` should include at least:

- installed `stable-worldmodel` version and source/release identifier;
- LeWM checkpoint revision and SHA-256;
- dataset/goal-source revisions and SHA-256 where applicable;
- exact environment source revision and success-function hash;
- frozen episode manifest hash or an explicit `unpaired_cross_implementation=true` flag;
- all planning-budget fields listed in B2;
- raw per-episode results hash;
- statistical test, alpha, confidence-interval method, and effect-size definition;
- protocol deviations and their timestamps.

## Current verdict

Proceed with the active run as a low-cost reproduction and harness feasibility test. Do not use its output as a ranking claim until the audit blockers are either resolved or explicitly accepted as limitations. This preserves useful engineering progress without turning a convenient side-by-side table into an unsupported financing claim.

## Local package evidence observed during audit

Read-only inspection of the environment installed by WorkBuddy found `stable-worldmodel==0.1.1` (package metadata license expression: MIT). The following hashes freeze the exact local evaluator/planner source inspected here:

| Local source | SHA-256 |
|---|---|
| `stable_worldmodel/policy.py` | `4967e7e3d5b20eb7a1d0b00e5d60fd701cce1c750ae7ec4a9b02529b9366db22` |
| `stable_worldmodel/world/world.py` | `39318b81ed151d8556d8540f460a63eedaed0ce4b2211ce0af9f8200e7d83bde` |
| `stable_worldmodel/envs/pusht/env.py` | `3c4df51ed01ed034386c24961caa638f02b5c0e31c88fec727a9f1cd5808e203` |
| `stable_worldmodel/solver/cem.py` | `a6662e7f1f2c6ecaf0a8527255d531aa05ed583339aabdf2798d0206382ef51b` |

Code-level findings:

- `PlanConfig.receding_horizon` has no default and is required.
- `WorldModelPolicy.get_action()` calls the solver whenever an environment's action buffer is empty, then executes `receding_horizon * action_block` environment actions before the next planning call.
- Push-T `eval_state()` uses the L2 difference over `goal_state[:4]` and `cur_state[:4]`, which includes agent and block position, plus the wrapped block-angle difference.
- `World.evaluate()` returns success in percent according to its docstring; the existing JEPA-WM ledger uses a fraction. The comparison script must normalize both values to the same unit before producing a table.

Audit document SHA-256 before this appendix was added: `ad47bf607cd7a7da7a439316dece7885b51283f6a36bc7e28d23a45800647c63`. The final hash must be recomputed after the document is closed.
