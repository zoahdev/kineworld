# KineWorld — evaluation methodology research on world models for physical AI

KineWorld is a single-machine research project (one RTX 5070 Ti Laptop, 12 GB) studying
**how world-model-driven agents should be evaluated**. Current focus: diagnostic studies on
the official Push-T checkpoints of two world models (Meta's `facebookresearch/jepa-wms` and
LeWM's `stable-worldmodel` release), with an emphasis on **confound auditing and honest
failure analysis** rather than leaderboard claims.

No training is involved at this stage. Everything here is diagnostic and statistical.

```bash
git clone https://github.com/zoahdev/kineworld.git
cd kineworld
```

Note: large artifacts are deliberately **not** in this repository. Model checkpoints, the Push-T
dataset, and raw run logs must be obtained separately (see the Level-2a and Level-3 sections
below); the archived evidence JSONs under `results/` are included and are what all reported
numbers are checked against.

## Evidence boundary (read this first)

- All results are **E1**: exploratory statistics, single seed, single checkpoint per model,
  single task suite (Push-T). Nothing here is a reproduction of an official benchmark, and
  nothing here has been validated by a third party.
- This is a **single-person project**. There are no co-authors, no customers, no external
  replications, and no expert endorsements. All claims are traceable to the archived result
  JSONs under `results/`; the internal evidence ledger (which also records what has *not*
  happened) is available on request.
- Passing any check below on your machine proves **internal consistency of this repo's
  artifacts**. It does **not** constitute third-party validation. Only a signed, independent
  Level-3 rerun can change that.

## What you can rerun (three levels)

### Level 1 — artifact integrity (no GPU, no special environment, ~1 s)

Verifies that every number quoted in the methodology report matches the archived result
JSONs (56 numeric + in-text checks; fails with non-zero exit on any mismatch):

```bash
python verification/scripts/verify_methodology_report_v1.py
```

Requires only the Python standard library. Expected output ends with
`56 ok / 0 fail`.

### Level 2 — CPU-only analyses (no GPU, no world-model checkpoint)

**2a. Model-free floor reference (`KW-LEWM-0005d`)** — the cheapest meaningful rerun in this
repo: the real Push-T environment, the same 50 dataset tasks, ~10,050 actual rollouts, no
world model at all. It establishes the random-policy floor that any planner claim must be
compared against.

```bash
python verification/scripts/kw_lewm_0005d_modelfree_reference.py
```

- Hardware: any modern CPU. Runtime ≈ 46 s. VRAM: none. Checkpoint download: none.
- Environment: Python 3.10, `numpy`, `stable_worldmodel==0.1.1` (recorded versions: numpy
  2.2.6, Python 3.10.21).
- Note: the Push-T task data is read from the local dataset cache under `data/`; see
  [`verification/experiments/KW-LEWM-0005d.md`](verification/experiments/KW-LEWM-0005d.md)
  for the full protocol and pre-registered decision rules.

**2b. Paired post-hoc analysis (`KW-LEWM-0006b`)** — pure log re-analysis (standard library
only) of the inference-time ablation logs:

```bash
python verification/scripts/kw_lewm_0006b_oop_analysis.py
```

### Level 3 — independent model rerun (GPU + checkpoints)

Requires a GPU, the world-model checkpoints, and the vendored environments. Start from the
existing release package:
[`verification/releases/KW-RISK-CARD-0001/README.md`](verification/releases/KW-RISK-CARD-0001/README.md).
That package separates integrity checks, deterministic regeneration, and true independent
reruns, and includes the attestation template a third-party evaluator would sign.

## Repository map

| Path | Contents |
| --- | --- |
| `AGENTS.md` | Operating manual for AI coding agents (Codex etc.) — hard rules & environment pitfalls |
| `docs/WORKLOG.md` | Append-only worklog — what was done, when, and why |
| `docs/research/METHODOLOGY_CONFOUNDS_v1.md` | Main external asset: two evaluation confounds + five converging evidence lines |
| `docs/research/` | Capability audits, research-ROI notes, harness documentation |
| `verification/experiments/` | Pre-registered experiment docs (`KW-EXP-*`, `KW-LEWM-*`) |
| `verification/manifests/` | Machine-readable manifests per experiment |
| `verification/scripts/` | All analysis / verification scripts (fixed seeds) |
| `verification/patches/` | Patches applied to vendored `external/jepa-wms` (base `13cf1d9`) |
| `verification/releases/` | Self-contained packages for external evaluators |
| `results/` | Archived result JSONs (referenced by the verification scripts) |
| `external/` | Vendored third-party repos (do not edit in place) |
| `company/` | Company / funding-track materials (separate from research claims) |

## Core findings (summary only — see the report for numbers)

Two evaluation-methodology findings, both at evidence level E1:

1. **Task reachability is a first-order confound.** On randomly-sampled goal pairs, planner
   success mostly measures task construction, not model capability.
2. **"Can move" ≠ "knows where to move".** Two agents with statistically indistinguishable
   displacement can have opposite goal-progress signs. Displacement/interaction-rate metrics
   alone will misclassify weak-action-condition models as having control.

Full treatment, caveats, and the reusable audit protocols (label-replication gate,
model-free three-arm floor, seed-count-independent primary criteria, effort-matched
controls) are in [`docs/research/METHODOLOGY_CONFOUNDS_v1.md`](docs/research/METHODOLOGY_CONFOUNDS_v1.md).

## Citation / contact norms

- If you rerun any level, keep raw logs and note every deviation from the documented
  protocol — deviations are expected (different hardware, versions) and should be reported,
  not hidden.
- Do not cite Level 1/2 outputs as validation. Cite the report as *single-person, E1,
  internally-consistent* research.
- Negative results are registered with the same care as positive ones; if you find a
  discrepancy, `company/EVIDENCE_LEDGER.md` is where it should be reconciled against.
