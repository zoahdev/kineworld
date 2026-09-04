# KW-LEWM-0001L — LeWM Push-T public-rows lite reproduction

- Pre-registered: 2026-09-02 11:52 +08:00
- Status: **CLOSED — INFRASTRUCTURE PASS / VALID CONTROL NOT SUPPORTED**
- Evidence target: E1 internal reproduction
- Training: none

## Why this downgraded protocol exists

The upstream LeWM evaluation requires `pusht_expert_train.h5.zst`, a 13.1 GB compressed HDF5 artifact. The official evaluator samples start/goal pairs from that dataset and uses dataset-wide standardization. KineWorld will not download that large artifact without Founder approval.

Hugging Face exposes a public rows API for a partial, official Lance conversion (`galilai-group/lewm-pusht`). This experiment uses a small deterministic sample of numeric rows and re-renders the disclosed states locally. It is a functional, low-resource reproduction only; it is not the upstream 50-task benchmark.

## Frozen upstream facts

- Upstream code: `lucas-maes/le-wm`.
- Model: `quentinll/lewm-pusht`, local `weights.pt` + `config.json`.
- Official Push-T config: horizon 5, receding horizon 5, action block 5, goal offset 25, evaluation budget 50, 50 evaluations, seed 42.
- Official CEM: 30 iterations, 300 samples, 30 elites.
- Local compatibility deviation: `transformers` pinned from 5.16.1 to 4.57.6 because the checkpoint uses the Transformers 4.x ViT state-dict layout. The model subsequently loaded strictly with 18,034,478 parameters.

## Lite sampling protocol

1. Dataset endpoint: `galilai-group/lewm-pusht`, config `default`, split `train`.
2. Public rows API currently reports 815,360 available rows and `partial=true`.
3. RNG seed: 42.
4. Fetch 16 deterministic pages of 100 rows. Store numeric columns only; do not download images.
5. Estimate action/proprio/state `StandardScaler` mean and population scale from all fetched rows.
6. Select pairs within a page where `episode_idx` matches and `goal.step_idx = start.step_idx + 25`.
7. Freeze the first eight valid pairs after deterministic page/row ordering.
8. Re-render start and goal pixels locally from the published states using installed `swm/PushT-v1`.

This sampling does not reproduce the official evaluator's exact 50 random starting rows or exact full-dataset scaler.

## Execution stages and kill gates

### Stage A — one-task reduced-budget smoke

- CEM: 10 iterations, 100 samples, 10 elites.
- PlanConfig: horizon 5, receding horizon 5, action block 5.
- Evaluation budget: 50 environment steps.
- Pass: strict model loading, two planning calls complete, finite actions/costs, no exception, raw terminal state stored.
- Kill: OOM, NaN/Inf, action-shape mismatch, or wall time above 15 minutes.

### Stage B — eight-task reduced-budget run

Run only if Stage A passes. Same configuration, eight frozen tasks. Report success count/rate, Wilson interval, wall time, peak VRAM, and every terminal state. This is the main lite result.

### Stage C — one-task upstream-CEM cost probe

Run only if Stage A finishes below 5 minutes and peak VRAM is below 10 GB. Use official CEM 30×300×30 on the same first task. This is a compute/feasibility probe, not an accuracy comparison.

## Claim rules

Allowed after success:

> KineWorld independently loaded the official LeWM Push-T checkpoint and executed closed-loop planning on a deterministic public-row subset under a disclosed lite protocol.

Not allowed:

- official LeWM benchmark reproduced;
- LeWM or KineWorld beats JEPA-WM;
- third-party validation;
- state of the art / world's best;
- the approximate scaler is equivalent to the official full-dataset scaler.

## Required outputs

- numeric sampling snapshot and SHA-256;
- checkpoint/config/source hashes;
- exact environment and dependency versions;
- per-task start/goal/final states and success flags;
- CEM and PlanConfig fields;
- wall time and peak VRAM;
- protocol deviations and failures, including the Transformers 5 incompatibility.

## Transport amendment — 2026-09-02 12:16 +08:00

The public rows API failed before returning the first pre-registered page, and the official statistics endpoint repeatedly returned HTTP 500. No sampled control result was observed from that path. The numeric preparation transport is therefore amended before any dataset-defined control run:

- use the Hugging Face generated Parquet shard `train/0.parquet`;
- require HTTP 206 range responses and reject a server that falls back to a full HTTP 200 body;
- read only `episode_idx`, `step_idx`, `action`, `proprio`, and `state` columns;
- never request the `pixels` column;
- abort if cumulative response bytes exceed 64 MiB;
- estimate scalers from the numeric rows in shard 0 only;
- select eight valid start/goal pairs with seed 42 from that shard.

This amendment is a transport/resource correction. It remains a partial-dataset lite protocol and does not upgrade the evidence claim.

## Result — 2026-09-02

Stage A completed as a functional infrastructure smoke test:

- the official checkpoint loaded strictly with **18,034,478 parameters** on CUDA;
- local `weights.pt` SHA-256 is `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607` (72,290,721 bytes);
- two reduced-budget CEM calls completed in 0.525 s and 0.179 s;
- total evaluation wall time was 0.869 s and peak CUDA allocation was 93,476,864 bytes;
- the single random task failed (0/1).

The control result is **invalid for model-performance interpretation**. With the official dataset unavailable, the policy used identity action normalization. The published evaluator instead fits `StandardScaler` transforms from the training dataset and inverse-transforms planned actions before environment execution. The unmodified CEM solver uses the environment action space for dimensionality but does not clamp sampled standardized actions to raw environment bounds. In the smoke run, the agent moved from approximately `(281.87, 104.94)` to `(120.21, 1202.36)` while the block did not move, demonstrating the resulting out-of-support control.

### Kill-gate decision

- **Stage B: KILLED.** Expanding to eight tasks would repeat a known invalid action transform.
- **Stage C: KILLED.** Increasing CEM compute cannot repair missing action semantics.
- No official benchmark, competitor comparison, or success-rate claim is permitted from this run.
- The retained evidence is limited to checkpoint provenance, strict-load compatibility, runtime feasibility, and an identified reproducibility dependency.

Raw result: `results/kw_lewm_0001l_smoke.json`  
Manifest: `verification/manifests/KW-LEWM-0001L_manifest.json`
