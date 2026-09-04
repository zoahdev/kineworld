# KineWorld Failure & Risk Card v0

> Card `KW-RISK-CARD-0001` · E1 internal reproduction · **not third-party certification**

## What was evaluated

One public JEPA-WM Push-T checkpoint was evaluated for **96 episodes** under the frozen KW-EXP-0006 protocol. KineWorld did not train this checkpoint. The purpose of this card is to expose failure structure hidden by an aggregate success rate, not to claim a new model record.

## Outcome and risk signals

| Measure | Observed value | What it supports |
|---|---:|---|
| Task success | **44/96 (45.8%)** | Reproduced task outcome for this checkpoint/protocol |
| Latent-error spikes | **18/576 (3.1%)** | Sparse prediction failures exist |
| Median error, horizon 1 → 6 | **0.0216 → 0.0337 (1.56×)** | Prediction error grows mildly with rollout horizon |
| Spike rate, high object motion | **7.5%** | Error concentration under stronger physical interaction |
| Spike rate, lower object motion | **1.4%** | Comparison group under the frozen slice |
| Relative spike risk | **5.40×** | Descriptive association; Fisher exact p=0.0018 |

## The important negative result

The physical signal was statistically clear but **not useful as a calibrated risk band**. At nominal 90% coverage, the object-displacement-normalized band was **22.22× wider** than the pooled band; at nominal 80%, it was **8.12× wider**. KineWorld therefore rejected this mechanism instead of presenting significance as product value.

## Reading boundary

This card supports a narrow claim: KineWorld can reproduce a public world-model control run, audit its hidden failure distribution, and reject a diagnostic that fails a predeclared utility gate. It does **not** show that KineWorld beats another model, that object motion causes error, or that this result has independent validation.

## Reproduce and verify

- Machine-readable card: `results/KW_RISK_CARD_v0.json`
- Frozen experiment: `verification/experiments/KW-EXP-0006.md`
- Experiment manifest: `verification/manifests/KW-EXP-0006_manifest.json` (`5e9ee7261eaa023d30bb584622c335f1dd96fb29e4f65141489c55d2f7619dcd`)
- Analysis artifact: `results/kw_exp_0006_analysis.json` (`9776eea31522c61ae15e9b354e54b7a38a2853120acea14eee40aeb4f695e4c5`)
- Builder: `verification/scripts/build_risk_card_v0.py`

## Next validation step

An external lab or evaluator should rerun the frozen protocol from the checkpoint hash, compare the generated manifest, and sign its own attestation. Until that happens, the evidence level remains E1.
