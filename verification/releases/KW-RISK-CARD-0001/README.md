# KW-RISK-CARD-0001 verification package

This package exposes the evidence boundary for KineWorld's first Failure & Risk Card. It separates three different checks that must not be conflated.

## Level 1 — artifact integrity

No GPU or network is required:

```bash
python verification/scripts/verify_risk_card_v0.py
```

Expected output begins with `KW-RISK-CARD-0001 integrity: PASS`. This proves only that the local card and frozen source artifacts match their published hashes and elementary arithmetic is consistent.

## Level 2 — internal analysis regeneration

Regenerate the card from the frozen KW-EXP-0006 evidence:

```bash
python verification/scripts/build_risk_card_v0.py
python -m unittest tests.test_risk_card_v0 -v
```

This checks deterministic derivation. It still does not independently reproduce model inference.

## Level 3 — independent model rerun

An external evaluator must independently obtain the checkpoint identified by SHA-256 `9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb`, reconstruct the frozen KW-EXP-0006 protocol, rerun all 96 episodes, retain raw logs, regenerate the analysis, document every deviation, and complete `verification/third_party/KW_RISK_CARD_ATTESTATION_TEMPLATE.md`.

Only Level 3 performed and signed by an independent party can upgrade the evidence beyond KineWorld's E1 internal reproduction. Passing Level 1 or Level 2 must never be advertised as third-party validation.

## Frozen artifacts

- `results/KW_RISK_CARD_v0.json`
- `docs/product/KINEWORLD_RISK_CARD_v0.md`
- `verification/manifests/KW-RISK-CARD-v0_manifest.json`
- `verification/manifests/KW-EXP-0006_manifest.json`
- `results/kw_exp_0006_analysis.json`
- `verification/scripts/build_risk_card_v0.py`
- `verification/scripts/verify_risk_card_v0.py`
- `tests/test_risk_card_v0.py`
- `verification/third_party/KW_RISK_CARD_ATTESTATION_TEMPLATE.md`
