# World-model audit notes: identifiability, contact dynamics, and window certification

**Author:** 潘奕成 (Yicheng Pan)  
**Version:** 1.1, 2026-10-03  
**Status:** Public technical reports. Not peer reviewed.

This collection studies three narrow questions about action-relevant world models:

1. What can behavior-policy observations identify when the policy uses hidden state?
2. How do one-step prediction error and open-loop planning quality compare in a small contact-dynamics system?
3. Under which explicit assumptions can prequential likelihood and finite-memory counts certify prediction and discounted planning error?

The reports provide explicit constructions, executable checks, and reproducible results. They do not claim a solution to general world-model learning, a state-of-the-art benchmark result, or established novelty. Each report separates known background from the contribution of its particular construction or diagnostic.

## 1. Exact finite-model causal audit

- [Technical report](../../../verification/releases/KW-WORLD-MODEL-AUDITS-2026-10/theory/report.md)
- [PDF](../../../verification/releases/KW-WORLD-MODEL-AUDITS-2026-10/theory/report.pdf)
- [Verification plan](../../../verification/releases/KW-WORLD-MODEL-AUDITS-2026-10/theory/verification_plan.md)
- [Exact checker](../../../verification/releases/KW-WORLD-MODEL-AUDITS-2026-10/theory/verify_causal_audit.py)
- [Machine-readable results](../../../verification/releases/KW-WORLD-MODEL-AUDITS-2026-10/theory/exact_results.json)

A finite, hidden-state example makes observational non-identifiability explicit. Exact arithmetic checks both a fixed balanced intervention design and an optimal adaptive-query formula. The scope is the specified known finite family with independent hidden-state resets, rather than general causal identification or a neural-world-model performance guarantee.

## 2. Synthetic contact-dynamics study

- [Technical report](../../../verification/releases/KW-WORLD-MODEL-AUDITS-2026-10/contact/REPORT.md)
- [PDF](../../../verification/releases/KW-WORLD-MODEL-AUDITS-2026-10/contact/REPORT.pdf)
- [Protocol and execution clarifications](../../../verification/releases/KW-WORLD-MODEL-AUDITS-2026-10/contact/protocol.md)
- [Experiment code](../../../verification/releases/KW-WORLD-MODEL-AUDITS-2026-10/contact/run.py)
- [Machine-readable results](../../../verification/releases/KW-WORLD-MODEL-AUDITS-2026-10/contact/results.json)

This controlled CPU-only experiment compares small learned dynamics models with matched candidate actions and an oracle evaluated on the same candidate set. A correct-family structural baseline has privileged knowledge of the functional family and wall locations. Its behavior must be interpreted with that prior in view. In this run, neither a reversal between ordinary one-step MSE and mean executed-cost rankings nor harmful effects from larger search budgets was demonstrated.

## 3. Prequential window certification

- [Technical report](../../../verification/releases/KW-WORLD-MODEL-AUDITS-2026-10/prequential/REPORT.md)
- [Reproduction package](../../../verification/releases/KW-WORLD-MODEL-AUDITS-2026-10/prequential/README.md)
- [Corrected fixed-family baseline](../../../verification/releases/KW-WORLD-MODEL-AUDITS-2026-10/prequential/FINITE_FAMILY_BASELINE.md)
- [Protocol](../../../verification/experiments/KW-PREQUENTIAL-CERTIFICATE-2026-10.md)
- [Manifest](../../../verification/manifests/KW-PREQUENTIAL-CERTIFICATE-2026-10_manifest.json)

This report separates a fixed finite realizable-family guarantee from a continuous-class extension using predictions committed before each outcome. It includes exact failure examples for fitted shortlists and global closure targets, plus a reproducible controlled-erasure-model illustration. The certificate requires realizability, predictable actions and predictions, exact support handling, and a frozen planning model. Five runs do not establish real-world policy improvement, neural-world-model superiority, or broad originality. Coverage is per trajectory under assumptions.

## Reproduce

Run the commands in each report from its own package directory. The exact audit uses only the Python standard library; the contact study lists its scientific-Python dependencies and recorded versions.

The original Push-T checkpoint studies elsewhere in this repository retain their original E1 evidence limits. These standalone exact/synthetic notes do not raise the evidence level of those studies.

## Authorship and AI assistance

OpenAI AI tools assisted with literature research, derivation, implementation, executable checks, and writing. Tool-based checks and AI-assisted review are not external peer review. No human expert endorsement or external replication is claimed. The reports provide more specific disclosure and limitations.

No private datasets, model checkpoints, company records, or unpublished implementation materials are included in this collection. No journal acceptance, arXiv identifier, or DOI is claimed for these reports.
