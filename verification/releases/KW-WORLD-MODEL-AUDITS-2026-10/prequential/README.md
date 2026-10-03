# World-model learned-confidence audit

**Author:** 潘奕成 (Yicheng Pan). **Version:** 1.0, 2026-10-03. Public technical report; not peer-reviewed.

Read [REPORT.md](REPORT.md) for the result, exact counterexample, repaired theorem, limits, and sources.

Reproduce with:

    python verify.py
    python check_high_precision.py

Requirements: Python 3, NumPy, SciPy, mpmath. Tested with NumPy 2.3.5 and SciPy 1.17.0. No network, accelerator, accounts, training jobs, or private repository code are needed. The scripts perform bounded synthetic checks only.

The local execution record states that PROTOCOL.md was fixed and hashed before execution. These hashes establish integrity, not independent proof of timing or public preregistration. results.json/run.log retain all five predetermined runs; replay.log is a separate automated repeat with a byte-identical results.json. high_precision.json checks all reported confidence intervals separately at 100 decimal digits. Exact rational enumeration checks support and adaptive sampling; finite numerical checks do not substitute for the proof.

No broad originality or real-world policy improvement is asserted. This self-contained publication includes a corrected finite-family baseline and does not modify earlier public reports.

## Files and integrity

- [Corrected finite-family baseline](FINITE_FAMILY_BASELINE.md)
- [Protocol](PROTOCOL.md) and [protocol digest](protocol.sha256)
- [Check implementation](verify.py) and [complete results](results.json)
- [High-precision cross-check](check_high_precision.py) and [its output](high_precision.json)
- [Publication manifest](manifest.sha256)

Run `sha256sum -c manifest.sha256` before reproduction. The scripts overwrite their corresponding output JSON files; preserve the published results for comparison. The `first_results.sha256` digest identifies the recorded results, without independently timestamping them.

Substantial OpenAI AI assistance was used for derivations, research, code, tests, and writing. Internal automated review is not external human peer review. No new license is granted by public visibility; the component-specific license decision is pending.
