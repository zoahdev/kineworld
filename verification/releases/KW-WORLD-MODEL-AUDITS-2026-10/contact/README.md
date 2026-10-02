# Contact world model diagnostic

A standalone CPU-only synthetic study by 潘奕成 (Yicheng Pan), produced with explicit OpenAI-assistant involvement.

Read [the technical note](REPORT.md) for the full method, data split, tables, limitations, prior work, and AI-assistance disclosure. The result is intentionally limited: ordinary one-step MSE and mean executed-cost rankings agreed, increasing the candidate budget helped every tested model, and a privileged correctly specified hybrid baseline matched the same-candidate oracle. No new general world-model method or open-problem solution is claimed.

## Reproduce

Use Python 3.12 and the versions in requirements.txt. From this directory:

    OPENBLAS_NUM_THREADS=1 python run.py --seeds 10 --output results_reproduced.json
    python verify.py --replay results_reproduced.json
    python -m unittest -v test_study.py
    python make_report.py --check

The experiment is independent of the surrounding repository. It does not read private data, checkpoints, credentials, or other repository files and performs no network calls. No GPU or paid compute is required.

## Files

- protocol.md: locally prespecified settings and transparent execution clarifications
- run.py: synthetic dynamics, learned baselines, evaluation, and JSON export
- results.json: all seed-level results, selected hyperparameters, aggregates, and package versions
- report_template.md and make_report.py: narrative source and deterministic numerical-table generation
- REPORT.md: complete technical note
- test_study.py and verify.py: simulator, metric, split, replay, and report invariants
- manifest.json: portable release allowlist, SHA-256 hashes, provenance, and validation metadata
- verification.log: recorded passing test and replay summary; not external replication

The protocol was not registered with a third party. The report includes the unexecuted balanced-probe clarification and the known-wall privilege of the hybrid baseline. There is no broad license change in this package.
