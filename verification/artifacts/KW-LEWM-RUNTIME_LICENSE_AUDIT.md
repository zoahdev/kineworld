# KW-LEWM runtime dependency and license audit

- Date: 2026-09-02
- Status: **YELLOW / direct dependencies inspected, full closure pending**
- Scope: the local Python environment used for the planned LeWM reproduction
- Legal note: engineering evidence, not legal advice

## Result

`pip check` reports no broken requirements after WorkBuddy installed:

- `stable-worldmodel==0.1.1` — package metadata license expression MIT;
- `stable-pretraining==0.1.8` — package metadata license MIT.

No direct dependency metadata inspected here indicates a non-commercial-only code license. This does **not** establish that the full product, checkpoint, dataset, or redistributed binary bundle is commercially cleared.

## Direct runtime dependency inventory

The following versions were read from the installed environment. Repeated dependencies are shown once in their license family.

| Metadata family | Installed direct dependencies |
|---|---|
| BSD / BSD-like | `torch 2.7.0+cu128`, `torchvision 0.22.0+cu128`, `numpy 2.2.6`, `omegaconf 2.3.1`, `pandas 2.3.3`, `requests-cache 1.3.3`, `prettytable 3.18.0`, `zstandard 0.25.0`, `scikit-learn 1.7.2` |
| MIT / MIT-like | `loguru 0.7.3`, `tabulate 0.10.0`, `gymnasium 1.3.0`, `einops 0.8.2`, `pillow 12.3.0`, `typer 0.27.2`, `rich 15.0.0`, `hydra-core 1.3.6`, `submitit 1.5.4`, `wandb 0.29.0`, `hydra-submitit-launcher 1.2.0`, `minari 0.5.3`, `richuru 0.1.1`, `opt-einsum 3.4.0` |
| Apache-2.0 / Apache-like | `lancedb 0.38.0`, `pylance 11.0.0`, `pyarrow 25.0.1`, `torchmetrics 1.9.0`, `lightning 2.6.5`, `timm 1.0.29`, `transformers 5.16.1`, `datasets 5.0.1`, `opencv-python-headless 5.0.0.93`, `kornia 0.8.2` |
| Mixed permissive / weak copyleft metadata | `tqdm 4.70.0` — `MPL-2.0 AND MIT` |
| Other permissive metadata | `matplotlib 3.10.9` — Python Software Foundation license |

## Unresolved checks before a commercial release

1. Freeze and hash the complete environment lock after the LeWM run stops changing dependencies.
2. Traverse the full transitive dependency graph, including bundled native libraries and their notice obligations.
3. Separate code licenses from model-weight, dataset, image/font, and benchmark terms.
4. Confirm the exact `quentinll/lewm-pusht` checkpoint revision and included license files; record file SHA-256.
5. Confirm that no JEPA-WM CC BY-NC artifacts are packaged into a commercial deliverable. JEPA-WM may remain an internal research comparator only under its current terms.
6. Generate `THIRD_PARTY_NOTICES` before distributing binaries or a hosted product image.

## Compatibility observation

The installed environment contains `transformers==5.16.1`. This satisfies the lower-bound-only dependency metadata but is newer than the 4.x series referenced by parts of the current ecosystem. Treat import/checkpoint conversion as unverified until the official LeWM conversion path succeeds. A successful import is not a license clearance.

## Decision

- Internal LeWM reproduction: **allowed to continue**.
- Public claim that the full KineWorld stack is commercially cleared: **blocked**.
- Commercial packaging: **blocked pending full lock, transitive audit, and checkpoint/dataset term verification**.
