# World-model audit technical reports

By 潘奕成 (Yicheng Pan), with disclosed AI assistance. Released as technical reports, not peer-reviewed articles. The public [collection index](../../../docs/research/world-model-audits-2026-10/README.md) explains their scope.

- `theory/`: exact causal-identifiability construction, proofs, and a rational-arithmetic checker
- `contact/`: CPU-only synthetic contact-dynamics diagnostic, complete results, and replay checks

Each study directory is self-contained. Follow the commands in its report or README. The exact checker requires Python's standard library; the contact package records pinned scientific-Python dependencies. No private data or model checkpoints are required.

## PDFs

The Markdown reports are the canonical editable sources. PDFs are formatted renderings of those sources. In the contact PDF, wide tables are split by metric to keep every entry legible; their values are unchanged. The PDF renderer also typesets the source's plain-text cost formula without changing the formula.

To rebuild both PDFs, install Pandoc and a working XeLaTeX distribution, plus the Noto Serif CJK SC and DejaVu Sans Mono fonts, then run:

    python render_pdfs.py

PDF bytes may differ with renderer versions and metadata timestamps. The release manifest records the hashes of the distributed files. Visual inspection is still necessary after rendering.

## Citation and evidence

Cite the report title, 潘奕成 (Yicheng Pan), 2026, this repository and the exact commit used. No DOI, arXiv submission, journal acceptance, external replication, or established novelty is asserted.

This collection does not change the evidentiary status of the original Push-T checkpoint studies elsewhere in the repository.
