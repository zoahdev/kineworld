# Episode Audit

A small, zero-dependency preflight check for paired world-model evaluations.

**Project author: 潘奕成 (Yicheng Pan), KineWorld.** The design, implementation,
synthetic fixtures, tests, and documentation were drafted with substantial
OpenAI-assistant involvement under the author's direction. This is a practical
engineering utility, not a new world-model method or a benchmark claim.

Episode Audit catches inconsistencies in declared evaluation records **before**
producing an effect summary. It runs offline with Python 3.10+ and the standard
library. No model, GPU, dataset download, account, package installation, or API
key is needed. Version 0.1.0 was tested on Python 3.12; other versions have not
been exercised in this release.

## Try it

From this directory:

```sh
python audit.py --manifest examples/manifest.json --records examples/valid.jsonl
python -m unittest -v test_audit.py
```

The valid example is entirely synthetic. It passes and has a median
candidate-minus-baseline difference of -0.5 over 12 episodes. That number is an
arithmetic fixture, not an experimental result. The report includes the exact
input SHA-256 hashes, tool/Python versions, bootstrap seed, and method.

Try the intentionally invalid fixture:

```sh
python audit.py --manifest examples/manifest.json --records examples/invalid.jsonl
```

It exits 1 and suppresses the effect report. It demonstrates cross-split episode
and seed reuse, mismatched candidate budgets, invalid metric values, and missing
required contact coverage. Its full machine-readable output is checked into
`examples/invalid_report.json`; the valid output is in `examples/valid_report.json`.

## What it checks

- Exact episode identity overlap across train, validation, and test splits,
  even when the evaluated windows do not overlap
- Reused episode content hashes and generator namespace/seed pairs, including
  aliases under different source IDs
- Shared generator namespaces across splits, as a warning rather than a claim
  that disjoint seeds leak
- Exact pairing on source ID, `(episode_idx, start_step, goal_step)`, and
  evaluation seed; no duplicate rows, silent inner join, or undeclared method
- Every declared test episode has evaluations; every evaluation refers to a
  registered test episode
- Finite numeric metrics, positive equal per-task candidate counts, and
  method-independent condition labels
- Predeclared rare-condition coverage counted in distinct paired test episodes,
  not multiplied by windows or repeated evaluation seeds

Schema and statistical semantics: [SCHEMA.md](SCHEMA.md). A machine-readable
record schema is [record.schema.json](record.schema.json), with a separate
[manifest schema](manifest.schema.json). The Python validator also checks
cross-record constraints that JSON Schema cannot express.

## Use your own logs

1. Export your complete episode inventory, including training and validation,
   plus two methods' test-task evaluations as JSONL. Use stable original episode
   IDs. Each line is one episode or evaluation object; see the examples.
2. Choose one scalar metric, its direction, two method labels, and minimum
   episode counts for important conditions in a JSON manifest. Fix these before
   inspecting the comparison. Counts are your design choices, not thresholds
   learned by the tool.
3. Run the checker. Any error blocks the effect report; do not repair the gate
   by removing adverse episodes or renaming shared IDs. Inspect the data and
   generation process and report legitimate deviations.

```sh
python audit.py --manifest my_manifest.json --records my_records.jsonl \
  --resamples 2000 --seed 0 --confidence 0.95 > my_audit.json
```

Exit codes: **0** = no audit errors (warnings may remain), **1** = audit error or
`--fail-on-warning` policy failure, **2** = malformed/unreadable input or invalid
configuration. Argument-parser failures also exit 2, with usage on stderr.
Reports otherwise go to stdout; the tool never writes to your source files.
For a warning-free CI gate, add `--fail-on-warning`.

There is no permissive mode that silently drops invalid records. Reports may
contain episode/source labels you supplied; review them before sharing.

## Effect and uncertainty

For each exactly paired task/seed, compute candidate metric minus baseline
metric. Take the median of those differences within each episode, then take
the median of the episode medians. Each episode has equal outer weight,
regardless of its number of windows/seeds. This estimand is **not** the overall
mean difference, a success-rate difference, the median of all windows, or the
difference of two independently computed medians.

The optional interval resamples these episode summaries with replacement and
uses percentile endpoints. Repeated windows or evaluation seeds do not inflate
the number of bootstrap units. One episode gets a descriptive point value and
no interval. Fewer than ten episodes and degenerate bootstrap distributions
produce warnings. Ten is a warning heuristic, not a guarantee of adequacy.
The procedure makes no p-value, significance, multiple-comparison, causal, or
superiority claim. Bootstrap confidence coverage is not guaranteed, especially
with tiny samples, dependence, discrete/tied metrics, or distribution shift.
Do not treat exploratory repeated comparisons as one preplanned interval.

## Honest limits

A pass means the **submitted declarations satisfy these checks**. It is not a
certificate of no leakage or a complete world-model evaluation.

The tool does not inspect raw HDF5/Parquet files, model weights, training code,
normalizers, visual near-duplicates, checkpoint pretraining, task reachability,
label correctness, held-out status, or the completeness of an inventory. A
missing training inventory produces a warning, not a fabricated clean split.
Different source IDs or inaccurate hashes can hide real duplication. Entire
omitted tasks inside an otherwise evaluated episode cannot be detected without
a separately frozen task list.

Content digests must be computed over a consistent canonical representation of
a whole original episode. Equal content may be legitimate repeated data, but
this paired-independent-episode workflow requires those aliases be resolved
explicitly. Hashes are declarations; this tool does not authenticate them.

A seed namespace identifies the generator, version, configuration, and stream
semantics; a seed pair must identify an entire original episode. Seed records
are not a substitute for provenance. Unknown/non-applicable provenance should
be null, not invented. Independence and representative sampling remain the
researcher's responsibility. Related episodes may require a higher-level
cluster design that this version does not implement.

Equal candidate counts do not establish equal FLOPs, wall time, model calls,
search horizons, candidate pools, training budgets, or privileged information.
Conditions must be defined independently of which method succeeded; coverage
is not evidence that a rare-condition risk is acceptably small.

## Relationship to KineWorld

The repository already provides `../success_comparison.py` for an **unpaired
binary-rate** comparison. This tool adds a separate declared-provenance gate
and a paired **continuous-metric, episode-clustered** descriptive report. It
does not replace that checker, change the harness contracts, or reanalyze any
previous checkpoint result. Existing reports and their evidence boundaries are
unchanged. Only public interfaces and synthetic examples informed this utility;
no private model implementation or training recipe is included.

## Credit, validation, and license status

Please credit **潘奕成 (Yicheng Pan)** and link to
[zoahdev/kineworld](https://github.com/zoahdev/kineworld) when discussing this work.
`CITATION.cff` contains software citation metadata. There is no DOI or external
peer-review/replication claim. `VERIFICATION_PLAN.md` describes local engineering
checks, and `verification.log` records their execution. Automated internal
review is not independent human validation.

No new open-source license is granted by this draft. A component-specific
license decision is pending; no existing repository or dependency license is
changed. Public visibility alone does not grant reuse permissions.
