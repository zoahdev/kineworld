# Schema v1 and audit semantics

UTF-8 JSON only. The manifest is one object; the records file is JSONL (one
object per LF-separated line, optional terminal LF, CRLF accepted). No comments,
blank lines, duplicate JSON keys, unknown fields, NaN, or Infinity. A JSON
exponent that overflows binary64 is rejected as a metric by the validator.
Manifest limit: 1 MiB. Records limit: 32 MiB, 100,000 lines, 1 MiB per line.

All identifier strings must be 1–256 characters, with no leading/trailing
whitespace, control characters below U+0020, or surrogate code points. Identifier/count fields require actual JSON integers, never booleans or floats,
in the range 0 to 2^53−1. Positive count fields start at 1. Keeping identifiers
in this range avoids common JSON interop loss. Metric values are exempt from
these count restrictions: they may be negative or contain integers above 2^53,
provided they are finite under the metric rules below.

## Manifest

All five fields are required; no extra fields are allowed.

```json
{
  "schema_version": 1,
  "baseline": "base",
  "candidate": "candidate",
  "metric": {"name": "executed_cost", "direction": "lower"},
  "required_conditions": {"contact": 10, "occlusion": 10}
}
```

`baseline` and `candidate` are distinct method labels. `metric.direction` is
`lower` or `higher` and affects only the report's `positive_favors` label;
it does not reverse the numeric difference. `required_conditions` maps up to
64 distinct labels to positive minimum counts of paired test **episodes**.
An empty map explicitly disables condition-coverage requirements.

Use one metric and two methods per invocation. The schema does not infer the
metric definition, units, or comparability; both methods must already share
those definitions. Fix design choices before looking at effect estimates.

## Episode inventory record

All seven fields are required.

```json
{"type":"episode","source_id":"my_dataset:immutable-revision","episode_idx":12,"split":"test","seed_namespace":"sim-v1-configA-test","seed":2012,"content_sha256":null}
```

- `source_id`: stable dataset identity and revision. Preserve original episode
  lineage across derived windows or augmentations. The checker cannot reconcile
  renamed IDs or dataset revisions without supplied shared content hashes.
- `episode_idx`: original episode's nonnegative integer identity in that source
- `split`: `train`, `validation`, or `test`; the inventory covers data used by
  either of the two methods. If training sets differ, declare their union. This
  version does not produce method-specific overlap exceptions.
- `seed_namespace`, `seed`: both null if unknown/inapplicable, or a string and
  nonnegative integer. The pair must identify a full independently generated
  original episode, not a run that generates many episodes from one stream.
  Namespace includes the generator/version/configuration/stream semantics;
  do not alter it merely to evade a warning or collision.
- `content_sha256`: null or a 64-character lowercase hexadecimal SHA-256 digest
  of a consistently canonicalized whole original episode. The tool does not
  compute or verify dataset content hashes itself.

Exactly one inventory record is permitted per `(source_id, episode_idx)`.
Reused identities in different splits fail. The same supplied generator pair
or nonnull content digest assigned to distinct episodes or splits fails, even
within test: those declarations do not support independent episode resampling.
Shared generator namespaces with disjoint seeds cause a warning only. Missing
both digest and seed produces an identity-only-provenance warning.

## Evaluation record

All ten fields are required.

```json
{"type":"evaluation","source_id":"my_dataset:immutable-revision","episode_idx":12,"start_step":10,"goal_step":25,"evaluation_seed":0,"method":"base","candidate_budget":64,"metric":1.25,"conditions":["contact"]}
```

- `source_id`, `episode_idx`: reference one registered test episode
- `start_step`, `goal_step`: nonnegative integers, with goal strictly later
- `evaluation_seed`: nonnegative integer identifying the matched evaluation
  randomization/repeat. Use 0 for a deterministic single evaluation. This is
  different from the episode generation seed. Both methods must evaluate the
  same declared seeds; matching integer labels cannot prove matching RNG use.
- `method`: exactly one of the two manifest labels
- `candidate_budget`: positive integer number of candidate action sequences
  considered for this evaluation. Pairwise equality is required; budgets may
  vary across different tasks. Do not substitute an incomparable budget unit.
- `metric`: finite numeric scalar in shared units, no booleans
- `conditions`: up to 64 distinct label strings, possibly empty. These describe
  the task, not method-dependent outcomes; both methods' labels must agree.

The complete pairing key is:
`(source_id, episode_idx, start_step, goal_step, evaluation_seed)`.

A method may occur exactly once per key. Both methods must appear on every
submitted key. All declared test episodes must have evaluations. Repeated
windows, different goals, and repeated evaluation seeds remain separate pairs
but are combined within their original episode for the outer analysis. You
must aggregate any sub-task repetitions not represented by this schema before
exporting them, or use distinct meaningful evaluation seeds.

## Errors, warnings, and effect gating

All schema errors are reported with a 1-based record/line index when reading
JSONL. Logical errors have stable codes, including `SPLIT_OVERLAP`,
`SEED_REUSE`, `CONTENT_REUSE`, `UNPAIRED_TASK`, `BUDGET_MISMATCH`,
`CONDITION_MISMATCH`, and `INSUFFICIENT_CONDITION_COVERAGE`. Any error suppresses
the effect report. There is no silent filtering into a favorable paired subset.
Counts shown in an invalid report describe only structurally valid pairs and
must not be read as a complete evaluation.

Warnings allow a report unless `--fail-on-warning` is set. Common warnings are
`NO_TRAIN_INVENTORY`, `IDENTITY_ONLY_PROVENANCE`, `SHARED_SEED_NAMESPACE`,
`SMALL_EPISODE_SAMPLE`, and `DEGENERATE_BOOTSTRAP`. A warning is an inspection
prompt, not an automatic finding of leakage.

Malformed JSON, unreadable files, unsupported schemas or invalid bootstrap
settings produce `status: input_error`, exit 2, and no numerical effect.

## Bootstrap and arithmetic

Let d(e,t) be candidate minus baseline for paired task/seed t in episode e.
The point statistic is median over e of [median over t of d(e,t)].
It gives every episode equal outer weight while using within-episode medians
for repeated tasks/seeds. It estimates a typical episode's typical paired
change, not a mean benchmark score. Even medians use the average of the two
central values; safe midpoint arithmetic preserves subnormal values and avoids
same-sign overflow. Paired subtraction preserves exact integer values, including
integers larger than 2^53, before converting the difference to binary64. JSON
decimal/exponent metrics are parsed as binary64, so original decimal precision
is not retained. If a paired difference overflows binary64, the audit fails and
asks for rescaled units. Percentile interpolation uses exact rational weights
for the parsed binary64 probability before rounding the endpoint to binary64.

The interval uses a local `random.Random(seed)` generator. For each replicate,
resample n episode summaries with replacement, then compute their median.
Sort the replicate medians and linearly interpolate percentiles at positions
`(B-1)*(1-confidence)/2` and `(B-1)*(1+confidence)/2`. Episode keys are sorted
first, making results invariant to JSONL row order when there are no errors.

CLI defaults: 2,000 replicates, seed 0, confidence 0.95. Valid replicates range
from 100 to 10,000; confidence is strictly between zero and one. The product of
episode count and replicates is bounded at 5,000,000. Reduce replicates when
necessary; the tool does not silently downsample episodes. One episode has no
interval. No bootstrap generalization guarantee or p-value is supplied.
