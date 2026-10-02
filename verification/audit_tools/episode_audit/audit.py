#!/usr/bin/env python3
"""Declared-provenance and paired-episode audit. Author: 潘奕成 (Yicheng Pan).

Implemented with OpenAI assistance. Standard library only; no network or code
execution from input. A passing report is not a certificate of no leakage.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import sys

VERSION = "0.1.0"
MAX_BYTES = 32 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
MAX_RECORDS = 100_000
MAX_BOOTSTRAP_WORK = 5_000_000
EPISODE_FIELDS = {"type", "source_id", "episode_idx", "split", "seed_namespace", "seed", "content_sha256"}
EVALUATION_FIELDS = {"type", "source_id", "episode_idx", "start_step", "goal_step", "evaluation_seed", "method", "candidate_budget", "metric", "conditions"}


class InputError(ValueError):
    """Malformed or unsupported declared input."""


def _string(value):
    return isinstance(value, str) and 0 < len(value) <= 256 and value == value.strip() and not any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value)


def _integer(value, minimum=0):
    return type(value) is int and minimum <= value <= 2**53 - 1


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _constant(value):
    raise InputError(f"non-standard JSON numeric constant: {value}")


def strict_json(text):
    try:
        return json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
    except (ValueError, RecursionError) as exc:
        raise InputError(str(exc)) from exc


def read_input(path, jsonl=False):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise InputError(f"input exceeds {MAX_BYTES} bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputError("input must be UTF-8") from exc
    if not jsonl:
        if len(raw) > MAX_LINE_BYTES:
            raise InputError("manifest exceeds 1 MiB")
        return strict_json(text), hashlib.sha256(raw).hexdigest()
    records = []
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # One terminal LF is allowed; interior blank lines are not.
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise InputError(f"line {line_number}: blank lines are not allowed")
        if len(line.encode("utf-8")) > MAX_LINE_BYTES:
            raise InputError(f"line {line_number}: exceeds 1 MiB")
        try:
            records.append(strict_json(line))
        except InputError as exc:
            raise InputError(f"line {line_number}: {exc}") from exc
        if len(records) > MAX_RECORDS:
            raise InputError(f"input exceeds {MAX_RECORDS} records")
    if not records:
        raise InputError("records file is empty")
    return records, hashlib.sha256(raw).hexdigest()


def validate_manifest(manifest):
    required = {"schema_version", "baseline", "candidate", "metric", "required_conditions"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise InputError(f"manifest fields must be exactly {sorted(required)}")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise InputError("schema_version must be integer 1")
    if not all(_string(manifest[k]) for k in ("baseline", "candidate")) or manifest["baseline"] == manifest["candidate"]:
        raise InputError("baseline and candidate must be distinct nonempty names")
    metric = manifest["metric"]
    if not isinstance(metric, dict) or set(metric) != {"name", "direction"} or not _string(metric["name"]) or metric["direction"] not in ("lower", "higher"):
        raise InputError("metric requires a name and direction: lower or higher")
    conditions = manifest["required_conditions"]
    if not isinstance(conditions, dict) or len(conditions) > 64 or any(not _string(k) or not _integer(v, 1) for k, v in conditions.items()):
        raise InputError("required_conditions must map up to 64 labels to positive episode counts")


def validate_record(record):
    if not isinstance(record, dict):
        return "record must be an object"
    kind = record.get("type")
    fields = EPISODE_FIELDS if kind == "episode" else EVALUATION_FIELDS if kind == "evaluation" else None
    if fields is None:
        return "type must be episode or evaluation"
    if set(record) != fields:
        return f"fields must be exactly {sorted(fields)}"
    if not _string(record["source_id"]) or not _integer(record["episode_idx"]):
        return "source_id must be a nonempty label and episode_idx a nonnegative safe integer"
    if kind == "episode":
        if record["split"] not in ("train", "validation", "test"):
            return "split must be train, validation or test"
        namespace, seed = record["seed_namespace"], record["seed"]
        if not ((namespace is None and seed is None) or (_string(namespace) and _integer(seed))):
            return "seed_namespace and seed must both be null or a nonempty label and nonnegative safe integer"
        digest = record["content_sha256"]
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            return "content_sha256 must be null or 64 lowercase hex characters"
    else:
        if not all(_integer(record[k]) for k in ("start_step", "goal_step", "evaluation_seed")) or record["goal_step"] <= record["start_step"]:
            return "steps and evaluation_seed must be nonnegative safe integers; goal_step must exceed start_step"
        if not _integer(record["candidate_budget"], 1) or not _string(record["method"]):
            return "method must be a nonempty label and candidate_budget a positive safe integer"
        if not _finite(record["metric"]):
            return "metric must be a finite JSON number (not a boolean)"
        conditions = record["conditions"]
        if not isinstance(conditions, list) or len(conditions) > 64 or any(not _string(c) for c in conditions) or len(set(conditions)) != len(conditions):
            return "conditions must be a list of up to 64 distinct nonempty labels"
    return None


def median(values):
    """Finite median with an overflow-safe even midpoint."""
    ordered = sorted(values)
    if not ordered or any(not _finite(v) for v in ordered):
        raise InputError("median needs finite observations")
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    low, high = float(ordered[midpoint - 1]), float(ordered[midpoint])
    total = low + high
    # Sum first to preserve subnormals; half-sums only when addition overflows.
    return total / 2 if math.isfinite(total) else low / 2 + high / 2


def percentile(ordered, probability):
    """Linear interpolation at (n-1)*p, avoiding subtraction overflow."""
    position = (len(ordered) - 1) * probability
    lo = math.floor(position)
    hi = math.ceil(position)
    fraction = position - lo
    # Only two endpoint calls per report: exact weighting avoids both
    # underflow of tiny weighted terms and overflow of a large difference.
    weight = Fraction(fraction)
    return float(Fraction(ordered[lo]) * (1 - weight) + Fraction(ordered[hi]) * weight)


def bootstrap_episode_median(episode_effects, resamples, seed, confidence):
    """Bootstrap independent episode summaries, never individual windows."""
    if len(episode_effects) < 2:
        return None
    rng = random.Random(seed)
    n = len(episode_effects)
    samples = sorted(median([episode_effects[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
    tail = (1 - confidence) / 2
    return {"low": percentile(samples, tail), "high": percentile(samples, 1 - tail),
            "confidence": confidence, "resamples": resamples, "seed": seed,
            "method": "percentile_bootstrap_over_episodes", "degenerate": samples[0] == samples[-1]}


def audit(manifest, records, *, resamples=2000, seed=0, confidence=0.95):
    validate_manifest(manifest)
    if not _integer(resamples, 100) or resamples > 10000 or not _integer(seed) or type(confidence) not in (int, float) or not 0 < confidence < 1:
        raise InputError("require 100 <= resamples <= 10000, a nonnegative safe seed, and 0 < confidence < 1")
    if not isinstance(records, list) or not 0 < len(records) <= MAX_RECORDS:
        raise InputError(f"records must be a nonempty list of at most {MAX_RECORDS} objects")
    issues = []

    def issue(severity, code, message, **context):
        issues.append({"severity": severity, "code": code, "message": message, **context})

    valid = []
    for row, record in enumerate(records, 1):
        error = validate_record(record)
        if error:
            issue("error", "SCHEMA", error, row=row)
        else:
            valid.append((row, record))
    episodes = defaultdict(list)
    seeds = defaultdict(list)
    hashes = defaultdict(list)
    namespaces = defaultdict(set)
    evaluations = defaultdict(dict)
    baseline, candidate = manifest["baseline"], manifest["candidate"]
    for row, record in valid:
        key = (record["source_id"], record["episode_idx"])
        if record["type"] == "episode":
            episodes[key].append(record)
            if record["seed"] is not None:
                seeds[(record["seed_namespace"], record["seed"])].append((key, record["split"]))
                namespaces[record["seed_namespace"]].add(record["split"])
            if record["content_sha256"] is not None:
                hashes[record["content_sha256"]].append((key, record["split"]))
        else:
            if record["method"] not in (baseline, candidate):
                issue("error", "UNKNOWN_METHOD", "evaluation method is not declared", row=row)
                continue
            task = (*key, record["start_step"], record["goal_step"], record["evaluation_seed"])
            if record["method"] in evaluations[task]:
                issue("error", "DUPLICATE_EVALUATION", "duplicate method/task/seed record", row=row, task=list(task))
            else:
                evaluations[task][record["method"]] = record
    for key, entries in sorted(episodes.items()):
        splits = sorted({e["split"] for e in entries})
        if len(splits) > 1:
            issue("error", "SPLIT_OVERLAP", "the same declared episode occurs in multiple splits", episode=list(key), splits=splits)
        elif len(entries) > 1:
            issue("error", "DUPLICATE_EPISODE", "episode inventory must contain exactly one record per identity", episode=list(key))
    for groups, code, label in ((seeds, "SEED_REUSE", "generator namespace/seed"), (hashes, "CONTENT_REUSE", "content digest")):
        for provenance, entries in sorted(groups.items()):
            identities = {entry[0] for entry in entries}
            splits = {entry[1] for entry in entries}
            if len(splits) > 1 or len(identities) > 1:
                issue("error", code, f"the same {label} labels multiple episodes or splits; independence is not established", splits=sorted(splits), episodes=[list(k) for k in sorted(identities)])
    for namespace, splits in sorted(namespaces.items()):
        if len(splits) > 1:
            issue("warning", "SHARED_SEED_NAMESPACE", "splits share a generator namespace; distinct seeds alone neither prove nor disprove independence", namespace=namespace, splits=sorted(splits))
    if not episodes:
        issue("error", "NO_EPISODES", "episode inventory is absent")
    if not any(e["split"] == "train" for entries in episodes.values() for e in entries):
        issue("warning", "NO_TRAIN_INVENTORY", "no training episodes are declared; training/test overlap cannot be checked")
    if any(e["seed"] is None and e["content_sha256"] is None for entries in episodes.values() for e in entries):
        issue("warning", "IDENTITY_ONLY_PROVENANCE", "some episodes have neither a generator seed nor a content digest; only declared identities can be compared")
    declared_test = {key for key, entries in episodes.items() if len(entries) == 1 and entries[0]["split"] == "test"}
    evaluated = {task[:2] for task in evaluations}
    for key in sorted(declared_test - evaluated):
        issue("error", "UNEVALUATED_TEST_EPISODE", "a declared test episode has no evaluations", episode=list(key))
    effects = defaultdict(list)
    coverage = defaultdict(set)
    paired_tasks = 0
    for task, arms in sorted(evaluations.items()):
        key = task[:2]
        entries = episodes.get(key, [])
        if len(entries) != 1 or entries[0]["split"] != "test":
            issue("error", "NONTEST_OR_UNKNOWN_EPISODE", "evaluations require one registered test episode", task=list(task))
            continue
        if set(arms) != {baseline, candidate}:
            issue("error", "UNPAIRED_TASK", "both declared methods must evaluate every task and evaluation seed", task=list(task), methods=sorted(arms))
            continue
        first, second = arms[baseline], arms[candidate]
        if first["candidate_budget"] != second["candidate_budget"]:
            issue("error", "BUDGET_MISMATCH", "paired methods declare unequal candidate budgets", task=list(task))
            continue
        if set(first["conditions"]) != set(second["conditions"]):
            issue("error", "CONDITION_MISMATCH", "paired task conditions must be method-independent", task=list(task))
            continue
        # Preserve large integer differences and mixed int/float differences.
        # JSON decimal/exponent values were already parsed as binary64.
        try:
            delta = float(Fraction(second["metric"]) - Fraction(first["metric"]))
        except OverflowError:
            delta = math.inf
        if not math.isfinite(delta):
            issue("error", "NONFINITE_DIFFERENCE", "finite metrics produced an overflowing paired difference; rescale units", task=list(task))
            continue
        paired_tasks += 1
        effects[key].append(delta)
        for condition in first["conditions"]:
            coverage[condition].add(key)
    for condition, minimum in sorted(manifest["required_conditions"].items()):
        count = len(coverage[condition])
        if count < minimum:
            issue("error", "INSUFFICIENT_CONDITION_COVERAGE", "too few distinct paired test episodes cover a required condition", condition=condition, observed=count, required=minimum)
    if not evaluations:
        issue("error", "NO_EVALUATIONS", "no declared-method evaluations are present")
    if not effects:
        issue("error", "NO_PAIRED_EPISODES", "no valid paired test episodes are available")
    error_count = sum(i["severity"] == "error" for i in issues)
    effect = None
    if not error_count:
        values = [median(effects[key]) for key in sorted(effects)]
        if len(values) * resamples > MAX_BOOTSTRAP_WORK:
            raise InputError(f"episode count times resamples exceeds {MAX_BOOTSTRAP_WORK}; reduce resamples")
        interval = bootstrap_episode_median(values, resamples, seed, confidence)
        if len(values) < 10:
            issue("warning", "SMALL_EPISODE_SAMPLE", "fewer than 10 episodes; uncertainty estimates may be unstable, and one episode has no interval", episodes=len(values))
        if interval is not None and interval["degenerate"]:
            issue("warning", "DEGENERATE_BOOTSTRAP", "bootstrap interval is degenerate; this does not establish zero population uncertainty")
        effect = {"statistic": "median_of_episode_median_paired_differences", "difference": "candidate_minus_baseline",
                  "value": median(values), "positive_favors": candidate if manifest["metric"]["direction"] == "higher" else baseline,
                  "episode_count": len(values), "paired_task_seed_count": paired_tasks, "interval": interval,
                  "episode_effects": [{"source_id": k[0], "episode_idx": k[1], "paired_task_seed_count": len(effects[k]), "value": median(effects[k])} for k in sorted(effects)]}
    return {"tool": "kineworld_episode_audit", "tool_version": VERSION, "schema_version": 1,
            "status": "fail" if error_count else "pass_with_warnings" if issues else "pass",
            "errors": error_count, "warnings": sum(i["severity"] == "warning" for i in issues), "issues": issues,
            "metric": manifest["metric"], "baseline": baseline, "candidate": candidate,
            "required_condition_coverage": {k: {"observed_episodes": len(coverage[k]), "required_episodes": v} for k, v in sorted(manifest["required_conditions"].items())},
            "effect": effect,
            "limits": ["Declared metadata only: no proof of absence of leakage or episode independence.",
                       "Equal candidate counts do not establish equal compute, candidate pools, or model privileges.",
                       "Percentile interval assumes independent representative episodes; no hypothesis test or superiority claim.",
                       "Coverage is declared task coverage, not a causal, safety, robustness, or benchmark certificate."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest, manifest_hash = read_input(args.manifest)
        records, records_hash = read_input(args.records, jsonl=True)
        result = audit(manifest, records, resamples=args.resamples, seed=args.seed, confidence=args.confidence)
        result["inputs_sha256"] = {"manifest": manifest_hash, "records": records_hash}
        result["runtime"] = {"python": sys.version.split()[0]}
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 1 if result["errors"] or (args.fail_on_warning and result["warnings"]) else 0
    except (InputError, OSError) as exc:
        print(json.dumps({"tool": "kineworld_episode_audit", "tool_version": VERSION, "status": "input_error", "error": str(exc)}, sort_keys=True, ensure_ascii=True, allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
