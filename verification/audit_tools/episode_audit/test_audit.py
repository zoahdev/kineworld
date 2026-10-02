"""Synthetic unit and CLI regression tests; no external packages or files."""
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import audit


def manifest():
    return {"schema_version": 1, "baseline": "base", "candidate": "candidate", "metric": {"name": "cost", "direction": "lower"}, "required_conditions": {"contact": 2}}


def episode(index, split="test", **changes):
    record = {"type": "episode", "source_id": "synthetic:v1", "episode_idx": index, "split": split,
              "seed_namespace": "generator-" + split, "seed": index, "content_sha256": None}
    return dict(record, **changes)


def evaluation(index, method, value, **changes):
    record = {"type": "evaluation", "source_id": "synthetic:v1", "episode_idx": index, "start_step": 0, "goal_step": 10,
              "evaluation_seed": 0, "method": method, "candidate_budget": 32, "metric": value, "conditions": ["contact"]}
    return dict(record, **changes)


def fixture(n=12):
    records = [episode(100, "train"), episode(101, "validation")]
    for i in range(n):
        records += [episode(i), evaluation(i, "base", 10), evaluation(i, "candidate", 10 + i - 6)]
    return records


def codes(result):
    return {item["code"] for item in result["issues"]}


class AuditTests(unittest.TestCase):
    def run_audit(self, rows=None, config=None, **kwargs):
        return audit.audit(config or manifest(), fixture() if rows is None else rows, resamples=200, **kwargs)

    def test_clean_and_known_effect(self):
        result = self.run_audit()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["effect"]["value"], -0.5)
        self.assertEqual(result["effect"]["positive_favors"], "base")
        self.assertEqual(result["effect"]["episode_count"], 12)
        self.assertEqual(result["required_condition_coverage"]["contact"]["observed_episodes"], 12)

    def test_direction_is_explicit(self):
        config = manifest(); config["metric"]["direction"] = "higher"
        self.assertEqual(self.run_audit(config=config)["effect"]["positive_favors"], "candidate")

    def test_reproducible_and_order_invariant(self):
        rows = fixture(); result = self.run_audit(rows)
        random.Random(99).shuffle(rows)
        self.assertEqual(result, self.run_audit(rows))

    def test_split_overlap_blocks_effect(self):
        rows = fixture() + [episode(0, "train")]
        result = self.run_audit(rows)
        self.assertIn("SPLIT_OVERLAP", codes(result)); self.assertIsNone(result["effect"])

    def test_same_episode_different_windows_still_overlap(self):
        rows = fixture() + [episode(0, "train"), evaluation(0, "base", 10, start_step=50, goal_step=60)]
        self.assertIn("SPLIT_OVERLAP", codes(self.run_audit(rows)))

    def test_duplicate_inventory(self):
        self.assertIn("DUPLICATE_EPISODE", codes(self.run_audit(fixture() + [episode(0)])))

    def test_seed_reuse_across_sources(self):
        rows = fixture() + [episode(200, "train", seed_namespace="generator-test", seed=0, source_id="other:v2")]
        self.assertIn("SEED_REUSE", codes(self.run_audit(rows)))

    def test_seed_alias_same_split(self):
        rows = fixture(); rows[5]["seed"] = 0
        self.assertIn("SEED_REUSE", codes(self.run_audit(rows)))

    def test_shared_namespace_disjoint_seeds_is_warning(self):
        rows = fixture(); rows[0]["seed_namespace"] = "generator-test"
        result = self.run_audit(rows)
        self.assertEqual(result["errors"], 0); self.assertIn("SHARED_SEED_NAMESPACE", codes(result))

    def test_digest_reuse_across_sources(self):
        rows = fixture(); rows[0]["content_sha256"] = "a" * 64; rows[2]["content_sha256"] = "a" * 64
        self.assertIn("CONTENT_REUSE", codes(self.run_audit(rows)))

    def test_identity_only_provenance_warning(self):
        rows = fixture(); rows[2]["seed_namespace"] = None; rows[2]["seed"] = None
        self.assertIn("IDENTITY_ONLY_PROVENANCE", codes(self.run_audit(rows)))

    def test_no_training_inventory_warning(self):
        self.assertIn("NO_TRAIN_INVENTORY", codes(self.run_audit(fixture()[1:])))

    def test_unpaired(self):
        rows = fixture(); rows.pop(4)
        self.assertIn("UNPAIRED_TASK", codes(self.run_audit(rows)))

    def test_duplicate_evaluation(self):
        rows = fixture(); rows.append(copy.deepcopy(rows[3]))
        self.assertIn("DUPLICATE_EVALUATION", codes(self.run_audit(rows)))

    def test_seed_and_both_steps_are_pairing_keys(self):
        for field in ("start_step", "goal_step", "evaluation_seed"):
            with self.subTest(field=field):
                rows = fixture(); rows[4][field] += 1
                self.assertIn("UNPAIRED_TASK", codes(self.run_audit(rows)))

    def test_source_is_pairing_key(self):
        rows = fixture(); rows[4]["source_id"] = "other:v1"; rows.append(episode(0, source_id="other:v1", seed_namespace="other"))
        self.assertIn("UNPAIRED_TASK", codes(self.run_audit(rows)))

    def test_unknown_or_non_test_episode(self):
        for idx in (100, 999):
            rows = fixture() + [evaluation(idx, "base", 1), evaluation(idx, "candidate", 2)]
            self.assertIn("NONTEST_OR_UNKNOWN_EPISODE", codes(self.run_audit(rows)))

    def test_unknown_method(self):
        rows = fixture(); rows[4]["method"] = "undeclared"
        self.assertIn("UNKNOWN_METHOD", codes(self.run_audit(rows)))

    def test_unequal_candidate_budget(self):
        rows = fixture(); rows[4]["candidate_budget"] = 64
        self.assertIn("BUDGET_MISMATCH", codes(self.run_audit(rows)))

    def test_conditions_must_be_method_independent(self):
        rows = fixture(); rows[4]["conditions"] = []
        self.assertIn("CONDITION_MISMATCH", codes(self.run_audit(rows)))

    def test_condition_order_irrelevant(self):
        rows = fixture(); rows[3]["conditions"] = ["contact", "rare"]; rows[4]["conditions"] = ["rare", "contact"]
        self.assertNotIn("CONDITION_MISMATCH", codes(self.run_audit(rows)))

    def test_rare_condition_absent(self):
        config = manifest(); config["required_conditions"]["unseen"] = 1
        self.assertIn("INSUFFICIENT_CONDITION_COVERAGE", codes(self.run_audit(config=config)))

    def test_coverage_counts_episodes_not_windows_or_seeds(self):
        rows = fixture(1)
        for i in range(1, 20):
            rows += [evaluation(0, "base", 10, evaluation_seed=i), evaluation(0, "candidate", 9, evaluation_seed=i)]
        result = self.run_audit(rows)
        self.assertEqual(result["required_condition_coverage"]["contact"]["observed_episodes"], 1)
        self.assertIn("INSUFFICIENT_CONDITION_COVERAGE", codes(result))

    def test_episode_weighting_not_window_weighting(self):
        rows = fixture(3)
        # Episode deltas are -6, -5, -4. Add many windows in episode 2.
        for i in range(1, 21):
            rows += [evaluation(2, "base", 10, start_step=i, goal_step=i+10), evaluation(2, "candidate", 6, start_step=i, goal_step=i+10)]
        result = self.run_audit(rows)
        self.assertEqual(result["effect"]["value"], -5)
        self.assertEqual(result["effect"]["episode_count"], 3)
        self.assertEqual(result["effect"]["paired_task_seed_count"], 23)

    def test_median_is_of_paired_differences(self):
        rows = [episode(100, "train"), episode(0)]
        for i, (base, candidate) in enumerate([(0, 1), (10, 100), (100, 101)]):
            rows += [evaluation(0, "base", base, evaluation_seed=i), evaluation(0, "candidate", candidate, evaluation_seed=i)]
        config = manifest(); config["required_conditions"] = {}
        self.assertEqual(self.run_audit(rows, config)["effect"]["value"], 1)

    def test_one_episode_no_interval(self):
        config = manifest(); config["required_conditions"] = {}
        result = self.run_audit(fixture(1), config)
        self.assertEqual(result["effect"]["value"], -6)
        self.assertIsNone(result["effect"]["interval"])
        self.assertIn("SMALL_EPISODE_SAMPLE", codes(result))

    def test_no_paired_episodes(self):
        result = self.run_audit([episode(100, "train")])
        self.assertIn("NO_EVALUATIONS", codes(result)); self.assertIsNone(result["effect"])

    def test_declared_test_episode_cannot_be_silently_dropped(self):
        rows = fixture() + [episode(500)]
        self.assertIn("UNEVALUATED_TEST_EPISODE", codes(self.run_audit(rows)))

    def test_no_inventory(self):
        rows = [r for r in fixture() if r["type"] == "evaluation"]
        self.assertIn("NO_EPISODES", codes(self.run_audit(rows)))

    def test_degenerate_bootstrap_warning(self):
        rows = fixture()
        for row in rows:
            if row["type"] == "evaluation": row["metric"] = 5
        result = self.run_audit(rows)
        self.assertIn("DEGENERATE_BOOTSTRAP", codes(result))
        self.assertEqual(result["effect"]["interval"]["low"], 0)

    def test_finite_inputs_overflowing_difference(self):
        rows = fixture(); rows[3]["metric"] = -1.7e308; rows[4]["metric"] = 1.7e308
        self.assertIn("NONFINITE_DIFFERENCE", codes(self.run_audit(rows)))

    def test_median_extreme_finite_values(self):
        self.assertEqual(audit.median([1.7e308, 1.7e308]), 1.7e308)
        self.assertEqual(audit.median([-1.7e308, 1.7e308]), 0)
        self.assertEqual(audit.median([1, 3, 100]), 3)

    def test_subnormal_midpoint_and_percentile(self):
        tiny = float.fromhex("0x0.0000000000001p-1022")
        self.assertEqual(audit.median([tiny, tiny]), tiny)
        self.assertEqual(audit.median([tiny, tiny * 2]), tiny * 2)
        self.assertEqual(audit.percentile([tiny, tiny], .5), tiny)
        self.assertEqual(audit.percentile([tiny, tiny * 2], .5), tiny * 2)
        self.assertEqual(audit.percentile([-1.7e308, 1.7e308], .5), 0)

    def test_integer_metrics_preserve_small_differences(self):
        for baseline, candidate, expected in ((2**53, 2**53+1, 1), (float(2**53), 2**53+1, 1), (2**53+1, float(2**53), -1)):
            rows = fixture()
            for row in rows:
                if row["type"] == "evaluation":
                    row["metric"] = baseline if row["method"] == "base" else candidate
            self.assertEqual(self.run_audit(rows)["effect"]["value"], expected)

    def test_bootstrap_matches_independent_reference(self):
        values = [-4, -1, 0, 8]
        rng = random.Random(42)
        # Separate reference arithmetic, including independent quantile calculation.
        samples = []
        for _ in range(200):
            selected = sorted(values[rng.randrange(4)] for _ in range(4))
            samples.append((selected[1] + selected[2]) / 2)
        samples.sort()
        def quantile(p):
            x = 199*p; i = int(x)
            return samples[i] + (samples[min(i+1, 199)] - samples[i]) * (x-i)
        actual = audit.bootstrap_episode_median(values, 200, 42, .95)
        self.assertAlmostEqual(actual["low"], quantile(.025)); self.assertAlmostEqual(actual["high"], quantile(.975))

    def test_bootstrap_full_enumeration_sanity(self):
        # Exact bootstrap law for two episodes [0,2]: 0,1,1,2.
        actual = audit.bootstrap_episode_median([0, 2], 10000, 42, .95)
        self.assertEqual((actual["low"], actual["high"]), (0, 2))

    def test_schema_rejects_nonfinite_and_bool_metrics(self):
        for value in (float("nan"), float("inf"), -float("inf"), True, None, "1", 10**400):
            with self.subTest(value=type(value).__name__):
                rows = fixture(); rows[4]["metric"] = value
                self.assertIn("SCHEMA", codes(self.run_audit(rows)))

    def test_schema_mutations(self):
        mutations = [("episode_idx", True), ("episode_idx", -1), ("episode_idx", 2**53), ("start_step", -1), ("goal_step", 0), ("evaluation_seed", 1.0), ("candidate_budget", False), ("candidate_budget", 0), ("conditions", ["x", "x"]), ("conditions", [{}]), ("conditions", "x"), ("source_id", " x"), ("method", ""), ("extra", 1)]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                rows = fixture(); rows[4][key] = value
                self.assertIn("SCHEMA", codes(self.run_audit(rows)))

    def test_inventory_schema_mutations(self):
        for changes in ({"seed": None}, {"seed_namespace": None}, {"seed": True}, {"split": []}, {"content_sha256": "A"*64}, {"content_sha256": 1}, {"seed_namespace": ""}):
            rows = fixture(); rows[2].update(changes)
            self.assertIn("SCHEMA", codes(self.run_audit(rows)))

    def test_missing_fields_and_wrong_record_types(self):
        for row in ({"type": "episode"}, [], None, {"type": "other"}):
            self.assertIn("SCHEMA", codes(self.run_audit(fixture() + [row])))

    def test_manifest_validation(self):
        invalid = [None, {}, dict(manifest(), schema_version=True), dict(manifest(), schema_version=2), dict(manifest(), candidate="base"), dict(manifest(), required_conditions={"rare": 0}), dict(manifest(), required_conditions={"rare": True}), dict(manifest(), metric={"name": "x", "direction": "sideways"}), dict(manifest(), extra=1)]
        for config in invalid:
            with self.assertRaises(audit.InputError): audit.audit(config, fixture(), resamples=100)

    def test_bootstrap_argument_validation(self):
        for kwargs in ({"resamples": 99}, {"resamples": 10001}, {"resamples": True}, {"seed": -1}, {"seed": True}, {"confidence": 0}, {"confidence": 1}, {"confidence": True}, {"confidence": float("nan")}):
            with self.assertRaises(audit.InputError): audit.audit(manifest(), fixture(), **kwargs)

    def test_empty_input_rejected(self):
        for rows in ([], None, {}):
            with self.assertRaises(audit.InputError): audit.audit(manifest(), rows)

    def test_bootstrap_work_bound(self):
        old = audit.MAX_BOOTSTRAP_WORK
        try:
            audit.MAX_BOOTSTRAP_WORK = 10
            with self.assertRaises(audit.InputError): self.run_audit()
        finally:
            audit.MAX_BOOTSTRAP_WORK = old

    def test_strict_json(self):
        for value in ('{"x":1,"x":2}', '{"x":{"a":1,"a":2}}', '{"x":NaN}', '{"x":Infinity}', '{} {}', '\ufeff{}'):
            with self.assertRaises(audit.InputError): audit.strict_json(value)

    def test_read_input_errors_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"input.jsonl"
            for payload in (b"", b"{}\n\n", b"\xff", b"{bad}", b"{" + b" " * audit.MAX_LINE_BYTES + b"}"):
                path.write_bytes(payload)
                with self.assertRaises(audit.InputError): audit.read_input(path, jsonl=True)
            path.write_text('{"x":1}\n', encoding="utf-8")
            rows, digest = audit.read_input(path, jsonl=True)
            self.assertEqual(rows, [{"x": 1}]); self.assertEqual(len(digest), 64)

    def test_jsonl_only_lf_is_record_separator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unicode.jsonl"
            path.write_text('{"label":"one\u2028two"}\n', encoding="utf-8")
            rows, _ = audit.read_input(path, jsonl=True)
            self.assertEqual(rows, [{"label": "one\u2028two"}])

    def test_json_serializable_report(self):
        json.dumps(self.run_audit(), allow_nan=False)


class CLITests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(Path(__file__).with_name("audit.py")), *map(str, args)], capture_output=True, text=True)

    def test_documented_examples(self):
        here = Path(__file__).parent / "examples"
        for example, expected in (("valid.jsonl", 0), ("invalid.jsonl", 1)):
            process = self.run_cli("--manifest", here / "manifest.json", "--records", here / example, "--resamples", 100)
            self.assertEqual(process.returncode, expected, process.stderr + process.stdout)
            self.assertEqual(process.stderr, "")
            report = json.loads(process.stdout)
            self.assertIn("inputs_sha256", report)
            self.assertEqual(report["effect"] is None, expected == 1)

    def test_checked_in_reports_are_fresh(self):
        here = Path(__file__).parent / "examples"
        for example in ("valid", "invalid"):
            process = self.run_cli("--manifest", here / "manifest.json", "--records", here / (example + ".jsonl"))
            actual = json.loads(process.stdout)
            expected = json.loads((here / (example + "_report.json")).read_text(encoding="utf-8"))
            actual.pop("runtime"); expected.pop("runtime")
            self.assertEqual(actual, expected)
        valid = json.loads((here / "valid_report.json").read_text(encoding="utf-8"))
        self.assertEqual(valid["effect"]["value"], -0.5)
        self.assertEqual(valid["effect"]["episode_count"], 12)

    def test_cli_input_error_is_json_and_exit_two(self):
        process = self.run_cli("--manifest", "does-not-exist", "--records", "does-not-exist")
        self.assertEqual(process.returncode, 2); self.assertEqual(json.loads(process.stdout)["status"], "input_error")

    def test_escaped_surrogate_is_clean_schema_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "manifest.json"; records = Path(tmp) / "records.jsonl"
            config.write_text(json.dumps(manifest()), encoding="utf-8")
            rows = fixture()
            for row in rows:
                row["source_id"] = "bad" + chr(0xD800)
            records.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            process = self.run_cli("--manifest", config, "--records", records)
            self.assertEqual(process.returncode, 1, process.stderr)
            self.assertEqual(process.stderr, "")
            result = json.loads(process.stdout)
            self.assertIn("SCHEMA", codes(result)); self.assertIsNone(result["effect"])

    def test_fail_on_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "manifest.json"; records = Path(tmp) / "records.jsonl"
            config.write_text(json.dumps(manifest()), encoding="utf-8")
            records.write_text("\n".join(json.dumps(r) for r in fixture(2)) + "\n", encoding="utf-8")
            normal = self.run_cli("--manifest", config, "--records", records, "--resamples", 100)
            strict = self.run_cli("--manifest", config, "--records", records, "--resamples", 100, "--fail-on-warning")
            self.assertEqual(normal.returncode, 0); self.assertEqual(strict.returncode, 1)


if __name__ == "__main__":
    unittest.main()
