import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "verification"
    / "audit_tools"
    / "success_comparison.py"
)
SPEC = importlib.util.spec_from_file_location("success_comparison", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SuccessComparisonTests(unittest.TestCase):
    def test_wilson_interval_contains_observed_rate(self):
        interval = MODULE.wilson_interval(44, 96)
        self.assertLess(interval.low, 44 / 96)
        self.assertGreater(interval.high, 44 / 96)

    def test_identical_tables_have_fisher_p_one(self):
        self.assertAlmostEqual(MODULE.fisher_exact_two_sided(44, 52, 44, 52), 1.0)

    def test_cross_implementation_never_allows_superiority_claim(self):
        result = MODULE.compare_unpaired(96, 96, 44, 96)
        self.assertTrue(result["statistical_gate_passed"])
        self.assertFalse(result["claim_allowed"])

    def test_invalid_count_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.compare_unpaired(97, 96, 44, 96)


if __name__ == "__main__":
    unittest.main()
