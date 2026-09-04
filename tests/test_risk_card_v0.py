from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "results" / "KW_RISK_CARD_v0.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RiskCardV0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.card = json.loads(CARD_PATH.read_text(encoding="utf-8"))

    def test_frozen_source_hashes_match(self) -> None:
        for source in self.card["source_artifacts"].values():
            path = ROOT / source["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(source["sha256"], sha256(path))

    def test_task_fraction_is_consistent(self) -> None:
        outcome = self.card["task_outcome"]
        self.assertAlmostEqual(
            outcome["success_rate"],
            outcome["success_count"] / outcome["episode_count"],
            places=12,
        )

    def test_relative_risk_is_consistent(self) -> None:
        risk = self.card["physical_interaction_slice"]
        expected = risk["high_interaction_spike_rate"] / risk["lower_interaction_spike_rate"]
        self.assertAlmostEqual(risk["relative_risk"], expected, places=4)

    def test_claim_boundary_is_explicit(self) -> None:
        forbidden = " ".join(self.card["claims"]["forbidden"]).lower()
        self.assertIn("third-party", forbidden)
        self.assertIn("state of the art", forbidden)
        self.assertIn("baize", forbidden)


if __name__ == "__main__":
    unittest.main()
