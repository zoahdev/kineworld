"""Offline integrity and consistency verifier for KW-RISK-CARD-0001.

This verifies frozen artifacts; it does not constitute an independent model
rerun or third-party attestation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASE_MANIFEST = ROOT / "verification" / "manifests" / "KW-RISK-CARD-v0_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    release = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    failures: list[str] = []
    for name, item in release["artifacts"].items():
        path = ROOT / item["path"]
        if not path.is_file():
            failures.append(f"missing {name}: {path}")
            continue
        actual = sha256(path)
        if actual != item["sha256"]:
            failures.append(f"hash mismatch {name}: {actual} != {item['sha256']}")

    card_path = ROOT / release["artifacts"]["machine_card"]["path"]
    if card_path.is_file():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        for name, item in card["source_artifacts"].items():
            path = ROOT / item["path"]
            if not path.is_file():
                failures.append(f"missing frozen source {name}: {path}")
            elif sha256(path) != item["sha256"]:
                failures.append(f"frozen source hash mismatch: {name}")
        outcome = card["task_outcome"]
        expected_rate = outcome["success_count"] / outcome["episode_count"]
        if abs(expected_rate - outcome["success_rate"]) > 1e-12:
            failures.append("success count/rate inconsistency")

    if failures:
        print("KW-RISK-CARD-0001 integrity: FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("KW-RISK-CARD-0001 integrity: PASS")
    print("Scope: artifact integrity and internal consistency only")
    print("Not established: independent rerun, third-party validation, or model superiority")


if __name__ == "__main__":
    main()
