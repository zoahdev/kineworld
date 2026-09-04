"""Prepare a tiny deterministic numeric Push-T evaluation snapshot via HF rows API."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path


DATASET = "galilai-group/lewm-pusht"
TOTAL_ROWS = 815_360
PAGE_SIZE = 100
GOAL_OFFSET = 25
SEED = 42


def fetch_page(offset: int, retries: int = 6) -> dict:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": PAGE_SIZE,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    last_error = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "KineWorld-evidence/0.1"})
            with urllib.request.urlopen(req, timeout=45) as response:
                return json.load(response)
        except Exception as exc:  # network errors are recorded by caller on final failure
            last_error = exc
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"failed page offset={offset}: {last_error}")


def vector_stats(rows: list[dict], key: str) -> dict:
    values = [row[key] for row in rows]
    width = len(values[0])
    mean = [sum(v[j] for v in values) / len(values) for j in range(width)]
    scale = [
        math.sqrt(sum((v[j] - mean[j]) ** 2 for v in values) / len(values))
        for j in range(width)
    ]
    return {"n": len(values), "mean": mean, "scale": scale}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/kw_lewm_0001l_snapshot.json"),
    )
    args = parser.parse_args()

    rng = random.Random(SEED)
    max_page = (TOTAL_ROWS - PAGE_SIZE) // PAGE_SIZE
    page_ids = sorted(rng.sample(range(max_page + 1), args.pages))
    offsets = [page_id * PAGE_SIZE for page_id in page_ids]

    numeric_rows: list[dict] = []
    pairs: list[dict] = []
    commits: set[str] = set()
    for offset in offsets:
        payload = fetch_page(offset)
        page_rows = []
        for entry in payload["rows"]:
            row = entry["row"]
            image_src = row.get("pixels", {}).get("src", "")
            marker = "/--/"
            if marker in image_src:
                parts = image_src.split(marker)
                if len(parts) > 2:
                    commits.add(parts[1])
            page_rows.append(
                {
                    "row_idx": int(entry["row_idx"]),
                    "episode_idx": int(row["episode_idx"]),
                    "step_idx": int(row["step_idx"]),
                    "action": [float(x) for x in row["action"]],
                    "proprio": [float(x) for x in row["proprio"]],
                    "state": [float(x) for x in row["state"]],
                }
            )
        numeric_rows.extend(page_rows)
        by_step = {(r["episode_idx"], r["step_idx"]): r for r in page_rows}
        for start in page_rows:
            goal = by_step.get((start["episode_idx"], start["step_idx"] + GOAL_OFFSET))
            if goal is not None:
                pairs.append({"start": start, "goal": goal})

    pairs.sort(key=lambda p: p["start"]["row_idx"])
    snapshot = {
        "protocol": "KW-LEWM-0001L",
        "dataset": DATASET,
        "dataset_commits_observed": sorted(commits),
        "api_total_rows_reported": TOTAL_ROWS,
        "api_partial": True,
        "seed": SEED,
        "page_size": PAGE_SIZE,
        "page_offsets": offsets,
        "goal_offset": GOAL_OFFSET,
        "numeric_rows_fetched": len(numeric_rows),
        "stats": {
            key: vector_stats(numeric_rows, key)
            for key in ("action", "proprio", "state")
        },
        "selected_pairs": pairs[:8],
        "candidate_pair_count": len(pairs),
    }
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    snapshot["content_sha256_without_this_field"] = hashlib.sha256(encoded).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "rows": len(numeric_rows),
        "candidate_pairs": len(pairs),
        "selected_pairs": len(snapshot["selected_pairs"]),
        "commits": sorted(commits),
        "content_sha256": snapshot["content_sha256_without_this_field"],
    }, indent=2))


if __name__ == "__main__":
    main()
