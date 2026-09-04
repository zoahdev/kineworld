"""Range-limited numeric-only reader for the public LeWM Push-T parquet shard."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests


PUBLIC_API_URL = "https://huggingface.co/api/datasets/galilai-group/lewm-pusht/parquet/default/train/0.parquet"
API_URL = os.environ.get("KW_PARQUET_URL", PUBLIC_API_URL)
MAX_TRANSFER = 64 * 1024 * 1024
OUTPUT = Path("results/kw_lewm_0001l_numeric_shard0.json")
COLUMNS = ["episode_idx", "step_idx", "action", "proprio", "state"]


class RangeReader(io.RawIOBase):
    def __init__(self, url: str, max_transfer: int):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "KineWorld-evidence/0.1"
        head = self.session.head(url, allow_redirects=True, timeout=45)
        head.raise_for_status()
        self.url = head.url
        self.size = int(head.headers["Content-Length"])
        self.etag = head.headers.get("ETag")
        self.max_transfer = max_transfer
        self.transferred = 0
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.position + offset
        elif whence == io.SEEK_END:
            target = self.size + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        if not 0 <= target <= self.size:
            raise ValueError("seek outside remote object")
        self.position = target
        return target

    def read(self, size=-1):
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size == 0:
            return b""
        if self.transferred + size > self.max_transfer:
            raise RuntimeError(
                f"range transfer guard: requested cumulative bytes would exceed {self.max_transfer}"
            )
        start = self.position
        end = start + size - 1
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=(20, 120),
        )
        if response.status_code != 206:
            raise RuntimeError(
                f"server did not honor Range (status={response.status_code}); refusing possible full download"
            )
        data = response.content
        if len(data) != size:
            raise RuntimeError(f"short range read: expected {size}, got {len(data)}")
        self.position += len(data)
        self.transferred += len(data)
        return data


def list_array(column) -> np.ndarray:
    values = column.combine_chunks()
    return np.asarray(values.values).reshape(len(values), values.type.list_size)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        type=int,
        default=8,
        help="number of start/goal pairs to freeze (default 8)",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT),
        help="output JSON path",
    )
    cli = parser.parse_args()
    n_pairs = cli.pairs
    out_path = Path(cli.output)

    reader = RangeReader(API_URL, MAX_TRANSFER)
    with pa.PythonFile(reader, mode="r") as source:
        parquet = pq.ParquetFile(source)
        table = parquet.read(columns=COLUMNS)

    episode = np.asarray(table["episode_idx"]).astype(np.int64)
    step = np.asarray(table["step_idx"]).astype(np.int64)
    action = list_array(table["action"]).astype(np.float64)
    proprio = list_array(table["proprio"]).astype(np.float64)
    state = list_array(table["state"]).astype(np.float64)

    valid = np.flatnonzero(
        (np.arange(len(step)) + 25 < len(step))
        & (np.roll(episode, -25) == episode)
        & (np.roll(step, -25) == step + 25)
    )
    valid = valid[valid + 25 < len(step)]
    rng = random.Random(42)
    selected = sorted(rng.sample(valid.tolist(), min(n_pairs, len(valid))))

    def stats(array):
        return {
            "n": int(len(array)),
            "mean": array.mean(axis=0).tolist(),
            "scale": array.std(axis=0, ddof=0).tolist(),
        }

    pairs = []
    for idx in selected:
        goal_idx = idx + 25
        pairs.append(
            {
                "start_row_in_shard": idx,
                "goal_row_in_shard": goal_idx,
                "episode_idx": int(episode[idx]),
                "start_step": int(step[idx]),
                "goal_step": int(step[goal_idx]),
                "start_action": action[idx].tolist(),
                "goal_action": action[goal_idx].tolist(),
                "start_proprio": proprio[idx].tolist(),
                "goal_proprio": proprio[goal_idx].tolist(),
                "start_state": state[idx].tolist(),
                "goal_state": state[goal_idx].tolist(),
            }
        )

    result = {
        "protocol": "KW-LEWM-0001L-TRANSPORT-AMENDMENT",
        "source_api": PUBLIC_API_URL,
        "resolved_etag": reader.etag,
        "remote_object_bytes": reader.size,
        "transferred_bytes": reader.transferred,
        "max_transfer_bytes": MAX_TRANSFER,
        "columns_read": COLUMNS,
        "pixels_requested": False,
        "rows": int(table.num_rows),
        "row_groups": parquet.metadata.num_row_groups,
        "stats": {
            "action": stats(action),
            "proprio": stats(proprio),
            "state": stats(state),
        },
        "valid_pair_count": int(len(valid)),
        "selection_seed": 42,
        "selected_pairs": pairs,
    }
    result["n_pairs_requested"] = n_pairs
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["content_sha256_without_this_field"] = hashlib.sha256(canonical).hexdigest()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(out_path),
        "remote_object_bytes": reader.size,
        "transferred_bytes": reader.transferred,
        "rows": table.num_rows,
        "valid_pairs": len(valid),
        "selected_pairs": len(pairs),
        "action_stats": result["stats"]["action"],
        "content_sha256": result["content_sha256_without_this_field"],
    }, indent=2))


if __name__ == "__main__":
    main()
