# -*- coding: utf-8 -*-
"""
Rebuild results/experiments.csv from the per-run metrics.json files.

Background: the CSV was polluted because two writers (a 16-column shell runner
and the 29-column python runner scripts/run_pt_bench.py) appended rows with
different column layouts, corrupting the header alignment. The authoritative
source of truth is the per-run metrics.json (written by run_pt_bench.py), which
contains every field needed. This script drops all malformed rows and rebuilds
the CSV in the canonical 29-column order used by run_pt_bench.py.

Usage:
  ./.venv310/Scripts/python.exe scripts/rebuild_experiments_csv.py
"""
import csv
import json
import os
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
RUNS = WS / "results" / "runs"
OUT = WS / "results" / "experiments.csv"

COLUMNS = [
    "experiment_id",
    "timestamp_utc",
    "status",
    "exit_code",
    "jepa_wms_git_commit",
    "config_path",
    "checkpoint",
    "checkpoint_bytes",
    "seed",
    "eval_episodes_requested",
    "quick_debug",
    "success_rate",
    "ep_end_dist",
    "reward",
    "total_emb_l2",
    "total_lpips",
    "official_total_time_s",
    "wall_time_s",
    "planning_latency_ms_mean",
    "planning_latency_ms_p95",
    "planning_calls",
    "inference_latency_ms_mean",
    "vram_baseline_mib",
    "vram_peak_mib",
    "vram_peak_delta_mib",
    "gpu_util_peak_pct",
    "failure_cases_path",
    "metrics_json",
    "stdout_log",
]


def load_summaries() -> list[dict]:
    rows = []
    for mf in sorted(RUNS.glob("*/metrics.json")):
        with open(mf, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        rows.append({
            "experiment_id": m.get("experiment_id"),
            "timestamp_utc": m.get("timestamp_utc"),
            "status": m.get("status", m.get("event")),
            "exit_code": m.get("exit_code"),
            "jepa_wms_git_commit": m.get("jepa_wms_git_commit"),
            "config_path": m.get("config_path"),
            "checkpoint": m.get("checkpoint"),
            "checkpoint_bytes": m.get("checkpoint_bytes"),
            "seed": m.get("seed"),
            "eval_episodes_requested": m.get("eval_episodes_requested"),
            "quick_debug": m.get("quick_debug"),
            "success_rate": m.get("success_rate"),
            "ep_end_dist": m.get("ep_end_dist"),
            "reward": m.get("reward"),
            "total_emb_l2": m.get("total_emb_l2"),
            "total_lpips": m.get("total_lpips"),
            "official_total_time_s": m.get("official_total_time_s"),
            "wall_time_s": m.get("wall_time_s"),
            "planning_latency_ms_mean": m.get("planning_latency_ms_mean"),
            "planning_latency_ms_p95": m.get("planning_latency_ms_p95"),
            "planning_calls": m.get("planning_calls"),
            "inference_latency_ms_mean": m.get("inference_latency_ms_mean"),
            "vram_baseline_mib": m.get("vram_baseline_mib"),
            "vram_peak_mib": m.get("vram_peak_mib"),
            "vram_peak_delta_mib": m.get("vram_peak_delta_mib"),
            "gpu_util_peak_pct": m.get("gpu_util_peak_pct"),
            "failure_cases_path": m.get("failure_cases_path"),
            "metrics_json": m.get("metrics_json"),
            "stdout_log": m.get("stdout_log"),
        })
    return rows


def main() -> int:
    rows = load_summaries()
    # sort by timestamp asc
    rows.sort(key=lambda r: (r["timestamp_utc"] or ""))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in COLUMNS})
    print(f"Rebuilt {OUT} with {len(rows)} valid rows "
          f"({len(COLUMNS)} columns) from per-run metrics.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
