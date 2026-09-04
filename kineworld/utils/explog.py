# -*- coding: utf-8 -*-
"""
KineWorld Phase-0 experiment logging (KineWorld directive §16).

Machine-readable, per-experiment records:
  results/<experiment_id>/metrics.jsonl   — one JSON object per record (append-only)
  results/experiments.csv                 — one row per experiment (flat summary, append-only)

Required fields per directive §16:
  experiment_id, timestamp, git_commit, seed, VRAM, latency, rollout errors,
  success rate, failure cases.

NO FABRICATION RULE: every numeric field must come from an actual measured
value. If a measurement was not taken, the field must be recorded as null —
never as a guessed number.
"""
import csv
import json
import os
import platform
import socket
import subprocess
import uuid
from datetime import datetime, timezone

import torch

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(WS, "results")
JEPA_REPO = os.path.join(WS, "external", "jepa-wms")

CSV_FIELDS = [
    "experiment_id", "timestamp_utc", "hostname", "gpu_name", "gpu_vram_total_mb",
    "torch_version", "cuda_version", "python_version", "os",
    "jepa_wms_git_commit", "config_path", "checkpoint", "checkpoint_bytes",
    "seed", "eval_episodes", "quick_debug",
    "success_rate", "ep_end_dist", "reward", "total_emb_l2", "total_lpips",
    "total_time_s", "planning_latency_ms_mean", "planning_latency_ms_p95",
    "inference_latency_ms_mean", "vram_peak_mb",
    "rollout_errors", "failure_cases_path", "metrics_jsonl", "notes",
]


def jepa_wms_commit():
    try:
        out = subprocess.run(
            ["git", "-C", JEPA_REPO, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def gpu_env():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {
            "gpu_name": props.name,
            "gpu_vram_total_mb": round(props.total_memory / 1024 / 1024),
            "cuda_version": torch.version.cuda,
        }
    return {"gpu_name": "cpu", "gpu_vram_total_mb": None, "cuda_version": None}


class ExperimentLogger:
    def __init__(self, notes="", config_path=None, checkpoint=None, seed=None,
                 eval_episodes=None, quick_debug=False):
        now = datetime.now(timezone.utc)
        self.experiment_id = f"{now.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.dir = os.path.join(RESULTS, self.experiment_id)
        os.makedirs(self.dir, exist_ok=True)
        self.jsonl = os.path.join(self.dir, "metrics.jsonl")
        self.env = {"os": f"{platform.system()} {platform.release()}"}
        self.env.update(gpu_env())
        self.base = {
            "experiment_id": self.experiment_id,
            "timestamp_utc": now.isoformat(timespec="seconds"),
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "jepa_wms_git_commit": jepa_wms_commit(),
            "config_path": config_path,
            "checkpoint": checkpoint,
            "checkpoint_bytes": os.path.getsize(checkpoint) if checkpoint and os.path.exists(checkpoint) else None,
            "seed": seed,
            "eval_episodes": eval_episodes,
            "quick_debug": quick_debug,
            "notes": notes,
            **self.env,
        }
        self._write_record(self.base)

    def _write_record(self, rec):
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def log_event(self, event: dict):
        rec = {"experiment_id": self.experiment_id,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        rec.update(event)
        self._write_record(rec)

    def finalize(self, summary: dict):
        """summary keys: success_rate, ep_end_dist, reward, total_emb_l2, total_lpips,
        total_time_s, planning_latency_ms_mean, planning_latency_ms_p95,
        inference_latency_ms_mean, vram_peak_mb, rollout_errors, failure_cases_path.
        All values must be measured; use None for not-measured."""
        final = dict(self.base)
        for k in CSV_FIELDS:
            if k not in final:
                final[k] = summary.get(k)  # None if absent — honesty by default
        self._write_record({"experiment_id": self.experiment_id,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "event": "finalize", "summary": {k: summary.get(k) for k in CSV_FIELDS if k in summary}})
        csv_path = os.path.join(RESULTS, "experiments.csv")
        exists = os.path.exists(csv_path)
        os.makedirs(RESULTS, exist_ok=True)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not exists:
                w.writeheader()
            w.writerow({k: ("" if final.get(k) is None else final.get(k)) for k in CSV_FIELDS})
        return final
