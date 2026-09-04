#!/usr/bin/env python3
"""Run one official JEPA-WM Push-T config and preserve measured evidence.

This wrapper does not change model, planner, environment, checkpoint, or metric
logic.  It only sets the documented local paths, captures stdout, samples the
GPU once per second, and converts official ``eval.csv`` plus timing log lines
into machine-readable evidence under ``results/runs``.
"""

from __future__ import annotations

import argparse
import csv
import json
import msvcrt
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


WS = Path(__file__).resolve().parents[1]
REPO = WS / "external" / "jepa-wms"
PYTHON = WS / ".venv310" / "Scripts" / "python.exe"
RUNS = WS / "results" / "runs"
EXPERIMENTS_CSV = WS / "results" / "experiments.csv"
EVAL_LOCK = WS / "results" / ".official_eval.lock"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip() if result.returncode == 0 else None


def gpu_snapshot() -> dict[str, float | str | None]:
    query = "name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {
            "gpu_name": None,
            "vram_total_mib": None,
            "vram_used_mib": None,
            "gpu_util_pct": None,
            "temperature_c": None,
            "power_w": None,
        }
    values = [part.strip() for part in result.stdout.splitlines()[0].split(",")]

    def number(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

    return {
        "gpu_name": values[0],
        "vram_total_mib": number(values[1]),
        "vram_used_mib": number(values[2]),
        "gpu_util_pct": number(values[3]),
        "temperature_c": number(values[4]),
        "power_w": number(values[5]),
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[index]


def official_result_dir(config: dict) -> Path:
    folder = Path(config["folder"])
    return folder / "simu_env_planning" / Path(config["tag"])


def read_last_eval_row(path: Path) -> dict[str, float | str | None]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    parsed: dict[str, float | str | None] = {}
    for key, value in rows[-1].items():
        try:
            parsed[key] = float(value) if value not in (None, "") else None
        except ValueError:
            parsed[key] = value
    return parsed


def append_experiment(summary: dict) -> None:
    columns = [
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
    exists = EXPERIMENTS_CSV.exists()
    with EXPERIMENTS_CSV.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: summary.get(key) for key in columns})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config path, absolute or relative to jepa-wms")
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO / config_path
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    EVAL_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = EVAL_LOCK.open("a+b")
    if EVAL_LOCK.stat().st_size == 0:
        lock_handle.write(b"0")
        lock_handle.flush()
    lock_handle.seek(0)
    try:
        msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print(f"another official evaluation owns {EVAL_LOCK}; refusing a concurrent launch", file=sys.stderr)
        lock_handle.close()
        return 73

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = f"{config_path.stem}_{timestamp}"
    run_dir = RUNS / experiment_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "stdout.log"
    gpu_path = run_dir / "gpu_samples.csv"
    metrics_path = run_dir / "metrics.json"
    jsonl_path = run_dir / "metrics.jsonl"

    checkpoint = Path(config["model_kwargs"]["checkpoint"])
    baseline = gpu_snapshot()
    start_record = {
        "event": "start",
        "experiment_id": experiment_id,
        "timestamp_utc": utc_now(),
        "config_path": str(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.exists() else None,
        "jepa_wms_git_commit": git_commit(),
        "seed": config.get("meta", {}).get("seed"),
        "eval_episodes_requested": config.get("meta", {}).get("eval_episodes"),
        "quick_debug": bool(config.get("meta", {}).get("quick_debug", False)),
        "host": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "gpu": baseline,
    }
    jsonl_path.write_text(json.dumps(start_record, ensure_ascii=False) + "\n", encoding="utf-8")

    stop_sampler = threading.Event()
    samples: list[dict] = []

    def sample_gpu() -> None:
        with gpu_path.open("w", encoding="utf-8", newline="") as handle:
            fields = [
                "timestamp_utc",
                "gpu_name",
                "vram_total_mib",
                "vram_used_mib",
                "gpu_util_pct",
                "temperature_c",
                "power_w",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            while not stop_sampler.is_set():
                sample = {"timestamp_utc": utc_now(), **gpu_snapshot()}
                samples.append(sample)
                writer.writerow(sample)
                handle.flush()
                stop_sampler.wait(args.sample_seconds)

    env = os.environ.copy()
    env.update(
        {
            "JEPAWM_DSET": (WS / "data").as_posix(),
            "JEPAWM_LOGS": (WS / "results" / "jepa_logs").as_posix(),
            "JEPAWM_HOME": WS.as_posix(),
            "JEPAWM_CKPT": (WS / "checkpoints").as_posix(),
            "WANDB_MODE": "disabled",
            "SDL_VIDEODRIVER": "dummy",
            "PYTHONUNBUFFERED": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            # This machine routes GitHub through the local Clash listener.
            # torch.hub still validates the DINOv2 default branch even when
            # both source and weights are already cached.
            "HTTP_PROXY": os.environ.get("HTTP_PROXY") or "http://127.0.0.1:7897",
            "HTTPS_PROXY": os.environ.get("HTTPS_PROXY") or "http://127.0.0.1:7897",
            "http_proxy": os.environ.get("http_proxy") or "http://127.0.0.1:7897",
            "https_proxy": os.environ.get("https_proxy") or "http://127.0.0.1:7897",
            "NO_PROXY": "localhost,127.0.0.1",
        }
    )
    command = [str(PYTHON), "-B", "-u", "-m", "evals.main", "--fname", str(config_path), "--debug"]
    try:
        sampler_thread = threading.Thread(target=sample_gpu, daemon=True)
        sampler_thread.start()
        wall_start = time.perf_counter()
        with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=REPO,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_handle.write(line)
                log_handle.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            exit_code = process.wait()
        wall_time = time.perf_counter() - wall_start
        stop_sampler.set()
        sampler_thread.join(timeout=max(5.0, args.sample_seconds * 2))
    finally:
        lock_handle.seek(0)
        msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
        lock_handle.close()

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    plan_seconds = [float(value) for value in re.findall(r"Action optim at step \d+ took ([0-9.]+) seconds", log_text)]
    used_values = [float(s["vram_used_mib"]) for s in samples if s.get("vram_used_mib") is not None]
    util_values = [float(s["gpu_util_pct"]) for s in samples if s.get("gpu_util_pct") is not None]
    official_dir = official_result_dir(config)
    eval_row = read_last_eval_row(official_dir / "eval.csv")
    failure_cases = sorted(str(p) for p in official_dir.rglob("*fail*.mp4")) if official_dir.exists() else []
    vram_baseline = baseline.get("vram_used_mib")
    vram_peak = max(used_values) if used_values else None

    summary = {
        **start_record,
        "event": "complete" if exit_code == 0 and eval_row else "failed",
        "status": "complete" if exit_code == 0 and eval_row else "failed",
        "exit_code": exit_code,
        "wall_time_s": round(wall_time, 6),
        "official_result_dir": str(official_dir),
        "official_eval_csv": str(official_dir / "eval.csv"),
        "official_metrics": eval_row,
        "success_rate": eval_row.get("episode_success"),
        "ep_end_dist": eval_row.get("ep_end_dist"),
        "reward": eval_row.get("episode_reward"),
        "total_emb_l2": eval_row.get("ep_total_emb_l2"),
        "total_lpips": eval_row.get("ep_total_lpips"),
        "official_total_time_s": eval_row.get("total_time"),
        "planning_calls": len(plan_seconds),
        "planning_latency_ms_mean": round(statistics.fmean(plan_seconds) * 1000, 3) if plan_seconds else None,
        "planning_latency_ms_p95": round(percentile(plan_seconds, 0.95) * 1000, 3) if plan_seconds else None,
        "planning_latency_ms_samples": [round(value * 1000, 3) for value in plan_seconds],
        "inference_latency_ms_mean": None,
        "vram_baseline_mib": vram_baseline,
        "vram_peak_mib": vram_peak,
        "vram_peak_delta_mib": round(vram_peak - float(vram_baseline), 3)
        if vram_peak is not None and vram_baseline is not None
        else None,
        "gpu_util_peak_pct": max(util_values) if util_values else None,
        "failure_cases": failure_cases,
        "failure_cases_path": json.dumps(failure_cases, ensure_ascii=False),
        "rollout_errors": None,
        "metrics_json": str(metrics_path),
        "metrics_jsonl": str(jsonl_path),
        "stdout_log": str(log_path),
        "gpu_samples_csv": str(gpu_path),
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    append_experiment(summary)
    print(json.dumps({key: summary[key] for key in [
        "status", "exit_code", "success_rate", "ep_end_dist", "planning_calls",
        "planning_latency_ms_mean", "vram_peak_mib", "vram_peak_delta_mib", "metrics_json"
    ]}, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
