"""KW-LEWM-0002 — action-scaler ablation for LeWM Push-T closed-loop control.

Single-variable ablation. Everything is held fixed at the KW-LEWM-0001L Stage A
setup (same checkpoint, same random environment tasks, same reduced CEM, same
seeds). The ONLY manipulated variable is the `process['action']` transform:

  condition identity : no transform (the KW-LEWM-0001L failure condition)
  condition scaler   : StandardScaler fitted from real LeWM Push-T actions

Real action statistics come from `results/kw_lewm_0001l_numeric_shard0.json`,
estimated on 80,384 numeric rows of the public partial Parquet shard (no images
and no full dataset download).

Why this experiment exists
--------------------------
KW-LEWM-0001L Stage A loaded the official checkpoint but produced an invalid
control result: the agent moved to ~(120, 1202) while the block never moved.
`stable_worldmodel/solver/cem.py` samples candidates with `torch.randn`, i.e.
N(0, 1), and never clamps them. The upstream evaluator instead fits a
StandardScaler on the training actions and inverse-transforms planned actions
back into raw environment units. This ablation tests whether that missing
inverse transform is the mechanism behind the out-of-support control.

Kill criteria
-------------
K1 (scaler does not matter): if the scaler condition does not materially reduce
   out-of-support agent excursions versus identity, the missing transform is NOT
   the mechanism and this line stops (report as negative result).
K2 (still invalid): if the scaler condition still leaves the agent outside the
   raw play area, the lite protocol remains unusable for control claims.

Usage:
    python kw_lewm_0002_scaler_ablation.py [--episodes 8] [--output results/...]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import transformers
from torchvision.transforms import v2 as transforms

from kw_lewm_0001_load import load_lewm


ROOT = Path(__file__).resolve().parents[2]
STATS_PATH = ROOT / "results" / "kw_lewm_0001l_numeric_shard0.json"
DEFAULT_OUTPUT = ROOT / "results" / "kw_lewm_0002_scaler_ablation.json"

# Push-T raw play area; agent/block coordinates are expected inside this box.
PLAY_LOW, PLAY_HIGH = 0.0, 512.0


class NumpyStandardScaler:
    """Minimal reversible StandardScaler satisfying swm.protocols.Transformable.

    `stable_worldmodel` post-processes actions as numpy arrays (policy.py
    converts to numpy before calling inverse_transform), so plain numpy is enough.
    """

    def __init__(self, mean, scale):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)

    def transform(self, x):
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.scale

    def inverse_transform(self, x):
        return np.asarray(x, dtype=np.float64) * self.scale + self.mean


def image_transform():
    # Exact order used by upstream lucas-maes/le-wm/eval.py.
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
        ]
    )


def serializable(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value).tolist()


def json_default(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def run_condition(
    *,
    model,
    condition: str,
    action_scaler: NumpyStandardScaler | None,
    episodes: int,
    base_seed: int,
):
    """Run `episodes` one-episode evaluations under one action-transform condition."""
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)

    world = swm.World(
        "swm/PushT-v1",
        num_envs=1,
        image_shape=(224, 224),
        max_episode_steps=50,
    )
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=1,
        num_samples=100,
        n_steps=10,
        topk=10,
        device="cuda",
        seed=base_seed,
    )
    plan = swm.PlanConfig(
        horizon=5,
        receding_horizon=5,
        action_block=5,
        history_len=1,
        warm_start=True,
    )
    transform = image_transform()
    process = {} if action_scaler is None else {"action": action_scaler}
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan,
        process=process,
        transform={"pixels": transform, "goal": transform},
    )
    world.set_policy(policy)
    world.reset(seed=base_seed)

    trials = []
    for episode_index in range(episodes):
        seed = base_seed + episode_index
        world.reset(seed=seed)
        initial_state = np.asarray(serializable(world.infos.get("state")), dtype=float).reshape(-1)
        goal_state = np.asarray(serializable(world.infos.get("goal_state")), dtype=float).reshape(-1)

        started = time.perf_counter()
        error = None
        metrics = None
        try:
            metrics = world.evaluate(episodes=1, seed=None, reset_mode="wait")
            torch.cuda.synchronize()
        except Exception as exc:  # keep the ablation running, record the failure
            error = {"type": type(exc).__name__, "message": str(exc)}
        wall_seconds = time.perf_counter() - started

        final_state = np.asarray(serializable(world.infos.get("state")), dtype=float).reshape(-1)

        agent_start = initial_state[:2]
        agent_final = final_state[:2]
        block_start = initial_state[2:4]
        block_final = final_state[2:4]
        agent_out_of_play = bool(
            np.any(agent_final < PLAY_LOW) or np.any(agent_final > PLAY_HIGH)
        )
        # success extraction: prefer the per-episode list, fall back to the
        # aggregate rate. `episode_successes` may be an empty list when an
        # episode does not terminate cleanly, so never treat falsy as usable.
        success = None
        if isinstance(metrics, dict):
            successes = metrics.get("episode_successes")
            if successes is not None and len(np.asarray(successes).ravel()) > 0:
                success = float(np.asarray(successes).ravel()[0])
            elif metrics.get("success_rate") is not None:
                success = float(metrics["success_rate"])

        trials.append(
            {
                "episode_index": episode_index,
                "seed": seed,
                "initial_state": initial_state.tolist(),
                "goal_state": goal_state.tolist(),
                "final_state": final_state.tolist(),
                "agent_start": agent_start.tolist(),
                "agent_final": agent_final.tolist(),
                "agent_out_of_play_area": agent_out_of_play,
                "block_displacement": float(np.linalg.norm(block_final - block_start)),
                "block_start": block_start.tolist(),
                "block_final": block_final.tolist(),
                "success": success,
                "metrics_raw": metrics,
                "wall_seconds": wall_seconds,
                "error": error,
            }
        )

    world.close()
    return trials


def summarize(trials: list[dict]) -> dict:
    out_of_play = [bool(t["agent_out_of_play_area"]) for t in trials]
    displacements = [t["block_displacement"] for t in trials]
    successes = [t["success"] for t in trials if t["success"] is not None]
    finals = np.asarray([t["agent_final"] for t in trials], dtype=float)
    errors = [t for t in trials if t["error"] is not None]
    return {
        "n_episodes": len(trials),
        "n_agent_out_of_play": int(sum(out_of_play)),
        "out_of_play_rate": float(np.mean(out_of_play)) if out_of_play else None,
        "block_displacement_mean": float(np.mean(displacements)) if displacements else None,
        "block_displacement_median": float(np.median(displacements)) if displacements else None,
        "success_rate": float(np.mean(successes)) if successes else None,
        "n_success": int(sum(1 for s in successes if s > 0)) if successes else None,
        "agent_final_abs_max": float(np.abs(finals).max()),
        "agent_final_mean": finals.mean(axis=0).tolist(),
        "n_errors": len(errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    action_stats = stats["stats"]["action"]
    scaler = NumpyStandardScaler(action_stats["mean"], action_stats["scale"])

    model, _ = load_lewm(device="cuda")
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    results = {}
    for condition, scaler_or_none in (("identity", None), ("scaler", scaler)):
        trials = run_condition(
            model=model,
            condition=condition,
            action_scaler=scaler_or_none,
            episodes=args.episodes,
            base_seed=args.base_seed,
        )
        results[condition] = {"summary": summarize(trials), "trials": trials}

    identity_summary = results["identity"]["summary"]
    scaler_summary = results["scaler"]["summary"]

    payload = {
        "experiment_id": "KW-LEWM-0002-SCALER-ABLATION",
        "question": "Is the missing action inverse_transform the mechanism behind invalid LeWM control?",
        "manipulated_variable": "process['action'] transform (identity vs dataset-fitted StandardScaler)",
        "held_fixed": [
            "official LeWM Push-T checkpoint (strict load)",
            "random environment tasks, seeds 42..42+N-1, identical across conditions",
            "reduced CEM 10x100x10",
            "PlanConfig horizon=5, receding_horizon=5, action_block=5",
            "max_episode_steps=50",
            "image_shape=(224,224)",
        ],
        "action_scaler_source": {
            "file": str(STATS_PATH.relative_to(ROOT)),
            "rows": action_stats["n"],
            "content_sha256": stats["content_sha256_without_this_field"],
            "mean": action_stats["mean"],
            "scale": action_stats["scale"],
            "scale_note": "population std (ddof=0); partial shard, not the full official dataset",
        },
        "episodes_per_condition": args.episodes,
        "base_seed": args.base_seed,
        "play_area": {"low": PLAY_LOW, "high": PLAY_HIGH},
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "stable_worldmodel": "0.1.1",
            "stable_pretraining": "0.1.8",
        },
        "conditions": results,
        "kill_criteria": {
            "K1_scaler_matters": (
                scaler_summary["out_of_play_rate"] is not None
                and identity_summary["out_of_play_rate"] is not None
                and scaler_summary["out_of_play_rate"] < identity_summary["out_of_play_rate"]
            ),
            "K2_scaler_condition_within_play_area": (
                scaler_summary["n_agent_out_of_play"] == 0
            ),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(out_path),
                "identity": identity_summary,
                "scaler": scaler_summary,
                "kill_criteria": payload["kill_criteria"],
            },
            indent=2,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
