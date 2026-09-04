"""KW-LEWM-0003 — LeWM Push-T on dataset-defined tasks (last confounder removed).

KW-LEWM-0002 proved the missing action inverse_transform is *part* of the
failure mechanism (out-of-play 8/8 -> 2/8) but not sufficient (K2 failed).
It also left one confounder unresolved: tasks were **randomly sampled**, with
start-to-goal distances up to 360.9 and a strict pose-alignment success rule.
Reading 0/8 success there as "model capability" would be invalid.

This experiment removes that confounder by driving `World.evaluate(dataset=...)`
with a **shim dataset** built from real LeWM Push-T rows:

  * start/goal pairs come from the public partial Parquet shard (same episode,
    25 steps apart) -> reachable by construction, like the official protocol;
  * `pixels` are **re-rendered locally** at 224x224 from the published states
    via `AddPixelsWrapper._get_pixels()`, because the numeric-only transfer
    deliberately never downloads the image column;
  * start state and goal state are pushed into the environment through
    `evaluate(callables=...)` -> `_set_state` / `_set_goal_state`.

The action inverse_transform found necessary in KW-LEWM-0002 is kept.

Kill criteria
-------------
K1 (task source matters): dataset-defined tasks must produce materially
   different behaviour from the random-task baseline in KW-LEWM-0002
   (e.g. block interaction rate or success).
K2 (control still legal): agent stays inside the [0, 512] play area.

Usage:
    python kw_lewm_0003_dataset_tasks.py [--pairs 8] [--output results/...]
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
from kw_lewm_0002_scaler_ablation import NumpyStandardScaler, image_transform


ROOT = Path(__file__).resolve().parents[2]
STATS_PATH = ROOT / "results" / "kw_lewm_0001l_numeric_shard0.json"
DEFAULT_OUTPUT = ROOT / "results" / "kw_lewm_0003_dataset_tasks.json"
PLAY_LOW, PLAY_HIGH = 0.0, 512.0
GOAL_OFFSET = 25


class ShimDataset:
    """Minimal stand-in for the upstream dataset consumed by `World.evaluate`.

    `_extract_init_goal` only needs `column_names` and `load_chunk(ep, s, e)`;
    it reads `arr[0]` as init and `arr[-1]` as goal. `pixels` must be a torch
    Tensor in (T, C, H, W) because the upstream code calls `.permute(0, 2, 3, 1)`.
    """

    def __init__(self, entries: dict[tuple[int, int], dict]):
        self._entries = entries
        self.column_names = ["pixels", "state", "proprio", "action"]

    def load_chunk(self, episodes_idx, start_steps, end_steps):
        out = []
        for ep, start, end in zip(
            np.asarray(episodes_idx).ravel(),
            np.asarray(start_steps).ravel(),
            np.asarray(end_steps).ravel(),
        ):
            item = self._entries[(int(ep), int(start))]
            # Only the first and last frames are consumed; emit exactly two.
            out.append(
                {
                    "pixels": torch.as_tensor(item["pixels"], dtype=torch.uint8),
                    "state": np.asarray(item["state"], dtype=np.float64),
                    "proprio": np.asarray(item["proprio"], dtype=np.float64),
                    "action": np.asarray(item["action"], dtype=np.float64),
                }
            )
        return out


def find_pixels_wrapper(env):
    """Walk the wrapper chain to the AddPixelsWrapper that can render."""
    cur = env
    while cur is not None:
        if hasattr(cur, "_get_pixels"):
            return cur
        cur = getattr(cur, "env", None)
    raise RuntimeError("no wrapper exposing _get_pixels() in the env chain")


def render_state(world, pixels_wrapper, state) -> np.ndarray:
    """Render `state` to a (224, 224, 3) uint8 image without stepping physics."""
    env0 = world.envs.envs[0]
    env0.unwrapped._set_state(np.asarray(state, dtype=float))
    pixels, _ = pixels_wrapper._get_pixels()
    return np.asarray(pixels["pixels"])


def build_entries(world, pairs: list[dict]) -> dict[tuple[int, int], dict]:
    pixels_wrapper = find_pixels_wrapper(world.envs.envs[0])
    entries: dict[tuple[int, int], dict] = {}
    for pair in pairs:
        start_state = np.asarray(pair["start_state"], dtype=float)
        goal_state = np.asarray(pair["goal_state"], dtype=float)
        start_img = render_state(world, pixels_wrapper, start_state)
        goal_img = render_state(world, pixels_wrapper, goal_state)
        # (T, C, H, W): first frame = start, last frame = goal
        pixels = np.stack([start_img, goal_img]).transpose(0, 3, 1, 2)
        entries[(int(pair["episode_idx"]), int(pair["start_step"]))] = {
            "pixels": pixels,
            "state": np.stack([start_state, goal_state]),
            "proprio": np.stack(
                [
                    np.asarray(pair["start_proprio"], dtype=float),
                    np.asarray(pair["goal_proprio"], dtype=float),
                ]
            ),
            "action": np.stack(
                [
                    np.asarray(pair["start_action"], dtype=float),
                    np.asarray(pair["goal_action"], dtype=float),
                ]
            ),
        }
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--stats", default=str(STATS_PATH), help="numeric shard JSON")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    stats_path = Path(args.stats)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    action_stats = stats["stats"]["action"]
    scaler = NumpyStandardScaler(action_stats["mean"], action_stats["scale"])
    pairs = stats["selected_pairs"][: args.pairs]

    model, _ = load_lewm(device="cuda")
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    world = swm.World(
        "swm/PushT-v1", num_envs=1, image_shape=(224, 224), max_episode_steps=50
    )
    world.reset(seed=42)
    entries = build_entries(world, pairs)

    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=1,
        num_samples=100,
        n_steps=10,
        topk=10,
        device="cuda",
        seed=42,
    )
    plan = swm.PlanConfig(
        horizon=5,
        receding_horizon=5,
        action_block=5,
        history_len=1,
        warm_start=True,
    )
    transform = image_transform()
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan,
        process={"action": scaler},
        transform={"pixels": transform, "goal": transform},
    )
    world.set_policy(policy)

    callables = [
        {"method": "_set_state", "args": {"state": {"in_dataset": True, "value": "state"}}},
        {
            "method": "_set_goal_state",
            "args": {"goal_state": {"in_dataset": True, "value": "goal_state"}},
        },
    ]

    trials = []
    for pair in pairs:
        ep_idx = int(pair["episode_idx"])
        start_step = int(pair["start_step"])
        dataset = ShimDataset(entries)

        start_state = np.asarray(pair["start_state"], dtype=float)
        goal_state = np.asarray(pair["goal_state"], dtype=float)
        goal_dist = float(np.linalg.norm(goal_state[:4] - start_state[:4]))

        started = time.perf_counter()
        error = None
        metrics = None
        try:
            metrics = world.evaluate(
                dataset=dataset,
                episodes_idx=[ep_idx],
                start_steps=[start_step],
                goal_offset=GOAL_OFFSET,
                eval_budget=50,
                callables=callables,
            )
            torch.cuda.synchronize()
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        wall_seconds = time.perf_counter() - started

        final_state = np.asarray(world.infos.get("state"), dtype=float).reshape(-1)
        agent_final = final_state[:2]
        block_start = start_state[2:4]
        block_final = final_state[2:4]

        success = None
        if isinstance(metrics, dict):
            successes = metrics.get("episode_successes")
            if successes is not None and len(np.asarray(successes).ravel()) > 0:
                success = float(np.asarray(successes).ravel()[0])
            elif metrics.get("success_rate") is not None:
                success = float(metrics["success_rate"])

        trials.append(
            {
                "episode_idx": ep_idx,
                "start_step": start_step,
                "goal_step": int(pair["goal_step"]),
                "start_state": start_state.tolist(),
                "goal_state": goal_state.tolist(),
                "final_state": final_state.tolist(),
                "start_goal_distance": goal_dist,
                "agent_final": agent_final.tolist(),
                "agent_out_of_play_area": bool(
                    np.any(agent_final < PLAY_LOW) or np.any(agent_final > PLAY_HIGH)
                ),
                "block_start": block_start.tolist(),
                "block_final": block_final.tolist(),
                "block_displacement": float(np.linalg.norm(block_final - block_start)),
                "success": success,
                "metrics_raw": metrics,
                "wall_seconds": wall_seconds,
                "error": error,
            }
        )

    world.close()

    out_of_play = [bool(t["agent_out_of_play_area"]) for t in trials]
    displacements = [t["block_displacement"] for t in trials]
    successes = [t["success"] for t in trials if t["success"] is not None]
    interacted = [d > 1.0 for d in displacements]

    payload = {
        "experiment_id": "KW-LEWM-0003-DATASET-TASKS",
        "question": "With the random-task confounder removed, does LeWM produce valid closed-loop control?",
        "design": "dataset-defined start/goal pairs from the public Parquet shard, driven through World.evaluate(dataset=...)",
        "evidence_level": "E1_INTERNAL_REPRODUCTION",
        "n_tasks": len(trials),
        "goal_offset": GOAL_OFFSET,
        "eval_budget": 50,
        "action_scaler": {
            "source": "results/kw_lewm_0001l_numeric_shard0.json",
            "content_sha256": stats["content_sha256_without_this_field"],
            "rows": action_stats["n"],
            "mean": action_stats["mean"],
            "scale": action_stats["scale"],
        },
        "protocol_notes": [
            "pixels re-rendered locally at 224x224 from published states; the image column is never downloaded",
            "only the first and last frames are materialised because _extract_init_goal reads arr[0] and arr[-1]",
            "approximations: partial-shard scaler, locally rendered pixels, reduced CEM 10x100x10",
            "this is NOT the official 13.1 GB dataset benchmark",
        ],
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "stable_worldmodel": "0.1.1",
            "stable_pretraining": "0.1.8",
        },
        "summary": {
            "success_rate": float(np.mean(successes)) if successes else None,
            "n_success": int(sum(1 for s in successes if s > 0)) if successes else None,
            "out_of_play_rate": float(np.mean(out_of_play)) if out_of_play else None,
            "n_out_of_play": int(sum(out_of_play)),
            "block_interaction_rate": float(np.mean(interacted)) if interacted else None,
            "block_displacement_mean": float(np.mean(displacements)) if displacements else None,
            "block_displacement_median": float(np.median(displacements)) if displacements else None,
            "start_goal_distance_mean": float(np.mean([t["start_goal_distance"] for t in trials])),
            "n_errors": int(sum(1 for t in trials if t["error"] is not None)),
        },
        "baseline_for_comparison": {
            "experiment": "KW-LEWM-0002 scaler condition (random tasks, same planner, same scaler)",
            "out_of_play_rate": 0.25,
            "block_displacement_median": 0.0,
            "success_rate": 0.0,
        },
        "kill_criteria": {
            "K1_task_source_matters": None,  # filled below
            "K2_control_within_play_area": bool(sum(out_of_play) == 0),
        },
        "trials": trials,
    }
    payload["kill_criteria"]["K1_task_source_matters"] = bool(
        payload["summary"]["block_interaction_rate"] is not None
        and payload["summary"]["block_interaction_rate"] > 3.0 / 8.0
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(out_path),
                "summary": payload["summary"],
                "kill_criteria": payload["kill_criteria"],
                "per_task": [
                    {
                        "ep": t["episode_idx"],
                        "start_step": t["start_step"],
                        "goal_dist": round(t["start_goal_distance"], 1),
                        "block_disp": round(t["block_displacement"], 2),
                        "success": t["success"],
                        "out_of_play": t["agent_out_of_play_area"],
                        "error": t["error"],
                    }
                    for t in trials
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
