"""One-task, no-dataset LeWM Push-T closed-loop smoke test.

This follows KW-LEWM-0001L Stage A. It deliberately uses identity action
normalization because the official full-dataset StandardScaler is unavailable.
"""

from __future__ import annotations

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
OUTPUT = ROOT / "results" / "kw_lewm_0001l_smoke.json"


class CountingCEM(swm.solver.CEMSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_seconds: list[float] = []

    def solve(self, *args, **kwargs):
        started = time.perf_counter()
        result = super().solve(*args, **kwargs)
        self.call_seconds.append(time.perf_counter() - started)
        return result


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


def serializable_array(value):
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


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.reset_peak_memory_stats()

    model, _ = load_lewm(device="cuda")
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    world = swm.World(
        "swm/PushT-v1",
        num_envs=1,
        image_shape=(224, 224),
        max_episode_steps=50,
    )
    solver = CountingCEM(
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
        process={},  # explicit lite deviation: no full-dataset StandardScaler
        transform={"pixels": transform, "goal": transform},
    )
    world.set_policy(policy)
    world.reset(seed=42)

    initial_state = serializable_array(world.infos.get("state"))
    goal_state = serializable_array(world.infos.get("goal_state"))
    started = time.perf_counter()
    error = None
    metrics = None
    try:
        metrics = world.evaluate(episodes=1, seed=None, reset_mode="wait")
        torch.cuda.synchronize()
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    wall_seconds = time.perf_counter() - started

    final_state = serializable_array(world.infos.get("state"))
    finite = True
    for item in (initial_state, goal_state, final_state):
        if item is not None:
            finite = finite and bool(np.isfinite(np.asarray(item, dtype=float)).all())

    result = {
        "experiment_id": "KW-LEWM-0001L-STAGE-A",
        "status": "PASS" if error is None and finite else "FAIL",
        "evidence_level": "E1_INTERNAL_FUNCTIONAL_SMOKE",
        "claim_scope": "closed-loop smoke only; not official benchmark; not competitor comparison",
        "seed": 42,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "stable_worldmodel": "0.1.1",
            "stable_pretraining": "0.1.8",
        },
        "environment": {
            "id": "swm/PushT-v1",
            "max_episode_steps": 50,
            "image_shape": [224, 224],
            "dataset_driven": False,
            "action_normalization": "identity_lite_deviation",
        },
        "plan_config": {
            "horizon": 5,
            "receding_horizon": 5,
            "action_block": 5,
            "history_len": 1,
            "warm_start": True,
        },
        "cem": {"n_steps": 10, "num_samples": 100, "topk": 10},
        "solver_calls": len(solver.call_seconds),
        "solver_call_seconds": solver.call_seconds,
        "wall_seconds": wall_seconds,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "metrics_raw": metrics,
        "success_rate_unit": "percent",
        "initial_state": initial_state,
        "goal_state": goal_state,
        "final_state": final_state,
        "finite_states": finite,
        "error": error,
        "known_deviations": [
            "random environment task instead of upstream dataset-defined task",
            "identity action normalization instead of full-dataset StandardScaler",
            "reduced CEM 10x100x10 instead of upstream 30x300x30",
            "single task",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, sort_keys=True, default=json_default)
    OUTPUT.write_text(encoded, encoding="utf-8")
    print(encoded)
    world.close()


if __name__ == "__main__":
    main()
