"""KW-LEWM-0005 debug — 检查执行期动作量级与每步 block 位移。

目的：确认 K4 fly-out 是「动作量级错误（缺逆归一化）」还是「模型本身不控」。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import stable_worldmodel as swm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external" / "jepa-wms"))
sys.path.insert(0, str(ROOT / "verification" / "scripts"))

from kw_lewm_0005_jepa_adapter import build_jepa_adapter, run_k4  # noqa: F401 (build only)
from kw_lewm_0005_jepa_adapter import JEPAWMAdapter
from kw_lewm_0003_dataset_tasks import ShimDataset, build_entries, find_pixels_wrapper
from kw_lewm_0002_scaler_ablation import NumpyStandardScaler


def main():
    device = "cuda"
    adapter, meta, stats, _ = build_jepa_adapter(device=device)
    action_stats = stats["stats"]["action"]
    scaler = NumpyStandardScaler(action_stats["mean"], action_stats["scale"])
    print(f"[dbg] action mean={np.asarray(action_stats['mean'])} scale={np.asarray(action_stats['scale'])}")

    solver = swm.solver.CEMSolver(
        model=adapter, batch_size=1, num_samples=100, n_steps=10, topk=10,
        device=device, seed=42,
    )
    plan = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5, history_len=1, warm_start=True)

    # 记录执行期动作（逆归一化前/后）
    exec_actions_raw = []
    exec_actions_den = []

    class DbgPolicy(swm.policy.WorldModelPolicy):
        def get_action(self, info_dict, **kwargs):
            out = super().get_action(info_dict, **kwargs)
            # out 是 numpy，已是逆归一化后的执行动作（2 维）
            exec_actions_den.append(np.asarray(out).ravel().copy())
            return out

    policy = DbgPolicy(solver=solver, config=plan, process={"action": scaler}, transform={})

    world = swm.World("swm/PushT-v1", num_envs=1, image_shape=(224, 224), max_episode_steps=50)
    world.reset(seed=42)
    pairs = stats["selected_pairs"][:1]
    entries = build_entries(world, pairs)
    world.set_policy(policy)

    pair = pairs[0]
    ep_idx = int(pair["episode_idx"]); start_step = int(pair["start_step"])
    dataset = ShimDataset(entries)
    callables = [
        {"method": "_set_state", "args": {"state": {"in_dataset": True, "value": "state"}}},
        {"method": "_set_goal_state", "args": {"goal_state": {"in_dataset": True, "value": "goal_state"}}},
    ]
    world.evaluate(dataset=dataset, episodes_idx=[ep_idx], start_steps=[start_step],
                   goal_offset=25, eval_budget=50, callables=callables)
    torch.cuda.synchronize()
    world.close()

    den = np.asarray(exec_actions_den)
    print(f"[dbg] n_exec_actions={len(den)} shape={den.shape}")
    print(f"[dbg] executed (denormalized) action: abs-mean={np.abs(den).mean():.4f} "
          f"max={np.abs(den).max():.4f} std={den.std():.4f}")
    print(f"[dbg] executed per-dim mean={den.mean(axis=0)}")
    # 若逆归一化生效，std 应≈0.19；若缺逆归一化，std≈1.0
    # 同时打印每步 block 位移
    print("[dbg] done.")


if __name__ == "__main__":
    main()
