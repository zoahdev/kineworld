"""KW-LEWM-0005 K4 诊断：代价是否被正确 goal 驱动。

目标：在烧 50 任务 K4 前，确认 adapter.get_cost 的代价地形对「朝目标 / 背离目标」
方向有正确响应。若代价不区分二者（或反向），说明 goal  conditioning 有 wiring 缺陷，
应修复后再跑全量；否则失败属模型质量，跑全量即可。

复用 kw_lewm_0005_jepa_adapter.build_jepa_adapter + render_state。
"""
from __future__ import annotations
import json
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "jepa-wms"))

import kw_lewm_0005_jepa_adapter as A


def main():
    device = "cuda"
    adapter, meta, stats, _ = A.build_jepa_adapter(device=device)
    pairs = stats["selected_pairs"][:8]

    # 渲染一个真实任务对的 start/goal 图像
    import stable_worldmodel as swm
    world = swm.World("swm/PushT-v1", num_envs=1, image_shape=(224, 224), max_episode_steps=50)
    world.reset(seed=42)
    pw = A.find_pixels_wrapper(world.envs.envs[0])

    rows = []
    for pi, pair in enumerate(pairs[:4]):
        start_state = np.asarray(pair["start_state"], dtype=float)
        goal_state = np.asarray(pair["goal_state"], dtype=float)
        start_img = A.render_state(world, pw, start_state)   # (224,224,3) uint8
        goal_img = A.render_state(world, pw, goal_state)

        start_prop = start_state[:4].reshape(1, 4).astype(float)
        goal_prop = goal_state[:4].reshape(1, 4).astype(float)
        f0 = start_img.transpose(2, 0, 1).astype(np.uint8)   # (3,224,224)
        f1 = goal_img.transpose(2, 0, 1).astype(np.uint8)

        B, N, H = 1, 4, 5
        pix_t = torch.from_numpy(np.broadcast_to(f0, (B, N, 1, 3, 224, 224)).copy()).to(torch.uint8)
        goal_t = torch.from_numpy(np.broadcast_to(f1, (B, N, 1, 3, 224, 224)).copy()).to(torch.uint8)
        prop_t = torch.from_numpy(np.repeat(start_prop, N, axis=0)[None]).float().to(device)
        gprop_t = torch.from_numpy(np.repeat(goal_prop, N, axis=0)[None]).float().to(device)

        # 朝向目标的「方向」：goal - start 在 agent 平面上的位移（归一化）
        d = goal_state[:2] - start_state[:2]
        dn = d / (np.linalg.norm(d) + 1e-6)

        # 构造候选：每个 N 一个 latent step 的 10 维动作 = 5 个 env action(2) 拼接
        # 候选 A：朝目标方向推；候选 B：背离目标方向推；候选 C：不动
        def make_cand(dirn):
            # 单 env action 沿 dirn，重复 5 次拼接成 10 维
            a = np.array([dirn[0], dirn[1]], dtype=np.float32) * 0.6
            block = np.concatenate([a, a, a, a, a])  # (10,)
            c = np.zeros((B, N, H, 10), dtype=np.float32)
            c[:, :, 0, :] = block   # 仅第一步非零，其余零
            return torch.from_numpy(c).to(device)

        cand_toward = make_cand(dn)
        cand_away = make_cand(-dn)
        cand_still = torch.zeros(B, N, H, 10, dtype=torch.float32, device=device)

        info = {"pixels": pix_t.to(device), "goal": goal_t.to(device),
                "proprio": prop_t, "goal_proprio": gprop_t}

        with torch.no_grad():
            cost_toward = adapter.get_cost(info, cand_toward).detach().cpu().numpy().ravel()
            cost_away = adapter.get_cost(info, cand_away).detach().cpu().numpy().ravel()
            cost_still = adapter.get_cost(info, cand_still).detach().cpu().numpy().ravel()
            # 额外：goal 图像换成 start 图像（验证 goal 是否真的影响代价）
            info_same = {"pixels": pix_t.to(device), "goal": pix_t.to(device),
                         "proprio": prop_t, "goal_proprio": prop_t}
            cost_same_goal = adapter.get_cost(info_same, cand_toward).detach().cpu().numpy().ravel()

        gpix = goal_img.astype(float)
        rows.append({
            "pair": pi,
            "episode_idx": int(pair["episode_idx"]),
            "start_step": int(pair["start_step"]),
            "goal_pix_mean": float(gpix.mean()),
            "goal_pix_std": float(gpix.std()),
            "cost_toward_mean": float(cost_toward.mean()),
            "cost_away_mean": float(cost_away.mean()),
            "cost_still_mean": float(cost_still.mean()),
            "cost_same_goal_mean": float(cost_same_goal.mean()),
            "toward_lt_away": bool(cost_toward.mean() < cost_away.mean()),
            "goal_changes_cost": bool(abs(cost_same_goal.mean() - cost_toward.mean()) > 1.0),
        })
    world.close()
    out = {"diagnosis": rows,
           "verdict": {
               "goal_drives_cost": bool(np.mean([r["goal_changes_cost"] for r in rows]) > 0.5),
               "toward_better_than_away": bool(np.mean([r["toward_lt_away"] for r in rows]) > 0.5),
           }}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
