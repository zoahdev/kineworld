"""KW-LEWM-0005 §8 根因深挖：代价地形平坦的来源。

核心诊断 —— 潜在空间迁移率（latent mobility）：
  该 checkpoint 在 H=5 latent step（=25 env step）内，最多能沿动作把预测 latent
  推多远？对比 start→goal 的 latent 间距（到达目标所需位移）。

  判据：
    gap_req   = ||z_goal - z_start||            （到达 goal 所需的 latent 位移）
    max_mob   = max_{scale,随机动作} ||pred_final - z_start||   （模型最大可达位移）
    ratio     = max_mob / gap_req
    - ratio < 0.5  → 低机动性（上下文主导预测，场景近乎静止）→ 根因=模型属性，不训练不可修
    - ratio ≈ 1    → 机动性足够，平坦来自代价度量（final-frame MSE）→ 推理期可救
  另报 context 主导度：do-nothing 位移 d(0) 占总可达位移的比例（高=模型预测近乎静止）。

复用 kw_lewm_0005_jepa_adapter.build_jepa_adapter + render_state。纯诊断，不训练。
"""
from __future__ import annotations
import json
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "jepa-wms"))
import kw_lewm_0005_jepa_adapter as A
from einops import rearrange


def main(output_path=None):
    device = "cuda"
    adapter, meta, stats, _ = A.build_jepa_adapter(device=device)
    pairs = stats["selected_pairs"][:5]  # 5 任务足够定性

    import stable_worldmodel as swm
    world = swm.World("swm/PushT-v1", num_envs=1, image_shape=(224, 224), max_episode_steps=50)
    world.reset(seed=42)
    pw = A.find_pixels_wrapper(world.envs.envs[0])

    H = 5  # latent steps（= frameskip 5，覆盖 goal_offset 25）
    N = 30  # 每 scale 随机动作样本数
    scales = [0.0, 0.5, 1.0, 2.0, 3.0]
    rng = torch.Generator(device=device).manual_seed(0)

    rows = []
    for pi, pair in enumerate(pairs):
        start_state = np.asarray(pair["start_state"], dtype=float)
        goal_state = np.asarray(pair["goal_state"], dtype=float)
        start_img = A.render_state(world, pw, start_state)
        goal_img = A.render_state(world, pw, goal_state)
        f0 = start_img.transpose(2, 0, 1).astype(np.float32)   # (3,224,224) 0-255
        f1 = goal_img.transpose(2, 0, 1).astype(np.float32)
        # 关键：z_start 的 batch 维必须 = N（与 act_suffix batch 一致），完全镜像 get_cost
        # （get_cost 里 _to_visual(pixels) 的 batch = BT = N）。用 batch=N 构造，避免 unroll 的
        # .expand 产生 stride-0 视图触发 predictor 内部 shape 错。
        pix_t = torch.from_numpy(np.broadcast_to(f0, (N, 1, 3, 224, 224)).copy()).to(device)
        goal_t = torch.from_numpy(np.broadcast_to(f1, (N, 1, 3, 224, 224)).copy()).to(device)
        prop_t = torch.from_numpy(np.broadcast_to(start_state[:4].reshape(1, 1, 4), (N, 1, 4)).copy()).float().to(device)
        gprop_t = torch.from_numpy(np.broadcast_to(goal_state[:4].reshape(1, 1, 4), (N, 1, 4)).copy()).float().to(device)

        # 关键修复：必须保留 encode 返回的完整 TensorDict（含 proprio），否则 unroll
        # 内部 forward_pred 跳 proprio concat -> 384 维输入撞 400 维 LayerNorm 崩溃。
        # 完全镜像 get_cost（adapter.py:203-219）：obs 是 TensorDict，z_start 是 TensorDict，
        # 仅在算 norm / 代价时用 z["visual"]。
        with torch.no_grad():
            z_start = adapter.wm.encode({"visual": pix_t, "proprio": prop_t})  # TensorDict
            z_goal = adapter.wm.encode({"visual": goal_t, "proprio": gprop_t})

        gap_req = float((z_goal["visual"] - z_start["visual"]).norm().item())

        scale_disp = {}
        for s in scales:
            cand = torch.randn(1, N, H, adapter.model_action_dim, generator=rng, device=device) * s
            cand = cand.clamp(-adapter.action_clip, adapter.action_clip)
            act_suffix = rearrange(cand, "b n h a -> h (b n) a")  # (H, B*N, 10)
            with torch.no_grad():
                pred_td = adapter.wm.unroll(z_start, act_suffix=act_suffix)
            pred_final = pred_td["visual"][-1]  # (N, V, g, g, D)  # batch=N，与 act_suffix 对齐
            # pred_final 已是 (N, V, g, g, D)（ndim=5）；z_start["visual"] 同为 (N, V, g, g, D)
            disp = (pred_final - z_start["visual"]).reshape(N, -1).norm(dim=1)  # (N,)
            cost_to_goal = (pred_final - z_goal["visual"]).reshape(N, -1).pow(2).mean(dim=1)  # (N,)
            scale_disp[s] = {
                "max_disp_from_start": float(disp.max().item()),
                "mean_disp_from_start": float(disp.mean().item()),
                "min_cost_to_goal": float(cost_to_goal.min().item()),
                "mean_cost_to_goal": float(cost_to_goal.mean().item()),
            }

        max_mob = max(scale_disp[s]["max_disp_from_start"] for s in scales if s > 0)
        ratio = max_mob / gap_req if gap_req > 0 else float("nan")
        d0 = scale_disp[0.0]["max_disp_from_start"]
        context_dom = d0 / max_mob if max_mob > 0 else float("nan")
        # 关键量：最大动作相对「不动」基线额外买到的 latent 位移。≈0 即动作对预测几乎无杠杆。
        action_leverage = max_mob - d0

        rows.append({
            "pair": pi,
            "episode_idx": int(pair["episode_idx"]),
            "start_step": int(pair["start_step"]),
            "gap_req": gap_req,
            "max_mobility": max_mob,
            "ratio_max_mob_over_gap": ratio,
            "do_nothing_disp": d0,
            "context_domination_ratio": context_dom,
            "action_leverage": float(action_leverage),
            "scale_detail": {str(s): scale_disp[s] for s in scales},
        })
    world.close()

    ratios = [r["ratio_max_mob_over_gap"] for r in rows]
    cdom = [r["context_domination_ratio"] for r in rows]
    lev = [r["action_leverage"] for r in rows]
    mean_ratio = float(np.mean(ratios))
    mean_cdom = float(np.mean(cdom))
    mean_lev = float(np.mean(lev))
    # 修正后的三元判别（原来 ratio<0.5 太粗，遗漏「机动充足但动作无杠杆」这第三种根因）：
    #   1) context_domination_ratio ≈ 1  → 预测被上下文主导、动作对最终 latent 几乎无杠杆
    #      → 根因 = 弱动作条件（weak action conditioning），模型属性，不训练不可修。
    #      （这正是 §7.3 toward_better_than_away=false 的来源：所有动作→几乎相同预测→几乎相同代价）
    #   2) mean_ratio < 0.5  → 低潜在机动性（场景近乎静止），模型属性。
    #   3) 否则 → 机动充足且动作有杠杆，平坦来自代价度量（final-frame MSE），推理期可救。
    if mean_cdom > 0.9:
        verdict = ("ACTION_INSENSITIVE_PREDICTION (weak action conditioning; model property, "
                   "NOT fixable at inference without retraining)")
    elif mean_ratio < 0.5:
        verdict = "LOW_LATENT_MOBILITY (model property, not fixable w/o training)"
    else:
        verdict = "MOBILITY_OK (flatness is cost-metric issue, inference-fixable)"
    out = {
        "diagnosis": "latent mobility + action-leverage (can the model traverse start->goal, and does the action steer it?)",
        "H_latent_steps": H,
        "mean_ratio_max_mob_over_gap": mean_ratio,
        "min_ratio": float(np.min(ratios)),
        "mean_context_domination_ratio": mean_cdom,
        "mean_action_leverage": mean_lev,
        "verdict": verdict,
        "rows": rows,
    }
    out_json = json.dumps(out, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if output_path:
        Path(output_path).write_text(out_json, encoding="utf-8")
        print(f"[rootcause] wrote {output_path}")
    else:
        print(out_json)
    # 简短摘要（避免被 dinov2 logging 淹没）
    print(f"[rootcause] verdict={verdict} mean_ratio={mean_ratio:.3f} "
          f"mean_context_domination_ratio={mean_cdom:.4f} mean_action_leverage={mean_lev:.1f}",
          file=sys.stderr)
    return out


if __name__ == "__main__":
    import argparse as _ap
    _a = _ap.ArgumentParser()
    _a.add_argument("--output", default=str(Path(__file__).resolve().parents[2] / "results" / "kw_lewm_0005_rootcause.json"))
    _ns = _a.parse_args()
    main(output_path=_ns.output)


if __name__ == "__main__":
    main()
