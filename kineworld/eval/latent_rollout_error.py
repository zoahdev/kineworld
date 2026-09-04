# -*- coding: utf-8 -*-
"""
KineWorld Phase-0: latent rollout prediction-error harness (directive §19).

Measures how the official JEPA-WM latent predictor degrades as it rolls forward
under the *real* actions from Push-T trajectories:

    z_t        = Encoder(o_t)                 (ground-truth latent)
    \hat z_{t+k} = Predictor^k(z_t, a_{t:t+k}) (imagination, open-loop)

For each sampled trajectory segment and each start frame, we roll the world
model forward up to H_max steps and compare predicted latent vs the real
encoded latent at horizons {1,5,10,20,50}. Error = normalized L2 on the
visual latent tokens.

This is a PREDICTION benchmark (world-model quality), distinct from the
planning benchmark (end-to-end task success). Both are required by Phase 0.

Reuses the official model build path (evals/simu_env_planning/eval.py) so the
result is directly comparable to the official baseline. No core algorithm is
rewritten.

Context convention (critical, matches official eval):
  The official Push-T evaluator (gc_agent.plan_step) encodes a SINGLE frame
  obs (`task_specification.num_frames == 1`) as z_init, then calls
  `model.unroll(z_init, act_suffix=actions)`. `ctxt_window` (=2) only sets the
  *internal sliding window* of the unroll loop, NOT the number of encode input
  frames. Passing ctxt_window frames to encode() over-seeds the context and
  desynchronises the AdaLN predictor's token counts (x: tau*256 vs z: tau)
  -> `a(512) must match b(256)`. We therefore encode a single context frame,
  exactly like the official evaluator.

Usage:
  python latent_rollout_error.py --episodes 12 --horizons 1,5,10,20,50 --out ../results/rollout_err.json
"""
import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from tensordict.tensordict import TensorDict

# --- repo paths ---
# __file__ = kineworld/kineworld/eval/latent_rollout_error.py
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))       # kineworld/kineworld/eval
_WS = os.path.abspath(os.path.join(_EVAL_DIR, "..", ".."))   # kineworld
REPO = os.path.abspath(os.path.join(_WS, "external", "jepa-wms"))
WS = _WS
sys.path.insert(0, REPO)

os.environ.setdefault("JEPAWM_DSET", os.path.join(WS, "data"))
os.environ.setdefault("JEPAWM_LOGS", os.path.join(WS, "results", "jepa_logs"))
os.environ.setdefault("JEPAWM_HOME", WS)
os.environ.setdefault("JEPAWM_CKPT", os.path.join(WS, "checkpoints"))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from evals.utils import make_datasets  # noqa: E402
from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module  # noqa: E402


def load_model(cfg):
    model_kwargs = cfg["model_kwargs"]
    cfgs_data = model_kwargs.get("data", {})
    cfgs_data_aug = model_kwargs.get("data_aug", {})
    wrapper_kwargs = model_kwargs.get("wrapper_kwargs", {})
    pretrain_kwargs = model_kwargs.get("pretrain_kwargs", {})
    checkpoint = model_kwargs.get("checkpoint")
    checkpoint_folder = cfg.get("checkpoint_folder", cfg.get("folder"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dset, preprocessor = make_datasets(cfgs_data, cfgs_data_aug, 1, 0)
    model = init_module(
        folder=checkpoint_folder,
        checkpoint=checkpoint,
        module_name=model_kwargs["module_name"],
        model_kwargs=pretrain_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        cfgs_data=cfgs_data,
        device=device,
        action_dim=dset.action_dim,
        proprio_dim=dset.proprio_dim,
        preprocessor=preprocessor,
    )
    return model, dset, preprocessor, device


def norm_l2(a, b):
    # normalized L2 across token dims; a,b: [D]
    return float((a - b).pow(2).sum().sqrt() / (a.pow(2).sum().sqrt() + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=os.path.join(
        REPO, "configs", "dump_online_evals", "pt",
        "pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep5_red10x100.yaml"))
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--horizons", default="1,5,10,20,50")
    ap.add_argument("--start_offsets", type=int, default=3,
                    help="sample N start frames per trajectory")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(WS, "results", "rollout_err.json"))
    args = ap.parse_args()

    import yaml
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    horizons = [int(h) for h in args.horizons.split(",")]
    hmax = max(horizons)

    model, dset, preprocessor, device = load_model(cfg)
    model.eval()
    print(f"[rollout-err] model loaded on {device}; action_dim={dset.action_dim} "
          f"proprio_dim={dset.proprio_dim}; ctxt_window={model.ctxt_window}")

    # per-horizon error accumulators
    errs = {h: [] for h in horizons}
    n_samples = 0
    n_traj = 0

    with torch.no_grad():
        for ep in range(args.episodes):
            traj_id = int(torch.randint(0, len(dset), (1,)).item())
            try:
                obs, act, state, reward, e_info = dset[traj_id]
            except Exception as e:
                print(f"  [skip] traj {traj_id}: {e}")
                continue
            visual = obs["visual"]  # [T, C, H, W]
            proprio = obs["proprio"]  # [T, P]
            T = visual.shape[0]
            if T < 1 + hmax + 1:
                continue
            n_traj += 1

            # sample start offsets; context = 1 frame, need off + hmax <= T
            max_offset = T - hmax - 1
            offsets = np.random.choice(max_offset + 1,
                                       size=min(args.start_offsets, max_offset + 1),
                                       replace=False)

            for off in offsets:
                # ---- ground-truth latents for all frames off..off+hmax (batchify encode) ----
                block = visual[off:off + 1 + hmax][None]        # [1, 1+hmax, C, H, W]
                prop_block = proprio[off:off + 1 + hmax][None]  # [1, 1+hmax, P]
                z_block = model.encode(TensorDict({
                    "visual": block,
                    "proprio": prop_block,
                }, batch_size=[1]))  # visual [1, 1+hmax, V, H, W, D]
                z_all = z_block["visual"][0]  # [1+hmax, V, H, W, D]; index h -> real frame off+h

                z_ctx = TensorDict({
                    "visual": z_all[0:1][None],                 # [1, 1, V, H, W, D]
                    "proprio": z_block["proprio"][:, 0:1],
                }, batch_size=[1])
                gt_vis = z_all  # gt latent for horizon h = off+h frame = index h

                # ---- actions to roll forward: hmax steps ----
                act_suffix = act[off:off + hmax]                # [T, A] -> [hmax, A]
                act_suffix = act_suffix[None].permute(1, 0, 2)  # [T, 1, A]
                if model.action_skip > 1:
                    act_suffix = act_suffix.repeat_interleave(model.action_skip, dim=0)[:hmax]
                act_suffix = act_suffix.to(device).contiguous()

                pred = model.unroll(z_ctx, act_suffix)  # TensorDict visual [T+tau, 1, V, H, W, D]
                pred_vis = pred["visual"][:, 0]         # [T+tau, V, H, W, D], tau=1
                # pred index = (tau-1) + h = h  (first predicted latent is at index 1)
                for h in horizons:
                    if h < pred_vis.shape[0]:
                        p = pred_vis[h].reshape(-1)      # predicted latent for horizon h
                        g = gt_vis[h].reshape(-1)        # real latent of frame off+h
                        errs[h].append(norm_l2(p, g))
                n_samples += 1
            print(f"  [ep {ep}] traj {traj_id} T={T} offsets={len(offsets)} "
                  f"(cum samples={n_samples})")

    # aggregate
    result = {
        "experiment": "kw_phase0_latent_rollout_error",
        "model": "jepa_wm_pusht (official)",
        "encoder": "DINOv2 ViT-S/14 (frozen)",
        "predictor_depth": cfg["model_kwargs"].get("pretrain_kwargs", {}).get("predictor", {}).get("pred_depth"),
        "n_trajectories": n_traj,
        "n_samples": n_samples,
        "seed": args.seed,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "horizon_errors": {
            str(h): {
                "mean_norm_l2": float(np.mean(errs[h])) if errs[h] else None,
                "std": float(np.std(errs[h])) if errs[h] else None,
                "n": len(errs[h]),
            }
            for h in horizons
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[rollout-err] wrote {args.out}")
    print(json.dumps(result["horizon_errors"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
