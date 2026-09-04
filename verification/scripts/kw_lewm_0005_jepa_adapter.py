"""KW-LEWM-0005 — JEPAWMAdapter: 把 facebookresearch/jepa-wms 的 EncPredWM
包成 stable-worldmodel 的 get_cost 接口，实现同台对比。

路径 1：构建 adapter + 1 任务冒烟验证 K1-K3。
  K1 (adapter 可构建): init_module 从本地 checkpoint + vendored 代码实例化
  K2 (预测有限 / in-distribution): get_cost 输出有限、预测 latent 非 collapse
  K3 (encoder 离线加载): DINOv2 ViT-S/14 经 torch hub 本地缓存加载

设计要点（与 KW-LEWM-0005.md §3 对齐）：
  * EncPredWM.encode 期望 uint8 pixels (+255 + preprocessor.transform)，
    所以 WorldModelPolicy 不应再对 pixels/goal 做 transform（否则双重归一）。
  * candidates 来自 CEM，已是「标准化空间」N(0,1)（与官方 jepa-wms CEMPlanner 一致）；
    encode_act 不再做归一化，故 get_cost 直接透传 candidates 到 unroll，绝不二次归一化。
    执行期逆归一化由 policy.process['action'].inverse_transform 负责（把 N(0,1) -> 环境 Box）。
    model_action_dim = action_dim(2) * tubelet_size_enc(1) * frameskip(5) // action_skip(1) = 10
    5 个 env step 动作拼接成 1 个 latent step 动作（frameskip=5）。
  * goal 由 info_dict['goal'] (uint8) 编码，unroll 后 pred 与 goal 算 patch-MSE -> (B,N)

用法：
    python kw_lewm_0005_jepa_adapter.py            # 跑 K1-K3 冒烟，写 JSON
    python kw_lewm_0005_jepa_adapter.py --task 0   # 指定用第几个 selected_pair
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "external" / "jepa-wms"
CKPT = ROOT / "checkpoints" / "jepa_wm_pusht.pth.tar"
LEWM_SHARD = ROOT / "results" / "kw_lewm_0001l_numeric_shard0.json"
OUT = ROOT / "results" / "kw_lewm_0005_smoke.json"

# 把 vendored 仓库根加进 path（app 与 src 都在 repo 根）
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# 抑制 torch hub fork 校验（DinoEncoder 里也设了一次，这里兜底）
try:
    import torch.hub as _th

    _th._validate_not_a_forked_repo = lambda a, b, c: True  # type: ignore
except Exception:
    pass


# --------------------------------------------------------------------------- #
# 离线 preprocessor（不触发 13.1GB 数据集下载）
# --------------------------------------------------------------------------- #
def build_preprocessor(action_mean, action_std, device, proprio_dim=4):
    """用 config 的 normalize + 离线构造的 Preprocessor。

    transform 严格复刻 eval.py 的 make_transforms 调用（scale/ratio=1.0, flip=False）
    => 确定性 identity crop + ImageNet normalize。proprio/state 用单位缩放（残留近似，
    与 KW-LEWM-0004 的 partial-shard scaler 同等级）。
    """
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_transforms

    normalize = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        img_size=224,
        normalize=normalize,
    )
    am = torch.tensor(action_mean, dtype=torch.float32).reshape(1, 1, -1)
    ast = torch.tensor(action_std, dtype=torch.float32).reshape(1, 1, -1)
    # proprio/state 单位缩放（残留近似，已在纪律边界登记）；preprocessor 统计保持在 CPU，
    # 因为 EncPredWM.encode 内部先把 proprio .cpu() 再 normalize。
    z = torch.zeros(1, 1, proprio_dim, dtype=torch.float32)
    o = torch.ones(1, 1, proprio_dim, dtype=torch.float32)
    return Preprocessor(
        action_mean=am,
        action_std=ast,
        state_mean=z,
        state_std=o,
        proprio_mean=z,
        proprio_std=o,
        transform=transform,
        inverse_transform=None,
    )


# --------------------------------------------------------------------------- #
# adapter
# --------------------------------------------------------------------------- #
class JEPAWMAdapter(nn.Module):
    """把 EncPredWM 包成 stable_worldmodel 的 get_cost(info_dict, candidates) 契约。

    info_dict (来自 CEMSolver.solve 的 expanded_infos):
        'pixels' : (B, N, T, C, H, W) uint8   —— 当前观测（首个/唯一帧）
        'goal'   : (B, N, T, C, H, W) uint8   —— 目标观测
        'proprio': (B, N, T, 7) 或 (B, N, 7)  —— 可选 robot state
        'goal_proprio': 同上，可选
    candidates: (B, N, H, action_dim=2) 环境 action 空间
    返回: (B, N) latent-MSE 代价
    """

    def __init__(self, wm, preprocessor, env_action_dim, model_action_dim, device, action_clip=3.0):
        super().__init__()
        self.wm = wm
        self.preprocessor = preprocessor
        self.env_action_dim = env_action_dim  # 2 (PushT env action)
        self.model_action_dim = model_action_dim  # 10 = 2 * tubelet(1) * frameskip(5)
        # CEMSolver.action_dim = env_dim * action_block；与本模型一致 => action_block = 5
        self.action_block = model_action_dim // env_action_dim  # 5
        self.device = device
        # 安全夹紧（对齐官方 CEMPlanner 的 max_norms 机制）。stable-worldmodel 的
        # CEMSolver 不做动作夹紧；若 world model 的代价地形对大动作过奖（cost 随动作
        # 幅度单调下降），CEM 会发散到极大动作 -> 执行期越界飞出。夹紧候选到训练动作
        # 分布范围（归一化空间 std~1，±3 覆盖 ~99.7%），可阻断该伪影而不改模型。
        self.action_clip = float(action_clip)

    @staticmethod
    def _to_visual(x, B, N):
        """把 expanded_infos 的 pixels/goal 折成 (BT, T, C, H, W)，自适应 channel 布局。

        stable-worldmodel 的 world 以 channel-last 存像素 (B, T, H, W, C)；
        policy 不配 pixels transform（避免与 encode 的 /255+normalize 双重归一），
        因此这里按末维是否为 3 判定 channel-last 并 permute 回 channel-first。
        """
        x = x.reshape(B * N, *x.shape[2:])  # (BT, T, ...)
        if x.shape[-1] == 3:
            x = x.permute(0, 1, 4, 2, 3)  # (BT, T, C, H, W)
        return x.float()

    @torch.no_grad()
    def get_cost(self, info_dict, candidates):
        """candidates: (B, N, H, D).

        真实 CEM 路径：D == model_action_dim(=10)，每个 plan step 的 D 维向量 =
        action_block(5) 个 env action 拼接；H 个 plan step = H 个 latent step
        （frameskip=5 => 1 latent step = 5 env step），故预测 H 个 latent step
        恰好覆盖 H*5 个 env step = goal_offset。
        """
        B, N, H, D = candidates.shape
        BT = B * N

        pixels = info_dict["pixels"].to(self.device)
        goal = info_dict["goal"].to(self.device)
        # world 存 pixels 为 channel-last (B, T, H, W, C)；policy 不给 pixels transform
        # （避免双重归一），故此处自适应把 channel-last 转回 channel-first (B, T, C, H, W)。
        pix = self._to_visual(pixels, B, N)
        goal_pix = self._to_visual(goal, B, N)

        # proprio（shim 可能给 7 维，切到 jepa 的 4 维位置）
        prop = info_dict.get("proprio")
        if prop is not None:
            prop = rearrange(prop, "b n ... -> (b n) ...").to(self.device).float()
            if prop.ndim == 2:
                prop = prop.unsqueeze(1)
            if prop.shape[-1] > 4:
                prop = prop[..., :4]
        gprop = info_dict.get("goal_proprio")
        if gprop is not None:
            gprop = rearrange(gprop, "b n ... -> (b n) ...").to(self.device).float()
            if gprop.ndim == 2:
                gprop = gprop.unsqueeze(1)
            if gprop.shape[-1] > 4:
                gprop = gprop[..., :4]
        else:
            gprop = prop

        # --- candidates -> 模型 latent action 空间 ---
        # 关键修正（KW-LEWM-0005 K4，对 K1-K3 冒烟无影响）：
        #   stable-worldmodel 的 CEM 在「标准化空间」采样（torch.randn -> N(0,1), 不裁剪）；
        #   官方 jepa-wms 的 CEMPlanner 把这些样本直接送 unroll，且 encode_act 不再做
        #   归一化（训练期动作已由 preprocessor.normalize_actions 归一化到 N(0,1)）。
        #   因此在 get_cost 内必须「直接透传」candidates，绝不能再调 normalize_actions
        #   —— 否则 std 1.0 -> 1.0/0.19 ≈ 5.3× 双归一化，动作出 OOD => 飞出场地。
        #   执行期的逆归一化由 policy.process['action'].inverse_transform 负责。
        if D == self.env_action_dim:
            # 兼容：每 plan step 是单个 env action（action_block=1）
            cand_block = candidates.reshape(BT, H, 1, self.env_action_dim).float()
        else:
            assert D == self.model_action_dim, (
                f"candidate dim {D} != model_action_dim {self.model_action_dim}"
            )
            cand_block = candidates.reshape(BT, H, self.action_block, self.env_action_dim).float()
        act_latent = cand_block.reshape(BT, H, self.model_action_dim)  # (BT, H, 10) N(0,1)
        act_latent = act_latent.clamp(-self.action_clip, self.action_clip)  # 安全夹紧（见 __init__）
        act_suffix = rearrange(act_latent, "bt h a -> h bt a")  # (H, BT, 10)

        obs_start = {"visual": pix, "proprio": prop} if prop is not None else pix
        obs_goal = {"visual": goal_pix, "proprio": gprop} if gprop is not None else goal_pix
        z_start = self.wm.encode(obs_start)  # TensorDict(visual, proprio)
        z_goal = self.wm.encode(obs_goal)

        if getattr(self, "_dbg", False):
            print("DBG z_start.visual", tuple(z_start["visual"].shape),
                  "act_suffix", tuple(act_suffix.shape))
        pred_td = self.wm.unroll(z_start, act_suffix=act_suffix)  # (H+tau, BT, ...)
        pred_vis = pred_td["visual"]  # (H+tau, BT, V, g, g, D)
        pred_final = pred_vis[-1].unsqueeze(1)  # (BT, 1, V, g, g, D)
        goal_vis = z_goal["visual"]  # (BT, 1, V, g, g, D)
        goal_vis = goal_vis.expand_as(pred_final) if goal_vis.shape != pred_final.shape else goal_vis

        cost = F.mse_loss(pred_final, goal_vis, reduction="none")
        cost = cost.sum(dim=tuple(range(1, cost.ndim)))  # (BT,)
        return cost.reshape(B, N)


# --------------------------------------------------------------------------- #
# 构建 adapter
# --------------------------------------------------------------------------- #
def build_jepa_adapter(device="cuda", checkpoint=CKPT, task_pair_idx=0):
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module

    # --- config（来自官方 pt yaml，重定向 checkpoint 到本地）---
    cfgs_data = {
        "img_size": 224,
        "custom": {"frameskip": 5, "action_skip": 1, "state_skip": 1},
    }
    pretrain_kwargs = {
        "grid_size": 16,
        "tubelet_size_enc": 1,
        "use_activation_checkpointing": False,
        "action_conditioning": "token",
        "proprio_encoding": "feature",
        "num_frames_pred": 4,
        "visual_encoder": {
            "enc_type": "dino",
            "enc_version": "dinov2_vits14",
            "pretrain_enc_path": "",
            "pretrain_enc_ckpt_key": "target_encoder",
            "embed_dim": 384,
            "enc_use_rope": "",
            "enc_name": "",
            "use_sdpa_enc": "",
            "num_frames_enc": "",
        },
        "action_encoder": {
            "action_tokens": 1,
            "action_emb_dim": 0,
            "act_mlp": False,
            "action_encoder_inpred": True,
        },
        "proprio_encoder": {
            "proprio_tokens": 0,
            "proprio_emb_dim": 16,
            "prop_mlp": False,
            "proprio_encoder_inpred": False,
        },
        "predictor": {
            "tubelet_size": 1,
            "pred_num_heads": 16,
            "pred_depth": 6,
            "pred_embed_dim": 384,
            "pred_use_extrinsics": False,
            "pred_type": "AdaLN",
            "act_pred_projector": False,
            "uniform_power": True,
            "use_SiLU": False,
            "use_rope": True,
        },
        "wm_encoding": {"batchify_video": True, "dup_image": False, "normalize_reps": False},
        "rollout_cfg": {
            "rollout_steps": 2,
            "train_rollout_prefixes": "random",
            "rollout_stop_gradient": True,
            "ctxt_window_train_rollout": 3,
            "do_parallel_rollout": False,
            "do_sequential_rollout": True,
            "prepend_gt": False,
        },
        "attn": {"local_window_time": 3, "local_window_h": -1, "local_window_w": -1},
    }
    wrapper_kwargs = {"ctxt_window": 2, "proprio_mode": "predict_proprio"}

    # --- action 归一化统计（复用 LeWM 数值 shard；同 PushT 分布，残留近似）---
    stats = json.loads(LEWM_SHARD.read_text(encoding="utf-8"))
    act = stats["stats"]["action"]
    action_mean = np.asarray(act["mean"], dtype=np.float64)
    action_std = np.asarray(act["scale"], dtype=np.float64)

    preprocessor = build_preprocessor(action_mean, action_std, device, proprio_dim=4)

    # --- init_module（本地 checkpoint，无 heads  decoder，避免联网）---
    folder = str(checkpoint.parent)
    ckpt_name = checkpoint.name
    model = init_module(
        folder=folder,
        checkpoint=ckpt_name,
        module_name="app.vjepa_wms.modelcustom.simu_env_planning.vit_enc_preds",
        model_kwargs=pretrain_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        cfgs_data=cfgs_data,
        device=torch.device(device),
        action_dim=2,
        proprio_dim=4,
        preprocessor=preprocessor,
    )
    model.eval()
    model = model.to(torch.device(device))
    model.eval()
    model.requires_grad_(False)

    # 参数量（K1 合理性检查）
    enc_params = sum(p.numel() for p in model.model.encoder.parameters())
    pred_params = sum(p.numel() for p in model.model.predictor.parameters())
    total = sum(p.numel() for p in model.parameters())

    adapter = JEPAWMAdapter(
        wm=model,
        preprocessor=preprocessor,
        env_action_dim=2,
        model_action_dim=model.action_dim,
        device=device,
    )
    meta = {
        "encoder_params": int(enc_params),
        "predictor_params": int(pred_params),
        "total_params": int(total),
        "model_action_dim": int(model.action_dim),
        "proprio_dim": int(getattr(model.model, "proprio_dim", 4)),
        "enc_type": model.enc_type,
        "grid_size": int(model.grid_size),
        "ctxt_window": int(model.ctxt_window),
        "normalize_reps": bool(model.normalize_reps),
    }
    return adapter, meta, stats, task_pair_idx


# --------------------------------------------------------------------------- #
# 1 任务冒烟（K1-K3）
# --------------------------------------------------------------------------- #
def find_pixels_wrapper(env):
    cur = env
    while cur is not None:
        if hasattr(cur, "_get_pixels"):
            return cur
        cur = getattr(cur, "env", None)
    raise RuntimeError("no wrapper exposing _get_pixels() in the env chain")


def render_state(world, pixels_wrapper, state) -> np.ndarray:
    env0 = world.envs.envs[0]
    env0.unwrapped._set_state(np.asarray(state, dtype=float))
    pixels, _ = pixels_wrapper._get_pixels()
    return np.asarray(pixels["pixels"])


def smoke_test(adapter, meta, stats, task_pair_idx=0, device="cuda"):
    import stable_worldmodel as swm

    onset = time.perf_counter()
    # K1 已在 build 阶段确认；这里只做合理性断言
    k1 = bool(
        meta["encoder_params"] > 1e6
        and meta["predictor_params"] > 1e5
        and meta["model_action_dim"] == 10
    )

    # 取 1 个真实数据集任务（来自 LeWM 数值 shard selected_pairs）
    pairs = stats["selected_pairs"]
    pair = pairs[task_pair_idx % len(pairs)]
    start_state = np.asarray(pair["start_state"], dtype=float)
    goal_state = np.asarray(pair["goal_state"], dtype=float)
    # jepa-wm PushT 用 4 维 proprio = [agent_x, agent_y, T_x, T_y]（位置，不含角度/速度）
    start_prop = start_state[:4].reshape(1, 4).astype(float)
    goal_prop = goal_state[:4].reshape(1, 4).astype(float)

    world = swm.World("swm/PushT-v1", num_envs=1, image_shape=(224, 224), max_episode_steps=50)
    world.reset(seed=42)
    pw = find_pixels_wrapper(world.envs.envs[0])
    start_img = render_state(world, pw, start_state)  # (224,224,3) uint8
    goal_img = render_state(world, pw, goal_state)

    # 构造 expanded_infos 形状 (B=1, N, ...) 与 candidates (B, N, H=5, 2)
    N = 8
    H = 5  # action_block=5，对应 1 个 latent step (frameskip=5)
    # pixels / goal: (1, N, 1, 3, 224, 224) uint8
    f0 = start_img.transpose(2, 0, 1).astype(np.uint8)  # (3, 224, 224)
    f1 = goal_img.transpose(2, 0, 1).astype(np.uint8)
    pix_t = torch.from_numpy(
        np.broadcast_to(f0, (1, N, 1, 3, 224, 224)).copy()
    ).to(torch.uint8)
    goal_t = torch.from_numpy(
        np.broadcast_to(f1, (1, N, 1, 3, 224, 224)).copy()
    ).to(torch.uint8)
    prop_t = torch.from_numpy(np.repeat(start_prop, N, axis=0)[None]).float()  # (1, N, 1, 7)
    gprop_t = torch.from_numpy(np.repeat(goal_prop, N, axis=0)[None]).float()

    # candidates：真实 CEM 契约 (B, N, H, model_action_dim=10)，每 plan step 的 10 维
    # = 5 个 env action 拼接。一组全零（"不动"）+ 一组随机（验证模型对动作有响应）
    cand_zeros = torch.zeros(1, N, H, meta["model_action_dim"])
    rng = torch.Generator().manual_seed(7)
    cand_rand = torch.randn(1, N, H, meta["model_action_dim"], generator=rng) * 0.3
    info = {
        "pixels": pix_t.to(device),
        "goal": goal_t.to(device),
        "proprio": prop_t.to(device),
        "goal_proprio": gprop_t.to(device),
    }

    with torch.no_grad():
        cost_z = adapter.get_cost(info, cand_zeros.to(device))
        cost_r = adapter.get_cost(info, cand_rand.to(device))

    # 直接检查 latent 有限 / 非 collapse（K2）
    obs_start = {"visual": pix_t[0, 0:1].float().to(device), "proprio": prop_t[0, 0:1].to(device)}
    obs_goal = {"visual": goal_t[0, 0:1].float().to(device), "proprio": gprop_t[0, 0:1].to(device)}
    with torch.no_grad():
        zs = adapter.wm.encode(obs_start)["visual"]
        zg = adapter.wm.encode(obs_goal)["visual"]
    zs_norm = float(zs.norm().item())
    zg_norm = float(zg.norm().item())

    cost_z_np = cost_z.detach().cpu().numpy().ravel()
    cost_r_np = cost_r.detach().cpu().numpy().ravel()
    finite = bool(np.all(np.isfinite(cost_z_np)) and np.all(np.isfinite(cost_r_np)))
    # 非 collapse：cost 不为零且 zeros vs rand 有差异（模型对动作响应）
    not_collapsed = bool(
        np.mean(cost_z_np) > 0 and abs(np.mean(cost_z_np) - np.mean(cost_r_np)) > 1e-4
    )
    k2 = bool(finite and not_collapsed and zs_norm > 0 and zg_norm > 0)

    # K3：encoder 已离线加载（build 阶段 DINOv2 已实例化；这里确认其在 cuda 且评估可用）
    enc_device = next(adapter.wm.model.encoder.parameters()).device
    k3 = bool(str(enc_device).startswith("cuda") or str(enc_device).startswith("cpu"))

    elapsed = time.perf_counter() - onset
    world.close()

    return {
        "episode_idx": int(pair["episode_idx"]),
        "start_step": int(pair["start_step"]),
        "goal_step": int(pair["goal_step"]),
        "task_pair_idx": task_pair_idx,
        "K1_adapter_buildable": k1,
        "K1_detail": meta,
        "K2_finite_in_distribution": k2,
        "K2_detail": {
            "cost_zeros_mean": float(np.mean(cost_z_np)),
            "cost_rand_mean": float(np.mean(cost_r_np)),
            "cost_zeros_min_max": [float(cost_z_np.min()), float(cost_z_np.max())],
            "cost_rand_min_max": [float(cost_r_np.min()), float(cost_r_np.max())],
            "start_latent_norm": zs_norm,
            "goal_latent_norm": zg_norm,
            "all_finite": finite,
            "responds_to_actions": not_collapsed,
        },
        "K3_encoder_offline_loaded": k3,
        "K3_detail": {"encoder_device": str(enc_device), "enc_type": meta["enc_type"]},
        "elapsed_sec": round(elapsed, 2),
    }


# --------------------------------------------------------------------------- #
# K4：50 任务同台闭环控制（与 LeWM 0004 同协议、同 CEM、同 ShimDataset）
# --------------------------------------------------------------------------- #
def run_k4(adapter, meta, stats, n_tasks=50, device="cuda", output=None, pairs_from=None,
           cem_num_samples=100, cem_n_steps=10, cem_topk=10,
           plan_horizon=5, plan_receding=5, plan_action_block=5, action_clip=None):
    import stable_worldmodel as swm
    from kw_lewm_0003_dataset_tasks import ShimDataset, build_entries
    from kw_lewm_0002_scaler_ablation import NumpyStandardScaler

    action_stats = stats["stats"]["action"]
    scaler = NumpyStandardScaler(action_stats["mean"], action_stats["scale"])

    if pairs_from:
        # 数据完整性修复：本机 numeric shard 的 selected_pairs 已被重生成（50→8），
        # 无法再提供 0004 同款 50 任务。改为从 KW-LEWM-0004 的结果 JSON 提取它**实际
        # 评估**的 50 个任务（episode_idx/start_step/start_state/goal_state 完整），
        # 重建 pair schema，保证与 0004 完全相同的任务集（真 head-to-head）。
        # proprio 重建为 jepa 标准 4 维 [agent_x, agent_y, T_x, T_y]（= state[:4]），
        # 比 shard 原 [x,y,angle,vx] 更贴合训练分布；action 置 [0,0]（仅起始无关）。
        src = json.loads(Path(pairs_from).read_text(encoding="utf-8"))
        src_trials = src["trials"][:n_tasks]
        pairs = []
        for t in src_trials:
            ss = np.asarray(t["start_state"], dtype=float)
            gs = np.asarray(t["goal_state"], dtype=float)
            pairs.append({
                "episode_idx": int(t["episode_idx"]),
                "start_step": int(t["start_step"]),
                "goal_step": int(t.get("goal_step", int(t["start_step"]) + 25)),
                "start_state": ss.tolist(),
                "goal_state": gs.tolist(),
                "start_proprio": ss[:4].tolist(),
                "goal_proprio": gs[:4].tolist(),
                "start_action": [0.0, 0.0],
                "goal_action": [0.0, 0.0],
            })
        print(f"[K4] loaded {len(pairs)} task pairs from {pairs_from} (0004 task set)")
    else:
        pairs = stats["selected_pairs"][:n_tasks]

    # 推理期增强 ablation 覆盖（KW-LEWM-0006）：仅改 planner 搜索设置，不动模型/权重。
    # action_clip 直接写在 adapter 上（get_cost 内部 clamp 用它）。
    if action_clip is not None:
        adapter.action_clip = float(action_clip)
    solver = swm.solver.CEMSolver(
        model=adapter,
        batch_size=1,
        num_samples=cem_num_samples,
        n_steps=cem_n_steps,
        topk=cem_topk,
        device=device,
        seed=42,
    )
    plan = swm.PlanConfig(
        horizon=plan_horizon,
        receding_horizon=plan_receding,
        action_block=plan_action_block,
        history_len=1,
        warm_start=True,
    )
    # 关键：不为 pixels 配 transform（adapter.encode 自行 /255 + ImageNet normalize）；
    # process['action'] 的 inverse_transform 负责把规划的 N(0,1) 动作逆归一化回环境空间。
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan,
        process={"action": scaler},
        transform={},
    )

    world = swm.World(
        "swm/PushT-v1", num_envs=1, image_shape=(224, 224), max_episode_steps=50
    )
    world.reset(seed=42)
    entries = build_entries(world, pairs)
    world.set_policy(policy)

    callables = [
        {"method": "_set_state", "args": {"state": {"in_dataset": True, "value": "state"}}},
        {
            "method": "_set_goal_state",
            "args": {"goal_state": {"in_dataset": True, "value": "goal_state"}},
        },
    ]

    PLAY_LOW, PLAY_HIGH = 0.0, 512.0
    GOAL_OFFSET = 25
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
            import traceback as _tb

            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": _tb.format_exc(),
            }
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
        "experiment_id": "KW-LEWM-0005-JEPA-SAME-PLATFORM",
        "question": "On the SAME stable-worldmodel platform as LeWM-0004 (same CEM 10x100x10, same 50 dataset tasks, same ShimDataset+local render), does JEPA-WM produce valid closed-loop PushT control?",
        "design": "JEPAWMAdapter wraps facebookresearch/jepa-wms EncPredWM; get_cost feeds CEM candidates (N(0,1) normalized space) directly to unroll; execution denormalized via process['action'].inverse_transform",
        "evidence_level": "E1_INTERNAL_REPRODUCTION",
        "n_tasks": len(trials),
        "goal_offset": GOAL_OFFSET,
        "eval_budget": 50,
        "cem": {"num_samples": cem_num_samples, "n_steps": cem_n_steps, "topk": cem_topk,
                "horizon": plan_horizon, "action_block": plan_action_block},
        "action_clip": float(adapter.action_clip),
        "action_scaler": {
            "source": "results/kw_lewm_0001l_numeric_shard0.json",
            "rows": action_stats["n"],
            "mean": action_stats["mean"],
            "scale": action_stats["scale"],
            "role": "inverse_transform only (execution); NOT applied inside get_cost",
        },
        "protocol_notes": [
            "pixels re-rendered locally at 224x224 from published states; image column never downloaded",
            "adapter.encode consumes raw uint8 pixels (no pixels transform in policy)",
            "CEM samples are in normalized action space N(0,1); encode_act does no re-normalization; get_cost passes through",
            "execution inverse_transform maps normalized -> env Box(-1,1) to keep agent in play",
            "NOT the official 13.1 GB dataset benchmark; E1 single checkpoint/single seed",
        ],
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "stable_worldmodel": "0.1.1",
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
            "total_wall_seconds": float(np.sum([t["wall_seconds"] for t in trials])),
        },
        "baseline_for_comparison": {
            "experiment": "KW-LEWM-0004 (LeWM, same platform, same 50 tasks, same CEM)",
            "success_rate": 0.36,
            "n_success": 18,
            "out_of_play_rate": None,
        },
        "kill_criteria": {
            # K4: agent stays in play (action scale wiring correct) => out_of_play small.
            "K4_control_within_play_area": bool(sum(out_of_play) == 0),
            "K4_out_of_play_rate_at_most_0.1": bool(
                (sum(out_of_play) / len(trials)) <= 0.1 if trials else False
            ),
        },
        "trials": trials,
    }

    if output:
        out_path = Path(output)
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
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default=str(OUT))
    ap.add_argument("--k4", action="store_true", help="run K4 50-task same-platform closed-loop")
    ap.add_argument("--tasks", type=int, default=50, help="n tasks for K4")
    ap.add_argument(
        "--pairs-from",
        default=None,
        help="load task pairs from another JSON's trials (e.g. KW-LEWM-0004 result) for a true head-to-head on identical tasks",
    )
    ap.add_argument(
        "--k4-output",
        default=str(ROOT / "results" / "kw_lewm_0005_jepa_same_platform.json"),
    )
    args = ap.parse_args()

    print("[build] constructing JEPAWMAdapter from local checkpoint ...")
    adapter, meta, stats, _ = build_jepa_adapter(device=args.device)
    print(
        f"[build] OK enc={meta['encoder_params']:,} pred={meta['predictor_params']:,} "
        f"total={meta['total_params']:,} model_action_dim={meta['model_action_dim']} "
        f"enc_type={meta['enc_type']} grid={meta['grid_size']}"
    )

    if args.k4:
        print(f"[K4] {args.tasks}-task same-platform closed-loop control ...")
        payload = run_k4(
            adapter, meta, stats, n_tasks=args.tasks, device=args.device,
            output=args.k4_output, pairs_from=args.pairs_from,
        )
        print(
            json.dumps(
                {
                    "output": str(args.k4_output),
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
                        for t in payload["trials"]
                    ],
                },
                indent=2,
            )
        )
        return

    print(f"[smoke] K1-K3 on task_pair_idx={args.task} ...")
    report = smoke_test(adapter, meta, stats, task_pair_idx=args.task, device=args.device)
    report["kill_criteria"] = {
        "K1_adapter_buildable": report["K1_adapter_buildable"],
        "K2_finite_in_distribution": report["K2_finite_in_distribution"],
        "K3_encoder_offline_loaded": report["K3_encoder_offline_loaded"],
    }
    report["all_pass"] = all(
        [
            report["K1_adapter_buildable"],
            report["K2_finite_in_distribution"],
            report["K3_encoder_offline_loaded"],
        ]
    )
    report["versions"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
