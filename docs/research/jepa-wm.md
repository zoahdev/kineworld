# jepa-wms 调研报告（Phase 0 §6）

> 调研日期：2026-09-01。所有条目基于本会话实际核实（GitHub 仓库文件逐个读取、HF API、arXiv 元数据），非凭印象。

## 1. 项目标识
- 仓库：`facebookresearch/jepa-wms`（468★，2026-09-01 时点）
- 论文：arXiv **2512.24497**，《What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?》（LeCun 组）
- WMS = World ModelS：同一框架下系统对比多种世界模型设计选择

## 2. 解决的问题
什么设计选择让 JEPA 隐空间世界模型真正能做规划（planning）？系统消融：编码器（DINOv2 / V-JEPA-2 / DINOv3）、预测器深度、action conditioning、rollout 训练、proprio 等。

## 3. 提供的三类 baseline
1. **JEPA-WM**（自训复现，完整 pipeline）
2. **DINO-WM**（复现 Zhou et al. 的 DINO-WM）
3. **V-JEPA-2-AC**（Meta 官方 fixed checkpoint，仅评测）

## 4. 规划公式（评测核心）
目标条件轨迹优化：ẑ₀=E(s₀)，ẑ_{t+1}=P(ẑ_t,a_t)，代价 C = Σ‖E(s_g) − P(ẑ_t,a_t)‖₂（L1/L2，可选 α 加权）。Planner：CEM / GD / Nevergrad-NG / Adam。评测 episode = (s₀,s_g) 对。
指标：success rate、ep_end_dist、reward、total_emb_l2、total_lpips。

## 5. 环境
Push-T（本仓库自带 simu_env 实现，pygame/pymunk/shapely，Windows 可跑）、PointMaze（mujoco-py，Linux-only，排除）、Wall、Metaworld、RoboCasa（20GB 资产）、DROID（5.6–8.7TB 真机数据）。
**Phase 0 选 Push-T**：数据小（2.79GB）、模型小（0.20GB）、环境轻量、配置齐全。

## 6. Push-T 模型规格（官方 checkpoint `jepa_wm_pusht.pth.tar`）
- 编码器：DINOv2 ViT-S/14（frozen，`torch.hub.load("facebookresearch/dinov2")` 在线拉取，~88MB）
- 预测器：AdaLN，depth 6，embed 384，16 heads，RoPE，action token conditioning
- 输入 224×224，frameskip 5，ctxt_window 2
- checkpoint 大小 211,639,615 B（202MB），已下载校验 ✓

## 7. 评测入口
```
python -m evals.main --fname <eval_cfg.yaml> --debug   # 单卡 cuda:0
```
- 官方模板：`configs/online_plan_evals/pt/pt_L2_cem_sourcedset_H6_nas6_ctxt2.yaml`（CEM: 30 iter × 300 samples × H6，96 episodes）
- `meta.quick_debug: true` → eval_episodes=1（smoke 模式）
- 评测配置由 `build_plan_eval_args()`（app/vjepa_wm/utils.py:164）从训练配置+模板合并生成；仓库内 `configs/dump_online_evals/` 为空，需自行生成（已复刻合并逻辑：`kineworld/scripts/gen_pt_eval_configs.py`）
- 模型构建：`app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py::init_module`，checkpoint 支持绝对路径

## 8. 数据
- 官方 HF 数据集 `facebook/jepa-wms` 为 **gated: auto**（需账号同意）；模型 checkpoint **不 gated**
- Push-T 数据 = `pusht_noise.zip` 2,785,304,515 B，源自 DINO-WM 项目（`vovw/dino_wm`）OSF 托管 `https://osf.io/download/k2d8w/`，**逐字节同源**（OSF 大小与 HF 完全一致），免 gate
- 2000 条 noisy episodes，239,900 帧，30fps，obs 96×96 RGB（评测时 resize 224），action 2 维
- 路径约定：`$JEPAWM_DSET/pusht_noise`（src/utils/cluster.py:77）

## 9. 必需环境变量（setup_macros.py 生成 macros.py）
- `JEPAWM_DSET`（数据根，设为 kineworld/data）
- `JEPAWM_LOGS`（日志根）
- `JEPAWM_HOME`
- 可选 `JEPAWM_OSSCKPT`（仅 DINOv3 需要）、`JEPAWM_CKPT`

## 10. 依赖（pyproject.toml 全量）与 Windows 最小化策略
- requires-python `>=3.10,<3.11`；torch>=2.7.0，torchvision==0.22.0（cu128 wheel 支持 Blackwell sm_120）
- Windows 阻塞项：`torchcodec<=0.5`（无 Windows wheel）、`decord`、`d4rl`+`mujoco-py`（Linux-only）
- **最小化安装**（已执行）：torch 2.7.0+cu128 + timm/einops/h5py/gym==0.23.1/gymnasium/pygame/pymunk==6.8.0/shapely 等；torchcodec/decord/d4rl 延后（Push-T npz 路径不需要）
- 注意 gym==0.23.1 是老 API，与 gymnasium 并存

## 11. License
**CC BY-NC 4.0（Non-Commercial）** — 仅限研究/非商业。用于申报材料中的"复现官方 baseline"没有问题；商业化需另行授权或换 baseline。（DINO-WM 上游代码 MIT，但本仓库整体按 CC BY-NC 对待。）

## 12. 单机可行性结论（RTX 5070 Ti Laptop 12GB）
- 模型 0.20GB + DINOv2 ViT-S ~0.1GB → 推理显存充裕（baseline ~0.6–3GB，缩减 CEM 规划峰值 6.8GB，完整 CEM 峰值 ~11.9GB，见下）
- **实测瓶颈（2026-09-01，非估算）**：官方 CEM 30 iter × 300 samples × 10 elites × H6 规划，单次 `Action optim` 在 5070 Ti Laptop 上 **>18 分钟未完成**（step0 被手动停止；GPU util 100% 持续，VRAM 冲到 11.88GB/12.2GB=97%，笔记本功耗墙内 ~45W）。96 episodes 需 **>200h**，不可行。
- **原因**：CEM 每一步 = 30 iter × 300 samples × 6 步潜空间 rollout = 90,000 次 world-model 前向；Push-T 每 episode 30 决策步。
- **`quick_debug:true` 非有效 benchmark**：官方会把 CEM 降为 2 iter × 2 samples × 2 elites（eval.py:180-183），规划精度完全失真，仅用于 smoke 冒烟。
- **Phase 0 实测解法**：缩减 CEM（10×100×10，约为官方 1/9 计算量）→ 单次规划降至 **~37s**，VRAM 峰值 **6.8GB（56%）**。因 Push-T 每个 episode 只规划一次（`planning_calls=1`，非每 6 步重规划），5 个 episode 缩减 CEM 总计仅 3m19s → **正式 benchmark 采用缩减 CEM + 小样本（5 episodes）**，成功率视为量级参考（官方 96-episode 不可行）。
- 全量训练不可行（论文用 4 nodes×8 GPU），但 Phase 0 只需 evaluation → 可行（受 CEM 规划耗时约束，已用缩减 CEM 规避）

## 13. 与 KineWorld 的关系
- 这是 Phase 0 的 Action-conditioned baseline：验证"预测真正参与动作选择"的最小闭环样板（CEM 在隐空间 rollout）
- KW-001 的评测协议（success rate / 轨迹误差 / latency）直接沿用本仓库口径，保证可对比
- 风险：CC BY-NC 限制、DINOv2 hub 在线下载依赖网络、gym 0.23.1 老版本冲突（已隔离在专用 venv）

## 14. 官方报告数字（Push-T，论文/README 口径，待本机复现对照）
- JEPA-WM Push-T CEM L2 sourcedset：README/论文表格口径（本机复现结果另记，不在此混写）
- 本机实测结果一律写入 `results/experiments.csv` 与 phase0_report，不凭记忆
