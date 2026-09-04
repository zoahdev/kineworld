# DEPENDENCY_LICENSES.md — 第三方依赖许可证台账

> 日期：2026-09-02。分类：GREEN=Commercial OK / YELLOW=Need Review / RED=Non-Commercial 或不兼容。
> RED 可用于 research/benchmark/architecture reference，禁止默认进入商业 Core。

| 依赖 | License | 商业使用 | 修改 | 分发 | 署名 | Checkpoint License | Dataset License | 分类 |
|---|---|---|---|---|---|---|---|---|
| facebookresearch/jepa-wms（代码） | CC BY-NC 4.0 | ❌ 禁止 | ✅ | ✅（同许可） | ✅ 必须 | — | — | **RED** |
| jepa_wm_pusht.pth.tar（checkpoint） | 随仓库 CC BY-NC 4.0 | ❌ 禁止 | — | — | ✅ | 非商业 | — | **RED** |
| DINOv2（Meta，编码器权重） | CC BY-NC 4.0 | ❌ 禁止 | — | — | ✅ | 非商业 | — | **RED** |
| pusht_noise 数据集（jepa-wms 镜像自 OSF） | 待核实（上游 Push-T/diffusion policy 数据） | ⚠️ | — | — | — | — | 待核实 | **YELLOW** |
| danijar/dreamerv3 | MIT | ✅ | ✅ | ✅ | ✅ | — | — | GREEN |
| nicklashansen/tdmpc2 | MIT | ✅ | ✅ | ✅ | ✅ | — | — | GREEN |
| NM512/r2dreamer | MIT（随 ICLR'26 公开） | ✅ | ✅ | ✅ | ✅ | — | — | GREEN |
| Hugging Face LeRobot | Apache 2.0 | ✅ | ✅ | ✅ | ✅ | — | 各数据集各异 | GREEN（数据集需逐个查） |
| MuJoCo | Apache 2.0 | ✅ | ✅ | ✅ | ✅ | — | — | GREEN |
| NVIDIA Cosmos | NVIDIA Open Model License（允商业，有限制条款） | ⚠️ 需逐条审 | ✅ | ✅ | ✅ | 模型许可 | — | **YELLOW** |
| AMI Labs / NYU LeWorldModel（代码） | MIT | ✅ | ✅ | ✅ | ✅ | — | — | **GREEN** |
| quentinll/lewm-pusht（官方权重，72.3 MB） | MIT（HF model card） | ✅ | — | ✅ | ✅ | MIT | — | **GREEN**（仍需保留上游声明） |
| quentinll/lewm-pusht（官方训练数据，13.1 GB） | MIT（HF dataset card） | ✅ | — | ✅ | ✅ | — | MIT | **GREEN**（下载前仍做来源/隐私复核） |
| stable-worldmodel 0.1.1（LeWM 直接依赖） | MIT（官方 PyPI） | ✅ | ✅ | ✅ | ✅ | — | — | **GREEN**（extras 另审） |
| stable-pretraining 0.1.8（LeWM 直接依赖） | MIT（官方 PyPI） | ✅ | ✅ | ✅ | ✅ | — | — | **GREEN**（传递依赖另审） |

## 关键结论

1. **当前整个 Phase 0/EXP-0001 链路是 RED**：JEPA-WM 代码+权重+DINOv2 全部 CC BY-NC 4.0。研究、申报、benchmark 没问题；**任何商业产品/收费服务不得直接包含这些组件**。
2. 商业 Core 的未来路径（三选一，届时决策）：
   a. 换成 GREEN 许可 baseline（如 dreamerv3/tdmpc2 系，需重训）；
   b. 自研训练替代模型（1M→100M 路线，宪法 §6）；
   c. 向权利方获取商业授权。
3. KineWorld 自产的评测 harness/工具脚本：我们持有版权，可自由选择开放许可（建议 Apache 2.0，但需注意若其强依赖 RED 组件则实际使用场景受限）。
4. LeWM 顶层代码、权重、数据及两个直接框架依赖均标注 MIT；但完整 `stable-worldmodel[train,env]` 传递依赖尚未锁定，故**模型资产为 GREEN、整体运行时暂为 YELLOW**。必须在安装后导出精确依赖树并逐项审计，才能进入商业 Core。
