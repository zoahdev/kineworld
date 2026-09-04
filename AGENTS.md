# AGENTS.md — KineWorld agent operating manual

任何 AI 编程代理（Codex / Claude Code / WorkBuddy 等）在本仓库工作前**必须读完本文件**。

## 0. 这个项目是什么

单机（RTX 5070 Ti Laptop, 12GB）物理 AI 世界模型**评测方法论**研究。当前阶段：基于 `facebookresearch/jepa-wms` 与 `stable-worldmodel`（LeWM）的官方 Push-T checkpoint 做诊断驱动研究。**没有训练**。

- 战略层：`docs/KINEWORLD_CONSTITUTION.md`（研究宪法，勿随意改）
- 单一事实来源：`company/EVIDENCE_LEDGER.md`（所有 claim 以台账为准，**含负面结果**）
  ⚠ `company/` **已被 .gitignore 排除**（含融资需求、股权口径、外联目标名单，不进版本控制）。
  从纯克隆环境工作时该目录不存在——此时以 `results/*.json` 证据工件 + `docs/research/` 报告为准。
- 对外主资产：`docs/research/METHODOLOGY_CONFOUNDS_v1.md`（所有数字受机器校验）
- 工作历史：`docs/WORKLOG.md`（append-only，每个工作日一条）

## 1. 硬纪律（违反 = 返工）

1. **证据分级**：当前所有结论为 **E1**（单 seed / 单 checkpoint / 单任务）。禁止把相关性写成因果，禁止声称复现官方 benchmark。
2. **实验三件套**：每个实验必须有 `verification/experiments/KW-*.md`（预注册）+ `verification/manifests/KW-*_manifest.json` + 固定 seed 脚本。先预注册判据，后计算。
3. **R0 判据自校验门禁**：复用既有日志做再分析的脚本，第一道门必须是用重构判据复现日志标签，<98% 直接判 VOID。纠错靠**读源码**，不是调阈值。
4. **配对键**：跨实验配对一律用三元组 `(episode_idx, start_step, goal_step)`。单 episode_idx 键会因采样重复丢对（实测 25→24）。
5. **多重比较**：检验 k 个变体必须报 Bonferroni 校正后 p（实例：0.0215×3=0.064，"显著"降级为"方向一致"）。
6. **统计表述**：重尾分布禁用均值（用中位数）；主判据必须与种子数 S 无关（用 Wilson 上界）；比方向进展前必须做等位移匹配。
7. **数字纪律**：对外文档的数字**必须机器校验不能手抄**（曾把 75.8468 抄成 75.9）。改 `docs/research/METHODOLOGY_CONFOUNDS_v1.md` 或其上游 JSON 后必跑：
   ```bash
   python verification/scripts/verify_methodology_report_v1.py   # 期望 56 ok / 0 fail, exit 0
   ```
8. **诚实边界**：团队是单人。禁止在任何材料里虚构合伙人/客户/收入/融资/外部复现/专家反馈。Level 1/2 校验通过 ≠ 第三方验证。
9. **vendored 代码**：`external/jepa-wms` 是 vendored 仓库，改动必须 `git diff` 存 `verification/patches/` 并记录 base commit（当前 `13cf1d9`）。

## 2. 环境坑（每个代理都会踩）

- **transformers 必须 4.x**（4.57.6 已验证）。5.x 的 ViT 结构与 LeWM 官方权重不兼容。
- **CEM 在标准化空间采样**：送环境前必须调 `process['action'].inverse_transform`，否则 agent 飞出场地。
- **Push-T success 判定**：`||goal[:4]−cur[:4]|| < 20` 且角度差 < π/9，`state[:4]` = [agent_x, agent_y, block_x, block_y] **联合预算**——不是只看方块。
- `near_object`/`obj_goal_dist`/`obj_lift` 在 Push-T 恒为占位值，不可用作信号。
- **HF 直连不通，用 `hf-mirror.com`**；parquet 端点支持 HTTP 206 Range 只读需要的列。
- 跑 jepa-wms 脚本前：`export JEPAWM_DSET/JEPAWM_LOGS/JEPAWM_HOME/JEPAWM_CKPT SDL_VIDEODRIVER=dummy WANDB_MODE=disabled`（见 `docs/WORKLOG.md` 与记忆中的常用命令块）。
- Python 双环境：`.venv310`（stable_worldmodel 线，Py3.10）/ 托管 venv（Py3.13，纯分析脚本）。

## 3. 当前状态（2026-09-03）

- LeWM lite 复现线闭环：0001L→0004（18/50=0.36，E1）。
- **弱动作条件证据链：五条独立证据线收敛**（0005c / block-only 反事实 / 0005d 三臂地板 / 等位移匹配 / 0006 干预 ablation）。0006 verdict = `CONFIRMS_SECTION_8`（推理期调参救不活）。0006b 越界率上升仅为**事后信号**（Bonferroni 后不显著）。
- 已关闭线程：frozen JEPA 上的廉价不确定性代理（4 候选全否定）。
- **版本控制状态（2026-09-04）**：已 git 化，公开仓库 `https://github.com/zoahdev/kineworld`。
  已排除：`company/`（融资/外联）、`data/`（9.7GB）、`checkpoints/`、`tools/`（uv 二进制）、`external/`（vendored，改动以 `verification/patches/` 形式保留）。
  ⚠ **本机 `git push` 会被环境静默拦截**（exit 0 但 stdout/stderr 全空，`GIT_TRACE` 也无输出），
  需改用 GitHub Git Data API 提交，见 `~/.workbuddy/skills/git-push-proxy-retry/`。
  注意 API 提交的 SHA 与本地 `git rev-parse HEAD` **不相等**（内容同、对象不同），校验用 `size_kb` / tree sha。

## 4. 修改本文件

只在与硬纪律冲突或环境坑清单需要增删时修改；改后须在 `docs/WORKLOG.md` 追加记录。策略问题改宪法，不改这里。
