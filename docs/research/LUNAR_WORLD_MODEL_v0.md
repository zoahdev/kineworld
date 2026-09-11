# 月球世界模型 v0 — 能严谨做到的那一种

- 实验编号：`KW-MOON-0001`
- 日期：2026-09-11
- 证据等级：**E1**（单一解析模型、单一数据源、**无第三方验证、无外部标定**）
- 预注册：`verification/experiments/KW-MOON-0001.md`（判据 K1–K7 在计算前锁定）
- 校验结果：`results/kw_moon_0001_validation.json`
- 本文件所有数字由 `verification/scripts/verify_lunar_report.py` 机器生成与核对，**不手抄**

---

## 1. 它是什么，不是什么

本仓库实现的月球世界模型是一个**解析式（analytic）世界模型**，即显式的状态转移函数：

```
s_{t+1} = f(s_t, a_t)
s = (x, y, ψ, v, battery_Wh, t_s)     局部东向/北向坐标、航向、速度、电量、时间
a = (v_cmd, ψ_rate)                    速度指令、转向角速度
f = 公开刚体力学 + 真实地形几何（解析给出）
```

**它是**：一个能回答「我在这里这样开 60 秒，会到哪、耗多少电、会不会卡住」的预测器，
可用于基于模型的规划与任务推演。

**它不是**：
- **不是学习型世界模型**。本模型**没有神经网络、没有训练、不能生成图像**。
  在基于模型的规划文献中，"已知动力学"与"学习动力学"是并列的两类；本工作属于前者。
- **不是 Operational 工具**。见 §5 分辨率限制。
- **不能替代 NASA SPICE**、不能替代轮-土力学（terramechanics）仿真。

## 2. 为什么不做学习型月球世界模型

世界模型的学习信号是「动作 → 结果」的因果序列。月球上该数据**不存在**：

| 世界模型需要 | 月球实际有 | 判定 |
|---|---|---|
| 精确几何高程 | LOLA 全球 118 m；南极区域到 5 m/px | 充足 |
| 海量连续影像序列 | LROC 静态影像 100 万张以上、0.5 m/px | 形态不对（轨道器俯视静态图） |
| 动作→结果因果对 | 人类共在月面开过 4 台车；玉兔二号累计约 1.6 km | **唯一死穴** |

⇒ 训练不可行，**且不是算力问题，是观测次数问题**。

**旁证**：火星已有同类工作 `MarsGen` / `M3arsSynth`
（*Martian World Models*，NeurIPS 2025，arXiv 2507.07978，项目页 `marsgenai.github.io`）。
其做法正是「先造数据（处理 NASA PDS 立体导航影像 → 10K+ 米级精度 3D 表面 → 渲染视频序列）→
再微调视频扩散模型」。其论文自述瓶颈为「高质量火星数据稀缺」「公开外星影像多为稀疏视角静态立体像对」。
该工作**代码未开源**（2026-09-11 实测：GitHub 上 `marsgen` / `M3arsSynth` 名下仓库均为无关项目）。
**月球没有对等物。**

## 3. 物理模型（全部闭式，可独立验算）

### 3.1 爬坡上限：三个独立约束取最小值

| 约束 | 物理条件 | 闭式解 | 依赖 |
|---|---|---|---|
| (a) 牵引 | `sinθ + Crr·cosθ ≤ μ·cosθ` | `θ_t = arctan(μ − Crr)` | **与质量无关** |
| (b) 扭矩 | `m·g·(sinθ+Crr·cosθ)·r_w / n_w ≤ τ_wheel` | `θ_T = arcsin(A_T/R) − φ` | ∝ 1/m |
| (c) 功率 | `m·g·(sinθ+Crr·cosθ)·v/η ≤ P_max` | `θ_P(v) = arcsin(A_P/R) − φ` | ∝ 1/(m·v) |

其中 `R = √(1+Crr²)`、`φ = arctan(Crr)`、`A_T = n_w·τ_wheel/(m·g·r_w)`、`A_P = P_max·η/(m·g·v)`。

推导要点：把 `sinθ + Crr·cosθ` 写成 `R·sin(θ+φ)`，即可对 (b)(c) 求闭式反解。

**默认参数下由哪一个约束决定？** 见校验块 `limits.binding_constraint`。
默认参数给出 `θ_t = arctan(μ − Crr) = arctan(0.6000 − 0.1000) = 26.5651°`，
**与质量无关**，且与公开文献中「月球车受牵引限制、爬坡上限约 26° 量级」一致（本模型独立推出，非拟合）。

### 3.2 能量与速度

```
E_traction = m·g·(sinθ + Crr·cosθ)·d / η
E_total    = E_traction + P_avionics·dt
v_available = P_max·η / (m·g·(sinθ + Crr·cosθ))      受功率限制
```

电量更新 `battery -= E_total/3600`，日照时按 `P_solar·sin(太阳高度角)` 充电。

### 3.3 参数来源声明

`RoverParams` 全部数值为**设计假设**（质量取玉兔二号量级 140 kg；
μ、Crr、P_max、τ_wheel 取公开仿真工具的常见量级），**未经任何车辆实测标定**。
本仓库没有任何车辆数据。

### 3.4 真实地形

主数据源为 NASA PDS LOLA GDR 柱面栅格 `ldem_16.img`
（5760×2880，int16 小端，比例因子 0.5 m，16 ppd，33,177,600 字节，2026-09-11 实测可直连）。
坡度用中心差分（Horn 式），基线 = 2 个栅格。
**不提交进版本库**（沿用 `data/` 排除惯例），由 `kw_moon_0001_fetch_lola.py` 按需拉取。
无网络环境自动回落到合成陨石坑场（标记 `units=synthetic`，**不得用于任何真实月面结论**）。

> **本次运行的诚实披露**：下述 §4 的 K1–K7 结果是在 `ldem_16.img`
> **未就位**（`results/kw_moon_0001_validation.json` 中 `dem_source.lola_present = false`）
> 的情况下产出的。其中 K1–K4 是**闭式解析自洽校验**，与 DEM 完全无关；
> K5–K7 使用测试替身 `ScriptedDEM`（坡度与方位直接给定，用于把「门控/终止/确定性」
> 逻辑与 DEM 分辨率解耦）。因此**没有任何一条判据依赖真实 LOLA 高程**，
> 本文件也不含任何由真实月面高程推出的结论。
> 真实 LOLA 高程路径已实现并可复现（§7 步骤 1），但**尚未运行过**。

## 4. 判据与结果（K1–K7）

判据已在预注册文档 §2 锁定；下述数值由 `kw_moon_0001_validate.py` 自动产出。

<!-- VERIFY:START -->
```text
verdict = PASS
n_checks = 7
n_passed = 7
limits.binding_constraint = traction
limits.theta_traction_deg = 26.565051
limits.theta_limit_deg = 26.565051
checks.K2.tol = 1e-12
checks.K2.worst_abs_deviation_rad = 0.0
checks.K3.worst_rel_error = 2.220446049250313e-16
checks.K4.rel_error = 6.661338147750939e-16
checks.K4.distance_got_m = 3.000000
checks.K5.steep_slope_deg = 32.294629
checks.K5.steep_displacement_m = 0.0
checks.K6.termination = battery_depleted
checks.K6.final_wh = 0.0
checks.K7.identical = true
checks.K7.random_scenarios = 200
checks.K7.scenarios_with_nan_or_inf = 0
rover_params.mass_kg = 140.00
rover_params.mu = 0.6000
rover_params.crr = 0.1000
rover_params.p_max_w = 30.00
rover_params.t_wheel_nm = 12.00
rover_params.v_ref = 0.0500
```
<!-- VERIFY:END -->

**结论**：`PASS` — 7/7 项判据通过。

**过程中修正的测试实例（判据未改）**：K6（电池终止）首版把测试放在 `t_s = 朔望月/2`，
该时刻 (0,0) 处太阳高度角 90°，太阳能输入 1.0 Wh ≫ 单步消耗 0.004 Wh，电量只升不降，
**该实例由构造决定不可能通过**。读源码定位后改为无日照时刻（太阳高度角 -90.0°），
预注册判据本身**未做任何改动**。修正记录留在 JSON 的 `checks.K6.correction_note`。

## 5. 限制（预注册时即声明，不可事后减少）

1. **分辨率是本 v0 的致命限制。** 16 ppd ⇒ 赤道约 **1.9 km/像素**；坡度基线（2 栅格）约 **3.8 km**。
   这**远大于漫游车尺度**，因此本模型算出的坡度是**区域地形趋势**，**不是车轮级可通过性判据**。
   真实工程用 0.02 m（导航相机 DEM）至 5 m/px（NASA PGDA Product 78）级产品。
   ⇒ **换 DEM 即可升级，但在当前分辨率下，任何车轮级结论都无效。**
2. **滑移模型是最弱一环**：`slip = min(slip_max, k·(θ/θ_limit)²)` 是**参数化假设**，
   不是轮-土力学仿真，无实测标定。
3. **光照为简化星历**：忽略轨道偏心率与章动，幅度精度不足 10°，只可判断「在不在照明区」，
   **不可用于精密能量预算**。真实工具用 NASA SPICE。
4. **无地形自遮挡**：不做 ray-marching，极区永久阴影区会被误判为受照。
5. **非学习型**：无神经网络、无训练、不能生成图像。
6. **E1**：单一解析模型、单一数据源、**无第三方验证、无外部标定**。
   K1–K7 通过只说明**内部自洽**，**不等于**模型正确，更不等于被第三方验证。

## 6. 对标基线（既有开源工作）

| 项目 | 星数（2026-09-11 实测） | 与本工作的关系 |
|---|---|---|
| `osh3276/cynthium` | 0 | GPL-3.0。功能覆盖更广（A*/Dijkstra 路径规划、光照/温度/陨石通量代价、四轮滑移转向仿真、NASA SPICE）。**本工作未使用其任何代码**；`θ = min(牵引, 功率, 扭矩)` 这一约束结构与其文档所述一致，物理公式本身来自公开力学。**它是"工具"，本工作是"可校验方法骨架 + 判据"** |
| `jasmeet0915/artemis_mission_simulator` | 17 | Apache-2.0。Gazebo + ROS 2 仿真环境，可自动下载 NASA LOLA/LROC 生成地形。偏"环境"，本工作偏"可校验的转移函数" |
| `chandrabhraman/awesome-lunar-gnc-resources` | 5 | 月面 GNC 资源索引 |
| `MarsGen` / `M3arsSynth` | 未开源 | 火星的学习型世界模型（NeurIPS 2025）。**月球无对等物** |

**本工作的差异化不在功能覆盖**（那不如 Cynthium），而在**预注册判据 + 机器校验 + 显式限制**：
即把「这个模型到底对不对、对到什么程度」变成可复算的检查，而不是文档里的一句话。

## 7. 复现

```bash
# 1) 取真实高程（33 MB，可跳过 → 自动回落合成高程）
python verification/scripts/kw_moon_0001_fetch_lola.py

# 2) 跑判据 K1–K7，产出 results/kw_moon_0001_validation.json
python verification/scripts/kw_moon_0001_validate.py     # 期望 7/7 PASS, exit 0

# 3) 校验本文档数字与 JSON 一致（AGENTS.md 硬纪律 7）
python verification/scripts/verify_lunar_report.py --check   # 期望 OK, 0 fail

# 4) 推演一条轨迹
python verification/scripts/kw_moon_0001_lunar_wm.py --steps 20 --dt 60 --v-cmd 0.05
```

依赖：**仅 Python 标准库**（`array` / `math` / `struct` / `json` / `random` / `urllib`）。

## 8. 下一步（未做，不声称已做）

- 换 5 m/px 级 DEM（NASA PGDA Product 78）重跑，消除 §5.1 的分辨率限制。
- 把本模型作为**评测出题机**：用 DEM 生成「从 A 到 B 能否通行、代价多大」的题目与标准答案，
  让学习型规划模型来答 —— 这才是本仓库「评测方法论」身份的正位。
- 用公开的玉兔二号行驶数据做**外部标定**（当前完全缺失，是本工作最大的空缺）。
