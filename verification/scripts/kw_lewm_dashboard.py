"""KW-LEWM 同台对比仪表盘生成器（单文件 HTML，离线、浅色主题、响应式卡片网格）。

读取已完成的两条评测线结果，生成可视化对比：
  - KW-LEWM-0004 (LeWM, lite 协议)  vs  KW-LEWM-0005 K4 (JEPA-WM, 同平台)
  - KW-LEWM-0005 §8 根因（弱动作条件）+ KW-LEWM-0005b 闭环佐证
  - KW-LEWM-0005c 配对行为学再分析（no-op 基线 / 同等位移相反符号 / 失败项归因）
输出 results/kw_lewm_head2head_dashboard.html（可直接双击打开）。

纯 CPU、不训练、E1。复现：python kw_lewm_dashboard.py
"""
from __future__ import annotations
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P_0004 = ROOT / "results" / "kw_lewm_0004_dataset_tasks_p50.json"
P_0005 = ROOT / "results" / "kw_lewm_0005_jepa_same_platform.json"
P_0005B = ROOT / "results" / "kw_lewm_0005b_k4_corroboration.json"
P_RC = ROOT / "results" / "kw_lewm_0005_rootcause.json"
P_0005C = ROOT / "results" / "kw_lewm_0005c_paired_noop_baseline.json"
P_0005D = ROOT / "results" / "kw_lewm_0005d_modelfree_reference.json"
OUT = ROOT / "results" / "kw_lewm_head2head_dashboard.html"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def pct(x):
    return f"{x*100:.0f}%"


def model_block(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    s = d["summary"]
    k = int(s["n_success"]); n = int(d["n_tasks"])
    ci = wilson(k, n)
    return {
        "n_tasks": n,
        "success": s["success_rate"],
        "n_success": k,
        "ci": [round(ci[0], 3), round(ci[1], 3)],
        "oop": s["out_of_play_rate"],
        "n_oop": int(s["n_out_of_play"]),
        "interaction": s["block_interaction_rate"],
        "disp_med": round(s["block_displacement_median"], 2),
        "disp_mean": round(s["block_displacement_mean"], 2),
        "n_errors": int(s["n_errors"]),
    }


def main():
    lewm = model_block(P_0004)
    jepa = model_block(P_0005)
    rc = json.loads(P_RC.read_text(encoding="utf-8"))
    corr = json.loads(P_0005B.read_text(encoding="utf-8"))

    # CI 是否重叠
    overlap = not (lewm["ci"][1] < jepa["ci"][0] or jepa["ci"][1] < lewm["ci"][0])
    # 这里 success 低=差；JEPA CI 上界 < LeWM CI 下界 -> JEPA 显著更低
    jepa_sig_lower = jepa["ci"][1] < lewm["ci"][0]

    data = {
        "lewm": lewm, "jepa": jepa,
        "ci_overlap": overlap, "jepa_sig_lower": jepa_sig_lower,
        "rootcause": {
            "verdict": rc["verdict"],
            "mean_ratio": round(rc["mean_ratio_max_mob_over_gap"], 2),
            "context_dom": round(rc["mean_context_domination_ratio"], 4),
            "action_leverage": round(rc["mean_action_leverage"], 1),
        },
        "corroboration": {
            "disp_vs_dist_spearman": corr["correlations"]["block_displacement_vs_goal_distance"]["spearman"],
            "succ_vs_dist_spearman": corr["correlations"]["success_vs_goal_distance"]["spearman"],
            "succ_vs_disp_spearman": corr["correlations"]["success_vs_block_displacement"]["spearman"],
            "near_but_failed": corr["near_but_failed"]["inplay_disp_gt_20_but_failed"],
            "near_total": corr["near_but_failed"]["inplay_disp_gt_20"],
            "oop_ratio": corr["out_of_play"]["goal_distance_ratio_oop_over_inplay_mean"],
        },
    }

    if P_0005C.exists():
        c = json.loads(P_0005C.read_text(encoding="utf-8"))
        pr = c["paired_progress_needs_work_stratum"]
        sr = c["success_rates"]["overall"]
        bo = c["counterfactual_block_only_criterion"]
        at = c["failure_term_attribution"]
        data["paired"] = {
            "verdict": c["verdict"],
            "n_tasks": c["pairing"]["n_tasks"],
            "criterion_agreement": c["criterion_validation"]["agreement_with_logged_success"],
            "solved_at_start": c["task_difficulty_audit"]["n_solved_at_start"],
            "noop": [sr["noop"]["k"], sr["noop"]["n"], round(sr["noop"]["lo"], 3), round(sr["noop"]["hi"], 3)],
            "mc_lewm_noop": c["paired_tests_exact_mcnemar"]["lewm_vs_noop"],
            "mc_jepa_noop": c["paired_tests_exact_mcnemar"]["jepa_vs_noop"],
            "mc_lewm_jepa": c["paired_tests_exact_mcnemar"]["lewm_vs_jepa"],
            "disp_l": round(pr["agent_disp"]["lewm"]["median"], 1),
            "disp_j": round(pr["agent_disp"]["jepa"]["median"], 1),
            "disp_diff": [round(pr["agent_disp"]["paired_diff_lewm_minus_jepa"]["mean"], 1),
                          round(pr["agent_disp"]["paired_diff_lewm_minus_jepa"]["lo"], 1),
                          round(pr["agent_disp"]["paired_diff_lewm_minus_jepa"]["hi"], 1)],
            "prog_l": round(pr["agent_progress"]["lewm"]["median"], 1),
            "prog_j": round(pr["agent_progress"]["jepa"]["median"], 1),
            "prog_diff": [round(pr["agent_progress"]["paired_diff_lewm_minus_jepa"]["mean"], 1),
                          round(pr["agent_progress"]["paired_diff_lewm_minus_jepa"]["lo"], 1),
                          round(pr["agent_progress"]["paired_diff_lewm_minus_jepa"]["hi"], 1)],
            "posprog_j": [round(pr["pos_progress"]["jepa"]["mean"], 1),
                          round(pr["pos_progress"]["jepa"]["lo"], 1),
                          round(pr["pos_progress"]["jepa"]["hi"], 1)],
            "attr_j": round(at["jepa"]["agent_position"] * 100),
            "attr_l": round(at["lewm"]["agent_position"] * 100),
            "agap_j": round(at["jepa"]["agent_gap_final_median"], 1),
            "agap_l": round(at["lewm"]["agent_gap_final_median"], 1),
            "agap_start": round(c["task_difficulty_audit"]["agent_gap_start_quartiles"][2], 1),
            "bo": {m: [bo[m]["k"], bo[m]["n"], round(bo[m]["lo"], 3), round(bo[m]["hi"], 3)] for m in ("lewm", "noop", "jepa")},
            "bo_p_jepa_noop": bo["mcnemar_jepa_vs_noop"]["p_exact_two_sided"],
            "bo_counts_jepa_noop": [bo["mcnemar_jepa_vs_noop"]["a_only"], bo["mcnemar_jepa_vs_noop"]["b_only"]],
        }

    if P_0005D.exists():
        m = json.loads(P_0005D.read_text(encoding="utf-8"))
        fl = m["model_free_floor"]
        arms = m["model_free_arms"]
        mc = m["paired_mcnemar_seed0"]
        me = m["matched_effort_control_POST_HOC"]
        fm = m["floor_mechanism"]
        an = m["analytic_noop_check"]
        ea = m["effort_axis"]

        def arm_rate(name):
            a = arms[name]
            w = a["success_pooled_wilson"] if "success_pooled_wilson" in a else None
            tl = a["success_task_level"]
            return {
                "rate": tl["rate"], "n_roll": tl["n_rollouts"],
                "task_lo": round(tl["lo"], 4), "task_hi": round(tl["hi"], 4),
                "pool_hi": round(w["hi"], 4) if w else None,
                "k": w["k"] if w else None,
                "disp": round(a["agent_disp"]["median_over_rollouts"], 2),
                "pos_lo": round(a["pos_progress"]["task_mean_ci"]["lo"], 2),
                "pos_hi": round(a["pos_progress"]["task_mean_ci"]["hi"], 2),
            }

        data["mf"] = {
            "verdict": m["verdict"],
            "verdict_post_hoc": m["verdict_post_hoc"],
            "n_tasks": m["n_tasks"], "n_seeds": m["n_seeds_stochastic_arms"],
            "U": round(fl["U"], 4),
            "jepa_lo_gt_U": fl["jepa_lo_gt_U"], "lewm_lo_gt_U": fl["lewm_lo_gt_U"],
            "jepa_w": [round(fl["jepa_wilson"]["lo"], 4), round(fl["jepa_wilson"]["hi"], 4), fl["jepa_wilson"]["p"]],
            "lewm_w": [round(fl["lewm_wilson"]["lo"], 4), round(fl["lewm_wilson"]["hi"], 4), fl["lewm_wilson"]["p"]],
            "arms": {k: arm_rate(k) for k in ("noop_zero_action", "cem_prior_random", "uniform_random")},
            "mc_j_cem": mc["jepa_vs_cem_prior_random_seed0"],
            "mc_l_cem": mc["lewm_vs_cem_prior_random_seed0"],
            "mc_l_noop": mc["lewm_vs_noop_zero_action_seed0"],
            "effort": {"jepa": round(ea["jepa_agent_disp_median"], 1),
                       "lewm": round(ea["lewm_agent_disp_median"], 1),
                       **{k: round(v, 1) for k, v in ea["model_free_agent_disp_medians"].items()}},
            "me_j_agent": [round(me["jepa"]["agent_progress_delta"]["mean"], 2),
                           round(me["jepa"]["agent_progress_delta"]["lo"], 2),
                           round(me["jepa"]["agent_progress_delta"]["hi"], 2)],
            "me_j_pos": [round(me["jepa"]["pos_progress_delta"]["mean"], 2),
                         round(me["jepa"]["pos_progress_delta"]["lo"], 2),
                         round(me["jepa"]["pos_progress_delta"]["hi"], 2)],
            "me_l_agent": [round(me["lewm"]["agent_progress_delta"]["mean"], 2),
                           round(me["lewm"]["agent_progress_delta"]["lo"], 2),
                           round(me["lewm"]["agent_progress_delta"]["hi"], 2)],
            "me_pool": int(me["jepa"]["matched_pool_size_median"]),
            "me_tol": me["jepa"]["tol_fraction"],
            "fm_guess_status": fm["initial_guess_status"],
            "fm_sp_block": round(fm["spearman_k_vs_block_gap_start"], 3),
            "fm_sp_agent": round(fm["spearman_k_vs_agent_gap_start"], 3),
            "fm_n_any": fm["n_tasks_with_any_random_success"],
            "noop_block_med": round(an["noop_block_disp_median"], 3),
            "noop_agent_med": round(an["noop_agent_disp_median"], 3),
            "noop_agent_max": round(an["noop_agent_disp_max"], 3),
            "noop_rolled": an["rolled_out_noop_successes"],
            "ppp": {k: round(v, 4) for k, v in m["progress_per_pixel_of_displacement"].items()},
        }

    html = build_html(data)
    OUT.write_text(html, encoding="utf-8")
    print(f"[dashboard] wrote {OUT} ({OUT.stat().st_size} bytes)")
    print(f"[dashboard] JEPA success {pct(jepa['success'])} CI {jepa['ci']} vs LeWM {pct(lewm['success'])} CI {lewm['ci']}")
    print(f"[dashboard] CI overlap={overlap} jepa_sig_lower={jepa_sig_lower}")
    if "mf" in data:
        mf = data["mf"]
        print(f"[dashboard] 0005d floor U={mf['U']} jepa_above={mf['jepa_lo_gt_U']} lewm_above={mf['lewm_lo_gt_U']}")
        print(f"[dashboard] 0005d verdict: {mf['verdict']}")
        print(f"[dashboard] 0005d post-hoc: {mf['verdict_post_hoc']}")
    else:
        print("[dashboard] WARNING: 0005d result not found, section omitted")


def diverging_bar(label, value, vmax, color):
    """0 在中轴的双向条。value 可正可负。"""
    frac = min(1.0, abs(value) / vmax) * 50.0
    if value >= 0:
        style = f"left:50%;width:{frac}%;background:{color};"
    else:
        style = f"right:50%;width:{frac}%;background:{color};"
    sign = "+" if value >= 0 else ""
    return f"""
      <div class="dbrow">
        <div class="dblab">{label}</div>
        <div class="dbtrack"><div class="dbaxis"></div><div class="dbbar" style="{style}"></div></div>
        <div class="dbval" style="color:{color}">{sign}{value}</div>
      </div>"""


def build_paired_section(p):
    if not p:
        return ""
    vmax = max(abs(p["prog_l"]), abs(p["prog_j"]), abs(p["disp_l"]), abs(p["disp_j"])) * 1.05
    bars = "".join([
        diverging_bar("LeWM · 智能体移动距离", p["disp_l"], vmax, "#9aa5b1"),
        diverging_bar("JEPA · 智能体移动距离", p["disp_j"], vmax, "#9aa5b1"),
        diverging_bar("LeWM · 智能体<b>朝目标</b>的进展", p["prog_l"], vmax, "#3b6fd4"),
        diverging_bar("JEPA · 智能体<b>朝目标</b>的进展", p["prog_j"], vmax, "#c0392b"),
    ])
    bo = p["bo"]
    return f"""
<section><h2>配对行为学再分析（KW-LEWM-0005c）：同等位移预算，相反进展符号</h2>
<div class="banner bad">
  <div class="v">两模型智能体移动距离统计不可区分，但进展方向符号相反</div>
  <div class="note">
    移动距离配对差 <b>{p['disp_diff'][0]}</b>，95% CI [{p['disp_diff'][1]}, {p['disp_diff'][2]}] <b>含 0</b>（能量消耗一样）；
    朝目标进展配对差 <b>+{p['prog_diff'][0]}</b>，CI [{p['prog_diff'][1]}, {p['prog_diff'][2]}] <b>排除 0</b>。
    JEPA 的 4 维联合进展均值 <b>{p['posprog_j'][0]}</b>，CI [{p['posprog_j'][1]}, {p['posprog_j'][2]}] 显著为负（no-op 恒为 0）。
    → 规划器不知道自己的动作把智能体带向哪个方向，正是弱动作条件的行为学签名。
  </div>
</div>
<div class="dbwrap">{bars}
  <div class="note">中位数（像素）。灰条＝走了多远，彩条＝朝目标缩小了多少差距。中轴为 0。</div>
</div>

<div class="rootgrid" style="margin-top:14px">
  <div class="rc"><div class="k">判据自校验（R0 门禁）</div><div class="vv">{pct(p['criterion_agreement'])}</div><div class="note">重构判据复现日志 success，100% 通过</div></div>
  <div class="rc"><div class="k">t=0 已满足容差的任务</div><div class="vv">{p['solved_at_start']}/{p['n_tasks']}</div><div class="note">no-op 基线 = 0/50 → 0004 的 0.36 非虚高</div></div>
  <div class="rc"><div class="k">JEPA 失败由智能体位置项主导</div><div class="vv">{p['attr_j']}%</div><div class="note">LeWM 仅 {p['attr_l']}%</div></div>
  <div class="rc"><div class="k">JEPA 终端智能体差距（中位）</div><div class="vv">{p['agap_j']}</div><div class="note">起始 {p['agap_start']} → 被<b>拉大</b>；LeWM 收到 {p['agap_l']}</div></div>
</div>

<div class="grid" style="margin-top:14px">
  <div class="metric"><div class="metric-h">真实判据下的 no-op 基线（配对精确 McNemar）</div>
    <table style="border:0"><tbody>
      <tr><td>LeWM vs no-op</td><td><b>p={p['mc_lewm_noop']['p_exact_two_sided']:.2g}</b>（{p['mc_lewm_noop']['a_only']} vs {p['mc_lewm_noop']['b_only']}）</td><td class="sub">显著优于不动 → 真控制</td></tr>
      <tr><td>JEPA vs no-op</td><td><b>p={p['mc_jepa_noop']['p_exact_two_sided']:.3g}</b>（{p['mc_jepa_noop']['a_only']} vs {p['mc_jepa_noop']['b_only']}）</td><td class="sub">与「什么都不做」统计不可区分</td></tr>
      <tr><td>LeWM vs JEPA</td><td><b>p={p['mc_lewm_jepa']['p_exact_two_sided']:.2g}</b>（{p['mc_lewm_jepa']['a_only']} vs {p['mc_lewm_jepa']['b_only']}）</td><td class="sub">LeWM 严格支配，无一例 JEPA 赢</td></tr>
    </tbody></table>
  </div>
  <div class="metric"><div class="metric-h">仅方块判据反事实（诊断分解，非基准）</div>
    <table style="border:0"><tbody>
      <tr><td>LeWM</td><td><b>{bo['lewm'][0]}/{bo['lewm'][1]}</b></td><td class="sub">CI [{bo['lewm'][2]}, {bo['lewm'][3]}]</td></tr>
      <tr><td>no-op</td><td><b>{bo['noop'][0]}/{bo['noop'][1]}</b></td><td class="sub">CI [{bo['noop'][2]}, {bo['noop'][3]}]</td></tr>
      <tr><td>JEPA</td><td><b style="color:var(--bad)">{bo['jepa'][0]}/{bo['jepa'][1]}</b></td><td class="sub">CI [{bo['jepa'][2]}, {bo['jepa'][3]}]</td></tr>
    </tbody></table>
    <div class="note">有余量处 JEPA <b>显著劣于不动</b>：McNemar p={p['bo_p_jepa_noop']:.4g}（{p['bo_counts_jepa_noop'][0]} vs {p['bo_counts_jepa_noop'][1]}）——破坏 8 个 no-op 本可满足的任务，零挽救。故失效同时覆盖导航与操作。</div>
  </div>
</div>
<div class="note" style="margin-top:8px">⚠ 须留的细微处：LeWM 的方块位置进展均值 +10.2、CI [−4.6, +24.7] <b>含 0</b>。其方块层优势主要来自修角度 + 不破坏已摆好的方块，<b>不可讲成「LeWM 能精确推物体到位」</b>。</div>
<div class="note" style="margin-top:6px">⚠ 已被 <b>KW-LEWM-0005d 修正</b>：本节的 LeWM vs JEPA 配对结论（同等位移、相反符号）不变，但「JEPA 主动远离目标」的措辞须改为「<b>JEPA 的进展与等位移随机游走不可区分</b>」——见下一节等位移匹配对照。本节 no-op 为<b>解析</b>基线，已由 0005d 的真实 rollout 验证成立（0/50、方块位移中位 0.0）。</div>
</section>"""


def hbar(label, value, vmax, color, right_txt):
    """左对齐单向条（用于成功率等非负量）。"""
    frac = 0.0 if vmax <= 0 else min(1.0, value / vmax) * 100.0
    return f"""
      <div class="hbrow">
        <div class="hblab">{label}</div>
        <div class="hbtrack"><div class="hbbar" style="width:{frac:.2f}%;background:{color}"></div></div>
        <div class="hbval" style="color:{color}">{right_txt}</div>
      </div>"""


def build_modelfree_section(m):
    """KW-LEWM-0005d：无模型参考臂（真实 CPU rollout）。"""
    if not m:
        return ""
    A = m["arms"]
    floor_pct = max(A["cem_prior_random"]["rate"], A["uniform_random"]["rate"])
    ratio = m["lewm_w"][2] / floor_pct if floor_pct else float("nan")
    vmax = max(m["lewm_w"][2], 0.05) * 1.05
    bars = "".join([
        hbar("LeWM (0004)", m["lewm_w"][2], vmax, "#3b6fd4", f"{m['lewm_w'][2]*100:.1f}%"),
        hbar("CEM 采样先验（纯随机）", A["cem_prior_random"]["rate"], vmax, "#9aa5b1",
             f"{A['cem_prior_random']['rate']*100:.2f}%"),
        hbar("均匀随机 U(−1,1)²", A["uniform_random"]["rate"], vmax, "#9aa5b1",
             f"{A['uniform_random']['rate']*100:.2f}%"),
        hbar("<b>JEPA-WM (0005 K4)</b>", m["jepa_w"][2], vmax, "#c0392b", f"{m['jepa_w'][2]*100:.1f}%"),
        hbar("零动作 no-op", A["noop_zero_action"]["rate"], vmax, "#c3cad3",
             f"{A['noop_zero_action']['rate']*100:.1f}%"),
    ])
    eff = m["effort"]
    emax = max(eff.values()) * 1.05
    ebars = "".join([
        hbar("零动作 no-op", eff["noop_zero_action"], emax, "#c3cad3", str(eff["noop_zero_action"])),
        hbar("CEM 采样先验", eff["cem_prior_random"], emax, "#9aa5b1", str(eff["cem_prior_random"])),
        hbar("LeWM", eff["lewm"], emax, "#3b6fd4", str(eff["lewm"])),
        hbar("JEPA-WM", eff["jepa"], emax, "#c0392b", str(eff["jepa"])),
        hbar("均匀随机", eff["uniform_random"], emax, "#9aa5b1", str(eff["uniform_random"])),
    ])
    ppp = m["ppp"]
    ppp_rows = "".join(
        f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"
        for k, v in sorted(ppp.items(), key=lambda kv: -kv[1])
    )
    return f"""
<section><h2>无模型地板（KW-LEWM-0005d）：JEPA 的“规划”没有高于随机动作</h2>
<div class="banner bad">
  <div class="v">JEPA-WM 与它自己的随机采样先验统计不可区分；LeWM 越过地板约 {ratio:.1f} 倍</div>
  <div class="note">
    在同一 {m['n_tasks']} 个任务、同 env step 预算下跑三个<b>不加载任何世界模型</b>的真实 rollout 臂（纯 CPU、零 GPU、共 {m['n_tasks']*m['n_seeds']*2 + m['n_tasks']} 条）。
    无模型地板（单侧 95% Wilson 上界）<b>U = {m['U']}</b>。
    JEPA {m['jepa_w'][2]*100:.0f}% CI [{m['jepa_w'][0]}, {m['jepa_w'][1]}] <b>未越过</b>地板（点估计甚至低于两个随机臂）；
    LeWM {m['lewm_w'][2]*100:.0f}% CI [{m['lewm_w'][0]}, {m['lewm_w'][1]}] <b>越过</b>地板。
    主判据刻意用 <b>S-无关</b> 的 Wilson 上界，而非随种子预算单调变化的 Poisson-binomial 尾概率（后者只作敏感性分析）。
  </div>
</div>
<div class="dbwrap">{bars}
  <div class="note">任务级成功率。地板非 0 是因为 4 维联合判据里的<b>智能体导航项</b>可被随机游走偶然满足——随机最高 {floor_pct*100:.2f}% ≪ 0.30，故任务集<b>不可</b>被随机动作刷开。</div>
</div>

<div class="grid" style="margin-top:14px">
  <div class="metric"><div class="metric-h">配对精确 McNemar（seed 0，同 50 任务）</div>
    <table style="border:0"><tbody>
      <tr><td>JEPA vs CEM 先验随机</td><td><b style="color:var(--bad)">p={m['mc_j_cem']['p_exact_two_sided']:.3g}</b>（{m['mc_j_cem']['a_only']} vs {m['mc_j_cem']['b_only']}）</td><td class="sub">不可区分</td></tr>
      <tr><td>LeWM vs CEM 先验随机</td><td><b style="color:var(--ok)">p={m['mc_l_cem']['p_exact_two_sided']:.3g}</b>（{m['mc_l_cem']['a_only']} vs {m['mc_l_cem']['b_only']}）</td><td class="sub">显著优于随机</td></tr>
      <tr><td>LeWM vs 零动作</td><td><b style="color:var(--ok)">p={m['mc_l_noop']['p_exact_two_sided']:.3g}</b>（{m['mc_l_noop']['a_only']} vs {m['mc_l_noop']['b_only']}）</td><td class="sub">显著优于不动</td></tr>
    </tbody></table>
  </div>
  <div class="metric"><div class="metric-h">0005c 的解析 no-op 假设已被真实 rollout 验证</div>
    <table style="border:0"><tbody>
      <tr><td>真实 no-op 成功</td><td><b>{m['noop_rolled']}/{m['n_tasks']}</b></td><td class="sub">与解析值一致</td></tr>
      <tr><td>方块位移（中位）</td><td><b>{m['noop_block_med']}</b></td><td class="sub">零动作不推方块</td></tr>
      <tr><td>智能体漂移（中位/最大）</td><td><b>{m['noop_agent_med']} / {m['noop_agent_max']}</b></td><td class="sub">初态速度注入 + 零阻尼导致滑行</td></tr>
    </tbody></table>
    <div class="note">判定 <code>ANALYTIC_NOOP_VALID</code>：0005c 用“终态=初态”解析推断 no-op 是<b>成立</b>的，仅方块判据下 16/50=0.32 逐位吻合。</div>
  </div>
</div>

<div class="dbwrap" style="margin-top:14px">
  <div class="metric-h" style="margin-bottom:8px">位移量不是区分轴（<code>EFFORT_NOT_DISCRIMINATIVE</code>）</div>
  {ebars}
  <div class="note">智能体移动距离中位数（像素）。两个模型臂（{eff['lewm']} / {eff['jepa']}）<b>夹在</b>两个随机臂（{eff['cem_prior_random']} / {eff['uniform_random']}）之间——“动得多/少”无法解释成败。</div>
</div>

<div class="banner" style="border-color:#f0d9a8;background:#fdf8ec;margin-top:14px">
  <div class="v" style="color:#8a6014">⚠ 对 KW-LEWM-0005c 措辞的修正（事后探索，非预注册）</div>
  <div class="note">
    0005c 曾写“JEPA <b>主动远离</b>目标”。但 JEPA 的移动距离（{eff['jepa']}）介于两个随机臂之间，而均匀随机臂的 <code>pos_progress</code> CI [{A['uniform_random']['pos_lo']}, {A['uniform_random']['pos_hi']}] <b>更负</b> → 负进展可能只是“动得多”的必然结果（努力混淆）。
    故补<b>等位移匹配对照</b>：逐任务只保留 <code>agent_disp</code> 与模型臂相差 ≤{int(m['me_tol']*100)}% 的无模型 rollout（每任务匹配池中位 {m['me_pool']} 条）后配对比较，得：
    JEPA <code>agent_progress</code> 配对差 <b>{m['me_j_agent'][0]:+}</b> CI [{m['me_j_agent'][1]}, {m['me_j_agent'][2]}] <b>含 0</b>、<code>pos_progress</code> <b>{m['me_j_pos'][0]:+}</b> CI [{m['me_j_pos'][1]}, {m['me_j_pos'][2]}] <b>含 0</b>；
    LeWM <code>agent_progress</code> 配对差 <b>{m['me_l_agent'][0]:+}</b> CI [{m['me_l_agent'][1]}, {m['me_l_agent'][2]}] <b>排除 0</b>。
    → 正确措辞是「<b>JEPA 的进展与等位移随机游走不可区分</b>」，即动作方向对进展的净贡献为 <b>0</b>，而不是“反向”。这<b>不削弱</b>弱动作条件结论，反而让它更纯粹。
  </div>
</div>

<div class="grid" style="margin-top:14px">
  <div class="metric"><div class="metric-h">地板机制：预写猜测被数据否证（按纪律保留）</div>
    <table style="border:0"><tbody>
      <tr><td>猜测「方块已在位驱动地板」</td><td><b style="color:var(--bad)">{m['fm_guess_status']}</b></td></tr>
      <tr><td>Spearman(随机成功数, block_gap_start)</td><td><b>{m['fm_sp_block']:+}</b> <span class="sub">无关</span></td></tr>
      <tr><td>Spearman(随机成功数, agent_gap_start)</td><td><b>{m['fm_sp_agent']:+}</b> <span class="sub">真实驱动项</span></td></tr>
      <tr><td>出现过随机成功的任务数</td><td><b>{m['fm_n_any']}/{m['n_tasks']}</b></td></tr>
    </tbody></table>
    <div class="note">真实机制是<b>智能体导航项</b>被随机游走偶然满足，与 0005c 的失败归因（85.7% 由智能体位置项主导）一致，是 4 维联合判据的性质而非物体操作能力。</div>
  </div>
  <div class="metric"><div class="metric-h">每像素位移换来的进展（仅描述性，<b>不作判据</b>）</div>
    <table style="border:0"><tbody>{ppp_rows}</tbody></table>
    <div class="note">⚠ 该聚合比值表面上说 JEPA 差于 CEM 先验，与上方逐任务<b>等努力配对含 0</b> 存在张力；因为是跨任务聚合比值（分母混合），一并登记但<b>不据此下结论</b>。</div>
  </div>
</div>
</section>"""


def build_html(d):
    L, J = d["lewm"], d["jepa"]
    rc, cb = d["rootcause"], d["corroboration"]
    paired_section = build_paired_section(d.get("paired"))
    modelfree_section = build_modelfree_section(d.get("mf"))
    n_lines = 4 if d.get("mf") else 3
    verdict_cls = "verdict-bad" if d["jepa_sig_lower"] else "verdict-ok"
    verdict_txt = ("JEPA-WM 任务成功率显著低于 LeWM（Wilson 95% CI 不重叠）" if d["jepa_sig_lower"]
                  else "两模型成功率 CI 重叠（无显著差异）")

    def card(title, lval, jval, sub_l="", sub_j="", better_high=True):
        # better_high: True 表示数值高更好（蓝高亮优胜方）
        lw = "win" if (better_high and L[title_key(title)] >= J[title_key(title)]) or (not better_high and L[title_key(title)] <= J[title_key(title)]) else ""
        jw = "win" if (better_high and J[title_key(title)] > L[title_key(title)]) or (not better_high and J[title_key(title)] < L[title_key(title)]) else ""
        return f"""
        <div class="metric">
          <div class="metric-h">{title}</div>
          <div class="metric-row"><span class="tag lewm {lw}">{lval}</span><span class="sub">{sub_l}</span></div>
          <div class="metric-row"><span class="tag jepa {jw}">{jval}</span><span class="sub">{sub_j}</span></div>
        </div>"""

    def title_key(t):
        return {"成功率": "success", "Agent 越界率": "oop", "Block 交互率": "interaction",
                "Block 位移中位数": "disp_med", "Block 位移均值": "disp_mean", "运行错误": "n_errors"}[t]

    cards = "\n".join([
        card("成功率", pct(L["success"]), pct(J["success"]),
             f"{L['n_success']}/{L['n_tasks']} · CI [{L['ci'][0]},{L['ci'][1]}]",
             f"{J['n_success']}/{J['n_tasks']} · CI [{J['ci'][0]},{J['ci'][1]}]", better_high=True),
        card("Agent 越界率", pct(L["oop"]), pct(J["oop"]),
             f"{L['n_oop']} 越界", f"{J['n_oop']} 越界", better_high=False),
        card("Block 交互率", pct(L["interaction"]), pct(J["interaction"]), "推到 block", "推到 block", better_high=True),
        card("Block 位移中位数", f"{L['disp_med']}", f"{J['disp_med']}", "粗推动能力", "粗推动能力", better_high=True),
        card("Block 位移均值", f"{L['disp_mean']}", f"{J['disp_mean']}", "右偏重尾", "右偏重尾", better_high=True),
        card("运行错误", str(L["n_errors"]), str(J["n_errors"]), "全链路", "全链路", better_high=False),
    ])

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KineWorld · World Model 同台对比仪表盘</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--ink:#1f2933;--mut:#6b7280;--line:#e5e7eb;
--lewm:#3b6fd4;--jepa:#d98a2b;--ok:#2e7d52;--bad:#c0392b;--accent:#3b6fd4;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;}}
.wrap{{max-width:1080px;margin:0 auto;padding:28px 20px 60px;}}
header h1{{margin:0 0 6px;font-size:24px;}}
header .meta{{color:var(--mut);font-size:13px;}}
.badge{{display:inline-block;background:#eef2fb;color:var(--accent);border:1px solid #dbe4f5;
border-radius:999px;padding:3px 10px;font-size:12px;margin:8px 6px 0 0;}}
.banner{{margin:18px 0;padding:14px 16px;border-radius:12px;border:1px solid var(--line);
background:var(--card);}}
.banner.bad{{border-color:#f3c9c2;background:#fdf3f1;}}
.banner .v{{font-weight:700;font-size:16px;color:var(--bad);}}
.banner.ok .v{{color:var(--ok);}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-top:6px;}}
.metric{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;}}
.metric-h{{font-weight:600;font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;margin-bottom:10px;}}
.metric-row{{display:flex;align-items:baseline;gap:8px;margin:6px 0;}}
.tag{{font-weight:700;font-size:18px;padding:2px 8px;border-radius:8px;}}
.tag.lewm{{color:var(--lewm);background:#eaf0fb;}}
.tag.jepa{{color:var(--jepa);background:#fbf0e0;}}
.tag.win{{outline:2px solid currentColor;outline-offset:1px;}}
.sub{{color:var(--mut);font-size:12px;}}
section{{margin-top:26px;}}
h2{{font-size:18px;border-left:4px solid var(--accent);padding-left:10px;margin:0 0 12px;}}
.rootgrid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;}}
.rc{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;}}
.rc .k{{font-size:12px;color:var(--mut);}}
.rc .vv{{font-size:22px;font-weight:700;margin-top:4px;}}
.note{{color:var(--mut);font-size:12.5px;margin-top:6px;}}
table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;}}
th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line);font-size:13px;}}
th{{background:#f0f2f5;color:var(--mut);}}
footer{{margin-top:30px;color:var(--mut);font-size:12px;}}
code{{background:#eef1f4;padding:1px 5px;border-radius:5px;font-size:12px;}}
.dbwrap{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;}}
.dbrow{{display:grid;grid-template-columns:200px 1fr 64px;align-items:center;gap:10px;margin:8px 0;}}
.dblab{{font-size:12.5px;color:var(--ink);}}
.dbtrack{{position:relative;height:20px;background:#f0f2f5;border-radius:6px;}}
.dbaxis{{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:#c3cad3;}}
.dbbar{{position:absolute;top:3px;bottom:3px;border-radius:4px;}}
.dbval{{font-size:13px;font-weight:700;text-align:right;}}
.hbrow{{display:grid;grid-template-columns:200px 1fr 72px;align-items:center;gap:10px;margin:8px 0;}}
.hblab{{font-size:12.5px;color:var(--ink);}}
.hbtrack{{position:relative;height:20px;background:#f0f2f5;border-radius:6px;overflow:hidden;}}
.hbbar{{position:absolute;left:0;top:3px;bottom:3px;border-radius:4px;min-width:2px;}}
.hbval{{font-size:13px;font-weight:700;text-align:right;}}
@media(max-width:640px){{.dbrow,.hbrow{{grid-template-columns:1fr;}}.dbval,.hbval{{text-align:left;}}}}
</style></head>
<body><div class="wrap">
<header>
  <h1>KineWorld · World Model 同台对比仪表盘</h1>
  <div class="meta">LeWM (KW-LEWM-0004) vs JEPA-WM (KW-LEWM-0005 K4) · 同一 stable-worldmodel 平台 / 同 50 数据集任务 / 同 CEM 10×100×10</div>
  <div>
    <span class="badge">E1 证据等级</span>
    <span class="badge">单 checkpoint</span>
    <span class="badge">单 seed</span>
    <span class="badge">无训练</span>
    <span class="badge">Wilson 95% CI</span>
  </div>
</header>

<div class="banner {('bad' if d['jepa_sig_lower'] else 'ok')}">
  <div class="v">{verdict_txt}</div>
  <div class="note">LeWM success {pct(L['success'])} CI [{L['ci'][0]},{L['ci'][1]}] &nbsp;vs&nbsp; JEPA-WM {pct(J['success'])} CI [{J['ci'][0]},{J['ci'][1]}].
  Agent 粗控制相当（越界 {pct(L['oop'])} vs {pct(J['oop'])}、位移中位 {L['disp_med']} vs {J['disp_med']}）→ 平坦代价伤及精度，非飞出/无控制。</div>
</div>

<section><h2>核心指标并排</h2>
<div class="grid">{cards}</div>
</section>

<section><h2>根因（KW-LEWM-0005 §8）：flat cost = 弱动作条件</h2>
<div class="rootgrid">
  <div class="rc"><div class="k">verdict</div><div class="vv" style="font-size:14px;line-height:1.3">{rc['verdict'].split('(')[0].strip()}</div></div>
  <div class="rc"><div class="k">mean ratio (max_mob/gap)</div><div class="vv">{rc['mean_ratio']}</div><div class="note">机动性非瓶颈（≈3× 所需）</div></div>
  <div class="rc"><div class="k">context_domination_ratio</div><div class="vv">{rc['context_dom']}</div><div class="note">≈1 → 零动作位移≈最大动作位移</div></div>
  <div class="rc"><div class="k">action_leverage</div><div class="vv">{rc['action_leverage']}</div><div class="note">≈0 → 动作对落点无净杠杆</div></div>
</div></section>

<section><h2>闭环佐证（KW-LEWM-0005b，真实 50 任务）</h2>
<div class="rootgrid">
  <div class="rc"><div class="k">block_disp vs 目标距离 (Spearman)</div><div class="vv">{cb['disp_vs_dist_spearman']}</div><div class="note">负 → 目标越远推越少（非精确收敛）</div></div>
  <div class="rc"><div class="k">success vs 目标距离 (Spearman)</div><div class="vv">{cb['succ_vs_dist_spearman']}</div><div class="note">≈0 → 成功非几何可exploited</div></div>
  <div class="rc"><div class="k">success vs 位移量 (Spearman)</div><div class="vv">{cb['succ_vs_disp_spearman']}</div><div class="note">≈0 → 推得多≠成功</div></div>
  <div class="rc"><div class="k">in-play 位移&gt;20 全失败</div><div class="vv">{cb['near_but_failed']}/{cb['near_total']}</div><div class="note">粗动有效、精度失效</div></div>
  <div class="rc"><div class="k">越界 ep 目标距离 / in-play</div><div class="vv">{cb['oop_ratio']}</div><div class="note">≈1.09 → 越界是漂移非更难目标</div></div>
</div></section>

{paired_section}

{modelfree_section}

<section><h2>明细</h2>
<table><thead><tr><th>指标</th><th>LeWM (0004)</th><th>JEPA-WM (0005 K4)</th></tr></thead><tbody>
<tr><td>成功率 (n/N)</td><td>{pct(L['success'])} ({L['n_success']}/{L['n_tasks']})</td><td>{pct(J['success'])} ({J['n_success']}/{J['n_tasks']})</td></tr>
<tr><td>Wilson 95% CI</td><td>[{L['ci'][0]}, {L['ci'][1]}]</td><td>[{J['ci'][0]}, {J['ci'][1]}]</td></tr>
<tr><td>Agent 越界率</td><td>{pct(L['oop'])}</td><td>{pct(J['oop'])}</td></tr>
<tr><td>Block 交互率</td><td>{pct(L['interaction'])}</td><td>{pct(J['interaction'])}</td></tr>
<tr><td>Block 位移中位 / 均值</td><td>{L['disp_med']} / {L['disp_mean']}</td><td>{J['disp_med']} / {J['disp_mean']}</td></tr>
<tr><td>运行错误</td><td>{L['n_errors']}</td><td>{J['n_errors']}</td></tr>
</tbody></table>
</section>

<footer>
  数据来源：<code>results/kw_lewm_0004_dataset_tasks_p50.json</code>、<code>results/kw_lewm_0005_jepa_same_platform.json</code>、
  <code>results/kw_lewm_0005_rootcause.json</code>、<code>results/kw_lewm_0005b_k4_corroboration.json</code>、<code>results/kw_lewm_0005c_paired_noop_baseline.json</code>、
  <code>results/kw_lewm_0005d_modelfree_reference.json</code>。
  0005c 为既有日志的<b>事后再分析</b>（假设与判据在计算前写死于脚本头、无新 rollout、零 GPU），其 no-op 为<b>解析</b>基线（终态=初态）；
  0005d 用<b>真实 CPU rollout</b>（不加载世界模型、零 GPU）验证该解析基线成立，并首次提供无模型控制对照臂。两者的判据自校验门禁 R0 均为 100%（环境 <code>eval_state</code> 标签 vs 重构判据）。
  0005d 的等位移匹配对照（H5）为<b>事后探索、非预注册</b>；Poisson-binomial 尾概率仅作敏感性分析，不作判据。JEPA 的 1/50 单次成功<b>不可解读</b>。
  仍 E1（单 checkpoint / 单 seed / 单任务集）；残留近似：partial-shard scaler、本地渲染像素、缩减 CEM、shard 0、单 goal_offset=25。
  禁止声称官方 benchmark 或任一模型优劣（除 CI 不重叠处）。生成脚本：<code>verification/scripts/kw_lewm_dashboard.py</code>。
</footer>
</div></body></html>"""


if __name__ == "__main__":
    main()
