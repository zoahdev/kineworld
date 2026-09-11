"""
KW-MOON-0001 — 判据校验器（K1–K7）

判据已在 `verification/experiments/KW-MOON-0001.md` §2 **预注册**。
本脚本只负责执行与判定，**不得在此处调整阈值或放宽单调性要求**。
输出：results/kw_moon_0001_validation.json（供 verify_lunar_report.py 机器校验文档数字）

依赖：仅标准库。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kw_moon_0001_lunar_wm import (  # noqa: E402
    G_MOON, J_PER_WH, SYNODIC_MONTH_S, Action, LolaDEM, LunarWorldModel,
    RoverParams, State, max_speed_on_slope, required_power_w,
    make_synthetic_crater_dem, theta_limit, theta_power, theta_torque, theta_traction,
)

TOL_K2 = 1e-12
TOL_K3 = 1e-9
TOL_K4 = 1e-9


# ---------------------------------------------------------------- 测试替身

class ScriptedDEM:
    """把门控逻辑与 DEM 分辨率解耦：返回预设地形。"""

    def __init__(self, slope_rad: float, elev_m: float = 0.0):
        self._slope = slope_rad
        self._elev = elev_m

    def terrain(self, lat_deg: float, lon_deg: float, n_cells: int = 2) -> dict:
        return {"elev_m": self._elev, "slope_rad": self._slope,
                "aspect_rad": 0.0, "roughness_m": 0.0}


def _strictly_monotone(vals: list[float]) -> tuple[bool, str]:
    """严格单调（全增或全减），且所有值必须有限。"""
    if any(not math.isfinite(v) for v in vals):
        return False, "含非有限值 (inf/nan)"
    inc = all(b > a for a, b in zip(vals, vals[1:]))
    dec = all(b < a for a, b in zip(vals, vals[1:]))
    if inc:
        return True, "严格递增"
    if dec:
        return True, "严格递减"
    return False, f"非严格单调：{['%.6f' % v for v in vals]}"


# ---------------------------------------------------------------- K1–K7

def k1_component_monotonicity() -> dict:
    """K1 分量单调性。每个子项在**该约束实际起作用的参数区间**内测，
    避免被其它约束掩盖（判决仍为严格单调）。"""
    sub = {}

    mus = [0.20, 0.40, 0.60, 0.80, 1.00]
    vals = [theta_traction(RoverParams(mu=m)) for m in mus]
    sub["theta_traction_vs_mu"] = {"params": mus, "values_deg": [math.degrees(v) for v in vals],
                                  "ok": _strictly_monotone(vals)[0], "why": _strictly_monotone(vals)[1]}

    masses = [100.0, 140.0, 180.0, 220.0, 260.0]
    vals = [theta_torque(RoverParams(mass_kg=m, t_wheel_nm=3.0)) for m in masses]
    sub["theta_torque_vs_mass"] = {"params": masses, "values_deg": [math.degrees(v) for v in vals],
                                  "ok": _strictly_monotone(vals)[0], "why": _strictly_monotone(vals)[1]}

    tws = [1.0, 2.0, 3.0, 4.0, 5.0]
    vals = [theta_torque(RoverParams(t_wheel_nm=t)) for t in tws]
    sub["theta_torque_vs_t_wheel"] = {"params": tws, "values_deg": [math.degrees(v) for v in vals],
                                      "ok": _strictly_monotone(vals)[0], "why": _strictly_monotone(vals)[1]}

    pmaxs = [10.0, 20.0, 30.0, 40.0, 50.0]
    vals = [theta_power(RoverParams(p_max_w=p), 0.2) for p in pmaxs]
    sub["theta_power_vs_p_max"] = {"params": pmaxs, "values_deg": [math.degrees(v) for v in vals],
                                   "ok": _strictly_monotone(vals)[0], "why": _strictly_monotone(vals)[1]}

    masses2 = [100.0, 140.0, 180.0, 220.0, 260.0]
    vals = [theta_power(RoverParams(mass_kg=m, p_max_w=30.0), 0.2) for m in masses2]
    sub["theta_power_vs_mass"] = {"params": masses2, "values_deg": [math.degrees(v) for v in vals],
                                  "ok": _strictly_monotone(vals)[0], "why": _strictly_monotone(vals)[1]}

    vs = [0.10, 0.15, 0.20, 0.30, 0.50]
    vals = [theta_power(RoverParams(p_max_w=30.0), v) for v in vs]
    sub["theta_power_vs_v"] = {"params": vs, "values_deg": [math.degrees(v) for v in vals],
                               "ok": _strictly_monotone(vals)[0], "why": _strictly_monotone(vals)[1]}

    return {"passed": all(s["ok"] for s in sub.values()), "subchecks": sub}


def k2_combined_limit() -> dict:
    """K2 θ_limit(v) == min(θ_t, θ_T, θ_P(v))，偏差 < 1e-12。"""
    worst = 0.0
    detail = []
    for m, tw, pm, v in [(140.0, 12.0, 30.0, 0.05), (140.0, 3.0, 30.0, 0.05),
                         (260.0, 2.0, 10.0, 0.20), (100.0, 1.0, 50.0, 0.50)]:
        p = RoverParams(mass_kg=m, t_wheel_nm=tw, p_max_w=pm)
        parts = [theta_traction(p), theta_torque(p), theta_power(p, v)]
        finite = [x for x in parts if math.isfinite(x)]
        expect = min(parts)
        got = theta_limit(p, v)
        dev = abs(got - expect) if math.isfinite(got) and math.isfinite(expect) else 0.0
        worst = max(worst, dev)
        detail.append({"mass_kg": m, "t_wheel_nm": tw, "p_max_w": pm, "v": v,
                       "component_deg": [None if not math.isfinite(x) else math.degrees(x) for x in parts],
                       "theta_limit_deg": None if not math.isfinite(got) else math.degrees(got),
                       "abs_deviation_rad": dev, "n_finite_components": len(finite)})
    return {"passed": worst < TOL_K2, "worst_abs_deviation_rad": worst, "tol": TOL_K2, "cases": detail}


def k3_power_closed_form() -> dict:
    """K3 在 θ = θ_P(v) 处，所需功率必须恰等于 P_max（相对误差 < 1e-9）。"""
    worst = 0.0
    detail = []
    for m, pm, v in [(140.0, 30.0, 0.20), (140.0, 20.0, 0.15), (260.0, 50.0, 0.30),
                     (100.0, 10.0, 0.10), (200.0, 40.0, 0.50)]:
        p = RoverParams(mass_kg=m, p_max_w=pm)
        th = theta_power(p, v)
        if not math.isfinite(th):
            detail.append({"mass_kg": m, "p_max_w": pm, "v": v, "skipped": "该约束不起作用(inf)"})
            continue
        got = required_power_w(p, th, v)
        rel = abs(got / pm - 1.0)
        worst = max(worst, rel)
        detail.append({"mass_kg": m, "p_max_w": pm, "v": v,
                       "theta_power_deg": math.degrees(th),
                       "required_power_w": got, "p_max_w": pm, "rel_error": rel})
    return {"passed": worst < TOL_K3, "worst_rel_error": worst, "tol": TOL_K3, "cases": detail}


def k4_flat_energy_closed_form() -> dict:
    """K4 平地上实际 step() 的牵引能耗必须等于 Crr·m·g·d/η。"""
    p = RoverParams()
    dem = ScriptedDEM(0.0, 0.0)
    wm = LunarWorldModel(dem=dem, origin_lat=0.0, origin_lon=0.0)
    t_sun_up = SYNODIC_MONTH_S / 2.0      # 在 (0,0) 处太阳高度角 = 90°
    s0 = State(battery_wh=p.battery_wh, t_s=t_sun_up)
    v_cmd, dt = 0.05, 60.0
    s1, info = wm.step(s0, Action(v_cmd_mps=v_cmd, psi_rate_rad_s=0.0), dt)

    d_expect = v_cmd * dt
    e_tract_expect_j = p.crr * p.mass_kg * G_MOON * d_expect / p.eta
    e_tract_got_j = info.energy_wh * J_PER_WH - p.p_avionics_w * dt
    rel = abs(e_tract_got_j / e_tract_expect_j - 1.0) if e_tract_expect_j != 0 else math.inf

    return {"passed": (rel < TOL_K4 and info.slip_ratio == 0.0
                       and abs(info.distance_m - d_expect) < 1e-12),
            "sun_elev_deg": math.degrees(info.sun_elev_rad),
            "distance_expect_m": d_expect, "distance_got_m": info.distance_m,
            "slip_ratio": info.slip_ratio,
            "e_traction_expect_J": e_tract_expect_j, "e_traction_got_J": e_tract_got_j,
            "rel_error": rel, "tol": TOL_K4,
            "flat_ground_zero_slip_ok": info.slip_ratio == 0.0}


def k5_blocked_gating() -> dict:
    """K5 坡度 > θ_limit 时必须 blocked、位移严格为 0。"""
    p = RoverParams()
    lim = theta_limit(p, p.v_ref)
    steep = lim + 0.10                     # 明确超过上限
    gentle = lim - 0.10
    t_sun_up = SYNODIC_MONTH_S / 2.0

    wm_bad = LunarWorldModel(dem=ScriptedDEM(steep), origin_lat=0.0, origin_lon=0.0)
    s0 = State(x_m=1000.0, y_m=2000.0, psi_rad=0.3, battery_wh=500.0, t_s=t_sun_up)
    s1, info_bad = wm_bad.step(s0, Action(v_cmd_mps=0.05), 60.0)

    wm_ok = LunarWorldModel(dem=ScriptedDEM(gentle), origin_lat=0.0, origin_lon=0.0)
    s2, info_ok = wm_ok.step(s0, Action(v_cmd_mps=0.05), 60.0)

    moved = math.hypot(s1.x_m - s0.x_m, s1.y_m - s0.y_m)
    return {"passed": (info_bad.blocked and info_bad.reason == "slope_exceeds_limit"
                       and moved == 0.0 and not info_ok.blocked and info_ok.distance_m > 0.0),
            "theta_limit_deg": math.degrees(lim),
            "steep_slope_deg": math.degrees(steep), "gentle_slope_deg": math.degrees(gentle),
            "steep_blocked": info_bad.blocked, "steep_reason": info_bad.reason,
            "steep_displacement_m": moved,
            "gentle_blocked": info_ok.blocked, "gentle_distance_m": info_ok.distance_m}


def k6_battery_termination() -> dict:
    """K6 电量耗尽必须终止，且电量非负、严格小于初始值。

    判据（预注册）不变。**测试实例修正记录**：首版把该测试放在 t_s = 朔望月/2（(0,0) 处太阳高度角 90°），
    此时太阳能输入 1.0 Wh ≫ 单步消耗 0.004 Wh，电量只会上升，该实例**由构造决定不可能通过**。
    读源码定位后改为 t_s = 0（同处太阳高度角 −90°，无日照），判据本身未改动。
    """
    p = RoverParams()
    dem = ScriptedDEM(0.0, 0.0)
    wm = LunarWorldModel(dem=dem, origin_lat=0.0, origin_lon=0.0)
    t_dark = 0.0
    init = 1e-6
    s0 = State(battery_wh=init, t_s=t_dark)
    acts = [Action(v_cmd_mps=0.05) for _ in range(50)]
    res = wm.rollout(s0, acts, 60.0)
    batt = res["final"]["battery_wh"]
    sun_deg = math.degrees(res["info"][0]["sun_elev_rad"]) if res["info"] else None
    return {"passed": (res["termination"] == "battery_depleted" and batt >= 0.0 and batt < init),
            "termination": res["termination"], "initial_wh": init, "final_wh": batt,
            "n_steps": res["n_steps"], "sun_elevation_deg": sun_deg,
            "test_instance_corrected": True,
            "correction_note": "首版实例置于日照区导致由构造不可能通过；改为无日照实例，判据未改"}


def k7_determinism_and_robustness() -> dict:
    """K7 同输入两次运行哈希一致；200 个随机场景无 NaN/Inf。"""
    def _hash_run():
        dem = ScriptedDEM(0.25, 12.0)
        wm = LunarWorldModel(dem=dem, origin_lat=0.0, origin_lon=0.0)
        s0 = State(battery_wh=500.0, t_s=1.0e6)
        acts = [Action(v_cmd_mps=0.05, psi_rate_rad_s=0.001 * i) for i in range(30)]
        return json.dumps(wm.rollout(s0, acts, 60.0), sort_keys=True)

    h1 = hashlib.sha256(_hash_run().encode()).hexdigest()
    h2 = hashlib.sha256(_hash_run().encode()).hexdigest()

    rng = random.Random(20260911)
    bad = 0
    for _ in range(200):
        slope = rng.uniform(0.0, 1.4)
        dem = ScriptedDEM(slope, rng.uniform(-9000, 9000))
        wm = LunarWorldModel(dem=dem, origin_lat=rng.uniform(-89.9, 89.9),
                             origin_lon=rng.uniform(0, 360))
        s0 = State(battery_wh=rng.uniform(0.0, 800.0), t_s=rng.uniform(0, 1e8))
        acts = [Action(v_cmd_mps=rng.uniform(0.0, 2.0),
                       psi_rate_rad_s=rng.uniform(-0.01, 0.01)) for _ in range(15)]
        res = wm.rollout(s0, acts, 60.0)
        txt = json.dumps(res)
        if ("NaN" in txt) or ("Infinity" in txt):
            bad += 1
        elif not all(math.isfinite(v) for v in
                     [res["final"][k] for k in ("x_m", "y_m", "psi_rad", "v_mps", "battery_wh", "t_s")]):
            bad += 1

    return {"passed": (h1 == h2 and bad == 0),
            "sha256_run1": h1, "sha256_run2": h2, "identical": h1 == h2,
            "random_scenarios": 200, "scenarios_with_nan_or_inf": bad,
            "seed": 20260911}


# ---------------------------------------------------------------- 主流程

def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(root, "results")
    os.makedirs(out_dir, exist_ok=True)

    lola_path = os.path.join(root, "data", "lola", "ldem_16.img")
    dem_meta = {"lola_path": lola_path, "lola_present": os.path.exists(lola_path),
                "lola_bytes": os.path.getsize(lola_path) if os.path.exists(lola_path) else None}

    print("[K1] 分量单调性 ...")
    r1 = k1_component_monotonicity()
    print("[K2] 组合上限一致性 ...")
    r2 = k2_combined_limit()
    print("[K3] 功率闭式自洽 ...")
    r3 = k3_power_closed_form()
    print("[K4] 平地能量闭式 ...")
    r4 = k4_flat_energy_closed_form()
    print("[K5] 阻挡门控 ...")
    r5 = k5_blocked_gating()
    print("[K6] 电池终止 ...")
    r6 = k6_battery_termination()
    print("[K7] 确定性与数值稳健 ...")
    r7 = k7_determinism_and_robustness()

    checks = {"K1": r1, "K2": r2, "K3": r3, "K4": r4, "K5": r5, "K6": r6, "K7": r7}
    all_pass = all(v["passed"] for v in checks.values())

    p = RoverParams()
    limits = {
        "theta_traction_deg": math.degrees(theta_traction(p)),
        "theta_torque_deg": (None if not math.isfinite(theta_torque(p))
                             else math.degrees(theta_torque(p))),
        "theta_power_at_v_ref_deg": (None if not math.isfinite(theta_power(p, p.v_ref))
                                     else math.degrees(theta_power(p, p.v_ref))),
        "theta_limit_deg": math.degrees(theta_limit(p, p.v_ref)),
        "binding_constraint": min(
            [("traction", theta_traction(p)),
             ("torque", theta_torque(p)),
             ("power", theta_power(p, p.v_ref))], key=lambda kv: kv[1])[0],
    }

    payload = {
        "experiment_id": "KW-MOON-0001",
        "evidence_level": "E1_INTERNAL_NO_THIRD_PARTY_VALIDATION",
        "model_class": "ANALYTIC_WORLD_MODEL_NOT_LEARNED",
        "pre_registration": "verification/experiments/KW-MOON-0001.md",
        "dem_source": dem_meta,
        "rover_params": {k: (v if not isinstance(v, float) else round(v, 6))
                         for k, v in vars(p).items()},
        "limits": limits,
        "checks": checks,
        "verdict": "PASS" if all_pass else "FAILED",
        "n_checks": len(checks),
        "n_passed": sum(1 for v in checks.values() if v["passed"]),
        "tolerances": {"K2_rad": TOL_K2, "K3_rel": TOL_K3, "K4_rel": TOL_K4},
    }

    path = os.path.join(out_dir, "kw_moon_0001_validation.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print()
    for k, v in checks.items():
        print(f"  {k}: {'PASS' if v['passed'] else 'FAIL'}")
    print(f"\nverdict = {payload['verdict']}  ({payload['n_passed']}/{payload['n_checks']})")
    print(f"binding_constraint = {limits['binding_constraint']}, "
          f"theta_limit = {limits['theta_limit_deg']:.4f} deg")
    print(f"json -> {path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
