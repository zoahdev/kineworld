"""
KW-MOON-0001 — 月球漫游车解析式世界模型（analytic / non-learned world model）

定位
----
本模块实现一个显式的状态转移函数

    s_{t+1} = f(s_t, a_t)

s = (x, y, psi, v, battery_Wh, t_s)   位姿 / 速度 / 电量 / 时间
a = (v_cmd, psi_rate)                 速度指令 / 转向角速度
f = 由公开文献的刚体力学 + 地形几何解析给出（**无神经网络、无训练**）

这是一类"已知动力学"世界模型（known-dynamics / analytic world model），
在基于模型的规划文献中与"学习动力学模型"（learned dynamics model）并列。
本文件**不是**学习型世界模型，不得如此表述。

证据等级：E1（单一解析模型 + 单一数据源 + 无第三方验证）

依赖：仅 Python 标准库（array / math / struct / json / random）
"""

from __future__ import annotations

import array
import json
import math
import os
import struct
from dataclasses import dataclass, asdict, field
from typing import Optional, Sequence

# ---------------------------------------------------------------- 物理常量

G_MOON = 1.62               # m/s^2，月球表面重力加速度
R_MOON = 1737400.0          # m，月球平均半径
SYNODIC_MONTH_S = 29.530589 * 86400.0     # 朔望月（月球太阳日）
LUNAR_OBLIQUITY_DEG = 1.5424              # 月球自转轴相对黄道倾角
YEAR_S = 365.256363 * 86400.0

J_PER_WH = 3600.0

# ---------------------------------------------------------------- 参数


@dataclass
class RoverParams:
    """漫游车参数。默认值取自公开的开源月球车仿真工具的量级（对标基线见 Cynthium），
    并标注为**设计假设**而非实测标定值 —— 本仓库无任何车辆实测数据。"""

    mass_kg: float = 140.0          # 玉兔二号量级（公开报道约 140 kg 级）
    mu: float = 0.60                # 轮-土附着系数（设计假设，未标定）
    crr: float = 0.10               # 滚动阻力系数（设计假设，未标定）
    p_max_w: float = 30.0           # 驱动总功率上限 W
    t_wheel_nm: float = 12.0        # 单轮**轮端**最大扭矩 N·m（已含减速比）
    n_wheels: int = 6               # 六轮
    r_wheel_m: float = 0.15         # 车轮半径 m
    eta: float = 0.70               # 传动效率
    p_avionics_w: float = 12.0      # 电子学常驻功率 W
    p_solar_w: float = 60.0         # 额定太阳能功率 W
    battery_wh: float = 600.0       # 初始电量 Wh
    v_ref: float = 0.05             # 用于定义爬坡上限的参考速度 m/s


# ---------------------------------------------------------------- 解析爬坡上限
#
# 三个独立上限，全部闭式可检：
#
# (a) 牵引（附着力）
#     匀速爬坡所需驱动力 F_req = m·g·(sinθ + Crr·cosθ)
#     可用附着力        F_avail = μ·m·g·cosθ
#     可爬 ⟺ sinθ + Crr·cosθ ≤ μ·cosθ ⟺ tanθ ≤ μ − Crr
#     ⇒ θ_t = arctan(μ − Crr)                （与质量无关）
#
# (b) 扭矩
#     F_req·r_wheel / n_wheels ≤ t_wheel
#     ⇒ sinθ + Crr·cosθ ≤ n_w·t_wheel / (m·g·r_wheel) ≡ A_T
#
# (c) 功率（速度相关）
#     m·g·(sinθ + Crr·cosθ)·v / η ≤ P_max
#     ⇒ sinθ + Crr·cosθ ≤ P_max·η / (m·g·v) ≡ A_P(v)
#
# (b)(c) 同形：R·sin(θ + φ) = A，其中 R = √(1+Crr²)、φ = arctan(Crr)
#     ⇒ θ = arcsin(A/R) − φ


def _solve_sin_plus_c_cos(A: float, crr: float) -> float:
    """解 sinθ + crr·cosθ = A 的 θ。A ≥ R 时无解（该约束不构成上限）→ 返回 +inf。"""
    R = math.sqrt(1.0 + crr * crr)
    phi = math.atan2(crr, 1.0)
    x = A / R
    if x >= 1.0:
        return math.inf
    if x <= -1.0:
        return -math.inf
    return math.asin(x) - phi


def theta_traction(p: RoverParams) -> float:
    """(a) 牵引上限。μ ≤ Crr 时不可爬任何坡 → 返回 -inf 便于上层判定。"""
    if p.mu <= p.crr:
        return -math.inf
    return math.atan(p.mu - p.crr)


def theta_torque(p: RoverParams) -> float:
    """(b) 扭矩上限（与速度无关）。"""
    A_T = (p.n_wheels * p.t_wheel_nm) / (p.mass_kg * G_MOON * p.r_wheel_m)
    return _solve_sin_plus_c_cos(A_T, p.crr)


def theta_power(p: RoverParams, v: float) -> float:
    """(c) 功率上限（速度相关）。v ≤ 0 时功率不构成上限 → +inf。"""
    if v <= 0.0:
        return math.inf
    A_P = (p.p_max_w * p.eta) / (p.mass_kg * G_MOON * v)
    return _solve_sin_plus_c_cos(A_P, p.crr)


def theta_limit(p: RoverParams, v: float) -> float:
    """综合爬坡上限 = 三者最小。"""
    return min(theta_traction(p), theta_torque(p), theta_power(p, v))


def required_power_w(p: RoverParams, theta: float, v: float) -> float:
    """在坡度 θ 上以速度 v 行驶所需的驱动功率 W。"""
    return p.mass_kg * G_MOON * (math.sin(theta) + p.crr * math.cos(theta)) * v / p.eta


def max_speed_on_slope(p: RoverParams, theta: float) -> float:
    """在坡度 θ 上功率可支持的最大速度 m/s。"""
    denom = p.mass_kg * G_MOON * (math.sin(theta) + p.crr * math.cos(theta))
    if denom <= 0.0:
        return math.inf
    return p.p_max_w * p.eta / denom


# ---------------------------------------------------------------- 高程栅格


class LolaDEM:
    """LOLA GDR 柱面投影栅格（PDS `ldem_*.img`）。

    格式：int16 小端、比例因子 0.5 m、行首为 +90° 北、列首为 0° 经、经度回绕。
    分辨率由栅格尺寸决定（16 ppd ⇒ 5760×2880 ⇒ 赤道约 1.9 km/像素）。
    """

    def __init__(self, path: str, ncols: int = 5760, nrows: int = 2880,
                 scale_m: float = 0.5, units: str = "real"):
        self.path = path
        self.ncols = ncols
        self.nrows = nrows
        self.scale_m = scale_m
        self.units = units                      # "real" | "synthetic"
        expect = ncols * nrows * 2
        size = os.path.getsize(path)
        if size != expect:
            raise ValueError(f"栅格尺寸不符：期望 {expect} 字节，实际 {size} 字节")
        a = array.array("h")
        with open(path, "rb") as fh:
            a.fromfile(fh, ncols * nrows)
        if struct.pack("<h", 1) != struct.pack("=h", 1):
            a.byteswap()                        # 大端主机兜底
        self._a = a

    # -- 本征 ---------------------------------------------------------

    def _at(self, row: int, col: int) -> float:
        row = min(max(row, 0), self.nrows - 1)
        col = col % self.ncols
        return self._a[row * self.ncols + col] * self.scale_m

    def elevation_m(self, lat_deg: float, lon_deg: float) -> float:
        """双线性插值高程（米）。"""
        lat = min(max(lat_deg, -90.0), 90.0)
        row_f = (90.0 - lat) / 180.0 * (self.nrows - 1)
        col_f = (lon_deg % 360.0) / 360.0 * self.ncols
        r0 = int(math.floor(row_f))
        c0 = int(math.floor(col_f))
        fr = row_f - r0
        fc = col_f - c0
        h00 = self._at(r0, c0)
        h01 = self._at(r0, c0 + 1)
        h10 = self._at(r0 + 1, c0)
        h11 = self._at(r0 + 1, c0 + 1)
        return ((1 - fr) * ((1 - fc) * h00 + fc * h01)
                + fr * ((1 - fc) * h10 + fc * h11))

    def pixel_spacing_m(self, lat_deg: float) -> tuple[float, float]:
        """返回 (dx 东向米/像素, dy 北向米/像素)，随纬度收缩。"""
        dlat = math.pi / (self.nrows - 1) * 1.0
        dlon = 2.0 * math.pi / self.ncols * 1.0
        dy = R_MOON * dlat
        dx = R_MOON * math.cos(math.radians(lat_deg)) * dlon
        return max(dx, 1.0), max(dy, 1.0)

    def terrain(self, lat_deg: float, lon_deg: float, n_cells: int = 2) -> dict:
        """采样地形：高程、坡度（rad）、坡向（rad）、粗糙度（m）。

        坡度用中心差分（Horn 式），基线 = n_cells 个栅格。**基线尺度必须如实披露**：
        16 ppd 下 n_cells=2 时基线约 3.8 km，远大于漫游车尺度，
        因此坡度是区域地形趋势，不是车轮级可通过性判据（见文档 §限制）。
        """
        row_c, col_c = self._rowcol(lat_deg, lon_deg)
        h = self._at(row_c, col_c)
        hE = self._at(row_c, col_c + n_cells)
        hW = self._at(row_c, col_c - n_cells)
        hN = self._at(row_c - n_cells, col_c)
        hS = self._at(row_c + n_cells, col_c)
        dx, dy = self.pixel_spacing_m(lat_deg)
        dzdx = (hE - hW) / (2.0 * n_cells * dx)
        dzdy = (hN - hS) / (2.0 * n_cells * dy)
        slope = math.atan(math.hypot(dzdx, dzdy))
        aspect = math.atan2(dzdy, dzdx)
        sample = [self._at(row_c + i, col_c + j)
                  for i in (-n_cells, 0, n_cells) for j in (-n_cells, 0, n_cells)]
        mean = sum(sample) / len(sample)
        var = sum((s - mean) ** 2 for s in sample) / len(sample)
        return {"elev_m": h, "slope_rad": slope, "aspect_rad": aspect,
                "roughness_m": math.sqrt(var)}

    def _rowcol(self, lat_deg: float, lon_deg: float) -> tuple[int, int]:
        lat = min(max(lat_deg, -90.0), 90.0)
        row_f = (90.0 - lat) / 180.0 * (self.nrows - 1)
        col_f = (lon_deg % 360.0) / 360.0 * self.ncols
        return int(round(row_f)), int(round(col_f)) % self.ncols


def make_synthetic_crater_dem(path: str, ncols: int = 288, nrows: int = 144,
                              seed: int = 20260911,
                              n_craters: int = 4000) -> str:
    """生成**合成**月面高程（陨石坑场），用于无网络/无数据环境下的可复现自检。

    坑的直径按公开的撞击坑尺度-频率分布取幂律 N(>D) ∝ D^-b（b≈2，陨石坑饱和面量级）。
    剖面用经典的碗形 + 提边。
    ⚠️ 这是合成数据，units 标记为 "synthetic"，**不是真实月面**，不得用于任何关于真实地形的结论。
    """
    import random
    rng = random.Random(seed)
    grid = [0.0] * (ncols * nrows)
    for i in range(nrows):
        lat = 90.0 - 180.0 * i / (nrows - 1)
        for j in range(ncols):
            grid[i * ncols + j] = 200.0 * math.sin(math.radians(lat) * 3.0)
    for _ in range(n_craters):
        d_km = 10.0 ** rng.uniform(math.log10(0.5), math.log10(60.0))
        r_px = max(1.0, (d_km * 1000.0 / 2.0) / (R_MOON * math.pi / nrows))
        cx = rng.uniform(0, ncols)
        cy = rng.uniform(0, nrows)
        depth = d_km * 1000.0 * 0.15
        x0, x1 = int(cx - 1.5 * r_px), int(cx + 1.5 * r_px)
        y0, y1 = int(cy - 1.5 * r_px), int(cy + 1.5 * r_px)
        for i in range(max(0, y0), min(nrows, y1 + 1)):
            for j in range(max(0, x0), min(ncols, x1 + 1)):
                r = math.hypot(j - cx, i - cy)
                u = r / r_px
                if u < 1.0:
                    grid[i * ncols + j] += -depth * (1.0 - u * u) + depth * 0.15 * (u ** 4)
                elif u < 1.3:
                    grid[i * ncols + j] += depth * 0.15 * math.exp(-((u - 1.0) / 0.12) ** 2)
    a = array.array("h")
    for v in grid:
        a.append(int(max(-32768, min(32767, round(v / 0.5)))))
    with open(path, "wb") as fh:
        a.tofile(fh)
    return path


# ---------------------------------------------------------------- 简化光照

def solar_elevation_rad(lat_deg: float, lon_deg: float, t_s: float) -> float:
    """简化月球太阳高度角（rad）。

    ⚠️ 简化模型：以朔望月为"月球日"周期，太阳赤纬随月球自转轴倾角（±1.54°）年周期摆动。
    忽略轨道偏心率、章动、以及精确历元。**仅用于判定"在不在照明区"，
    幅度精度不足 10°，不可用于精密能量预算。** 真实工具用 NASA SPICE（见对标基线）。
    """
    phase = 2.0 * math.pi * (t_s % SYNODIC_MONTH_S) / SYNODIC_MONTH_S
    lon_sub = 180.0 - math.degrees(phase)
    lat_sub = LUNAR_OBLIQUITY_DEG * math.sin(2.0 * math.pi * t_s / YEAR_S)
    la = math.radians(lat_deg)
    ls = math.radians(lat_sub)
    dl = math.radians(lon_deg - lon_sub)
    s = math.sin(la) * math.sin(ls) + math.cos(la) * math.cos(ls) * math.cos(dl)
    return math.asin(max(-1.0, min(1.0, s)))


# ---------------------------------------------------------------- 状态与转移


@dataclass
class State:
    x_m: float = 0.0            # 以起点为原点的局部东向坐标
    y_m: float = 0.0            # 局部北向坐标
    psi_rad: float = 0.0        # 航向（0 = 东，逆时针）
    v_mps: float = 0.0
    battery_wh: float = 600.0
    t_s: float = 0.0


@dataclass
class Action:
    v_cmd_mps: float = 0.0
    psi_rate_rad_s: float = 0.0


@dataclass
class StepInfo:
    blocked: bool = False
    reason: str = ""
    slope_rad: float = 0.0
    theta_limit_rad: float = 0.0
    slip_ratio: float = 0.0
    distance_m: float = 0.0
    energy_wh: float = 0.0
    solar_wh: float = 0.0
    sun_elev_rad: float = 0.0
    ele_m: float = 0.0


@dataclass
class LunarWorldModel:
    """解析式月球世界模型：s_{t+1} = f(s_t, a_t)。"""

    dem: LolaDEM
    params: RoverParams = field(default_factory=RoverParams)
    origin_lat: float = 0.0
    origin_lon: float = 0.0
    slip_k: float = 0.25        # ⚠️ 最弱假设：滑移随坡度比平方增长
    slip_max: float = 0.60      # ⚠️ 滑移比上限

    DM_PER_DEG_LAT = 111319.49  # 由 R_MOON 推得的近似（仅用于局部平面展开）

    def _latlon(self, x_m: float, y_m: float) -> tuple[float, float]:
        lat = self.origin_lat + math.degrees(y_m / R_MOON)
        coslat = max(math.cos(math.radians(lat)), 1e-3)
        lon = self.origin_lon + math.degrees(x_m / (R_MOON * coslat))
        return lat, lon

    def _slip(self, slope_rad: float, theta_lim: float) -> float:
        """滑移比（0=无滑移）。⚠️ 这是一个**参数化假设**，不是轮-土力学仿真；
        本模型最弱的一环，已在文档中列为首要限制。"""
        if not math.isfinite(theta_lim) or theta_lim <= 0.0:
            return min(self.slip_max, self.slip_k)
        ratio = min(slope_rad / theta_lim, 1.0)
        return min(self.slip_max, self.slip_k * ratio * ratio)

    def step(self, s: State, a: Action, dt: float) -> tuple[State, StepInfo]:
        info = StepInfo()
        lat, lon = self._latlon(s.x_m, s.y_m)
        ter = self.dem.terrain(lat, lon)
        info.slope_rad = ter["slope_rad"]
        info.ele_m = ter["elev_m"]

        lim = theta_limit(self.params, self.params.v_ref)
        if not math.isfinite(lim):
            lim = math.pi / 2.0
        info.theta_limit_rad = lim

        if ter["slope_rad"] > lim:
            info.blocked = True
            info.reason = "slope_exceeds_limit"
            info.distance_m = 0.0
            s2 = State(s.x_m, s.y_m, s.psi_rad, 0.0, s.battery_wh, s.t_s + dt)
            return s2, info

        sun = solar_elevation_rad(lat, lon, s.t_s)
        info.sun_elev_rad = sun

        v_avail = 0.0 if sun <= 0.0 else max_speed_on_slope(self.params, ter["slope_rad"])
        v = max(0.0, min(a.v_cmd_mps, v_avail))
        info.slip_ratio = self._slip(ter["slope_rad"], lim)
        d = v * dt * (1.0 - info.slip_ratio)
        info.distance_m = d

        e_tract_j = self.params.mass_kg * G_MOON * (
            math.sin(ter["slope_rad"]) + self.params.crr * math.cos(ter["slope_rad"])
        ) * d / self.params.eta
        e_avionics_j = self.params.p_avionics_w * dt
        e_wh = (e_tract_j + e_avionics_j) / J_PER_WH
        solar_wh = 0.0
        if sun > 0.0:
            solar_wh = self.params.p_solar_w * max(0.0, math.sin(sun)) * dt / J_PER_WH
        info.energy_wh = e_wh
        info.solar_wh = solar_wh

        batt = s.battery_wh - e_wh + solar_wh
        if batt < 0.0:
            batt = 0.0

        psi = s.psi_rad + a.psi_rate_rad_s * dt
        x = s.x_m + d * math.cos(psi)
        y = s.y_m + d * math.sin(psi)
        s2 = State(x, y, psi, v, batt, s.t_s + dt)
        if batt <= 0.0:
            info.reason = "battery_depleted"
        return s2, info

    def rollout(self, s0: State, actions: Sequence[Action], dt: float) -> dict:
        """前向推演：预测未来状态序列。这是"世界模型"能力的核心体现。"""
        s = State(**asdict(s0))
        traj, infos = [], []
        for a in actions:
            s, info = self.step(s, a, dt)
            traj.append(asdict(s))
            infos.append(asdict(info))
            if info.reason == "battery_depleted":
                break
        return {
            "trajectory": traj,
            "info": infos,
            "n_steps": len(traj),
            "final": asdict(s),
            "termination": infos[-1]["reason"] if infos else "horizon",
        }


# ---------------------------------------------------------------- CLI

def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="KW-MOON-0001 月球世界模型（解析式）")
    ap.add_argument("--dem", default=os.environ.get("KW_MOON_DEM", ""),
                    help="LOLA GDR .img 路径；留空则自动探测 data/lola/ldem_16.img")
    ap.add_argument("--synthetic", action="store_true", help="强制使用合成陨石坑高程")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--dt", type=float, default=60.0)
    ap.add_argument("--v-cmd", type=float, default=0.05)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dem_path = args.dem or os.path.join(root, "data", "lola", "ldem_16.img")
    units = "real"
    if args.synthetic or not os.path.exists(dem_path):
        dem_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_synthetic_dem.img")
        make_synthetic_crater_dem(dem_path)
        units = "synthetic"
        dem = LolaDEM(dem_path, ncols=288, nrows=144)
        origin = (0.0, 0.0)
    else:
        dem = LolaDEM(dem_path)
        origin = (-89.0, 0.0)      # 南极区，Shackleton 一带
    wm = LunarWorldModel(dem=dem, origin_lat=origin[0], origin_lon=origin[1])
    s0 = State(battery_wh=wm.params.battery_wh)
    actions = [Action(v_cmd_mps=args.v_cmd, psi_rate_rad_s=0.0) for _ in range(args.steps)]
    res = wm.rollout(s0, actions, args.dt)
    out = {"dem_source": dem_path, "dem_units": units,
           "theta_limit_deg": math.degrees(theta_limit(wm.params, wm.params.v_ref)),
           **res}
    txt = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(txt[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
