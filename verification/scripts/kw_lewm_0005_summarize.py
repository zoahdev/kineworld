"""KW-LEWM-0005 K4 结果后处理：输出与 KW-LEWM-0004 并排的对比表 + Wilson CI。

读 results/kw_lewm_0005_jepa_same_platform.json，打印：
  - 关键判据（out_of_play_rate, block_interaction_rate, displacement 中位数/均值, success）
  - 与 0004 基线（success 0.36, n_success 18, out_of_play 0.0）并排
  - K4 Kill 判定
  - 若 success 有值，给 Wilson 95% CI
不依赖任何外部统计库（纯 numpy）。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "results" / "kw_lewm_0005_jepa_same_platform.json"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (float(p), float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    data = json.loads(path.read_text(encoding="utf-8"))
    s = data["summary"]
    kc = data["kill_criteria"]
    n = data["n_tasks"]

    out_of_play = s.get("out_of_play_rate")
    n_oop = s.get("n_out_of_play")
    interacted = s.get("block_interaction_rate")
    disp_mean = s.get("block_displacement_mean")
    disp_med = s.get("block_displacement_median")
    succ = s.get("success_rate")
    n_succ = s.get("n_success")
    n_err = s.get("n_errors")

    # success 三重统计（中位数优先，均值并报；n 小用 Wilson）
    print("=" * 72)
    print("KW-LEWM-0005 K4 — JEPA-WM 同台闭环（同平台/同任务/同 CEM 10×100×10）")
    print("=" * 72)
    print(f"n_tasks            = {n}")
    print(f"goal_offset        = {data.get('goal_offset')}")
    print(f"eval_budget        = {data.get('eval_budget')}")
    print(f"n_errors           = {n_err}")
    print("-" * 72)
    print("关键判据:")
    print(f"  agent 越界率      = {out_of_play}  (n_out_of_play={n_oop})")
    print(f"  block 交互率(>1) = {interacted}")
    print(f"  block 位移均值    = {disp_mean}")
    print(f"  block 位移中位数  = {disp_med}")
    print(f"  success_rate      = {succ}  (n_success={n_succ})")
    if succ is not None and n_succ is not None:
        p, lo, hi = wilson(int(n_succ), n)
        print(f"  Wilson 95% CI     = [{lo:.3f}, {hi:.3f}]  (width {hi-lo:.3f})")
    print("-" * 72)
    print("K4 Kill 判定:")
    for k, v in kc.items():
        print(f"  {k:42s} = {v}")
    print("-" * 72)
    print("与 KW-LEWM-0004（LeWM，同平台/同任务/同 CEM）并排:")
    print(f"  {'指标':<22}{'0004 LeWM':>16}{'0005 JEPA-WM':>18}")
    print(f"  {'success':<22}{'0.36 (18/50)':>16}{str(succ):>18}")
    if succ is not None and n_succ is not None:
        p, lo, hi = wilson(int(n_succ), n)
        print(f"  {'success Wilson CI':<22}{'[0.241,0.499]':>16}{f'[{lo:.3f},{hi:.3f}]':>18}")
    print(f"  {'agent 越界率':<22}{'0.00':>16}{str(out_of_play):>18}")
    print(f"  {'block 交互率':<22}{'0.88':>16}{str(interacted):>18}")
    print(f"  {'block 位移中位数':<22}{'31.54':>16}{str(disp_med):>18}")
    print(f"  {'block 位移均值':<22}{'51.56':>16}{str(disp_mean):>18}")
    print("=" * 72)

    verdict = "PASS" if (kc.get("K4_control_within_play_area") or kc.get("K4_out_of_play_rate_at_most_0.1")) else "FAIL/NEGATIVE"
    print(f"K4 整体判定倾向: {verdict}")


if __name__ == "__main__":
    main()
