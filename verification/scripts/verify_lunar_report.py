"""
KW-MOON-0001 — 文档数字机器校验（AGENTS.md 硬纪律 7）

对外文档里的数字**必须由脚本生成，不得手抄**（历史事故：把 75.8468 抄成 75.9）。

机制：
  * 文档中所有数字写成占位符 `{{N:点路径:小数位}}`，例如
        {{N:limits.theta_limit_deg:4}}
  * `--write`  从 results/kw_moon_0001_validation.json 取值，回填占位符，
               并重写 `<!-- VERIFY:START --> ... <!-- VERIFY:END -->` 校验块。
  * `--check`  重算并逐项比对文档现值，任一不符即退出码 1。

用法：
  python verification/scripts/verify_lunar_report.py --write
  python verification/scripts/verify_lunar_report.py --check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

DOC_REL = os.path.join("docs", "research", "LUNAR_WORLD_MODEL_v0.md")
JSON_REL = os.path.join("results", "kw_moon_0001_validation.json")

TOKEN = re.compile(r"\{\{N:([A-Za-z0-9_.]+):(\d+)\}\}")
BLOCK = re.compile(r"<!-- VERIFY:START -->(.*?)<!-- VERIFY:END -->", re.S)

# 校验块里逐行核对的项目（顺序即文档顺序）
BLOCK_KEYS = [
    ("verdict", "verdict", None),
    ("n_checks", "n_checks", None),
    ("n_passed", "n_passed", None),
    ("limits.binding_constraint", "limits.binding_constraint", None),
    ("limits.theta_traction_deg", "limits.theta_traction_deg", 6),
    ("limits.theta_limit_deg", "limits.theta_limit_deg", 6),
    ("checks.K2.tol", "checks.K2.tol", None),
    ("checks.K2.worst_abs_deviation_rad", "checks.K2.worst_abs_deviation_rad", None),
    ("checks.K3.worst_rel_error", "checks.K3.worst_rel_error", None),
    ("checks.K4.rel_error", "checks.K4.rel_error", None),
    ("checks.K4.distance_got_m", "checks.K4.distance_got_m", 6),
    ("checks.K5.steep_slope_deg", "checks.K5.steep_slope_deg", 6),
    ("checks.K5.steep_displacement_m", "checks.K5.steep_displacement_m", None),
    ("checks.K6.termination", "checks.K6.termination", None),
    ("checks.K6.final_wh", "checks.K6.final_wh", None),
    ("checks.K7.identical", "checks.K7.identical", None),
    ("checks.K7.random_scenarios", "checks.K7.random_scenarios", None),
    ("checks.K7.scenarios_with_nan_or_inf", "checks.K7.scenarios_with_nan_or_inf", None),
    ("rover_params.mass_kg", "rover_params.mass_kg", 2),
    ("rover_params.mu", "rover_params.mu", 4),
    ("rover_params.crr", "rover_params.crr", 4),
    ("rover_params.p_max_w", "rover_params.p_max_w", 2),
    ("rover_params.t_wheel_nm", "rover_params.t_wheel_nm", 2),
    ("rover_params.v_ref", "rover_params.v_ref", 4),
]


def resolve(data: dict, path: str):
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def fmt(value, decimals) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if decimals is None:
            return repr(value)
        return f"{value:.{int(decimals)}f}"
    if isinstance(value, int) and decimals is not None:
        return f"{value:.{int(decimals)}f}"
    return str(value)


def build_block(data: dict) -> str:
    lines = ["<!-- VERIFY:START -->", "```text"]
    for label, path, dec in BLOCK_KEYS:
        lines.append(f"{label} = {fmt(resolve(data, path), dec)}")
    lines.append("```")
    lines.append("<!-- VERIFY:END -->")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    doc_path = os.path.join(root, DOC_REL)
    json_path = os.path.join(root, JSON_REL)
    if not os.path.exists(json_path):
        print(f"缺 JSON：{json_path}；先跑 kw_moon_0001_validate.py")
        return 1
    data = json.load(open(json_path, encoding="utf-8"))
    doc = open(doc_path, encoding="utf-8").read()

    fails: list[str] = []
    missing: list[str] = []

    # 1) 校验块
    new_block = build_block(data)
    m = BLOCK.search(doc)
    if not m:
        fails.append("文档缺少 VERIFY 校验块")
    else:
        if args.check and m.group(0) != new_block:
            old_lines = m.group(0).splitlines()
            new_lines = new_block.splitlines()
            for a, b in zip(old_lines, new_lines):
                if a != b:
                    fails.append(f"校验块不一致：文档『{a}』 vs 实际『{b}』")
            if len(old_lines) != len(new_lines):
                fails.append("校验块行数不一致")
        doc = doc[:m.start()] + new_block + doc[m.end():]

    # 2) 正文占位符
    def repl(mo: re.Match) -> str:
        path, dec = mo.group(1), int(mo.group(2))
        val = resolve(data, path)
        if val is None:
            missing.append(path)
            return mo.group(0)
        return fmt(val, dec)

    doc_new = TOKEN.sub(repl, doc)

    if args.write:
        open(doc_path, "w", encoding="utf-8").write(doc_new)
        left = TOKEN.findall(doc_new)
        print(f"已回填 {doc_path}")
        print(f"  剩余未解析占位符：{len(left)} {left[:5] if left else ''}")
        return 1 if (left or missing) else 0

    # --check
    if doc_new != doc:
        n = sum(1 for a, b in zip(doc.splitlines(), doc_new.splitlines()) if a != b)
        fails.append(f"正文占位符已过期（{n} 行需更新）；跑 --write 重新生成")
    if missing:
        fails.append(f"无法解析的路径：{sorted(set(missing))}")

    if fails:
        print("LUNAR REPORT CHECK: FAIL")
        for f in fails[:25]:
            print("  -", f)
        return 1
    n_num = len(BLOCK_KEYS)
    print(f"LUNAR REPORT CHECK: OK ({n_num} 校验项全部一致, 0 fail)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
