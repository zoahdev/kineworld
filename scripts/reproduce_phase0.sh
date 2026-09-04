#!/usr/bin/env bash
# -*- coding: utf-8 -*-
"""
KineWorld Phase 0 — 一键复现脚本（Windows Git Bash / Linux 兼容）。

按 docs/phase0_report.md §8 复现完整流程：
  1) 校验环境（venv、数据集、checkpoint、GPU）
  2) 生成评测配置（缩减 CEM 10×100×10）
  3) 运行规划 benchmark（run_pt_bench.py，N 个 episodes）
  4) 运行 rollout-error 评测（隐空间预测误差 1/5/10/20/50 步）
  5) 重建统一 experiments.csv

用法（在 kineworld 根目录）：
  bash scripts/reproduce_phase0.sh [--episodes 15] [--quick-debug]

说明：
  - 默认正式 benchmark 用缩减 CEM（10 iter × 100 samples × 10 elites），
    因为完整 CEM（30×300）本机单次规划 >18min 不可行（见报告 §12.2）。
  - 所有 JEPAWM_* 环境变量由脚本自动设置，无需手动 export。
"""
set -euo pipefail

# --- 定位工作目录 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$WS/.venv310/Scripts/python.exe"
# Linux fallback（.venv310/bin/python）
[ -x "$PY" ] || PY="$WS/.venv310/bin/python"
REPO="$WS/external/jepa-wms"

# --- 参数 ---
EPISODES=15
QUICK_DEBUG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes) EPISODES="$2"; shift 2 ;;
    --quick-debug) QUICK_DEBUG="--quick-debug"; shift ;;
    --help|-h) sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

# --- 环境变量 ---
export JEPAWM_DSET="$WS/data"
export JEPAWM_LOGS="$WS/results/jepa_logs"
export JEPAWM_HOME="$WS"
export JEPAWM_CKPT="$WS/checkpoints"
export SDL_VIDEODRIVER=dummy
export WANDB_MODE=disabled

echo "======================================"
echo " KineWorld Phase 0 一键复现"
echo " 工作目录 : $WS"
echo " episodes : $EPISODES  $QUICK_DEBUG"
echo "======================================"

# --- 1) 环境校验 ---
echo ""
echo "[1/5] 环境校验 ..."
[ -f "$PY" ] || { echo "✗ 找不到 venv python: $PY"; exit 1; }
[ -d "$WS/data/pusht_noise" ] || { echo "✗ 数据集缺失: data/pusht_noise"; exit 1; }
[ -f "$WS/checkpoints/jepa_wm_pusht.pth.tar" ] || { echo "✗ checkpoint 缺失"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "⚠ GPU 不可用（将跑 CPU，极慢）"
echo "✓ 环境就绪"

# --- 2) 生成配置 ---
echo ""
echo "[2/5] 生成评测配置 (缩减 CEM 10×100×10) ..."
SUFFIX="red10x100"
if [ -n "$QUICK_DEBUG" ]; then
  SUFFIX="smoke"
fi
"$PY" scripts/gen_pt_eval_configs.py \
  --planner cem --cost L2 --episodes "$EPISODES" --suffix "$SUFFIX" \
  --cem_iter 10 --cem_samples 100 --cem_elites 10 $QUICK_DEBUG

CFG="$REPO/configs/dump_online_evals/pt/pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep${EPISODES}_${SUFFIX}.yaml"
[ -f "$CFG" ] || { echo "✗ 配置生成失败"; exit 1; }
echo "✓ 配置: $CFG"

# --- 3) 规划 benchmark ---
echo ""
echo "[3/5] 运行规划 benchmark (${EPISODES} episodes) ..."
"$PY" scripts/run_pt_bench.py --config "$CFG"
echo "✓ benchmark 完成（结果在 results/runs/ 与 results/experiments.csv）"

# --- 4) rollout-error 评测 ---
echo ""
echo "[4/5] 运行 rollout-error 评测 (1/5/10/20/50 步) ..."
"$PY" kineworld/eval/latent_rollout_error.py \
  --cfg "$CFG" \
  --episodes 12 --horizons 1,5,10,20,50 \
  --out "$WS/results/rollout_err.json"
echo "✓ rollout-error 完成（结果在 results/rollout_err.json）"

# --- 5) 重建 experiments.csv ---
echo ""
echo "[5/5] 重建统一 experiments.csv ..."
"$PY" scripts/rebuild_experiments_csv.py

echo ""
echo "======================================"
echo " ✅ Phase 0 复现完成"
echo " 报告   : docs/phase0_report.md"
echo " 实验日志: results/experiments.csv"
echo " rollout: results/rollout_err.json"
echo "======================================"
