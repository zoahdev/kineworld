#!/usr/bin/env bash
# KineWorld Phase-0 Push-T benchmark runner (single-GPU official JEPA-WM baseline)
#   - launches nvidia-smi sampler for peak VRAM + GPU util
#   - runs evals.main (single-GPU --debug, gloo dist fallback patched)
#   - teardown sampler, compute metrics, append a row to results/experiments.csv
# Usage:
#   bash run_pt_bench.sh --episodes 96 --tag full96
#   bash run_pt_bench.sh --episodes 1  --tag smoke (default)
set -euo pipefail

WS="C:/Users/zoah/WorkBuddy/2026-09-01-10-59-54/kineworld"
REPO="$WS/external/jepa-wms"
PY="$WS/.venv310/Scripts/python.exe"
RESULTS_ROOT="$WS/results/phase0_jepa_wm_pusht"

EPISODES=1
TAG="smoke"
while [[ $# -gt 0 ]]; do
  case $1 in
    --episodes) EPISODES=$2; shift 2;;
    --tag) TAG=$2; shift 2;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

NAME="pt_L2_cem_sourcedset_H6_nas6_ctxt2_ep${EPISODES}"
if [[ "$TAG" != "smoke" ]]; then
  NAME="${NAME}_${TAG}"
else
  # smoke config is generated with --suffix smoke
  NAME="${NAME}_smoke"
fi
CFG="$REPO/configs/dump_online_evals/pt/${NAME}.yaml"
LOG="$RESULTS_ROOT/${NAME}.log"

# env for jepa-wms
export JEPAWM_DSET="$WS/data"
export JEPAWM_LOGS="$RESULTS_ROOT/jepa_logs"
export JEPAWM_HOME="$WS"
export WANDB_MODE=disabled
# proxy routing: workbuddy session proxy (127.0.0.1:57875) 502s on github; 7897 is open
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy 2>/dev/null || true
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897

echo "[runner] episodes=$EPISODES tag=$TAG cfg=$CFG"
[[ -f "$CFG" ]] || { echo "[runner] config missing: $CFG — run scripts/gen_pt_eval_configs.py first"; exit 1; }

# The Python runner owns the single-evaluation lock, GPU sampling, stdout log,
# and JSON/CSV evidence.  Delegating here prevents WorkBuddy and Codex from
# accidentally launching two memory-heavy official evaluations.
exec "$PY" "$WS/scripts/run_pt_bench.py" --config "$CFG"
