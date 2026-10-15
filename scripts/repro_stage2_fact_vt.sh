#!/usr/bin/env bash
# Reproduce Fact LatentCoT VT0→VT1→VT2→VT3 (train-only + val=test diagnostic).
# Main paper table for implicit tricks; requires Qwen3-0.6B + GPU + Stage1/events.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export MMD_PY="${MMD_PY:-/root/anaconda3/envs/mmd-tfs/bin/python}"
export HF_HOME="${HF_HOME:-/tmp/hf_qwen06}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FACT_EPOCHS="${FACT_EPOCHS:-8}"
export FACT_BS="${FACT_BS:-48}"
export FACT_BS_FALLBACKS="${FACT_BS_FALLBACKS:-48,40,32,24,16,8}"
export FACT_N_LATENT="${FACT_N_LATENT:-12}"
export FACT_LR="${FACT_LR:-1e-4}"
export FACT_DOMAINS="${FACT_DOMAINS:-agriculture,economy,energy,security,social_good,public_health,traffic}"

mkdir -p artifacts/stage2_fact_valtest_tricks
echo "[repro] Fact VT0–VT3 → artifacts/stage2_fact_valtest_tricks"
nohup "$MMD_PY" -u scripts/run_fact_valtest_tricks_bg.py \
  > artifacts/stage2_fact_valtest_tricks/run.log 2>&1 &
echo $! | tee artifacts/stage2_fact_valtest_tricks/run.pid
echo "PID=$(cat artifacts/stage2_fact_valtest_tricks/run.pid)"
echo "tail -f artifacts/stage2_fact_valtest_tricks/run.log"
