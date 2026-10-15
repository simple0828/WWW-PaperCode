#!/usr/bin/env bash
# (Optional) Re-extract events.jsonl per domain. Shipped repo already includes events.
# Requires vLLM server.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PY="${MMD_PY:-/root/anaconda3/envs/mmd-tfs/bin/python}"
export LLM_PROVIDER="${LLM_PROVIDER:-vllm}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://10.30.1.2:8001}"
export VLLM_MODEL="${VLLM_MODEL:-/model}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

DOMAINS="${STAGE2_DOMAINS:-traffic,agriculture,economy,energy,public_health,security,social_good}"
IFS=',' read -ra DS <<< "$DOMAINS"
for d in "${DS[@]}"; do
  d="$(echo "$d" | xargs)"
  echo "===== extract_events $d ====="
  "$PY" -u scripts/extract_events.py \
    --domain "$d" \
    --llm-provider vllm \
    --base-url "$VLLM_BASE_URL" \
    --model "$VLLM_MODEL" \
    --resume
done
