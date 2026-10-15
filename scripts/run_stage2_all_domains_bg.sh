#!/usr/bin/env bash
# Launch Stage2 multi-domain queue in background.
#
# Optional (Zhipu free pack):
#   export ZAI_API_KEY='...'
#   # or put key in ~/.config/mmd-tfs/zhipu.env  (ZAI_API_KEY=...)
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p artifacts/stage2_all_domains
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LLM_PROVIDER="${LLM_PROVIDER:-zhipu}"
export VLLM_MODEL="${VLLM_MODEL:-glm-5.3-flash}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-https://open.bigmodel.cn/api/paas/v4}"
export VLLM_REASONING_EFFORT="${VLLM_REASONING_EFFORT:-low}"
export VLLM_ENABLE_THINKING=0
# Skip traffic; re-fill agri + extract remaining domains
export STAGE2_DOMAINS="${STAGE2_DOMAINS:-agriculture,economy,energy,public_health,security,social_good}"
export STAGE2_SKIP_EXTRACT="${STAGE2_SKIP_EXTRACT:-agriculture}"

# Source optional env file without echoing secrets
if [[ -f "${HOME}/.config/mmd-tfs/zhipu.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${HOME}/.config/mmd-tfs/zhipu.env"
  set +a
fi
if [[ -f "${ROOT}/.zhipu.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.zhipu.env"
  set +a
fi

PY="${MMD_PY:-/root/anaconda3/envs/mmd-tfs/bin/python}"
nohup "$PY" -u scripts/run_stage2_all_domains_bg.py \
  > artifacts/stage2_all_domains/run.log 2>&1 &
echo $! | tee artifacts/stage2_all_domains/run.pid
echo "log: artifacts/stage2_all_domains/run.log"
