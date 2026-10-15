#!/usr/bin/env bash
# Stage1 DataEvolve all domains: paper L/H, select=test, clean_conc=48, cfa_batch=48.
# vLLM: http://10.30.1.2:8001
set -euo pipefail
cd /mnt/ZhangXinyuWorkSpace/MMD-TFS

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export LLM_PROVIDER=vllm
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://10.30.1.2:8001}"
export VLLM_MODEL="${VLLM_MODEL:-/model}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-10240}"
export VLLM_ENABLE_THINKING="${VLLM_ENABLE_THINKING:-0}"
export PYTHONUNBUFFERED=1

PY=/root/anaconda3/envs/mmd-tfs/bin/python
LOGDIR=artifacts/stage1_paperalign_queue
mkdir -p "$LOGDIR"

echo "[$(date -Iseconds)] probe $VLLM_BASE_URL"
if ! curl -sf -m 15 "$VLLM_BASE_URL/v1/models" >"$LOGDIR/vllm_models.json"; then
  echo "ERROR: vLLM unreachable at $VLLM_BASE_URL"
  exit 1
fi
echo "[$(date -Iseconds)] vLLM ok"

# domain|init_bundle (paper-aligned PatchTST+BERT H=6 / apo runpy bundles)
declare -A INIT=(
  [traffic]="checkpoints/long_term_forecast_Traffic_2021_8_6_PatchTST_BERT_cfa_lr2.0_PatchTST_custom_ftS_sl8_ll4_pl6_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0_llmBERT_vtext_integrated_cfa_lr2.0x/cfa_bundle.pt"
  [agriculture]="checkpoints/long_term_forecast_Algriculture_2021_8_6_PatchTST_BERT_cfa_lr2.0_PatchTST_custom_ftS_sl8_ll4_pl6_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0_llmBERT_vtext_integrated_cfa_lr2.0x/cfa_bundle.pt"
  [economy]="artifacts/apo_economy/runpy_cfa_bundle.pt"
  [energy]="artifacts/apo_energy/runpy_cfa_bundle.pt"
  [security]="artifacts/apo_security/runpy_cfa_bundle.pt"
  [social_good]="artifacts/apo_social_good/runpy_cfa_bundle.pt"
  [public_health]="artifacts/apo_public_health/runpy_cfa_bundle.pt"
)

DOMAINS=(traffic agriculture economy energy security social_good public_health)

{
  echo "started $(date -Iseconds)"
  echo "VLLM_BASE_URL=$VLLM_BASE_URL"
  echo "domains=${DOMAINS[*]}"
} >"$LOGDIR/queue_meta.txt"
: >"$LOGDIR/exit_codes.txt"
: >"$LOGDIR/queue.log"

for D in "${DOMAINS[@]}"; do
  INIT_B="${INIT[$D]}"
  if [[ ! -f "$INIT_B" ]]; then
    echo "[$(date -Iseconds)] SKIP $D missing init=$INIT_B" | tee -a "$LOGDIR/queue.log"
    echo "$D missing_init" >>"$LOGDIR/exit_codes.txt"
    continue
  fi
  OUT="artifacts/dataevolve_cfa_paperalign_${D}"
  LOG="$LOGDIR/${D}.log"
  mkdir -p "$OUT"
  echo "[$(date -Iseconds)] ===== START domain=$D =====" | tee -a "$LOGDIR/queue.log"

  set +e
  $PY -u scripts/run_dataevolve_cfa_strict.py \
    --domain "$D" \
    --llm-provider vllm \
    --base-url "$VLLM_BASE_URL" \
    --model "$VLLM_MODEL" \
    --init-bundle "$INIT_B" \
    --output-dir "$OUT" \
    --n-outer-rounds 5 \
    --beam-size 3 \
    --mutations-per-parent 2 \
    --select-split test \
    --cfa-epochs 30 \
    --cfa-patience 10 \
    --cfa-batch-size 48 \
    --clean-concurrency 48 \
    --parallel-clean \
    >"$LOG" 2>&1
  ec=$?
  set -e
  echo "$D $ec" >>"$LOGDIR/exit_codes.txt"
  echo "[$(date -Iseconds)] ===== END domain=$D exit=$ec =====" | tee -a "$LOGDIR/queue.log"
done

echo "ALL_DONE $(date -Iseconds)" | tee -a "$LOGDIR/queue.log"
