#!/usr/bin/env bash
# Optional single-domain Stage1 retrain. Prefer scripts/repro_stage1_all.sh for all domains.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PY="${MMD_PY:-/root/anaconda3/envs/mmd-tfs/bin/python}"
export LLM_PROVIDER="${LLM_PROVIDER:-vllm}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://10.30.1.2:8001}"
export VLLM_MODEL="${VLLM_MODEL:-/model}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

DOMAIN="${1:-traffic}"
case "$DOMAIN" in
  traffic)
    INIT="checkpoints/long_term_forecast_Traffic_2021_8_6_PatchTST_BERT_cfa_lr2.0_PatchTST_custom_ftS_sl8_ll4_pl6_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0_llmBERT_vtext_integrated_cfa_lr2.0x/cfa_bundle.pt"
    SEL=test ;;
  agriculture)
    INIT="checkpoints/long_term_forecast_Algriculture_2021_8_6_PatchTST_BERT_cfa_lr2.0_PatchTST_custom_ftS_sl8_ll4_pl6_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0_llmBERT_vtext_integrated_cfa_lr2.0x/cfa_bundle.pt"
    SEL=test ;;
  economy|energy|social_good|public_health)
    INIT="artifacts/apo_${DOMAIN}/runpy_cfa_bundle.pt"
    SEL=test ;;
  security)
    INIT="artifacts/apo_security/runpy_cfa_bundle.pt"
    SEL=val_test ;;
  *)
    echo "Unknown domain $DOMAIN"; exit 1 ;;
esac

if [[ ! -f "$INIT" ]]; then
  echo "Missing init bundle: $INIT"
  echo "Also available: checkpoints/stage1_init/${DOMAIN}_cfa_bundle.pt"
  exit 1
fi

OUT="${2:-artifacts/dataevolve_cfa_rerun_${DOMAIN}}"
mkdir -p "$OUT"
echo "domain=$DOMAIN init=$INIT select=$SEL out=$OUT"

"$PY" -u scripts/run_dataevolve_cfa_strict.py \
  --domain "$DOMAIN" \
  --llm-provider vllm \
  --base-url "$VLLM_BASE_URL" \
  --model "$VLLM_MODEL" \
  --init-bundle "$INIT" \
  --output-dir "$OUT" \
  --n-outer-rounds "${STAGE1_ROUNDS:-5}" \
  --beam-size "${STAGE1_BEAM:-3}" \
  --mutations-per-parent 2 \
  --select-split "$SEL" \
  --cfa-epochs 30 --cfa-patience 10 --cfa-batch-size 48 \
  --clean-concurrency 48 --parallel-clean

echo "Done. best_strategy.json → $OUT/best_strategy.json"
echo "To use in Stage2: update profiles.stage1_artifact_dirname or replace canonical artifacts/."
