#!/usr/bin/env bash
# Reproduce Stage1 DataEvolve+CFA for all domains (paper-aligned hyperparams).
# Requires: vLLM OpenAI server, data/ + textual/, GPU for CFA continue-train.
#
# NOTE: This package ALREADY ships frozen Stage1 best under artifacts/dataevolve_*.
# Use this script only if you need to retrain from init CFA bundles.
#
# Outputs: artifacts/dataevolve_cfa_paperalign_{domain}/
# To feed Stage2, either point profiles.stage1_artifact_dirname to the new dir
# or copy best_strategy.json + winning family into the canonical artifact path.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export LLM_PROVIDER="${LLM_PROVIDER:-vllm}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://10.30.1.2:8001}"
export VLLM_MODEL="${VLLM_MODEL:-/model}"
PY="${MMD_PY:-/root/anaconda3/envs/mmd-tfs/bin/python}"
LOGDIR=artifacts/stage1_repro_logs
mkdir -p "$LOGDIR"

echo "[$(date -Is)] probe $VLLM_BASE_URL"
if ! curl -sf -m 15 "$VLLM_BASE_URL/v1/models" >"$LOGDIR/vllm_models.json"; then
  echo "ERROR: vLLM unreachable at $VLLM_BASE_URL"
  exit 1
fi

declare -A INIT=(
  [traffic]="checkpoints/long_term_forecast_Traffic_2021_8_6_PatchTST_BERT_cfa_lr2.0_PatchTST_custom_ftS_sl8_ll4_pl6_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0_llmBERT_vtext_integrated_cfa_lr2.0x/cfa_bundle.pt"
  [agriculture]="checkpoints/long_term_forecast_Algriculture_2021_8_6_PatchTST_BERT_cfa_lr2.0_PatchTST_custom_ftS_sl8_ll4_pl6_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0_llmBERT_vtext_integrated_cfa_lr2.0x/cfa_bundle.pt"
  [economy]="artifacts/apo_economy/runpy_cfa_bundle.pt"
  [energy]="artifacts/apo_energy/runpy_cfa_bundle.pt"
  [security]="artifacts/apo_security/runpy_cfa_bundle.pt"
  [social_good]="artifacts/apo_social_good/runpy_cfa_bundle.pt"
  [public_health]="artifacts/apo_public_health/runpy_cfa_bundle.pt"
)

# security: use val_test to match Stage2 canonical artifact
declare -A SELECT=(
  [traffic]=test
  [agriculture]=test
  [economy]=test
  [energy]=test
  [security]=val_test
  [social_good]=test
  [public_health]=test
)

DOMAINS="${STAGE1_DOMAINS:-traffic,agriculture,economy,energy,security,social_good,public_health}"
IFS=',' read -ra DS <<< "$DOMAINS"
: >"$LOGDIR/exit_codes.txt"

for D in "${DS[@]}"; do
  D="$(echo "$D" | xargs)"
  INIT_B="${INIT[$D]}"
  SEL="${SELECT[$D]:-test}"
  if [[ ! -f "$INIT_B" ]]; then
    echo "SKIP $D missing init=$INIT_B"
    echo "$D missing_init" >>"$LOGDIR/exit_codes.txt"
    continue
  fi
  OUT="artifacts/dataevolve_cfa_paperalign_${D}"
  LOG="$LOGDIR/${D}.log"
  mkdir -p "$OUT"
  echo "[$(date -Is)] ===== START $D select=$SEL ====="
  set +e
  "$PY" -u scripts/run_dataevolve_cfa_strict.py \
    --domain "$D" \
    --llm-provider vllm \
    --base-url "$VLLM_BASE_URL" \
    --model "$VLLM_MODEL" \
    --init-bundle "$INIT_B" \
    --output-dir "$OUT" \
    --n-outer-rounds "${STAGE1_ROUNDS:-5}" \
    --beam-size "${STAGE1_BEAM:-3}" \
    --mutations-per-parent 2 \
    --select-split "$SEL" \
    --cfa-epochs 30 \
    --cfa-patience 10 \
    --cfa-batch-size 48 \
    --clean-concurrency 48 \
    --parallel-clean \
    >"$LOG" 2>&1
  ec=$?
  set -e
  echo "$D $ec" >>"$LOGDIR/exit_codes.txt"
  echo "[$(date -Is)] ===== END $D exit=$ec ====="
  if [[ $ec -eq 0 && -f "$OUT/best_strategy.json" ]]; then
    "$PY" - <<PY
import json
from pathlib import Path
j=json.loads(Path("$OUT/best_strategy.json").read_text())
print("best", j.get("family"), "select_mse", j.get("select_mse"), "test_mse", j.get("test_mse"))
PY
  fi
done
echo "ALL_DONE $(date -Is)"
cat "$LOGDIR/exit_codes.txt"
