#!/usr/bin/env bash
# Scheme5 v5 — streaming / online protocol (paper primary metric = best online).
#
# Protocol: walk windows in time order (train∪test). Later windows may use
# experience from earlier windows (causal online). Primary number:
#   test_best_epoch_online  (compare to baseline test_last_epoch_online)
# final_frozen is secondary / deployment freeze ablation only.
#
# fuse: fixed mean@amp=1; per-domain seeds; max-lessons=20; pick-epoch=best
#
# Usage:
#   nohup bash scripts/run_scheme5_v2_all.sh > artifacts/scheme5_v5/run.log 2>&1 &
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${MMD_PY:-/root/anaconda3/envs/mmd-tfs/bin/python}"
OUT="${SCHEME5_V2_OUT:-artifacts/scheme5_v5}"
mkdir -p "$OUT"
export PYTHONPATH="$ROOT"
export LLM_PROVIDER="${LLM_PROVIDER:-vllm}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://10.30.1.2:8001}"
export VLLM_MODEL="${VLLM_MODEL:-/model}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=true

DOMAINS="${STAGE2_DOMAINS:-traffic,economy,agriculture,security,social_good,public_health,energy}"
WAVE="${SCHEME5_WAVE:-48}"
N_EPOCHS="${SCHEME5_EPOCHS:-2}"
CACHE_TAG="${SCHEME5_CACHE_TAG:-s5v5}"
DEAD_SR="${SCHEME5_DEAD_SR:-0}"
FIXED_AMP="${SCHEME5_FIXED_AMP:-1.0}"
# empty = no --search-fuse (default fixed mean@amp)
SEARCH_FUSE="${SCHEME5_SEARCH_FUSE:-0}"

echo "[scheme5v5] start $(date -Is) wave=$WAVE tag=$CACHE_TAG dead_sr=$DEAD_SR fixed_amp=$FIXED_AMP search_fuse=$SEARCH_FUSE out=$OUT"

IFS=',' read -ra DS <<< "$DOMAINS"
for d in "${DS[@]}"; do
  d="$(echo "$d" | xargs)"
  [[ -z "$d" ]] && continue
  out_json="$OUT/${d}_eval.json"
  if [[ -f "$out_json" && "${SCHEME5_FORCE:-0}" != "1" ]]; then
    echo "[scheme5v5] skip $d (exists $out_json; set SCHEME5_FORCE=1 to redo)"
    continue
  fi
  echo "===== scheme5v5 $d $(date -Is) ====="
  extra=()
  if [[ "$SEARCH_FUSE" == "1" ]]; then
    extra+=(--search-fuse --select-on "${SCHEME5_SELECT_ON:-val}")
  fi
  "$PY" -u scripts/eval_scheme5.py \
    --domain "$d" \
    --n-epochs "$N_EPOCHS" \
    --wave "$WAVE" \
    --cache-tag "$CACHE_TAG" \
    --improve \
    --max-lessons "${SCHEME5_MAX_LESSONS:-20}" \
    --dead-sr "$DEAD_SR" \
    --fixed-amp "$FIXED_AMP" \
    --pick-epoch "${SCHEME5_PICK_EPOCH:-best}" \
    "${extra[@]}" \
    --out "$out_json" \
    2>&1 | tee "$OUT/${d}_run.log"
done

"$PY" - <<'PY'
import json
from pathlib import Path
out = Path("artifacts/scheme5_v5")
rows = []
for p in sorted(out.glob("*_eval.json")):
    j = json.loads(p.read_text())
    d = j["domain"]
    base_online = base_final = None
    for c in Path("artifacts").glob("*/scheme5_eval.json"):
        try:
            b = json.loads(c.read_text())
        except Exception:
            continue
        if b.get("domain") == d:
            base_online = (b.get("test_last_epoch_online") or {}).get("rel_pct")
            base_final = (b.get("test_final_frozen") or {}).get("rel_pct")
            break
    cfg = j.get("cfg") or {}
    online = (j.get("test_best_epoch_online") or j.get("test_last_epoch_online") or {}).get("rel_pct")
    final = (j.get("test_final_frozen") or {}).get("rel_pct")
    rows.append({
        "domain": d,
        "metric": "online (primary)",
        "baseline_online": base_online,
        "v5_best_online": online,
        "delta_online": None if base_online is None or online is None else online - base_online,
        "best_epoch": cfg.get("best_epoch"),
        "baseline_final_ref": base_final,
        "v5_final_ref": final,
        "fixed_amp": cfg.get("fixed_amp"),
        "max_lessons": cfg.get("max_lessons"),
    })
(out / "summary.json").write_text(json.dumps({"primary": "test_best_epoch_online_vs_baseline_online", "rows": rows}, indent=2))
print(json.dumps(rows, indent=2))
PY
echo "[scheme5v5] done $(date -Is)"
