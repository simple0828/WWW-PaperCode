#!/usr/bin/env bash
# Reproduce Scheme5 v5 best-online (test_best_epoch_online) on all domains.
# Requires: Stage1 best CFA (shipped under artifacts/), events.jsonl (shipped),
#           vLLM OpenAI-compatible server for protocol infer/rewrite.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export MMD_PY="${MMD_PY:-/root/anaconda3/envs/mmd-tfs/bin/python}"
export LLM_PROVIDER="${LLM_PROVIDER:-vllm}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://10.30.1.2:8001}"
export VLLM_MODEL="${VLLM_MODEL:-/model}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=true
export SCHEME5_V2_OUT="${SCHEME5_V2_OUT:-artifacts/scheme5_v5}"
export SCHEME5_FORCE="${SCHEME5_FORCE:-0}"
export SCHEME5_PICK_EPOCH=best
export SCHEME5_CACHE_TAG=s5v5
export SCHEME5_WAVE=48
export SCHEME5_EPOCHS=2
export SCHEME5_DEAD_SR=0
export SCHEME5_FIXED_AMP=1.0
export SCHEME5_MAX_LESSONS=20

echo "[repro] Scheme5 v5 → $SCHEME5_V2_OUT"
bash "$ROOT/scripts/run_scheme5_v2_all.sh"

"$MMD_PY" - <<'PY'
import json
from pathlib import Path
out = Path("artifacts/scheme5_v5")
print("\n=== Scheme5 v5 test_best_epoch_online ===")
rows = []
for p in sorted(out.glob("*_eval.json")):
    j = json.loads(p.read_text())
    online = j.get("test_best_epoch_online") or {}
    rel = online.get("rel_pct", online.get("rel"))
    print(f"  {j.get('domain')}: {rel}")
    if isinstance(rel, (int, float)):
        rows.append(float(rel))
if rows:
    print(f"  mean: {sum(rows)/len(rows):+.4f}")
PY
