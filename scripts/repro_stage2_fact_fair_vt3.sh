#!/usr/bin/env bash
# Fair (honest) Fact LatentCoT VT3: train-only, chrono val for ckpt/α (NO test leak).
# Use this for paper "main fair table"; VT val=test is diagnostic only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PY="${MMD_PY:-/root/anaconda3/envs/mmd-tfs/bin/python}"
export HF_HOME="${HF_HOME:-/tmp/hf_qwen06}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DOMAINS="${FACT_DOMAINS:-agriculture,economy,energy,security,social_good,public_health,traffic}"
OUT=artifacts/stage2_fact_fair_vt3
mkdir -p "$OUT"
MODEL="${FACT_MODEL:-/tmp/hf_qwen06/Qwen3-0.6B}"
BS="${FACT_BS:-48}"
EPOCHS="${FACT_EPOCHS:-8}"

IFS=',' read -ra DS <<< "$DOMAINS"
for d in "${DS[@]}"; do
  d="$(echo "$d" | xargs)"
  [[ -z "$d" ]] && continue
  tag="fair_vt3"
  log="$OUT/${d}.log"
  echo "===== fair VT3 $d ====="
  "$PY" -u scripts/fair_branch_resid.py \
    --domain "$d" \
    --tag "$tag" \
    --skip-soft --lora \
    --val-mode chrono --val-frac 0.2 --amp-select val \
    --epochs "$EPOCHS" --batch-size "$BS" --lr-fact 1e-4 --n-latent 12 \
    --coconut-recycle --adaptive-halt --n-thought-groups 3 \
    --model "$MODEL" \
    2>&1 | tee "$log"
done

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("artifacts/stage2_fact_improve_ablate/fair_vt3")
print("\n=== Fair VT3 gated rel ===")
vals=[]
for p in sorted(root.glob("*/summary.json")):
    s=json.loads(p.read_text())
    r=(s.get("B_fact_latentcot_alone") or {}).get("rel")
    print(f"  {p.parent.name}: {None if r is None else f'{r*100:+.2f}%'}")
    if isinstance(r,(int,float)): vals.append(r)
if vals:
    print(f"  mean: {sum(vals)/len(vals)*100:+.2f}%")
PY
