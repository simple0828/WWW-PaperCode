# TimeMMD Residual Paper Code

Self-contained reproduction package for the **full paper pipeline**:

| Part | Method | Primary metric / artifact |
|------|--------|---------------------------|
| **Stage1** | DataEvolve + CFA continue-train | `best_strategy.json` + `cfa_bundle.pt`（已冻结；也可从 init 重训） |
| **Stage2 Explicit** | Scheme5 **v5** best-online | `test_best_epoch_online` |
| **Stage2 Implicit** | Fact LatentCoT **VT1→VT2→VT3** | gated `rel` vs CFA（VT 诊断 + fair chrono） |

包含：完整 Stage1/Stage2 **代码**、七域 Stage1 **best**、**init CFA bundles**（用于重跑 Stage1）、`events.jsonl`、参考结果 JSON、三份技术说明文档。

---

## 0. Quick start (recommended: Stage2 only)

Stage1 best CFA bundles are already under `artifacts/dataevolve_cfa_*`. You do **not** need to retrain Stage1 to reproduce Scheme5 / Fact.

```bash
cd TimeMMD-Residual-Paper

# 1) Environment
conda create -n mmd-tfs python=3.10 -y   # or use existing
conda activate mmd-tfs
# install torch for your CUDA, then:
pip install -r requirements.txt

# 2) Data mounts (same layout as Time-MMD / MM-TSFlib)
#    This package ships symlinks; fix if broken:
# ln -sfn /path/to/MM-TSFlib/data data
# ln -sfn /path/to/Time-MMD/textual textual

# 3) Models
export HF_ENDPOINT=https://hf-mirror.com   # optional mirror
export HF_HOME=/tmp/hf_qwen06
# download Qwen3-0.6B into HF_HOME (Fact LatentCoT)
# huggingface-cli download Qwen/Qwen3-0.6B --local-dir $HF_HOME/Qwen3-0.6B

# 4a) Scheme5 v5 (needs vLLM OpenAI server for infer/rewrite)
export MMD_PY=$(which python)
export VLLM_BASE_URL=http://127.0.0.1:8001
export VLLM_MODEL=/model
bash scripts/repro_stage2_scheme5_v5.sh

# 4b) Fact VT0–VT3 (GPU; no vLLM required for training)
bash scripts/repro_stage2_fact_vt.sh
# fair (honest) VT3 only:
bash scripts/repro_stage2_fact_fair_vt3.sh
```

Reference numbers (from the source run) live in:

- `artifacts/scheme5_v5/summary.json` / `*_eval.json`
- `artifacts/stage2_fact_valtest_tricks/summary_valtest_tricks.json`
- `REPRO_MANIFEST.json` (Stage1 family paths)

---

## 1. Repository layout

```
TimeMMD-Residual-Paper/
├── README.md                          # this file
├── REPRO_MANIFEST.json                # Stage1 best + events inventory
├── requirements.txt / environment.yml
├── run.py                             # CFA training entry (Stage1)
├── language/
│   ├── apo/                           # CFA continue / scorer
│   ├── dataevolve_vmt/                # Stage1 DataEvolve loop
│   ├── stage2/                        # Scheme5 + Fact LatentCoT
│   ├── lamp_vmt/ lamp_agri/ lamp_timemmd/
│   └── vllm_client.py
├── models/ layers/ exp/ data_provider/ utils/   # CFA backbone
├── scripts/
│   ├── eval_scheme5.py / run_scheme5_v2_all.sh
│   ├── fair_branch_resid.py
│   ├── run_fact_valtest_tricks_bg.py
│   ├── repro_stage2_scheme5_v5.sh     # ★ Scheme5 one-click
│   ├── repro_stage2_fact_vt.sh        # ★ Fact VT0–VT3 one-click
│   ├── repro_stage2_fact_fair_vt3.sh  # ★ Fair VT3
│   ├── repro_extract_events.sh        # optional re-extract
│   └── repro_stage1_optional.sh       # optional Stage1 retrain
├── docs/
│   ├── Scheme5_v5_Technical_Specification.md
│   └── FactLatentCoT_Implicit_Branch_Technical_Specification.md
├── data/ → Time-MMD numeric (symlink)
├── textual/ → Time-MMD text (symlink)
└── artifacts/
    ├── dataevolve_cfa_*/              # ★ Stage1 BEST only (per domain)
    ├── *_events_mech*/events.jsonl    # ★ events
    ├── scheme5_v5/                    # reference Scheme5 results
    └── stage2_fact_*_tricks/          # reference Fact VT/leak summaries
```

Domains: `traffic`, `agriculture`, `economy`, `energy`, `public_health`, `security`, `social_good`.

---

## 2. Stage1 (CFA + DataEvolve)

技术说明：`docs/Stage1_DataEvolve_CFA_Technical_Specification.md`

### 2.1 Frozen best（Stage2 默认使用，已打包）

| Domain | Artifact | Family |
|--------|----------|--------|
| traffic | `artifacts/dataevolve_cfa_selfevo_traffic_reprompt_v2` | `F1_r4c2` |
| agriculture | `artifacts/dataevolve_cfa_selfevo_agri_reprompt` | `F5_r5c1` |
| economy | `artifacts/dataevolve_cfa_selfevo_economy_reprompt` | `F2_r4c4` |
| energy | `artifacts/dataevolve_cfa_selfevo_energy_reprompt` | `F2_r3c4` |
| public_health | `artifacts/dataevolve_cfa_selfevo_public_health` | `F5_r5c1` |
| security | `artifacts/dataevolve_cfa_selfevo_security_valtest` | `F1_r2c2` |
| social_good | `artifacts/dataevolve_cfa_selfevo_social_good_reprompt` | `F5_r3c5` |

解析：`language/stage2/resolve.py`。

### 2.2 从 init 重跑 Stage1（复现最优实验流程）

Init bundles 已打包：

- `checkpoints/long_term_forecast_Traffic_*/cfa_bundle.pt`
- `checkpoints/long_term_forecast_Algriculture_*/cfa_bundle.pt`
- `artifacts/apo_{economy,energy,security,social_good,public_health}/runpy_cfa_bundle.pt`
- 短别名：`checkpoints/stage1_init/{domain}_cfa_bundle.pt`

```bash
export VLLM_BASE_URL=... VLLM_MODEL=...
# 全域（耗时长；security 自动用 select_split=val_test）
bash scripts/repro_stage1_all.sh

# 单域
bash scripts/repro_stage1_optional.sh agriculture
```

产物：`artifacts/dataevolve_cfa_paperalign_{domain}/` 或 `dataevolve_cfa_rerun_*`。  
若要让 Stage2 用新结果，替换对应 canonical `artifacts/dataevolve_cfa_selfevo_*` 的 `best_strategy.json` + winning family 目录。

---

## 3. Stage2 Explicit — Scheme5 v5 best-online

### Algorithm (one line)

Per-channel evolvable protocol text; online walk train∪test; mean fuse × amp=1; metric = **`test_best_epoch_online`**.

Full spec: `docs/Scheme5_v5_Technical_Specification.md`.

### Reproduce

```bash
export VLLM_BASE_URL=...   # OpenAI-compatible chat API
export VLLM_MODEL=...
bash scripts/repro_stage2_scheme5_v5.sh
```

Equivalent core flags (do not change for “v5 best”):

```text
--improve --max-lessons 20 --dead-sr 0 --fixed-amp 1.0
--n-epochs 2 --wave 48 --cache-tag s5v5 --pick-epoch best
```

Outputs: `artifacts/scheme5_v5/{domain}_eval.json` → field `test_best_epoch_online`.

Set `SCHEME5_FORCE=1` to overwrite existing JSONs.

---

## 4. Stage2 Implicit — Fact LatentCoT VT1→VT2→VT3

### Stack (cumulative)

| Tag | Flags |
|-----|-------|
| VT0 | baseline Fact LatentCoT + LoRA |
| VT1 | `--coconut-recycle` |
| VT2 | VT1 + `--adaptive-halt` |
| VT3 | VT2 + `--n-thought-groups 3` |

Full spec: `docs/FactLatentCoT_Implicit_Branch_Technical_Specification.md`.

### A) Diagnostic table (val=test, train-only) — matches shipped `summary_valtest_tricks.json`

```bash
bash scripts/repro_stage2_fact_vt.sh
# → artifacts/stage2_fact_valtest_tricks/summary_valtest_tricks.json
```

Reference means (source run): VT0 −2.5% → VT1 −2.7% → VT2 −3.2% → VT3 −3.6%.

### B) Fair table (chrono val, no test for α/ckpt) — paper “honest” setting

```bash
bash scripts/repro_stage2_fact_fair_vt3.sh
```

### C) Single-domain example (VT3 fair)

```bash
PYTHONPATH=. python -u scripts/fair_branch_resid.py \
  --domain agriculture --tag fair_vt3 \
  --skip-soft --lora \
  --val-mode chrono --val-frac 0.2 --amp-select val \
  --epochs 8 --batch-size 48 --lr-fact 1e-4 --n-latent 12 \
  --coconut-recycle --adaptive-halt --n-thought-groups 3 \
  --model $HF_HOME/Qwen3-0.6B
```

---

## 5. Prerequisites checklist

| Need | Stage1 retrain | Scheme5 v5 | Fact VT/fair |
|------|:---:|:---:|:---:|
| `data/` + `textual/` | ✓ | ✓ | ✓ |
| Shipped Stage1 best | — | ✓ | ✓ |
| Shipped `events.jsonl` | — | ✓ | ✓ |
| vLLM chat server | ✓ | ✓ | ✗ |
| GPU + Qwen3-0.6B | CFA train | LLM calls | ✓ |
| BERT (`bert-base-uncased`) | ✓ | via CFA scorer | via CFA scorer |

---

## 6. What we intentionally did **not** ship

- Full Stage1 beam trees (multi-GB)
- CFA init `checkpoints/` hyperparameter sweeps
- Regenerable Scheme5 infer/rewrite caches
- HuggingFace weight files (download locally)

---

## 7. Citation / docs

- Stage1 DataEvolve+CFA: `docs/Stage1_DataEvolve_CFA_Technical_Specification.md`
- Scheme5 v5: `docs/Scheme5_v5_Technical_Specification.md`
- Fact LatentCoT VT: `docs/FactLatentCoT_Implicit_Branch_Technical_Specification.md`
- Pipeline sketch: `docs/PAPER_PIPELINE.md`
- CFA baseline: arXiv:2603.22372; Time-MMD benchmark

---

## 8. Smoke test (no GPU LLM)

```bash
cd TimeMMD-Residual-Paper
PYTHONPATH=. python - <<'PY'
from language.stage2.profiles import list_stage2_domains, get_stage2_profile
from language.stage2.resolve import resolve_stage1_candidate
from pathlib import Path
root=Path('.')
for d in list_stage2_domains():
    p=get_stage2_profile(d)
    c=resolve_stage1_candidate(root, p)
    ev=root/p.events_dir/'events.jsonl'
    assert ev.is_file() and ev.stat().st_size>100, d
    assert (Path(c['dir'])/'cfa_bundle.pt').is_file(), d
    print('OK', d, c['family'])
print('all domains resolve')
PY
```
