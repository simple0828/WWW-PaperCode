# Stage Freeze (VMT / Time-MMD)

Policy: **delete only abandoned ideas**. Keep anything needed to reuse
structured events / cleaning / CFA on other datasets.

## Frozen Stage 1 — self-evolving text cleaning → CFA

- Entry: `scripts/run_dataevolve_cfa_strict.py`
- Lib: `language/dataevolve_vmt/`
- Best: `artifacts/dataevolve_cfa_selfevo_emergent_jls/` → family **F1_r4c2**
  (`cleaned_texts.jsonl`, `US_VMT_Month_cleaned.csv`, `cfa_bundle.pt`)
- Reusable helpers (KEEP): `prepare_dataevolve_*`, `clean_dataevolve_*`,
  `eval_dataevolve_*`, `finetune_cfa_on_cleaned.py`, `compare_cfa_*`,
  `repro_cfa_*.sh`, `extract_vmt_events.py`, `reclass_event_channels.py`

## Frozen Stage 2 — residual vs CFA (three complementary Δ)

ŷ = y_CFA + Δ

| Branch | Role | Canonical code |
|--------|------|----------------|
| SoftLSTM | multi-channel soft/presence **temporal** residual | `fair_branch_resid.py`, `improve_latent_fuse.py` |
| Explicit | S5 dual ⊕ LAMP driver (display / gated mech) | `cfa_resid_schemes.py`, `lamp_on_cfa.py`, `fusion_shell_cfa.py` |
| Fact LatentCoT | fact_text prefix + latent tokens (prefer longer facts) | `latent_cot_residual.py`, `improve_latent_fuse.py` |

Best usable fuse (current):  
`Δ = Soft + 0.4·Explicit + 0.25·Fact` only if Soft and Fact **same sign**.

Results: `artifacts/event_ts_assoc/latent_cot/improve_latent_fuse.json`,
`summary_fair_branches.json`, `summary.json` (numeric LatentCoT),
`summary_text.json` (text prefix ablations).

Caches KEEP: `scheme1_event_v1.jsonl`, `scheme1_window_v1.jsonl`,
`artifacts/vmt_events_mech/`.

## Abandoned (do not revive)

- Handcrafted **A4 / graph / soft_pool** numeric features
- Auto feature-evolution Ridge (`evolve_resid_features`)
- Event-transformer / LLM-gen-seq residual experiments
- **Joint TripleFuse raw-head “solo” scores** (`triple_resid_fuse.py`) —
  misleading; use fair solo + `improve_latent_fuse` instead

## Stage 3 — open (design next)

Cross-dataset transfer of Stage1+2, learned residual router, or
joint Stage1↔Stage2 feedback.

### Multi-dataset Stage1 (started)

Stage1 entry now accepts `--domain` (see `language/dataevolve_vmt/domain_profiles.py`).

Agriculture (running / to run):
```bash
python scripts/run_dataevolve_cfa_strict.py \
  --domain agriculture \
  --init-bundle checkpoints/...Algriculture...cfa.../cfa_bundle.pt \
  --output-dir artifacts/dataevolve_cfa_selfevo_agri \
  --n-outer-rounds 3 --beam-size 2 --mutations-per-parent 2
```
Log: `logs/stage1_agriculture.log`

Climate / other Time-MMD domains need `build_domain_args` + CFA init bundle before Stage1.
