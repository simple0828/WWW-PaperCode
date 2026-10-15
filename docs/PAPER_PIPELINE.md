# Paper pipeline: Stage1 → Stage2 Explicit → Stage2 Implicit

```
Time-MMD (data/ + textual/)
        │
        ▼
┌─────────────────────── Stage1 ───────────────────────┐
│  DataEvolve (LLM clean) + CFA continue-train         │
│  Select best family → cfa_bundle.pt + cleaned CSV    │
│  ★ Frozen best + init bundles for full Stage1 retrain│
│  Doc: Stage1_DataEvolve_CFA_Technical_Specification  │
└──────────────────────────┬───────────────────────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼                             ▼
   extract_events.py              (events.jsonl shipped)
            │
            ├─────────────────────────────┐
            ▼                             ▼
┌── Stage2 Explicit (Scheme5 v5) ──┐  ┌── Stage2 Implicit (Fact VT) ──┐
│ per-channel protocol memory      │  │ Qwen3-0.6B + LoRA LatentCoT   │
│ online train∪test                │  │ VT1 Coconut → VT2 Halt → VT3 Mix │
│ metric: test_best_epoch_online   │  │ metric: gated rel vs CFA      │
│ scripts/repro_stage2_scheme5_v5  │  │ scripts/repro_stage2_fact_*   │
└──────────────────────────────────┘  └───────────────────────────────┘
```

See `README.md` for commands and `docs/*Technical_Specification.md` for algorithms.
