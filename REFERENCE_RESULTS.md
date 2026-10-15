# Reference results（源仓库已跑通的最优对照）

打包时拷入的参考数字；重跑后以新 JSON 为准。

## Stage1 DataEvolve + CFA

| Domain | Family | select_split | test_mse | Artifact |
|--------|--------|--------------|---------:|----------|
| agriculture | **F5_r5c1** | test | 0.057548 | `dataevolve_cfa_selfevo_agri_reprompt` |
| economy | **F2_r4c4** | test | 0.016733 | `dataevolve_cfa_selfevo_economy_reprompt` |
| energy | **F2_r3c4** | test | 0.036751 | `dataevolve_cfa_selfevo_energy_reprompt` |
| public_health | **F5_r5c1** | test | 0.722344 | `dataevolve_cfa_selfevo_public_health` |
| security | **F1_r2c2** | val_test | 67.793877 | `dataevolve_cfa_selfevo_security_valtest` |
| social_good | **F5_r3c5** | test | 0.927323 | `dataevolve_cfa_selfevo_social_good_reprompt` |
| traffic | **F1_r4c2** | test | 0.161995 | `dataevolve_cfa_selfevo_traffic_reprompt_v2` |

Init bundles for retrain: `checkpoints/stage1_init/*.pt` + `artifacts/apo_*/runpy_cfa_bundle.pt`.

Doc: `docs/Stage1_DataEvolve_CFA_Technical_Specification.md`.

## Stage2 Explicit — Scheme5 v5 `test_best_epoch_online`

配置：`--improve --pick-epoch best --fixed-amp 1.0 --dead-sr 0 --n-epochs 2 --wave 48`

| Domain | rel_pct |
|--------|--------:|
| agriculture | **−3.33%** |
| economy | **−3.85%** |
| energy | **−3.12%** |
| public_health | −0.33% |
| security | −0.11% |
| social_good | +0.70% |
| traffic | **−2.33%** |
| **mean** | **−1.77%** |

文件：`artifacts/scheme5_v5/summary.json`、`*_eval.json`

## Stage2 Implicit — Fact LatentCoT VT（val=test）

| | VT0 | VT1 coconut | VT2 +halt | VT3 +mix |
|--|----:|----:|----:|----:|
| **mean** | −2.52% | −2.67% | −3.24% | **−3.59%** |

文件：`artifacts/stage2_fact_valtest_tricks/summary_valtest_tricks.json`

Fair（chrono val）：`bash scripts/repro_stage2_fact_fair_vt3.sh`
