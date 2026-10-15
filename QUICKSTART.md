# Quickstart（论文代码一键复现）

路径：`/mnt/ZhangXinyuWorkSpace/TimeMMD-Residual-Paper`（约 300MB，含 Stage1 best CFA + events）。

## 1. 环境

```bash
cd /mnt/ZhangXinyuWorkSpace/TimeMMD-Residual-Paper
conda activate mmd-tfs   # 或按 requirements.txt 新建
export PYTHONPATH=$PWD
export MMD_PY=$(which python)
export HF_HOME=/tmp/hf_qwen06
export HF_ENDPOINT=https://hf-mirror.com
```

确认数据链接：

```bash
ls -la data textual
# 若断链：
# ln -sfn /path/to/MM-TSFlib/data data
# ln -sfn /path/to/Time-MMD/textual textual
```

冒烟（无需 GPU）：

```bash
PYTHONPATH=. python - <<'PY'
from language.stage2.profiles import list_stage2_domains, get_stage2_profile
from language.stage2.resolve import resolve_stage1_candidate
from pathlib import Path
root=Path('.')
for d in list_stage2_domains():
    c=resolve_stage1_candidate(root, get_stage2_profile(d))
    assert (Path(c['dir'])/'cfa_bundle.pt').is_file()
print('OK')
PY
```

## 2. 三部分最优结果怎么跑

### A. Stage1（DataEvolve + CFA）

- **直接用冻结 best**（推荐，Stage2 已对齐）：`artifacts/dataevolve_cfa_selfevo_*/`
- **从 init 重跑全流程**（需 vLLM + GPU，很慢）：

```bash
export VLLM_BASE_URL=... VLLM_MODEL=...
bash scripts/repro_stage1_all.sh
# 或单域：bash scripts/repro_stage1_optional.sh traffic
```

Init 已在 `checkpoints/` 与 `artifacts/apo_*/`。说明见 `docs/Stage1_DataEvolve_CFA_Technical_Specification.md`。

### B. Stage2 显式 — Scheme5 v5 best-online

```bash
export VLLM_BASE_URL=http://127.0.0.1:8001
export VLLM_MODEL=/model
bash scripts/repro_stage2_scheme5_v5.sh
# 主指标：artifacts/scheme5_v5/*_eval.json → test_best_epoch_online
```

参考汇总：`artifacts/scheme5_v5/summary.json`

### C. Stage2 隐式 — Fact LatentCoT VT1→VT2→VT3

```bash
# 诊断表（val=test，与文档/参考 JSON 一致）
bash scripts/repro_stage2_fact_vt.sh

# 诚实 fair 表（chrono val，推荐写主表）
bash scripts/repro_stage2_fact_fair_vt3.sh
```

参考：`artifacts/stage2_fact_valtest_tricks/summary_valtest_tricks.json`  
（VT0 −2.5% → VT1 −2.7% → VT2 −3.2% → VT3 −3.6%）

## 3. 文档

- `README.md` — 完整说明  
- `docs/Scheme5_v5_Technical_Specification.md` — 显式  
- `docs/FactLatentCoT_Implicit_Branch_Technical_Specification.md` — 隐式  
- `docs/PAPER_PIPELINE.md` — 总流程图  
- `REPRO_MANIFEST.json` — Stage1 family 清单  
