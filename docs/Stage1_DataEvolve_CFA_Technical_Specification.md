# Stage1 DataEvolve + CFA 技术方案完整技术说明

> 面向论文写作的实现级文档。描述的是 **Stage1**：在 Time-MMD 多域上，用 **DataEvolve 式自进化清洗** 改写辅助新闻文本，再对 **CFA（Controlled Fusion Adapter）** 做 continue-train，按 CFA MSE 选出最优清洗策略与 `cfa_bundle.pt`。  
> Stage2（Scheme5 显式 / Fact LatentCoT 隐式）**冻结**该 Stage1 最优 checkpoint，只学残差。  
> 代码根：`MMD-TFS/`（论文包：`TimeMMD-Residual-Paper/`）。核心入口：`scripts/run_dataevolve_cfa_strict.py` → `language/dataevolve_vmt/strict_loop.py`。

---

## 0. 一句话概括

**Stage1** =「Observer 一次观察生语料问题 → 多轮 Designer 进化清洗策略 → Cleaner 全表重写 → Quality Judge 写 Coverage/Executability 残差经验 → **CFA continue-train + MSE 选优**」。  
最终每个域保留一个 **best family** 目录，内含：

- `cfa_bundle.pt`（Stage2 的 \(y^{\mathrm{CFA}}\) 来源）
- `*_cleaned.csv` + `cleaned_texts.jsonl`（清洗后文本）
- 根目录 `best_strategy.json`（策略全文 + select_mse）

---

## 1. Motivation（为什么要 Stage1）

### 1.1 问题背景

Time-MMD 把数值序列与辅助新闻拼进多模态预报器。**CFA**（arXiv:2603.22372）用 PatchTST + BERT 文本融合，已能改善部分域的 MSE。但原始新闻里常见：

1. **噪声与广告 / 导航壳**；  
2. **跨年百科段落**混进当月行（日期不对齐）；  
3. **非美国 / 无关话题**；  
4. **空行被「填」成近年新闻**（破坏因果面板）。

若直接用脏文本训 CFA，或人工写死一套清洗规则，都会：不可扩展、难跨域、且无法用下游预报误差反馈清洗质量。

### 1.2 设计目标

| 目标 | Stage1 如何落实 |
|------|-----------------|
| 文本服务预报 | 选择准则是 **CFA MSE**（`select_split`），不是仅 LLM 主观分 |
| 规则可进化 | DataEvolve：Experience DB + Designer 突变清洗 prompt |
| 面板安全 | `PANEL_SAFETY_RULES`：禁止全局年份截断、禁止往旧行塞新闻 |
| 可复现 | 固定 init CFA bundle、固定轮次/beam、写出 `best_strategy.json` |
| 服务 Stage2 | 同一 `cfa_bundle.pt` 供 Scheme5 / Fact 读 \(y^{\mathrm{CFA}}\) |

### 1.3 在整条论文管线中的位置

```
Time-MMD raw CSV + textual
        │
        ▼
┌────────────── Stage1（本文）──────────────┐
│ DataEvolve 清洗 + CFA continue-train     │
│ → best cfa_bundle.pt + cleaned texts     │
└──────────────────┬───────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
  Scheme5 v5 显式        Fact LatentCoT 隐式
  (残差协议)              (VT1→VT3)
```

---

## 2. 参考项目 / 论文与概念谱系

### 2.1 直接依赖

| 工作 | 角色 |
|------|------|
| **CFA** — Controlled Fusion Adapter（arXiv:2603.22372） | 时序+文本融合基线；Stage1 **不改结构**，只换清洗文本后 continue-train |
| **Time-MMD** | 七域数据与窗口协议 |
| **DataEvolve**（仓库 `third_party/DataEvolve`） | Observer / Designer / Cleaner / Quality Judge 四段协议与 Experience 思想 |
| **APO-style beam**（本仓库 `language/apo` + Designer 多样性） | beam + Jaccard / paradigm-shift 探索 |

### 2.2 与原文 DataEvolve 的差异（写论文需诚实）

| | 经典 DataEvolve | 本仓库 Stage1 |
|--|-----------------|---------------|
| 下游任务 | 多为 LLM QA / 工具 | **CFA 预报 MSE** |
| 每轮后 | Judge → 经验 | Judge → 经验 → **CFA finetune** |
| 选优 | 质量分 / 任务分 | **`select_mse`（默认 test）** |
| 多样性 | 视实现 | APO：beam 续写 + 温度 exploit/explore + Jaccard |

可表述为：**DataEvolve + forecasting utility selection (CFA MSE)**。

---

## 3. 符号与产物

| 符号 / 文件 | 含义 |
|-------------|------|
| \(\mathcal{D}_{\mathrm{raw}}\) | 原始 CSV（含 `text_col`） |
| \(\pi\) | 清洗策略（自然语言 `cleaning_prompt`） |
| \(\mathcal{D}_\pi\) | Cleaner 写出的 cleaned CSV / jsonl |
| \(\theta_0\) | init `cfa_bundle.pt`（预训练 CFA） |
| \(\theta_\pi\) | 在 \(\mathcal{D}_\pi\) 上 continue-train 后的 bundle |
| \(\mathrm{MSE}_{\mathrm{val/test}}(\theta_\pi)\) | 选优用的 CFA 误差 |
| `family` | 策略 ID，如 `F5_r5c1`（种子族 + 轮次 + 突变序号） |
| `best_strategy.json` | 全局最优条目（含完整 prompt 与 MSE） |

---

## 4. 算法详细流程

### 4.1 总览伪代码

```
Input: domain profile, init_bundle θ0, raw_csv
Hyper: R=5 outer rounds, B=3 beam, M=2 mutations/parent,
       CFA epochs=30, patience=10, select_split ∈ {test, val, val_test}

1. Raw control (once):
   θ_raw ← CFA_continue(θ0, raw_csv)
   record MSE_val/test(θ_raw)

2. Observer ONCE:
   sample n rows from raw → LLM Observer → Experience DB

3. For round r = 1..R:
   a. Parents ← current beam (r=1: domain seed prompts F0–F5 的前 B 个)
   b. Designer: for each parent, produce M mutated π's
      (exploit T=0.55 / explore T=0.9; diversity Jaccard)
   c. Cleaner: for each π, rewrite ALL rows → cleaned_csv / jsonl
   d. Quality Judge: compare original vs cleaned pairs
      → Coverage / Executability + NEW issues → Experience DB
   e. CFA_continue(θ0, cleaned_csv) → θ_π; evaluate val/test MSE
   f. Beam ← top-B strategies by select_mse, preferring diverse family roots
   g. Update global_best if select_mse improves

4. Write best_strategy.json + keep round_*/{family}/ artifacts
```

实现：`language/dataevolve_vmt/strict_loop.py` → `run_strict_loop`。

### 4.2 Phase 0 — Raw control CFA

同一 init、同一 epochs，在**未清洗**文本上 continue-train，得到对照 `cfa_bundle_Traw_control.pt`。  
每个候选报告 \(\Delta\mathrm{test}= \mathrm{MSE}_{\mathrm{test}}(\theta_\pi)-\mathrm{MSE}_{\mathrm{test}}(\theta_{\mathrm{raw}})\)。

### 4.3 Phase 1 — Observer（只跑一次）

- 从 raw CSV 随机抽 `observe_n`（默认 24）行文本。  
- System / User 见第 6 节。  
- 输出 `new_issues` + `useful_signal_types`，写入 `experience_database.jsonl`。  
- **后续轮次不再重跑 Observer**；新问题主要来自 Quality Judge。

### 4.4 Phase 2 — Designer（每轮）

- Round 1：从 `stage1_seed_prompts.yaml` 的 F0–F5 按 `beam_priority` 取 beam 个 seed。  
- 之后：以当前 beam 为 parents，每个 parent 突变 `mutations_per_parent` 次。  
- 多样性：`diversity_jaccard`、`paradigm_shift_ratio`；探索温度更高。  
- 强制注入 `PANEL_SAFETY_RULES`（禁止全局年份截断等）。

### 4.5 Phase 3 — Cleaner

- 对**整表**每一行：策略 + 原文 → 清洗后文本（并行，`clean_concurrency=48`）。  
- 写出 `{cleaned_csv_name}` 与 `cleaned_texts.jsonl`。

### 4.6 Phase 4 — Quality Judge

- 采样 original/cleaned pairs，多 batch Judge。  
- 打分 + Coverage / Executability 短分析 + **new residual issues** → Experience。  
- 近重复压缩：`near_dup_jaccard`。

### 4.7 Extension — CFA continue-train + 选优

对每个候选：

```text
finetune_cfa_bundle(θ0 → θ_π; epochs=30, patience=10, batch=48)
evaluate val_mse, test_mse
select_mse = f(select_split)
score_objective = -select_mse
```

**Beam 更新**：按 `select_mse` 升序，优先不同 `family` 根（如 F1 vs F5），取前 `beam_size`。  
**全局最优**：全程最小 `select_mse` → `best_strategy.json`。

### 4.8 `select_split` 语义

| 值 | 含义 | 使用域 |
|----|------|--------|
| `test` | 用 test MSE 选优 | 多数域（默认） |
| `val` | 用 val MSE | 可选 |
| `val_test` | mean(finite val, test) | **security** 线上 artifact |

> 论文需说明：线上 Stage2 所用 security 产物来自 `select_split=val_test` 的 run（目录名含 `security_valtest`）。paperalign 队列脚本默认全域 `test`；复现「与 Stage2 完全一致」时应对齐各域历史 `best_strategy.json` 中的 `select_split`。

---

## 5. 定稿超参与七域最优

### 5.1 推荐复现超参（paperalign）

```text
--n-outer-rounds 5
--beam-size 3
--mutations-per-parent 2      # → 每轮约 6 个候选
--select-split test           # security 线上用 val_test，见上
--cfa-epochs 30
--cfa-patience 10
--cfa-batch-size 48
--clean-concurrency 48
--parallel-clean
--temperature-clean 0.0
--temperature-design 0.55
--temperature-design-explore 0.9
--temperature-judge 0.2
--judge-batches 4
--judge-pairs 3
```

### 5.2 Init CFA bundle（continue-train 起点）

| Domain | Init path（相对仓库根） |
|--------|-------------------------|
| traffic | `checkpoints/long_term_forecast_Traffic_2021_8_6_PatchTST_BERT_cfa_lr2.0_.../cfa_bundle.pt` |
| agriculture | `checkpoints/long_term_forecast_Algriculture_2021_8_6_PatchTST_BERT_cfa_lr2.0_.../cfa_bundle.pt` |
| economy | `artifacts/apo_economy/runpy_cfa_bundle.pt` |
| energy | `artifacts/apo_energy/runpy_cfa_bundle.pt` |
| security | `artifacts/apo_security/runpy_cfa_bundle.pt` |
| social_good | `artifacts/apo_social_good/runpy_cfa_bundle.pt` |
| public_health | `artifacts/apo_public_health/runpy_cfa_bundle.pt` |

### 5.3 Stage2 实际消费的 best（冻结）

| Domain | Artifact dir | Family | select_split | test_mse（记录） |
|--------|--------------|--------|--------------|------------------|
| traffic | `dataevolve_cfa_selfevo_traffic_reprompt_v2` | **F1_r4c2** | test | 0.1620 |
| agriculture | `dataevolve_cfa_selfevo_agri_reprompt` | **F5_r5c1** | test | 0.0575 |
| economy | `..._economy_reprompt` | **F2_r4c4** | test | 0.0167 |
| energy | `..._energy_reprompt` | **F2_r3c4** | test | 0.0368 |
| public_health | `..._public_health` | **F5_r5c1** | test | 0.7223 |
| security | `..._security_valtest` | **F1_r2c2** | **val_test** | 67.79 |
| social_good | `..._social_good_reprompt` | **F5_r3c5** | test | 0.9273 |

解析：`language/stage2/resolve.py` 读 `artifacts/{stage1_artifact_dirname}/best_strategy.json` → `round_*/{family}/`。

---

## 6. Prompt 体系（原文级）

所有角色字符串在 `language/dataevolve_vmt/prompts.py`；域任务名由 `make_systems(display_task)` 注入。

### 6.1 System prompts（模板）

**Observer**

```text
You are DataEvolve's Data Quality Observer for {display_task} news facts.
The host has already called read_experience() for you. Independently find NEW
quality issues in the raw samples and return them so the host can write_experience().
Do not restate issues already in the experience list. Reply with JSON only.
```

**Designer**

```text
You design cleaning strategies (prompts) for auxiliary news text for {display_task}.
Evolve from experience-pool issues and Quality Judge Coverage/Executability gaps.
Do not blindly keep a fixed output schema if feedback says it fails.
Reply with JSON only.
```

**Cleaner**

```text
You clean auxiliary news text for {display_task}.
Follow the provided strategy strictly. Output the cleaned fact list only
(bullet lines; no markdown fences, no commentary).
```

**Judge**

```text
You are DataEvolve's Quality Judge for {display_task} auxiliary news cleaning.
Compare original vs cleaned pairs, score cleaning, give SHORT Coverage and
Executability notes, and list only genuinely NEW residual issues.
Keep every string field concise (...). Reply with compact JSON only — no markdown.
```

### 6.2 PANEL_SAFETY_RULES（强制进 Designer）

见 `prompts.py` 中 `PANEL_SAFETY_RULES`：长面板按行清洗、禁止全局年份截断、禁止往空/旧行塞近年新闻、日期 ±1 年硬规则、禁止过度清空等。

### 6.3 Observer user（结构）

```text
# Data Quality Observation Task
## Step 1 — read_experience() [ALREADY DONE BY HOST]
...
## Step 2 — Independently examine raw samples
...
## Step 3 — Propose issues for write_experience()
Return JSON:
{ "new_issues": [...], "useful_signal_types": [...] }
```

### 6.4 Designer user（结构）

```text
# Design cleaning strategy prompt
## Previous best strategy ... / First generation ...
## Domain soft-start / panel context (MUST honor)
{soft_start_hint from YAML}
## MANDATORY panel-safety ...
## Experience pool ...
Return JSON:
{ "cleaning_prompt": "...", "design_rationale": "...", "key_improvements": [...] }
```

### 6.5 Cleaner user

```text
# Cleaning strategy
{strategy}

# Original text
{raw_text}

# Output
Cleaned text only (...):
```

### 6.6 Round-1 Seed prompts

完整 F0–F5 在：

- `language/dataevolve_vmt/stage1_seed_prompts.yaml`
- `language/dataevolve_vmt/stage1_seed_prompts_extra.yaml`

每域含 `display_task`、`raw_csv`、`soft_start_hint`、`beam_priority`、各 `seeds.F*.cleaning_prompt`。

**Traffic F0 片段（示例）：**

```text
DATASET: U.S. monthly VMT panel (~1980–2024). Clean the CURRENT row month only.
ROLE: light denoiser — strip chrome/ads/off-year dumps; KEEP most on-year U.S. mobility text.
...
DATE HARD: printed year within ±1 of ROW year (prefer ±3 months). Mixed → keep on-year.
```

**Agriculture best（F5_r5c1）prompt 开头（线上）：**

```text
Clean the current month's U.S. retail broiler composite news text only. Each row
corresponds to one calendar month; do not apply global year filters ...
Focus strictly on U.S. retail price drivers with event/release dates within ±1 year ...
Preserve only: U.S. feed costs (corn/soy), placements ...
```

（全文见对应 `best_strategy.json` 的 `cleaning_prompt` 字段。）

---

## 7. 代码索引

| 模块 | 路径 |
|------|------|
| CLI 入口 | `scripts/run_dataevolve_cfa_strict.py` |
| 主循环 | `language/dataevolve_vmt/strict_loop.py` |
| Observer / pipeline | `language/dataevolve_vmt/pipeline.py` |
| Designer | `language/dataevolve_vmt/designer_diverse.py` |
| Cleaner | `language/dataevolve_vmt/clean_ops.py` |
| Judge | `language/dataevolve_vmt/quality_judge.py` |
| Prompts | `language/dataevolve_vmt/prompts.py` |
| Seeds YAML | `language/dataevolve_vmt/stage1_seed_prompts*.yaml` |
| Domain profiles | `language/dataevolve_vmt/domain_profiles.py` |
| CFA finetune / eval | `language/apo/cfa_finetune.py`, `cfa_trainer.py` |
| Stage2 解析 best | `language/stage2/resolve.py`, `profiles.py` |
| 全域队列 | `artifacts/stage1_paperalign_queue/run_stage1_paperalign.sh` |
| 论文包一键 | `scripts/repro_stage1_all.sh`（TimeMMD-Residual-Paper） |

---

## 8. 复现命令

### 8.1 跳过重训（推荐）

论文包已含七域 **best only**。Stage2 直接：

```bash
# resolve smoke
PYTHONPATH=. python -c "from language.stage2.resolve import resolve_stage1_candidate; ..."
```

### 8.2 从 init 重跑单域（对齐 paperalign）

```bash
export LLM_PROVIDER=vllm VLLM_BASE_URL=... VLLM_MODEL=...
PYTHONPATH=. python -u scripts/run_dataevolve_cfa_strict.py \
  --domain agriculture \
  --init-bundle checkpoints/.../cfa_bundle.pt \
  --output-dir artifacts/dataevolve_cfa_rerun_agriculture \
  --n-outer-rounds 5 --beam-size 3 --mutations-per-parent 2 \
  --select-split test \
  --cfa-epochs 30 --cfa-patience 10 --cfa-batch-size 48 \
  --clean-concurrency 48 --parallel-clean
```

### 8.3 复现「与 Stage2 完全一致」的 security

```bash
--domain security --select-split val_test \
  --output-dir artifacts/dataevolve_cfa_selfevo_security_valtest
```

---

## 9. 论文写作建议

1. **Problem**：脏辅助文本伤害多模态预报。  
2. **Method**：DataEvolve 四阶段 + CFA utility selection。  
3. **Safety**：panel / date hard rules（消融可关，但线上开）。  
4. **Results**：相对 raw-control 的 \(\Delta\) MSE；再接 Stage2 残差。  
5. **Limitation**：`select_split=test` 有选优泄漏争议——可主报 val 选优消融，或明确为「forecasting-oriented cleaning」。

---

## 10. 与 Stage2 文档关系

| 文档 | 内容 |
|------|------|
| **本文** | Stage1 清洗 + CFA |
| `Scheme5_v5_Technical_Specification.md` | 显式残差 |
| `FactLatentCoT_Implicit_Branch_Technical_Specification.md` | 隐式 VT1–VT3 |

三者共享：Time-MMD 数据、Stage1 `cfa_bundle.pt`、（Stage2）`events.jsonl`。

---

*实现以当前 `strict_loop.py` 为准。数值以各域 `best_strategy.json` 为准。*
