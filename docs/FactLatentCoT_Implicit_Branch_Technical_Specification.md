# Fact LatentCoT 隐式残差分支技术方案（VT1 → VT2 → VT3）

> 面向论文写作的实现级文档。描述的是 **Stage2 隐式分支**最终定稿方案：在 CFA 预报上学习事件驱动的残差修正 \(\Delta\)，骨干为 **Fact LatentCoT**，并按 **VT1 Coconut → VT2 Adaptive Halt → VT3 Mixture-of-Thoughts** 逐层累加。  
> 代码根目录：`MMD-TFS/`。核心实现：`language/stage2/models.py`（`FactLatentCoT`）、`language/stage2/explicit_prompts.py`（`build_fact_prompt`）、`scripts/fair_branch_resid.py`（训练/评估）、`scripts/run_fact_valtest_tricks_bg.py`（VT 消融）。  
> 本文尽量做到：动机清晰、符号统一、流程可复现、prompt 原文可抄。

---

## 0. 一句话概括

**Fact LatentCoT（隐式分支）** 冻结 Qwen3-0.6B 主体权重（仅训 LoRA + 少量头），把「近期事件事实」编成自然语言前缀，再在连续隐空间里做多步 latent reasoning，读出 horizon 残差 \(\Delta\in\mathbb{R}^H\)，得到

\[
\hat{y} = y^{\mathrm{CFA}} + \alpha\cdot\Delta
\]

其中 \(\alpha\) 由验证集网格搜索选择（幅度门控）。相对 **Scheme5 显式协议分支**（自然语言 IF–THEN），本分支**不生成可读 CoT 文本**，而是在 LLM 隐状态里完成「想几步 → 再读出 \(\Delta\)」。

最终定稿按累加配置：

| 代号 | 相对上一档新增 | 开关 |
|------|----------------|------|
| **VT0**（消融基线） | — | 无 Coconut / Halt / Mix |
| **VT1 Coconut** | 连续 thought 回灌 | `--coconut-recycle` |
| **VT2 +halt** | 自适应软停机 | `--coconut-recycle --adaptive-halt` |
| **VT3 +mix** | 多组 thought 注意力融合 | `--coconut-recycle --adaptive-halt --n-thought-groups 3` |

**定稿主线是 VT1 → VT2 → VT3（逐层递增）**；VT0 仅作对照。论文可报「主方案 = VT3」，或「主方案 = VT2 / VT3 择优」并附 VT0–VT3 消融表。

---

## 1. Motivation（为什么需要隐式分支）

### 1.1 问题设定

Time-MMD 多域上，Stage1 **CFA** 已给出数值预报 \(y^{\mathrm{CFA}}\in\mathbb{R}^H\)（\(H=6\)）。Stage2 的目标不是重训预报器，而是学残差：

\[
r^\star = y^{\mathrm{true}} - y^{\mathrm{CFA}},\qquad
\hat{y} = y^{\mathrm{CFA}} + \Delta,\qquad
\Delta \approx r^\star.
\]

文本侧已有结构化事件（`events.jsonl`：fact + 可选机制假设）。显式分支（Scheme5）用自然语言协议做可解释修正；隐式分支要回答的是另一问题：

> **事件事实能否被压进 LLM 的连续隐空间，用更短、更稠密的「latent CoT」直接回归残差，并在不写长文 CoT 的情况下仍获得稳定增益？**

### 1.2 朴素做法的不足

| 朴素做法 | 问题 |
|----------|------|
| 把事件文本拼进 CFA 输入再端到端训 | 改动 Stage1；难与「只改残差」解耦；可复现性差 |
| 显式文本 CoT → 再解析成 \(\Delta\) | 长、慢、解析脆；与 Scheme5 职责重叠 |
| 单次 forward、无 latent 迭代 | 表达力弱；难以对应「多步推理」叙事 |
| 固定步数、固定融合（mean） | 空事实 / 稠密事实一刀切；多假设路径无法并行 |

### 1.3 设计目标

| 目标 | 如何落实 |
|------|----------|
| 残差而非重报 | \(\hat{y}=y^{\mathrm{CFA}}+\alpha\Delta\)；loss = 原始空间 MSE |
| 隐式而非显式 CoT | 连续 latent tokens 迭代；不 decode 中文/英文推理句 |
| 事件条件化 | 文本前缀只含 focus 通道事实；数值侧可选 soft aux |
| 可训练且省显存 | 冻结 LLM + LoRA（q/k/v/o）+ gradient checkpointing |
| 幅度可控 | 全局 \(\mu/\sigma\) 标准化 + tanh 裁剪 + val 选 \(\alpha\) |
| 可消融的创新点 | Coconut / Halt / Mix 三层开关，对应 VT1–VT3 |

### 1.4 与显式分支（Scheme5）的分工

| | **隐式 Fact LatentCoT（本文）** | **显式 Scheme5** |
|--|--------------------------------|------------------|
| 推理介质 | 连续 hidden / latent tokens | 自然语言协议 `prompt_c` |
| LLM 权重 | 冻结 + LoRA 微调 | 完全冻结，只改协议文本 |
| 输出 | 直接回归 \(\Delta\in\mathbb{R}^H\) | 门控 horizon 修正 + 可解释规则 |
| 优势 | 稠密、端到端可微、步数短 | 可审计、可积累经验 |
| 论文叙事 | Latent CoT / Coconut 系 | Agent memory / protocol learning |

二者可独立报告，也可后期 fuse；**最终定稿的隐式技术方案以 VT1–VT3 为准**。

---

## 2. 参考项目 / 论文与概念谱系

> 说明：本方案**不是**某一篇论文的逐字复现，而是把 Latent CoT 文献中的几个小 trick **工程化到 Time-MMD 残差预报**。写 Related Work 时建议按「范式 → 本仓库适配」叙述。

### 2.1 直接依赖（本仓库）

1. **CFA（Controlled Fusion Adapter）** — Stage1 基线预报 \(y^{\mathrm{CFA}}\)。  
2. **Time-MMD** — 七域多模态时序；事件与通道定义见 `language/stage2/profiles.py`。  
3. **Stage1 事件管线** — `events.jsonl`（`fact_text`、`channel`、`mechanisms`…）；Fact 前缀默认**只用 fact 文本**，不把机制链写进 prompt（`include_mech=False`）。  
4. **Qwen3-0.6B** — 作为 frozen backbone；路径默认 `/tmp/hf_qwen06/Qwen3-0.6B`。

### 2.2 Latent / Continuous CoT 主线（VT1 的来源）

| 工作 | 核心想法 | 我们怎么用 |
|------|----------|------------|
| **Coconut** — *Training Large Language Models to Reason in a Continuous Latent Space*（Hao et al., arXiv:2412.06769） | 不把中间推理 decode 成词；把 **最后一层 hidden 直接当作下一步输入 embedding**（continuous thought） | **VT1**：`coconut_recycle=True` 时，`lat ← thought_proj(h_last[:, -n_latent:])`，去掉旧的 residual blend |
| **Pause / filler tokens**（Goyal et al. 等） | 插入可学习占位 token 增加计算深度 | 对照概念；我们用 **latent tokens**（从 ctx 投影）代替 pause |
| **CODI**（Shen et al., 2025） | 显式 CoT 蒸馏到连续空间 | 未进定稿；可作为 future work（与 Scheme5 对齐蒸馏） |
| **SIM-CoT**（2025） | 逐步监督隐式 token，防 collapse | 未进定稿；我们目前用最终 MSE 监督 |

**Motivation（对应 VT1）**：原先实现是「每步把 latent 拼回序列，再 `lat ← σ·h+(1-σ)·lat`」。这更像 residual RNN，不是文献里的 continuous thought。Coconut 的关键差异是：**下一步输入 = 上一步隐状态本身**，使推理轨迹留在连续空间，并可能编码多条候选路径（superposition）。

### 2.3 Adaptive computation / Halt（VT2 的来源）

| 工作 | 核心想法 | 我们怎么用 |
|------|----------|------------|
| **ACT**（Graves, Adaptive Computation Time） | 每步预测 halt 概率，软加权输出 | **VT2**：`halt_mlp` 输出 \(p_{\mathrm{exit}}\)，ACT 式累积质量 |
| **Adaptive Latent Reasoning**（arXiv:2511.21581 等） | 样本自适应决定 latent 步数 | 我们用 **软 halt**（训练全程跑满 \(T\) 步，用权重混合），避免硬 early-exit 不可微 |
| **Token-Level Adaptive Latent CoT**（2026） | Router 控制 continue/exit | 思想对齐；我们用事实密度 prior 调制 \(p_{\mathrm{exit}}\) |

**Motivation（对应 VT2）**：空事实或弱事实窗口不需要 3 步满算；稠密事实需要更深。固定 \(T=3\) 要么浪费算力、要么干扰。Halt 让模型学会「早停 / 晚停」。

### 2.4 Mixture / Superposition（VT3 的来源）

| 工作 | 核心想法 | 我们怎么用 |
|------|----------|------------|
| **Emergence of Superposition in Continuous CoT**（Zhu et al., 2025） | continuous thought 可编码多条并行路径 | **VT3**：把 \(n_{\mathrm{latent}}\) 切成 \(K=3\) 组，组内 mean，组间 attention 融合 |
| Coconut 分析中的 BFS 直觉 | 隐空间并行探索，而非单路径 greedy | Mix 提供「多假设再加权」的可实现形式 |

**Motivation（对应 VT3）**：单一 mean pooling 会抹平冲突假设。分组成「多条 thought」，用文本 ctx 做 query 做 soft routing。

### 2.5 与上述文献的差异（写论文时务必诚实）

1. 任务是 **回归 residual 向量**，不是 GSM8k 式答案 token。  
2. 无 Coconut 原文的多阶段 curriculum（language CoT → 逐步替换 latent）；我们 **直接端到端训 residual MSE**。  
3. Halt 是 **软加权**，推理时仍跑满 `n_steps`（实现简单、梯度稳）。  
4. Mix 是 **latent 分组 + 注意力**，不是多条独立 Coconut 轨迹的完整并行解码。

---

## 3. 符号与数据对象

| 符号 | 含义 |
|------|------|
| \(H=6\) | 预报 horizon 长度 |
| \(C\) | focus 通道数（域相关） |
| \(y^{\mathrm{CFA}}, y^{\mathrm{true}}\in\mathbb{R}^H\) | CFA 预报 / 真值 |
| \(r^\star = y^{\mathrm{true}}-y^{\mathrm{CFA}}\) | 目标残差 |
| \(\Delta\in\mathbb{R}^H\) | 模型预测残差（解码后） |
| \(\alpha\in\{0,0.05,\ldots,1\}\) | 幅度门控 |
| \(n_{\mathrm{latent}}=12\) | latent token 数 |
| \(n_{\mathrm{aux}}=4\) | soft 数值 aux token 数（默认开） |
| \(T=n_{\mathrm{steps}}=3\) | latent 迭代步数 |
| \(K=n_{\mathrm{thought\_groups}}\) | VT3 时 \(K=3\)；否则 \(K=1\) |
| \(d\) | LLM hidden size（Qwen3-0.6B） |
| \(\mu,\sigma\) | 训练集残差全局均值 / 标准差 |

每个训练样本（window）含：

- `prompt`：事实文本前缀（见第 6 节）  
- `ev`：\(\mathbb{R}^{L\times C\times 2}\) soft 事件矩阵（signed + presence）  
- `yc`, `yt`：CFA / true  
- `resid_mean`, `resid_scale`：即 \(\mu,\sigma\)（全局挂到每个样本）

---

## 4. 总体流水线（端到端）

```
Time-MMD 窗口
    │
    ├─ Stage1 CFA ──────────────────────────────► y_CFA
    │
    ├─ events.jsonl + focus 过滤 + top-k/月
    │         │
    │         └─ build_fact_prompt ─────────────► prompt 文本
    │
    ├─ soft_matrix(signed, presence) ───────────► ev
    │
    └─ FactLatentCoT(prompt, ev)
              │
              ├─ tokenize → text embeddings
              ├─ ctx = masked mean(text)
              ├─ init latents from ctx (+ bias)
              ├─ optional aux from ev
              ├─ for t = 1..T:  LLM(text, aux?, lat) → update lat
              │       VT1: Coconut recycle
              │       VT2: soft halt 累加 readout
              │       VT3: group attention fuse
              └─ raw → z = 3·tanh(raw/3) → Δ = z·σ + μ
                        │
                        └─ α* = argmin_α MSE(y_CFA + αΔ, y_true) on VAL
                                   │
                                   └─ test: ŷ = y_CFA + α* Δ
```

**相对 CFA 的相对误差**（主指标）：

\[
\mathrm{rel} = \frac{\mathrm{MSE}(\hat{y}, y^{\mathrm{true}}) - \mathrm{MSE}(y^{\mathrm{CFA}}, y^{\mathrm{true}})}
{\mathrm{MSE}(y^{\mathrm{CFA}}, y^{\mathrm{true}})}.
\]

负值 = 优于 CFA。

---

## 5. 算法详细流程

### 5.1 残差标准化（全局 \(\mu/\sigma\)）

在 **训练划分** 上拟合（val/test **不参与**拟合）：

\[
\mu = \mathrm{mean}(r^\star_{\mathrm{train}}),\quad
\sigma = \mathrm{std}(r^\star_{\mathrm{train}}).
\]

每个样本挂上相同 \(\mu,\sigma\)。模型输出 `raw`，解码：

\[
z = 3\cdot\tanh(\mathrm{raw}/3),\qquad
\Delta = z\cdot\sigma + \mu.
\]

（`Z_CLIP=3.0`，见 `fair_branch_resid.py`。）

**训练损失**（原始空间 MSE，与 checkpoint 一致）：

\[
\mathcal{L} = \big\| y^{\mathrm{CFA}} + \Delta - y^{\mathrm{true}} \big\|_2^2.
\]

### 5.2 FactLatentCoT 前向（VT0 基线，无三 trick）

1. **文本**：`input_ids, attention_mask` → embedding `text_emb`；  
   \(\mathrm{ctx} = \sum_i m_i e_i / \sum_i m_i\)。  
2. **初始化 latent**：  
   \(\mathrm{lat}_0 = \mathrm{reshape}(W_{\mathrm{ctx}}\mathrm{ctx}) + b_{\mathrm{lat}}\)，形状 \([B,12,d]\)。  
3. **Aux（默认开）**：用 `ev` 的全局均值与近 3 步均值投影为 \(n_{\mathrm{aux}}=4\) 个 soft token。  
4. **迭代** \(t=0..T-1\)：拼接序列  
   \(\mathrm{embeds}=[\mathrm{text\_emb};\,\mathrm{aux};\,\mathrm{lat}]\)，过 LLM，取最后 \(n_{\mathrm{latent}}\) 个位置的 last-layer hidden 为 `new_lat`。  
5. **VT0 更新**（非 Coconut）：  
   \(\mathrm{lat} \leftarrow \sigma(g)\cdot\mathrm{new\_lat} + (1-\sigma(g))\cdot\mathrm{lat}\)，\(g\) 为可学习标量。  
6. **读出**：\(\mathrm{fused}=\mathrm{mean}(\mathrm{lat})\)（或 VT3 的注意力融合）→ MLP → `raw`∈\(\mathbb{R}^H\)。

### 5.3 VT1 — Coconut recycle

**改动点**：关闭 residual blend，改为连续 thought 回灌。

\[
\mathrm{lat} \leftarrow W_{\mathrm{thought}}\,\mathrm{new\_lat}
\]

- \(W_{\mathrm{thought}}\)：`nn.Linear(d,d,bias=False)`，**初始化为单位阵**，再可微调。  
- 下一步 LLM 输入里的 latent 段 = 上一步真正的 hidden（经线性映射），语义上对齐 Coconut 的 continuous thought。

伪代码：

```
for t in 1..T:
    h = LLM([text, aux?, lat]).hidden_states[-1][:, -n_latent:]
    if coconut:
        lat = thought_proj(h)          # VT1
    else:
        lat = σ(g)*h + (1-σ(g))*lat  # VT0
```

### 5.4 VT2 — Adaptive soft halt（叠在 VT1 上）

每步在更新 `lat` 之后：

1. 构造 halt 特征  
   \([\mathrm{ctx};\; \overline{\mathrm{lat}}+0.1\cdot e_t;\; \rho;\; t/(T-1)]\)，  
   其中 \(\rho=\|\mathrm{mask}\|_1/L\) 为事实长度密度代理，\(e_t\) 为 step embedding。  
2. \(p_{\mathrm{exit}}=\sigma(\mathrm{MLP}(\cdot))\)，再乘密度 prior：  
   \(p_{\mathrm{exit}}\leftarrow p_{\mathrm{exit}}\cdot(0.35+0.65\rho)\)，并 clamp 到 \((10^{-3},1-10^{-3})\)。  
3. 最后一步强制 \(p_{\mathrm{exit}}=1\)。  
4. ACT 累积：  
   \(w_t = R_t\cdot p_{\mathrm{exit},t},\quad
   R_{t+1}=R_t(1-p_{\mathrm{exit},t}),\quad R_0=1\)。  
5. 每步 \(\hat{\Delta}_t=\mathrm{Readout}(\mathrm{Fuse}(\mathrm{lat}_t))\)，  
   最终 \(\mathrm{raw}=\sum_t w_t\hat{\Delta}_t\)。

直觉：稀疏事实 → \(\rho\) 小 → 更倾向早停；稠密事实 → 质量往后移。

### 5.5 VT3 — Mixture-of-Thoughts（叠在 VT1+VT2 上）

- \(n_{\mathrm{latent}}=12\)，\(K=3\) ⇒ 每组 4 个 latent。  
- 组表示：\(g_k=\mathrm{mean}(\mathrm{lat}^{(k)})\in\mathbb{R}^d\)。  
- Query：\(q=W_q\mathrm{ctx}\)；Key：\(k_k=W_k g_k\)。  
- \(\alpha=\mathrm{softmax}(q^\top k/\sqrt{d})\)，\(\mathrm{fused}=\sum_k\alpha_k g_k\)。  
- 该 `Fuse` 用于 VT2 每步 readout 与非 halt 最终 readout。

### 5.6 幅度门控 \(\alpha\)

解码得到 ungated \(\Delta\) 后，在 **验证集** 上网格搜索：

\[
\alpha^\star=\arg\min_{\alpha\in\mathcal{G}}
\mathrm{MSE}\big(y^{\mathrm{CFA}}+\alpha\Delta,\,y^{\mathrm{true}}\big),
\]

\(\mathcal{G}=\{0,0.05,0.1,0.15,0.2,0.25,0.35,0.5,0.75,1.0\}\)。

测试时用同一 \(\alpha^\star\)（主指标）。另可记录 test 上选 \(\alpha\) 的 oracle，**仅诊断，不作正式分**。

### 5.7 训练超参（定稿实验）

| 项 | 值 |
|----|-----|
| Backbone | Qwen3-0.6B，bf16，冻结 |
| LoRA | r=8, α=16, dropout=0.05，目标 `q/k/v/o_proj` |
| Epochs | 8 |
| Batch size | 48（OOM 则 40→32→…→8） |
| LR | \(1\times10^{-4}\)（AdamW） |
| Weight decay | \(1\times10^{-4}\) |
| Grad clip | 1.0 |
| Gradient checkpointing | 开 |
| Val 协议（VT 实验） | `--val-on-test`（train-only；ckpt/α 用 test，**部分 oracle**） |
| 完整 leak 诊断 | 另加 `--train-includes-test`（上界，非正式） |

Checkpoint：选 **val 原始 MSE 最低** 的 epoch 权重。

---

## 6. Prompt 设计（全部原文模板）

Fact 分支 **没有多轮 chat**；每个窗口一条前缀字符串，由 `build_fact_prompt` 生成。

### 6.1 模板结构（所有域共用）

```
{fact_prompt_lead}
Target: {target_series_short}.
Origin: {origin}.
Facts:
[{ym_1}]
  {channel}: {fact_text truncated to 120 chars}
  {channel}: ...
[{ym_2}]
  ...
Think about residual vs CFA in latent space.
```

规则（实现细节）：

1. 只保留 `channel ∈ profile.focus`。  
2. 取距离 Origin 最近的 `fact_max_months` 个月。  
3. 每月 top-`fact_max_per_month` 条；排序：非 report 通道优先，再按 `|soft(effect×plaus)|` 降序。  
4. **默认不写入 mechanisms**（`include_mech=False`）。  
5. Tokenizer `truncation=True`，`max_length=profile.fact_max_length`。

### 6.2 各域 `fact_prompt_lead` 与超参

| Domain | Lead（原文） | max_length | max_months | max_per_month |
|--------|--------------|------------|------------|---------------|
| agriculture | `Infer CFA residual for U.S. monthly retail broiler composite from event facts only.` | 640 | 4 | 2 |
| economy | `Infer CFA residual for U.S. monthly goods & services trade balance from event facts only.` | 576 | 4 | 2 |
| energy | `Infer CFA residual for U.S. weekly retail gasoline price from event facts only.` | 576 | 4 | 2 |
| security | `Infer CFA residual for U.S. monthly FEMA grant outlays from event facts only.` | 576 | 4 | 2 |
| social_good | `Infer CFA residual for U.S. monthly unemployment rate from event facts only.` | 576 | 4 | 2 |
| public_health | `Infer CFA residual for U.S. weekly influenza-like illness (ILI) ratio from event facts only.` | 640 | 4 | 2 |
| traffic | `Infer CFA residual for U.S. monthly VMT from event facts only.` | 576 | **6** | 2 |

通用 fallback lead（代码备用）：

```
Infer CFA residual for {target_series_short} from event facts only.
```

共享尾句（所有域）：

```
Think about residual vs CFA in latent space.
```

共享常量（`LEAD_RESIDUAL`，可供其它显式/消融复用，**Fact 主路径用的是 profile lead**）：

```
Predict the residual (TRUE minus CFA) for the forecast horizon.
Use only dated event facts that can move this series after Origin.
Do not restate the CFA level. If facts are weak or conflicting, output near-zero residual.
```

### 6.3 完整示例（agriculture，真实拼装风格）

```
Infer CFA residual for U.S. monthly retail broiler composite from event facts only.
Target: U.S. monthly retail broiler composite.
Origin: 1991-06.
Facts:
[1991-03]
  demand: In early 1991, the market saw an increase in the variety of chicken types available, including 5- to 6-pound stewing hen
[1991-04]
  demand: The United States market for chicken and eggs has voluntary limits to ensure a high return.
  demand: There was an increase in variety of chicken types available in the market, including 5- to 6-pound stewing hens.
[1991-05]
  demand: U.S. retail demand for curry chicken is priced about 15% less than standard broiler price (1991-04-22).
  demand: U.S. beef prices reached record highs due to a cattle shortage, prompting retailers to hold down prices to retain custom
[1991-06]
  demand: U.S. broiler market had demand for curry chicken priced about 15% less than standard broiler price in April 1991.
Think about residual vs CFA in latent space.
```

### 6.4 Traffic focus 通道列表（示例）

`holiday, fuel, weather, policy, macro, transit, labor, remote_work, freight, infrastructure, vmt_report`

Agriculture focus：

`feed, supply, stocks, price, disease, trade, demand, weather, policy, macro, price_report`

（其它域见 `language/lamp_timemmd` / `profiles.py`。）

### 6.5 隐式分支 **不使用** 的 prompt（避免混淆）

下列属于 **显式分支 / 其它候选**，**不是** Fact LatentCoT 输入：

- Scheme5 `prompt_infer` / `prompt_rewrite` / `seed_prompt`  
- `prompt_dual`（direct/indirect JSON）  
- `prompt_lamp_driver`（LAMP residual-evidence JSON）

隐式分支 **不会** 要求模型输出 JSON；\(\Delta\) 完全由回归头产生。

---

## 7. VT0 / VT1 / VT2 / VT3 对照（是否逐层递增？）

**是，严格累加。**

| | Coconut recycle | Adaptive halt | Mixture K=3 |
|--|:---:|:---:|:---:|
| VT0 | ✗ | ✗ | ✗ |
| **VT1** | ✓ | ✗ | ✗ |
| **VT2** | ✓ | ✓ | ✗ |
| **VT3** | ✓ | ✓ | ✓ |

CLI 对应：

```bash
# VT0
python -u scripts/fair_branch_resid.py --domain $D --tag VT0_baseline \
  --skip-soft --lora --val-on-test --amp-select val \
  --epochs 8 --batch-size 48 --lr-fact 1e-4 --n-latent 12

# VT1
... --tag VT1_coconut --coconut-recycle

# VT2
... --tag VT2_coconut_halt --coconut-recycle --adaptive-halt

# VT3
... --tag VT3_coconut_halt_mix --coconut-recycle --adaptive-halt --n-thought-groups 3
```

一键后台：`scripts/run_fact_valtest_tricks_bg.py`。

---

## 8. 推荐实验协议与已有结果（写论文用）

### 8.1 正式叙述建议

1. **主表（诚实）**：train-only + **chrono/stride val**（不碰 test）上的 gated rel。  
2. **诊断上界 A（本文 VT）**：train-only + **val=test**（选 ckpt/α 泄漏）——展示「若验证对齐测试，trick 是否仍有益」。  
3. **诊断上界 B（leak）**：train∪test + val=test——展示容量上界，**不可当正式分数**。

### 8.2 VT（val=test，train-only）七域结果摘要

| Domain | VT0 | VT1 | VT2 | VT3 |
|--------|----:|----:|----:|----:|
| agriculture | −2.0% | −2.3% | **−8.5%** | −6.1% |
| economy | −3.5% | −3.9% | −2.6% | −2.6% |
| energy | **−4.4%** | −1.3% | −1.5% | −1.4% |
| security | −0.9% | −0.9% | −0.9% | −1.3% |
| social_good | −0.5% | −0.9% | −1.7% | −2.9% |
| public_health | −6.3% | −9.3% | −7.3% | **−9.6%** |
| traffic | ~0% | −0.2% | −0.3% | −1.3% |
| **mean** | **−2.5%** | **−2.7%** | **−3.2%** | **−3.6%** |

解读：逐层累加 **没有整体崩坏**；mean 从 −2.5% 到 −3.6%。增益集中在 agri / PH / social_good；energy 上 VT0 最好（trick 非全域单调）。

### 8.3 对照：full leak（train∪test）同结构

| | L0 | L1 | L2 | L3 |
|--|---:|---:|---:|---:|
| mean | −17.2% | −23.1% | **−27.5%** | −26.9% |

说明：**大头增益来自见到 test residual 分布**；但同一套 VT1–VT3 在 leak 上仍进一步改善，支撑「机制有容量」。

产物路径：

- `artifacts/stage2_fact_valtest_tricks/summary_valtest_tricks.json`  
- `artifacts/stage2_fact_leak_tricks/summary_leak_tricks.json`

---

## 9. 实现索引（方便写「Reproducibility」）

| 模块 | 路径 |
|------|------|
| 模型 + VT1/2/3 | `language/stage2/models.py` → `FactLatentCoT` |
| Fact prompt | `language/stage2/explicit_prompts.py` → `build_fact_prompt` |
| 域配置 / lead | `language/stage2/profiles.py` |
| 样本收集 | `language/stage2/collect.py` |
| 训练 / α 门控 / 指标 | `scripts/fair_branch_resid.py` |
| VT 后台扫 | `scripts/run_fact_valtest_tricks_bg.py` |
| Leak 后台扫 | `scripts/run_fact_leak_tricks_bg.py` |

关键类参数：

```python
FactLatentCoT(
    ...,
    n_latent=12, n_steps=3, n_aux=4, use_aux=True,
    use_lora=True, lora_r=8, lora_alpha=16,
    coconut_recycle=True,   # VT1+
    adaptive_halt=True,     # VT2+
    n_thought_groups=3,     # VT3
)
```

---

## 10. 论文写作建议（结构草稿）

1. **Problem**：CFA 残差 + 事件条件。  
2. **Method overview**：隐式 LatentCoT 回归 \(\Delta\)；与 Scheme5 并列。  
3. **Base FactLatentCoT**：prompt、标准化、LoRA、α-gate。  
4. **Enhancements**：  
   - 4.1 Continuous thought recycle（Coconut）  
   - 4.2 Density-aware soft halt（ACT）  
   - 4.3 Mixture-of-thoughts（superposition）  
5. **Ablation**：VT0–VT3；可选 leak 上界。  
6. **Limitation**：soft halt 仍跑满 \(T\) 步；无 curriculum；val=test 仅诊断。

---

## 11. 附录：算法伪代码（VT3 全开）

```
Input: prompt tokens X, soft event tensor E, y_CFA, (μ, σ)
Hyper: T=3, n_lat=12, K=3, use_aux=True

e ← Embed(X);  ctx ← masked_mean(e);  ρ ← mean(mask)
lat ← reshape(W_ctx ctx) + b_lat
aux ← AuxProj(E) if use_aux else ∅
R ← 1;  out ← 0

for t = 0 .. T-1:
    h ← LLM([e; aux; lat]).last_hidden[:, -n_lat:]
    lat ← W_thought(h)                          # VT1 Coconut
    p ← σ(HaltMLP([ctx; mean(lat)+0.1 e_t; ρ; t/(T-1)]))
    p ← clip(p · (0.35+0.65ρ))
    if t == T-1: p ← 1
    fused ← AttnFuse(lat, ctx; K=3)             # VT3
    Δ̂_t ← Readout(fused)
    w ← R · p;  out ← out + w · Δ̂_t;  R ← R·(1-p)   # VT2

z ← 3 tanh(out/3);  Δ ← z·σ + μ
return Δ
```

测试：\(\hat{y}=y^{\mathrm{CFA}}+\alpha^\star\Delta\)。

---

## 12. 版本与定稿声明

- **隐式分支定稿配置**：VT1 → VT2 → VT3 累加栈（主报告建议 **VT3**，消融含 VT0–VT2）。  
- **文档对应代码**：以当前仓库 `FactLatentCoT` 实现为准（含 `thought_proj` / `halt_mlp` / `group_query|key`）。  
- **与 Scheme5 v5 文档关系**：显式见 `docs/Scheme5_v5_Technical_Specification.md`；隐式见本文。二者共享 CFA / Time-MMD / 事件库，方法正交。

---

*文档生成用途：论文方法章节底稿。数值以 `artifacts/stage2_fact_*_tricks/` 下 JSON 为准；复现请固定种子与 Stage1 checkpoint。*
