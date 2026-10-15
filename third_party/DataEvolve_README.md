<h1 align="center">Data Darwinism Part II: DataEvolve</h1>

<p align="center">AI can autonomously evolve pretraining data curation.</p>

<p align="center">
  📄 <a href="./report.pdf">Paper</a> |
  🤗 <a href="https://huggingface.co/datasets/GAIR/Darwin-CC">Dataset</a> |
  💻 <a href="https://github.com/GAIR-NLP/DataEvolve">Code</a>
</p>

<p align="center">
  <img src="./pic.png" alt="DataEvolve Overview" width="1000" />
</p>

This repository contains the implementation behind **DataEvolve**, the framework described in [Data Darwinism Part II: DataEvolve](./report.pdf). Instead of relying on a single manually designed cleaning recipe, DataEvolve improves data curation through an iterative loop: observe the data, generate a strategy, execute it, evaluate the result, and refine the next strategy from feedback.

🔗 **Dataset**: [Darwin-CC on Hugging Face](https://huggingface.co/datasets/GAIR/Darwin-CC)

## 🧭 Overview

Large pretraining corpora are heterogeneous. Different domains, content types, and quality levels exhibit different failure modes, which makes static filtering rules hard to scale.

DataEvolve addresses that problem by evolving a cleaning strategy for each category. The system is organized around four core components:

- **Data Observer** identifies recurring quality issues in raw data
- **Strategy Designer** proposes the next cleaning strategy
- **Data Cleaner** applies that strategy to sample data
- **Quality Judge** scores the result and surfaces remaining issues

Across iterations, the system keeps track of:

- an **experience pool** of discovered data issues
- a **strategy pool** of prompts, scores, and diagnostic feedback

This gives the pipeline memory, so later iterations can improve on earlier ones rather than restart from scratch.

## ✨ Why It Matters

The report motivates DataEvolve as a way to replace expensive manual strategy design with automated, category-specific refinement. The key idea is not just cleaning data with an LLM, but **evolving the cleaning strategy itself** through execution and feedback.

This repository captures that core mechanism.

## 📦 What This Repo Covers

This codebase focuses on the **strategy-evolution loop**:

- issue discovery on sampled data
- prompt evolution across iterations
- sample-based cleaning
- quality evaluation and feedback accumulation

It is best understood as the pipeline layer of the larger research system, not a full one-command reproduction of all large-scale cleaning, pretraining, and benchmark experiments from the report.

## 🔄 Workflow

At a high level, one iteration looks like this:

1. Observe raw samples and record category-specific quality issues.
2. Generate a new cleaning prompt using accumulated issues and prior prompt history.
3. Run the data cleaner on sample data.
4. Evaluate raw/cleaned pairs and assign quality scores.
5. Feed the new analysis back into the next iteration.

Over time, the best strategy becomes the parent for subsequent refinements.

## 🚀 Quick Start

Set the required API key:

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

Optional endpoint override:

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Prepare category-specific sample data under:

```text
data/level={level}--domain={domain}--content={content}/sample_data/sample_data.jsonl
```

Then run:

```bash
python pipeline.py
```

The pipeline will run data observation, strategy design, sample cleaning, and quality evaluation, while saving prompt history and discovered issues to disk.

🤗 Darwin-CC dataset: https://huggingface.co/datasets/GAIR/Darwin-CC

## Repository Layout

```text
.
├── pipeline.py
├── config.py
├── clean/
├── observer/
├── prompt_designer/
├── quality_judge/
├── executor/
├── tools/
├── utils/
├── report.pdf
└── pic.png
```

## 📝 Outputs

During execution, the pipeline writes category-scoped artifacts such as:

- `experience_database.jsonl`
- `prompt.jsonl`
- `sample_cleaned_data/{prompt_id}.jsonl`

Together, these files represent the accumulated memory of the evolutionary loop.

## 📚 Citation

```bibtex
@misc{mi2026dataevolve,
  title        = {Data Darwinism Part II: DataEvolve: AI can Autonomously Evolve Pretraining Data Curation},
  author       = {Tiantian Mi and Dongming Shan and Zhen Huang and Yiwei Qin and Muhang Xie and Yuxuan Qiao and Yixiu Liu and Chenyang Zhou and Pengfei Liu},
  year         = {2026},
  note         = {Technical report and code repository},
  howpublished = {\url{https://github.com/GAIR-NLP/DataEvolve}}
}
```