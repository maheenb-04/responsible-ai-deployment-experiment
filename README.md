# Responsible AI Deployment — Experiment Codebase

**Paper:** Responsible AI Deployment: Sustainable AI, Green Cybersecurity and Ethical Governance for a Resilient Digital Future  
**Authors:** Abu Kamruzzaman & Maheen Bilal | York College / CUNY  
**Conference:** ACDSA 2027

---

## What This Repo Contains

This repository contains the full experimental code supporting the paper's empirical findings on the relationship between Generative AI workload intensity and computational sustainability metrics (execution time, CPU utilization, CO₂ emissions).

---

## Experiment Design

| Component | Details |
|---|---|
| **Model** | `distilgpt2` — real distilled GPT-2 LLM (~82M parameters) via HuggingFace |
| **Prompt corpus** | WikiText-2 (real text dataset, HuggingFace) |
| **Conditions** | Low (5 prompts), Medium (20 prompts), High (50 prompts) |
| **Trials per condition** | 5 (for mean ± SD statistical reporting) |
| **Metrics** | Execution time (s), CPU utilization (%), CO₂ emissions (kg) |
| **Monitoring tools** | `psutil` (CPU), `CodeCarbon` (CO₂ estimation) |
| **Hardware** | CPU-only inference for cross-hardware reproducibility |

---

## Why These Choices Are Credible

- **Real model, not a math loop:** distilgpt2 performs actual language model inference — tokenization, attention, generation — making it a genuine proxy for LLM workloads at small scale.
- **Real dataset:** WikiText-2 is a standard NLP benchmark corpus. Prompts are sampled from real English text, not synthetic inputs.
- **Established methodology:** This proxy approach is consistent with Oliveira et al. (2026), Morrison et al. (2025), and Singh et al. (2025), all of which use system-level proxy metrics due to proprietary infrastructure constraints.
- **Reproducible:** Fixed random seed (42), deterministic decoding, identical system conditions per trial.

---

## How to Run

### 1. Install dependencies

```bash
pip install codecarbon transformers torch datasets numpy psutil
```

### 2. Run the experiment

```bash
python experiment.py
```

### 3. Copy output into paper

The terminal will print a formatted summary table with mean ± SD values and real-world scaling estimates. Paste those directly into Table 1.

---

## Repository Structure

```
├── experiment.py       # Main experiment script
├── README.md           # This file
└── requirements.txt    # All dependencies
```

---

## Citing This Code

If referencing this repository in the paper, add to the manuscript:

> Experimental code is publicly available at: https://github.com/[your-username]/responsible-ai-deployment-experiment

---

## Methodology Note

This simulation does not directly access proprietary data center infrastructure. It provides a controlled, reproducible approximation of relative computational impact across workload intensities, consistent with accepted practice in sustainable AI research.
