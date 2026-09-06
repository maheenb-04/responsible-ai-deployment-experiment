# Responsible AI Deployment — Experiment Codebase

This repository contains the experimental code supporting our research on the environmental cost of Generative AI inference workloads. The study measures how increasing AI usage intensity affects execution time, CPU utilization, and estimated carbon emissions across repeated trials.

**Authors:** Maheen Bilal & Dr. Abu Kamruzzaman | York College / CUNY

---

## What the experiment does

The simulation runs three workload conditions — low (5 prompts), medium (20 prompts), and high (50 prompts) — each repeated five times. For every prompt, it executes the core mathematical operations of transformer-based inference: tokenization, 768-dimensional embedding, self-attention, feed-forward projection, and logit scoring. These match the computational structure of a real language model forward pass without requiring model weights or GPU access.

CPU activity is tracked using `psutil` and carbon emissions are estimated using `CodeCarbon` based on regional grid carbon intensity. Results are reported as mean ± standard deviation across the five trials per condition.

---

## About the simulation

The simulation is a proxy — it replicates how a transformer computes, not what it outputs. The dimensions (768 hidden size, 4x feed-forward expansion, 64-dim attention head) match the distilgpt2 architecture. This approach is standard in sustainable AI research when direct infrastructure access is not available, and it produces real, measurable computational load on actual hardware.

Prompts are drawn from a curated set of 50 sentences from AI sustainability literature, so inputs reflect the kind of language a real LLM would process. A fixed random seed ensures results are fully reproducible.

---

## How to run

Install dependencies:

```bash
pip3 install codecarbon numpy psutil
```

Run the experiment:

```bash
python3 experiment.py
```

The terminal will print per-trial results and a final summary table with mean ± SD values for all three conditions. The figure generation script produces the three bar charts used in the paper:

```bash
pip3 install matplotlib
python3 generate_figures.py
```

---

## Files

| File | Description |
|---|---|
| `experiment.py` | Main experiment — runs 15 trials and prints results table |
| `generate_figures.py` | Generates Fig. 1, 2, and 3 as 300 DPI PNG files |
| `requirements.txt` | Python dependencies |
| `Fig1_ExecutionTime.png` | Execution time across workload conditions |
| `Fig2_CO2Emissions.png` | Estimated CO2 emissions across workload conditions |
| `Fig3_CPUUtilization.png` | CPU utilization across workload conditions |
