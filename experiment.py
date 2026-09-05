# Responsible AI Deployment — Experiment
# Sustainable AI, Green Cybersecurity and Ethical Governance
# Authors: Abu Kamruzzaman & Maheen Bilal | York College / CUNY
#
# This experiment measures the relationship between GenAI inference
# workload intensity and computational sustainability metrics.
# Uses a real HuggingFace language model (distilgpt2) for genuine
# GenAI-like inference, monitored via psutil and CodeCarbon.
#
# Methodology aligns with: Oliveira et al. (2026), Morrison et al. (2025),
# Singh et al. (2025) — all of which use proxy/simulation approaches
# with system-level monitoring due to proprietary infrastructure constraints.
#
# Dataset backing: HuggingFace wikitext-2 (real text corpus) used as
# prompt source, ensuring inference reflects realistic language inputs
# rather than synthetic computation.

import time
import numpy as np
import psutil
import warnings
warnings.filterwarnings("ignore")

# ── dependency check ──────────────────────────────────────────────────────────
try:
    from codecarbon import EmissionsTracker
except ImportError:
    raise SystemExit("Run:  pip install codecarbon transformers torch datasets numpy psutil")

try:
    from transformers import pipeline
    from datasets import load_dataset
except ImportError:
    raise SystemExit("Run:  pip install transformers torch datasets")

# ── load real backing dataset (wikitext-2) ────────────────────────────────────
print("Loading WikiText-2 dataset for prompt corpus...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# Filter to sentences of reasonable length for prompting
prompts_pool = [
    row["text"].strip()
    for row in dataset
    if len(row["text"].strip()) > 40 and len(row["text"].strip()) < 200
]
print(f"  Prompt pool size: {len(prompts_pool)} real text samples\n")

# ── load real small language model ───────────────────────────────────────────
# distilgpt2: real GPT-2 distilled model, ~82M params, publicly available,
# small enough to run on CPU, large enough to be meaningful inference.
print("Loading distilgpt2 model (real HuggingFace LLM)...")
generator = pipeline(
    "text-generation",
    model="distilgpt2",
    device=-1,          # CPU — matches constrained/edge deployment scenario
    max_new_tokens=30,  # short generation keeps trials fast but real
    do_sample=False,    # deterministic output for reproducibility
    truncation=True,
)
print("  Model loaded.\n")

# ── experiment configuration ──────────────────────────────────────────────────
CONDITIONS = {
    "Low":    5,   # 5 inference calls
    "Medium": 20,  # 20 inference calls
    "High":   50,  # 50 inference calls
}
NUM_TRIALS      = 5       # repeated trials per condition for statistical validity
SCALE_QUERIES   = 1_000_000  # daily query volume for real-world scaling estimate
RANDOM_SEED     = 42
rng             = np.random.default_rng(RANDOM_SEED)

# ── trial runner ──────────────────────────────────────────────────────────────
def run_trial(num_prompts: int) -> tuple[float, float, float]:
    """
    Run one trial: sample `num_prompts` real text prompts from WikiText-2,
    run distilgpt2 inference on each, record execution time, CPU %, CO₂.
    Returns (elapsed_seconds, cpu_percent, co2_kg).
    """
    # Sample prompts without replacement for this trial
    indices = rng.choice(len(prompts_pool), size=num_prompts, replace=False)
    trial_prompts = [prompts_pool[i] for i in indices]

    tracker = EmissionsTracker(
        log_level="error",
        save_to_file=False,
        tracking_mode="process",
    )

    cpu_before = psutil.cpu_percent(interval=0.1)
    tracker.start()
    t_start = time.perf_counter()

    # Real LLM inference — this is the actual workload
    _ = generator(trial_prompts)

    elapsed   = time.perf_counter() - t_start
    emissions = tracker.stop()        # kg CO₂
    cpu_after = psutil.cpu_percent(interval=0.1)

    cpu_avg = (cpu_before + cpu_after) / 2.0
    co2     = emissions if (emissions and emissions > 0) else 0.0
    return elapsed, cpu_avg, co2

# ── main experiment loop ──────────────────────────────────────────────────────
results = {}

for condition, num_prompts in CONDITIONS.items():
    print(f"{'='*60}")
    print(f"Condition: {condition} ({num_prompts} prompts) — {NUM_TRIALS} trials")
    print(f"{'='*60}")

    times, cpus, co2s = [], [], []

    for trial_num in range(1, NUM_TRIALS + 1):
        elapsed, cpu, co2 = run_trial(num_prompts)
        times.append(elapsed)
        cpus.append(cpu)
        co2s.append(co2)
        print(f"  Trial {trial_num}: "
              f"time={elapsed:.4f}s  "
              f"cpu={cpu:.1f}%  "
              f"co2={co2:.3e} kg")

    results[condition] = {
        "n_prompts":  num_prompts,
        "time_mean":  np.mean(times),
        "time_sd":    np.std(times, ddof=1),    # sample SD
        "cpu_mean":   np.mean(cpus),
        "cpu_sd":     np.std(cpus, ddof=1),
        "co2_mean":   np.mean(co2s),
        "co2_sd":     np.std(co2s, ddof=1),
        "co2_scaled": np.mean(co2s) * SCALE_QUERIES,
    }
    print()

# ── results summary ───────────────────────────────────────────────────────────
print("\n" + "="*80)
print("RESULTS SUMMARY — Copy these values into Table 1")
print("="*80)
header = (f"{'Condition':<10} {'Time (s) mean±SD':<24} "
          f"{'CPU (%) mean±SD':<22} "
          f"{'CO₂ (kg) mean±SD':<32} "
          f"CO₂ @ 1M queries/day (kg)")
print(header)
print("-"*len(header))

for cond, r in results.items():
    print(
        f"{cond:<10} "
        f"{r['time_mean']:.4f} ± {r['time_sd']:.4f}       "
        f"{r['cpu_mean']:.1f} ± {r['cpu_sd']:.1f}            "
        f"{r['co2_mean']:.3e} ± {r['co2_sd']:.3e}    "
        f"{r['co2_scaled']:.4f}"
    )

print("\nNOTE: SD computed with ddof=1 (sample standard deviation, n=5 trials).")
print(f"Scaling: mean CO₂ per query × {SCALE_QUERIES:,} queries/day.")
print("Prompts sourced from WikiText-2 (HuggingFace), a real text corpus.")
print("Model: distilgpt2 — a real distilled GPT-2 LLM (~82M parameters).")
print("Inference run on CPU for reproducibility across hardware configurations.")
