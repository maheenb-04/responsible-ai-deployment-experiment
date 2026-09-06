# Figure Generation Script
# Responsible AI Deployment: Sustainable AI, Green Cybersecurity
# Authors: Abu Kamruzzaman & Maheen Bilal | York College / CUNY
#
# Generates Figures 1, 2, and 3 from experimental results (n=5 trials)
# Saves as high-resolution PNG files suitable for IEEE paper submission

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Experimental data (Mean ± SD, n=5 trials per condition) ──────────────────

conditions = ["Low\n(5 prompts)", "Medium\n(20 prompts)", "High\n(50 prompts)"]
x = np.arange(len(conditions))
bar_width = 0.5

# Execution time
time_mean = [0.1142, 0.4345, 1.1336]
time_sd   = [0.0115, 0.0138, 0.0521]

# CPU utilization
cpu_mean  = [16.6, 16.9, 20.1]
cpu_sd    = [18.5, 16.8, 16.0]

# CO2 emissions
co2_mean  = [1.250e-7, 3.858e-7, 1.182e-6]
co2_sd    = [7.294e-8, 7.269e-8, 2.549e-7]

# ── Shared style settings ─────────────────────────────────────────────────────

TITLE_SIZE   = 13
LABEL_SIZE   = 11
TICK_SIZE    = 10
CAPTION_SIZE = 9
DPI          = 300

COLOR_LOW    = "#4C72B0"
COLOR_MED    = "#55A868"
COLOR_HIGH   = "#C44E52"
COLORS       = [COLOR_LOW, COLOR_MED, COLOR_HIGH]
EDGE         = "white"
CAPSIZE      = 6
ERROR_KW     = dict(elinewidth=1.4, capthick=1.4, capsize=CAPSIZE, color="black")

def style_ax(ax, ylabel, title):
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=TICK_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_axisbelow(True)

# ── FIGURE 1: Execution Time ──────────────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(6.5, 4.5))

bars1 = ax1.bar(
    x, time_mean, width=bar_width,
    color=COLORS, edgecolor=EDGE, linewidth=0.8, zorder=3
)
ax1.errorbar(
    x, time_mean, yerr=time_sd,
    fmt="none", **ERROR_KW, zorder=4
)

# Annotate bar tops with mean value
for bar, mean, sd in zip(bars1, time_mean, time_sd):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        mean + sd + 0.015,
        f"{mean:.4f}s",
        ha="center", va="bottom", fontsize=9, fontweight="bold"
    )

style_ax(ax1, "Execution Time (seconds)", "Fig. 1 — Execution Time Across GenAI Workload Conditions")
ax1.set_xlabel("Workload Condition", fontsize=LABEL_SIZE)

legend_patches = [
    mpatches.Patch(color=COLOR_LOW,  label="Low (5 prompts)"),
    mpatches.Patch(color=COLOR_MED,  label="Medium (20 prompts)"),
    mpatches.Patch(color=COLOR_HIGH, label="High (50 prompts)"),
]
ax1.legend(handles=legend_patches, fontsize=9, loc="upper left", framealpha=0.9)

fig1.text(
    0.5, -0.02,
    "Error bars represent ±1 SD across n=5 independent trials per condition.",
    ha="center", fontsize=CAPTION_SIZE, style="italic", color="#444444"
)

plt.tight_layout()
plt.savefig("Fig1_ExecutionTime.png", dpi=DPI, bbox_inches="tight")
print("Saved: Fig1_ExecutionTime.png")
plt.close()

# ── FIGURE 2: CO2 Emissions ───────────────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))

# Convert to scientific notation labels
co2_labels = ["1.250e-7", "3.858e-7", "1.182e-6"]

bars2 = ax2.bar(
    x, co2_mean, width=bar_width,
    color=COLORS, edgecolor=EDGE, linewidth=0.8, zorder=3
)
ax2.errorbar(
    x, co2_mean, yerr=co2_sd,
    fmt="none", **ERROR_KW, zorder=4
)

for bar, mean, sd, label in zip(bars2, co2_mean, co2_sd, co2_labels):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        mean + sd + 1.5e-8,
        label + " kg",
        ha="center", va="bottom", fontsize=8.5, fontweight="bold"
    )

style_ax(ax2, "Estimated CO\u2082 Emissions (kg)", "Fig. 2 — Estimated CO\u2082 Emissions Across GenAI Workload Conditions")
ax2.set_xlabel("Workload Condition", fontsize=LABEL_SIZE)
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda val, _: f"{val:.2e}")
)

ax2.legend(handles=legend_patches, fontsize=9, loc="upper left", framealpha=0.9)

fig2.text(
    0.5, -0.02,
    "Error bars represent ±1 SD across n=5 independent trials per condition. Estimated via CodeCarbon.",
    ha="center", fontsize=CAPTION_SIZE, style="italic", color="#444444"
)

plt.tight_layout()
plt.savefig("Fig2_CO2Emissions.png", dpi=DPI, bbox_inches="tight")
print("Saved: Fig2_CO2Emissions.png")
plt.close()

# ── FIGURE 3: CPU Utilization ─────────────────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(6.5, 4.5))

bars3 = ax3.bar(
    x, cpu_mean, width=bar_width,
    color=COLORS, edgecolor=EDGE, linewidth=0.8, zorder=3
)
ax3.errorbar(
    x, cpu_mean, yerr=cpu_sd,
    fmt="none", **ERROR_KW, zorder=4
)

for bar, mean, sd in zip(bars3, cpu_mean, cpu_sd):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        mean + sd + 0.8,
        f"{mean:.1f}%",
        ha="center", va="bottom", fontsize=9, fontweight="bold"
    )

style_ax(ax3, "CPU Utilization (%)", "Fig. 3 — CPU Utilization Across GenAI Workload Conditions")
ax3.set_xlabel("Workload Condition", fontsize=LABEL_SIZE)
ax3.set_ylim(0, max(cpu_mean) + max(cpu_sd) + 12)

# Add note about high SD being a finding
ax3.text(
    0.98, 0.97,
    "Note: High SD reflects dynamic\nOS-level scheduling variance\n(see Results discussion)",
    transform=ax3.transAxes,
    ha="right", va="top", fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8E7", edgecolor="#CCAA00", alpha=0.9)
)

ax3.legend(handles=legend_patches, fontsize=9, loc="upper left", framealpha=0.9)

fig3.text(
    0.5, -0.02,
    "Error bars represent ±1 SD across n=5 independent trials per condition. Monitored via psutil.",
    ha="center", fontsize=CAPTION_SIZE, style="italic", color="#444444"
)

plt.tight_layout()
plt.savefig("Fig3_CPUUtilization.png", dpi=DPI, bbox_inches="tight")
print("Saved: Fig3_CPUUtilization.png")
plt.close()

print("\nAll three figures saved as high-resolution PNG files (300 DPI).")
print("Insert into your paper as Fig. 1, Fig. 2, and Fig. 3.")
print("Caption each with [self-drawn] per IEEE figure attribution standard.")
EOF