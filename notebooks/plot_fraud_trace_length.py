"""
Regenerate the fraud case trace-length distribution plot (right panel of C_fraud_case_analysis).
Output: figures/C_fraud_trace_length.png
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

warnings.filterwarnings("ignore")

LOG_PATH    = r"C:\Users\sophi\OneDrive\Documents\uni\Master\SS26\praktikum\bpic17-process-mining\data\BPIChallenge2017.xes.gz"
FRAUD_ACT   = "W_Assess potential fraud"
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "C_fraud_trace_length.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

COLORS = {"primary": "#1a5276", "accent": "#e74c3c"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#eaecee", "grid.linewidth": 0.8,
    "figure.dpi": 150,
})

# ── Load & filter ─────────────────────────────────────────────────────────────
print("Loading event log …")
log = xes_importer.apply(LOG_PATH)
df  = pm4py.convert_to_dataframe(log)
df  = df[df["lifecycle:transition"].fillna("").str.upper() == "COMPLETE"]

fraud_ids     = set(df[df["concept:name"] == FRAUD_ACT]["case:concept:name"])
fraud_len     = df[df["case:concept:name"].isin(fraud_ids)].groupby("case:concept:name").size()
non_fraud_len = df[~df["case:concept:name"].isin(fraud_ids)].groupby("case:concept:name").size()

print(f"  Fraud cases    : {len(fraud_len)}   mean length = {fraud_len.mean():.1f}")
print(f"  Non-fraud cases: {len(non_fraud_len)}  mean length = {non_fraud_len.mean():.1f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
import numpy as np
bins = range(0, max(int(fraud_len.max()), int(non_fraud_len.max())) + 2, 2)

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(non_fraud_len, bins=bins, alpha=0.7, color=COLORS["primary"],
        label=f"Normal cases (n={len(non_fraud_len)})", density=True)
ax.hist(fraud_len, bins=bins, alpha=0.7, color=COLORS["accent"],
        label=f"Fraud cases (n={len(fraud_len)})", density=True)
ax.axvline(non_fraud_len.mean(), color=COLORS["primary"], linestyle="--",
           linewidth=1.5, label=f"Normal mean = {non_fraud_len.mean():.1f}")
ax.axvline(fraud_len.mean(), color=COLORS["accent"], linestyle="--",
           linewidth=1.5, label=f"Fraud mean = {fraud_len.mean():.1f}")

ax.set_xlabel("Trace length (COMPLETE events per case)")
ax.set_ylabel("Density")
ax.set_title("Step 2c — Trace Length Distribution: Fraud vs. Normal Cases",
             fontweight="bold")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT_PATH, bbox_inches="tight")
plt.close(fig)
print(f"  [saved] {OUTPUT_PATH}")
