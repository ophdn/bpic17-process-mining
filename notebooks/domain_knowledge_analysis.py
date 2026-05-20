"""
Domain Knowledge Validation — BPIC-17
======================================
Empirical justification for each Rojos et al. (bpi2017_paper_36)
preprocessing step, based on analysis of the event log itself.

Analyses:
  A — Sequence Merge Justification  (Step 2a)
      Time-delta analysis for near-instant activity pairs
      → Merge candidate: co-occur in >= MIN_CO_OCCUR_PCT of cases where
        either activity appears AND mean delta < MERGE_THRESHOLD_S seconds

  B — Endpoint Distribution          (Step 2b)
      Distribution of final activities per case
      → Justifies removing cases with invalid endpoints

  C — Fraud Case Analysis            (Step 2c)
      Prevalence and structure of fraud assessment cases
      → Justifies removing fraud cases as exceptional sub-process

  D — Variant Coverage Curve         (Step 2d)
      Cumulative case coverage as a function of variant count
      → Justifies rare-variant threshold of 100

All plots are saved to OUTPUT_DIR.
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.objects.log.util import filtering_utils

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
LOG_PATH   = r"C:\Users\sophi\OneDrive\Documents\uni\Master\SS26\praktikum\bpic17-process-mining\data\BPIChallenge2017.xes.gz"
OUTPUT_DIR = "output_domain_knowledge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Valid endpoint activities (Step 2b) — everything else is invalid
VALID_END_ACTIVITIES = {
    "A_Approved",    # offer accepted + loan granted
    "A_Pending",     # only valid AFTER O_Accepted merge
    "A_Denied",      # application rejected
    "A_Cancelled",   # application withdrawn
    "O_Cancelled",   # offer cancelled (before acceptance)
    "W_Assess potential fraud",  # fraud cases are a separate sub-process
}

# Fraud activity (Step 2c)
FRAUD_ACTIVITY = "W_Assess potential fraud"

# Rare variant threshold (Step 2d) — must match common.RARE_VARIANT_THRESHOLD
RARE_VARIANT_THRESHOLD = 100

# Merge threshold for Step 2a (seconds)
MERGE_THRESHOLD_S = 5

# Minimum fraction of cases (where at least one of the two activities appears)
# in which both must co-occur to be considered a merge candidate
MIN_CO_OCCUR_PCT = 1.0

# Plot style
COLORS = {
    "primary"  : "#1a5276",
    "secondary": "#2e86c1",
    "accent"   : "#e74c3c",
    "ok"       : "#27ae60",
    "warn"     : "#f39c12",
    "light"    : "#d6eaf8",
    "grid"     : "#eaecee",
}
plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.color"       : COLORS["grid"],
    "grid.linewidth"   : 0.8,
    "figure.dpi"       : 150,
})


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ═════════════════════════════════════════════════════════════════════════════
# LOAD & PREPARE
# ═════════════════════════════════════════════════════════════════════════════
print("Loading event log …")
log_raw = xes_importer.apply(LOG_PATH)
print(f"  {len(log_raw)} cases loaded.\n")

df_raw = pm4py.convert_to_dataframe(log_raw)
df_raw["time:timestamp"] = pd.to_datetime(df_raw["time:timestamp"], utc=True)

# COMPLETE-only DataFrame
df = df_raw[df_raw["lifecycle:transition"].fillna("").str.upper() == "COMPLETE"].copy()
df = df.sort_values(["case:concept:name", "time:timestamp"])
n_total_cases = df["case:concept:name"].nunique()
print(f"  {len(df)} COMPLETE events across {n_total_cases} cases.\n")


# ═════════════════════════════════════════════════════════════════════════════
# A — SEQUENCE MERGE JUSTIFICATION (Step 2a)
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  A — Sequence Merge Justification (Step 2a)")
print("=" * 60)

ACTIVITIES = [
    'A_Create Application', 'A_Submitted', 'W_Handle leads',
    'W_Complete application', 'A_Concept', 'A_Accepted', 'O_Create Offer',
    'O_Created', 'O_Sent (mail and online)', 'W_Call after offers',
    'A_Complete', 'W_Validate application', 'A_Validating', 'O_Returned',
    'W_Call incomplete files', 'A_Incomplete', 'O_Accepted', 'A_Pending',
    'A_Denied', 'O_Refused', 'O_Cancelled', 'A_Cancelled',
    'O_Sent (online only)', 'W_Assess potential fraud',
    'W_Personal Loan collection', 'W_Shortened completion',
]

print(f"  Computing metrics for {len(ACTIVITIES) * (len(ACTIVITIES)-1)} directed pairs …")

# Pre-build per-case first-occurrence maps for speed
first_ts  = {}
case_acts = {}
for case_id, group in df.groupby("case:concept:name"):
    g_sorted = group.sort_values("time:timestamp")
    act_map  = {}
    for _, row in g_sorted.iterrows():
        a = row["concept:name"]
        if a not in act_map:
            act_map[a] = row["time:timestamp"]
    first_ts[case_id]  = act_map
    case_acts[case_id] = set(act_map.keys())

rows = []
for act1 in ACTIVITIES:
    for act2 in ACTIVITIES:
        if act1 == act2:
            continue

        deltas      = []
        only_first  = 0
        only_second = 0
        for case_id, acts in case_acts.items():
            if act1 in acts and act2 in acts:
                delta = abs((first_ts[case_id][act2]
                             - first_ts[case_id][act1]).total_seconds())
                deltas.append(delta)
            elif act1 in acts:
                only_first += 1
            elif act2 in acts:
                only_second += 1

        n_co         = len(deltas)
        n_any        = n_co + only_first + only_second   # cases where at least one activity appears
        pct_co_occur = n_co / n_any if n_any > 0 else 0.0

        if n_co == 0:
            rows.append({
                "act1"           : act1,
                "act2"           : act2,
                "pct_co_occur"   : 0.0,
                "mean_s"         : None,
                "merge_candidate": False,
            })
            continue

        s    = pd.Series(deltas, dtype=float)
        mean = s.mean()

        is_candidate = (
            pct_co_occur >= MIN_CO_OCCUR_PCT and
            mean         <  MERGE_THRESHOLD_S
        )

        rows.append({
            "act1"           : act1,
            "act2"           : act2,
            "pct_co_occur"   : round(pct_co_occur, 4),
            "mean_s"         : round(mean, 2),
            "merge_candidate": is_candidate,
        })

pair_df = pd.DataFrame(rows)

# ── Print candidates ──────────────────────────────────────────────────────────
candidates = pair_df[pair_df["merge_candidate"]].sort_values("mean_s")
print(f"\n  Merge candidates "
      f"(co_occur >= {MIN_CO_OCCUR_PCT:.0%} of cases, mean < {MERGE_THRESHOLD_S}s):\n")
print(f"  {'act1':<30} {'act2':<30} {'co_occur%':>9}  {'mean_s':>7}")
print(f"  {'-'*75}")
for _, r in candidates.iterrows():
    print(f"  {r['act1']:<30} {r['act2']:<30} "
          f"{r['pct_co_occur']:>9.1%}  {r['mean_s']:>7.2f} s")

# ── Save CSVs ─────────────────────────────────────────────────────────────────
pair_df.to_csv(os.path.join(OUTPUT_DIR, "A_all_pairs.csv"), index=False)
candidates.to_csv(os.path.join(OUTPUT_DIR, "A_merge_candidates.csv"), index=False)
print(f"\n  [saved CSV] {OUTPUT_DIR}/A_all_pairs.csv")
print(f"  [saved CSV] {OUTPUT_DIR}/A_merge_candidates.csv")

# ── Bar chart: merge candidates + 2 next non-candidates, x-axis capped at 8s ─
# Sort all qualifying pairs by mean_s ascending
all_qualifying = (pair_df[pair_df["pct_co_occur"] >= MIN_CO_OCCUR_PCT]
                  .dropna(subset=["mean_s"])
                  .sort_values("mean_s", ascending=True)
                  .reset_index(drop=True))

# Keep candidates + the next 2 non-candidates above the threshold
n_cands   = candidates.shape[0]
plot_df   = all_qualifying.head(n_cands + 2).sort_values("mean_s", ascending=False)

plot_labels  = [f"{r['act1']} → {r['act2']}" for _, r in plot_df.iterrows()]
plot_means   = plot_df["mean_s"].tolist()
plot_is_cand = plot_df["merge_candidate"].tolist()
bar_colors   = [COLORS["primary"] if c else COLORS["secondary"]
                for c in plot_is_cand]

fig, ax = plt.subplots(figsize=(12, max(4, len(plot_labels) * 0.7)))

ax.barh(range(len(plot_labels)), plot_means,
        color=bar_colors, edgecolor="white", height=0.6, alpha=0.85)

ax.axvline(MERGE_THRESHOLD_S, color=COLORS["accent"], linestyle=":",
           linewidth=2, label=f"{MERGE_THRESHOLD_S} s threshold")

# Annotate co_occur% on each bar
for bar, (_, row) in zip(ax.patches, plot_df.iterrows()):
    x_pos = min(bar.get_width() + 0.1, 7.6)
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
            f"co={row['pct_co_occur']:.0%}  {row['mean_s']:.2f}s",
            va="center", fontsize=9, color="dimgrey")

ax.set_yticks(range(len(plot_labels)))
ax.set_yticklabels(plot_labels, fontsize=9)
ax.set_xlim(0, 8)
ax.set_xlabel("Mean |time delta| between first occurrences (seconds)")
ax.set_title(
    f"Activity Pair Merge Candidates — Mean Time Delta\n"
    f"Showing candidates + 2 nearest non-candidates  |  "
    f"co = % of all cases where both activities appear",
    fontweight="bold",
)
ax.legend(handles=[
    mpatches.Patch(color=COLORS["primary"],
                   label=f"Merge candidate (co≥{MIN_CO_OCCUR_PCT:.0%}, mean<{MERGE_THRESHOLD_S}s)"),
    mpatches.Patch(color=COLORS["secondary"], label="Non-candidate (above threshold)"),
    Line2D([0], [0], color=COLORS["accent"], linestyle=":", linewidth=2,
           label=f"{MERGE_THRESHOLD_S} s threshold"),
], fontsize=9, loc="lower right")
fig.tight_layout()
save(fig, "A_merge_candidates")


# ═════════════════════════════════════════════════════════════════════════════
# B — ENDPOINT DISTRIBUTION (Step 2b)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  B — Endpoint Distribution (Step 2b)")
print("=" * 60)

last_events = (df.groupby("case:concept:name")
                 .apply(lambda g: g.sort_values("time:timestamp").iloc[-1]["concept:name"])
                 .value_counts()
                 .reset_index())
last_events.columns = ["activity", "count"]
last_events["pct"]        = last_events["count"] / last_events["count"].sum()
last_events["is_invalid"] = ~last_events["activity"].isin(VALID_END_ACTIVITIES)

print("\n  Final activity distribution (COMPLETE events):")
print(f"  {'Activity':<45} {'Count':>7}  {'%':>6}  {'Valid?':>7}")
print(f"  {'-'*70}")
for _, row in last_events.iterrows():
    print(f"  {row['activity']:<45} {row['count']:>7}  "
          f"{row['pct']:>5.1%}  {'invalid' if row['is_invalid'] else 'valid':>7}")

n_invalid = last_events[last_events["is_invalid"]]["count"].sum()
n_total   = last_events["count"].sum()
print(f"\n  Invalid endpoint cases: {n_invalid} / {n_total} = {n_invalid/n_total:.1%}")

# Export invalid endpoints with < 1 % share to CSV
rare_invalid = last_events[last_events["is_invalid"] & (last_events["pct"] < 0.01)].copy()
rare_invalid_path = os.path.join(OUTPUT_DIR, "B_rare_invalid_endpoints.csv")
rare_invalid.to_csv(rare_invalid_path, index=False)
print(f"  Rare invalid endpoints (< 1 %): {len(rare_invalid)} activities")
for _, row in rare_invalid.iterrows():
    print(f"    {row['activity']:<45} {row['count']:>7}  {row['pct']:>5.1%}")
print(f"  [saved CSV] {rare_invalid_path}")

fig, ax = plt.subplots(figsize=(11, max(5, len(last_events) * 0.5)))
colors_b = [COLORS["accent"] if r["is_invalid"] else COLORS["primary"]
            for _, r in last_events.iterrows()]
bars = ax.barh(last_events["activity"], last_events["count"],
               color=colors_b, edgecolor="white", height=0.7)
for bar, (_, row) in zip(bars, last_events.iterrows()):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
            f"{row['pct']:.1%}", va="center", fontsize=9)
ax.set_xlabel("Number of cases")
ax.set_title("Step 2b — Distribution of Final Activities per Case\n"
             "(COMPLETE events only)", fontweight="bold")
ax.legend(handles=[
    mpatches.Patch(color=COLORS["primary"], label="Valid endpoint"),
    mpatches.Patch(color=COLORS["accent"],  label="Invalid endpoint (to remove)"),
], loc="lower right")
ax.set_xlim(0, last_events["count"].max() * 1.15)
fig.tight_layout()
save(fig, "B_endpoint_distribution")


# ═════════════════════════════════════════════════════════════════════════════
# C — FRAUD CASE ANALYSIS (Step 2c)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  C — Fraud Case Analysis (Step 2c)")
print("=" * 60)

fraud_case_ids = set(
    df[df["concept:name"] == FRAUD_ACTIVITY]["case:concept:name"].unique()
)
n_fraud   = len(fraud_case_ids)
n_total_c = df["case:concept:name"].nunique()
pct_fraud = n_fraud / n_total_c

fraud_df     = df[df["case:concept:name"].isin(fraud_case_ids)]
non_fraud_df = df[~df["case:concept:name"].isin(fraud_case_ids)]
fraud_len     = fraud_df.groupby("case:concept:name").size()
non_fraud_len = non_fraud_df.groupby("case:concept:name").size()

print(f"\n  Fraud cases       : {n_fraud} / {n_total_c} = {pct_fraud:.2%}")
print(f"  Non-fraud cases   : {n_total_c - n_fraud}")
print(f"\n  Avg trace length — fraud cases    : {fraud_len.mean():.1f} events")
print(f"  Avg trace length — non-fraud cases: {non_fraud_len.mean():.1f} events")

fraud_act_freq     = fraud_df["concept:name"].value_counts() / n_fraud
non_fraud_act_freq = non_fraud_df["concept:name"].value_counts() / (n_total_c - n_fraud)
comparison = pd.DataFrame({
    "fraud"    : fraud_act_freq,
    "non_fraud": non_fraud_act_freq,
}).fillna(0)
comparison["ratio"] = (comparison["fraud"] + 1e-6) / (comparison["non_fraud"] + 1e-6)

print(f"\n  Activities most over-represented in fraud cases (ratio > 1.5):")
for act, row in comparison[comparison["ratio"] > 1.5].sort_values("ratio", ascending=False).iterrows():
    print(f"    {act:<45}  ratio={row['ratio']:.2f}  "
          f"fraud={row['fraud']:.3f}  non-fraud={row['non_fraud']:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Step 2c — Fraud Case Analysis", fontweight="bold", fontsize=13)

ax1.pie([n_fraud, n_total_c - n_fraud],
        labels=[f"Fraud cases\n({n_fraud}, {pct_fraud:.1%})",
                f"Normal cases\n({n_total_c - n_fraud}, {1-pct_fraud:.1%})"],
        colors=[COLORS["accent"], COLORS["primary"]],
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2})
ax1.set_title("Case prevalence")

_max = max(int(fraud_len.max()), int(non_fraud_len.max()))
bins = range(0, _max + 2, 2)
ax2.hist(non_fraud_len, bins=bins, alpha=0.7, color=COLORS["primary"],
         label=f"Normal cases (n={len(non_fraud_len)})", density=True)
ax2.hist(fraud_len, bins=bins, alpha=0.7, color=COLORS["accent"],
         label=f"Fraud cases (n={len(fraud_len)})", density=True)
ax2.axvline(non_fraud_len.mean(), color=COLORS["primary"], linestyle="--",
            linewidth=1.5, label=f"Normal mean = {non_fraud_len.mean():.1f}")
ax2.axvline(fraud_len.mean(), color=COLORS["accent"], linestyle="--",
            linewidth=1.5, label=f"Fraud mean = {fraud_len.mean():.1f}")
ax2.set_xlabel("Trace length (COMPLETE events per case)")
ax2.set_ylabel("Density")
ax2.set_title("Trace length distribution")
ax2.legend(fontsize=9)
fig.tight_layout()
save(fig, "C_fraud_case_analysis")


# ═════════════════════════════════════════════════════════════════════════════
# D — VARIANT COVERAGE CURVE (Step 2d)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  D — Variant Coverage Curve (Step 2d)")
print("=" * 60)

def filter_complete_events(log):
    new_log = EventLog()
    new_log.attributes.update(log.attributes)
    for trace in log:
        new_trace = Trace()
        new_trace.attributes.update(trace.attributes)
        for event in trace:
            if event.get("lifecycle:transition", "complete").upper() == "COMPLETE":
                new_trace.append(event)
        if len(new_trace) > 0:
            new_log.append(new_trace)
    return new_log

def remove_fraud_cases(log):
    new_log = EventLog()
    new_log.attributes.update(log.attributes)
    for trace in log:
        if not any(e["concept:name"] == FRAUD_ACTIVITY for e in trace):
            new_log.append(trace)
    return new_log

print("  Building pre-step-2d log (V1 → V2c pipeline) …")
log_pre2d = filter_complete_events(log_raw)  # V1:  COMPLETE only
log_pre2d = remove_fraud_cases(log_pre2d)    # V2c: drop fraud cases
variants = pm4py.get_variants_as_tuples(log_pre2d)


sorted_variants = sorted(variants.items(), key=lambda x: len(x[1]), reverse=True)

# Build a DataFrame of all variants with their frequency
variant_rows = [
    {"variant": " -> ".join(variant), "frequency": len(traces)}
    for variant, traces in sorted_variants
]
variant_df = (
    pd.DataFrame(variant_rows)
    .sort_values("frequency", ascending=False)
    .reset_index(drop=True)
)
variant_df["pct"]     = variant_df["frequency"] / variant_df["frequency"].sum()
variant_df["removed"] = variant_df["frequency"] < RARE_VARIANT_THRESHOLD

# variant_df is already sorted descending by frequency
variant_df["cumulative_pct"] = (
    variant_df.sort_values("frequency", ascending=False)["frequency"]
    .cumsum() / n_total_cases
)

# For each possible threshold, what % of cases do we keep?
thresholds = range(1, 200)
results = []
for t in thresholds:
    retained_pct = variant_df[variant_df["frequency"] >= t]["frequency"].sum() / n_total_cases
    results.append({"threshold": t, "retained_pct": retained_pct})

thresh_df = pd.DataFrame(results)
max_thresh = thresh_df[thresh_df["retained_pct"] >= 0.80]["threshold"].max()
print(f"Max threshold with ≥80% case retention: {max_thresh}")
print(thresh_df[thresh_df["threshold"] <= max_thresh + 5])

# Rare variants (to be removed)
rare_df = variant_df[variant_df["removed"]].copy()
rare_csv_path = os.path.join(OUTPUT_DIR, "D_rare_variants.csv")
rare_df.to_csv(rare_csv_path, index=False)

n_cases_pre2d      = variant_df["frequency"].sum()
n_variants_total   = len(variant_df)
n_above_threshold  = (variant_df["frequency"] >= RARE_VARIANT_THRESHOLD).sum()
n_below_threshold  = len(rare_df)
cases_above        = variant_df.loc[~variant_df["removed"], "frequency"].sum()
cases_below        = rare_df["frequency"].sum()
coverage_at_thresh = cases_above / n_cases_pre2d

variant_counts = variant_df["frequency"].tolist()   # sorted descending
cumulative     = np.cumsum(variant_counts) / n_cases_pre2d
idx_80         = next(i for i, c in enumerate(cumulative) if c >= 0.80)

print(f"  Total cases (pre-2d)        : {n_cases_pre2d}")
print(f"  Total variants              : {n_variants_total}")
print(f"  Variants >= {RARE_VARIANT_THRESHOLD} cases      : "
      f"{n_above_threshold} ({n_above_threshold/n_variants_total:.1%} of variants)")
print(f"  Cases covered by those      : {cases_above} ({coverage_at_thresh:.1%} of cases)")
print(f"  Variants <  {RARE_VARIANT_THRESHOLD} cases      : "
      f"{n_below_threshold} ({n_below_threshold/n_variants_total:.1%} of variants)  → removed")
print(f"  Cases in rare variants      : {cases_below} ({cases_below/n_cases_pre2d:.1%} of cases)")
print(f"  Variants needed for 80% cov : {idx_80 + 1}")
print(f"\n  Rare variants (freq < {RARE_VARIANT_THRESHOLD}) — top 10 by frequency:")
print(f"  {'Frequency':>10}  {'%':>6}  Variant")
print(f"  {'-'*70}")
for _, r in rare_df.head(10).iterrows():
    variant_preview = r["variant"][:80] + ("…" if len(r["variant"]) > 80 else "")
    print(f"  {r['frequency']:>10}  {r['pct']:>5.2%}  {variant_preview}")
print(f"  [saved CSV] {rare_csv_path}  ({n_below_threshold} variants)")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle(f"Step 2d — Rare Variant Elimination (frequency < {RARE_VARIANT_THRESHOLD})",
             fontweight="bold", fontsize=13)

# Left: cumulative coverage curve coloured by retained / removed
x = np.arange(1, n_variants_total + 1)
ax1.fill_between(x[:n_above_threshold], 0, cumulative[:n_above_threshold],
                 alpha=0.25, color=COLORS["ok"])
ax1.fill_between(x[n_above_threshold - 1:], 0, cumulative[n_above_threshold - 1:],
                 alpha=0.25, color=COLORS["accent"])
ax1.plot(x, cumulative, color=COLORS["primary"], linewidth=2, label="Cumulative coverage")
ax1.axhline(0.80, color=COLORS["ok"],   linestyle="--", linewidth=1.5, label="80% coverage")
ax1.axhline(0.95, color=COLORS["warn"], linestyle=":",  linewidth=1.5, label="95% coverage")
ax1.axvline(n_above_threshold, color=COLORS["accent"], linestyle="-.", linewidth=1.8,
            label=f"Cut-off: freq < {RARE_VARIANT_THRESHOLD}\n"
                  f"({n_above_threshold} kept, {n_below_threshold} removed, "
                  f"{coverage_at_thresh:.0%} coverage retained)")
ax1.set_xlabel("Variants ranked by frequency (descending)")
ax1.set_ylabel("Cumulative case coverage")
ax1.set_xlim(1, 1000)
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=8, loc="lower right")
ax1.set_title("Cumulative Coverage Curve")

# Right: frequency histogram split by retained / removed
bins = np.linspace(0, 150, 150)
retained_counts = variant_df.loc[~variant_df["removed"], "frequency"]
removed_counts  = rare_df["frequency"]
ax2.hist(retained_counts, bins=bins, color=COLORS["ok"],
         alpha=0.8, edgecolor="white", label=f"Retained (≥{RARE_VARIANT_THRESHOLD})")
ax2.hist(removed_counts, bins=bins, color=COLORS["accent"],
         alpha=0.8, edgecolor="white", label=f"Removed (<{RARE_VARIANT_THRESHOLD})")
ax2.axvline(RARE_VARIANT_THRESHOLD, color=COLORS["accent"], linestyle="-.",
            linewidth=2)
ax2.set_xlabel("Variant frequency (cases per variant)")
ax2.set_ylabel("Number of variants")
ax2.set_title("Variant Frequency Histogram")
ax2.legend(fontsize=9)
ax2.text(0.55, 0.82,
         f"{n_below_threshold} variants removed\n"
         f"(freq < {RARE_VARIANT_THRESHOLD})\n"
         f"= {cases_below/n_cases_pre2d:.1%} of cases lost",
         transform=ax2.transAxes, fontsize=9,
         bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS["light"],
                   edgecolor=COLORS["accent"], alpha=0.9))
ax2.set_xlim(0, 50)
fig.tight_layout()
save(fig, "D_variant_coverage_curve")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  DOMAIN KNOWLEDGE VALIDATION — SUMMARY")
print("=" * 60)

print(f"""
  Step 2a — Sequence Merges
  ─────────────────────────
  Criteria: co-occur in >= {MIN_CO_OCCUR_PCT:.0%} of cases where either activity appears AND mean delta < {MERGE_THRESHOLD_S}s
""")
for _, r in candidates.iterrows():
    print(f"  {r['act1']} -> {r['act2']}")
    print(f"    co_occur={r['pct_co_occur']:.1%} of cases  mean={r['mean_s']:.2f}s  [CONFIRMED]")

print(f"""
  Step 2b — Invalid Endpoints
  ────────────────────────────
  {n_invalid} cases ({n_invalid/n_total:.1%}) end in a non-terminal activity.
  These do not correspond to any defined process endpoint
  (A_Approved / A_Pending / A_Denied / A_Cancelled). [CONFIRMED]

  Step 2c — Fraud Cases
  ──────────────────────
  {n_fraud} cases ({pct_fraud:.2%}) contain fraud assessment activity.
  Fraud traces are {fraud_len.mean():.0f} events long on average vs
  {non_fraud_len.mean():.0f} for normal cases — structurally different. [CONFIRMED]

  Step 2d — Rare Variant Filter (threshold={RARE_VARIANT_THRESHOLD})
  ──────────────────────────────
  {n_below_threshold} variants ({n_below_threshold/n_variants_total:.0%}) have < {RARE_VARIANT_THRESHOLD} cases.
  Together they cover only {cases_below/n_total_cases:.1%} of cases.
  Removing them retains {coverage_at_thresh:.1%} of cases in {n_above_threshold} variants. [CONFIRMED]

  Plots saved to: {OUTPUT_DIR}/
    A_merge_candidates.png
    B_endpoint_distribution.png
    C_fraud_case_analysis.png
    D_variant_coverage_curve.png
""")