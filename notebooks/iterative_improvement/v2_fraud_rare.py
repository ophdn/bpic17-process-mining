"""
v2_fraud_rare.py — Alternative pipeline: V1 → V2c (fraud) → V2d (rare variants)
==================================================================================
Skips V2a (merge sequences) and V2b (endpoint filter) to isolate the impact of
the two filters with the strongest effect.

Metrics are evaluated against the original raw log at every step and written to
a dedicated CSV (separate from the main iteration_metrics.csv):

    output/iteration_metrics_fraud_rare.csv

Intermediate logs are stored under distinct keys so the standard V2 pipeline is
not disturbed:

    output/log_fr_v1.pkl    — lifecycle-filtered (same as V1)
    output/log_fr_v2c.pkl   — after fraud filter
    output/log_fr_v2d.pkl   — after rare-variant filter  (final)

Run from any directory:
    python iterative_improvement/v2_fraud_rare.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import pandas as pd

# ── Dedicated output CSV ───────────────────────────────────────────────────────
METRICS_CSV_FR = os.path.join(common.OUTPUT_DIR, "iteration_metrics_fraud_rare.csv")

# ── Dedicated pickle paths (do not overwrite standard pipeline logs) ───────────
LOG_FR_V2C_PKL = os.path.join(common.OUTPUT_DIR, "log_fr_v2c.pkl")
LOG_FR_V2D_PKL = os.path.join(common.OUTPUT_DIR, "log_fr_v2d.pkl")


def upsert_metrics(records, version_prefix, csv_path):
    """Upsert records into the fraud-rare metrics CSV."""
    new_df = pd.DataFrame(records)
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing = existing[~existing["version"].str.startswith(version_prefix)]
        df = pd.concat([existing, new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(csv_path, index=False)
    print(f"  [saved] {csv_path}")


def make_record(version, n_cases, n_raw, m):
    return {
        "version"            : version,
        "algo"               : "IMf",
        "cases_log"          : n_cases,
        "pct_removed_vs_raw" : round((n_raw - n_cases) / n_raw * 100, 2),
        **m,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 0. Raw log
# ═════════════════════════════════════════════════════════════════════════════
log_raw, n_raw, v_raw, a_raw = common.load_raw_log()

# ═════════════════════════════════════════════════════════════════════════════
# V1: Reuse already-computed lifecycle-filtered log from standard pipeline
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*68}")
print("  Step V1 — reusing existing log_v1.pkl (lifecycle filter)")
print(f"{'='*68}\n")

print(f"Loading V1 log from {common.LOG_V1_PKL} …")
log_v1 = common.load_log_pkl(common.LOG_V1_PKL)
n_v1, v_v1, a_v1 = common.log_summary(log_v1, "V1")

# Copy V1 row from the main metrics CSV into our dedicated CSV
main_csv = common.METRICS_CSV
if os.path.exists(main_csv):
    main_df = pd.read_csv(main_csv)
    v1_rows = main_df[main_df["version"].str.startswith("V1")]
    if not v1_rows.empty:
        upsert_metrics(v1_rows.to_dict("records"), "V1", METRICS_CSV_FR)
        print("  [copied] V1 metrics from iteration_metrics.csv")
    else:
        print("  [warn]  No V1 row found in iteration_metrics.csv — skipping V1 metrics copy")

# ═════════════════════════════════════════════════════════════════════════════
# V2c: Fraud filter  (applied directly to V1 — skipping V2a/V2b)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*68}")
print("  Step V2c — fraud filter  (skipping V2a / V2b)")
print(f"{'='*68}\n")

print(f"  Removing cases containing '{common.FRAUD_ACTIVITY}' …")
log_v2c = common.remove_fraud_cases(log_v1)
n_v2c, v_v2c, a_v2c = common.log_summary(log_v2c, "V2c")
common.print_removal_stats(n_v1, n_v2c, n_raw, label_before="V1")

print("\nDiscovering IMf on V2c log …")
net, im, fm = common.discover_imf(log_v2c)
m_v2c = common.compute_metrics(log_v1, net, im, fm, log_train=log_v2c)

print("\n  V2c — QUALITY METRICS  (vs V1 lifecycle-filtered log)")
common.print_metrics_table(m_v2c)

print("\nSaving V2c outputs …")
common.save_log_pkl(log_v2c, LOG_FR_V2C_PKL)
common.save_pnml("FR_V2c_IMf", net, im, fm)
common.save_petri_png("FR_V2c_IMf", net, im, fm)

upsert_metrics([make_record("V2c — fraud filter (direct)", n_v2c, n_raw, m_v2c)],
               "V2c", METRICS_CSV_FR)

# ═════════════════════════════════════════════════════════════════════════════
# V2d: Rare-variant filter
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*68}")
print(f"  Step V2d — rare variant filter  (threshold < {common.RARE_VARIANT_THRESHOLD})")
print(f"{'='*68}\n")

log_v2d = common.remove_rare_variants(log_v2c, threshold=common.RARE_VARIANT_THRESHOLD)
n_v2d, v_v2d, a_v2d = common.log_summary(log_v2d, "V2d")
common.print_removal_stats(n_v2c, n_v2d, n_raw, label_before="V2c")
print(f"  Variants: {v_v2c} -> {v_v2d}")
print(f"\n  Total coverage: {n_v2d} / {n_raw} cases retained "
      f"({n_v2d / n_raw:.1%} of raw log)")

print("\nDiscovering IMf on V2d log …")
net, im, fm = common.discover_imf(log_v2d)
m_v2d = common.compute_metrics(log_v1, net, im, fm, log_train=log_v2d)

print("\n  V2d — QUALITY METRICS  (vs V1 lifecycle-filtered log)")
common.print_metrics_table(m_v2d)

print("\nSaving V2d outputs …")
common.save_log_pkl(log_v2d, LOG_FR_V2D_PKL)
common.save_pnml("FR_V2d_IMf", net, im, fm)
common.save_petri_png("FR_V2d_IMf", net, im, fm)

upsert_metrics([make_record("V2d — rare variant filter (direct)", n_v2d, n_raw, m_v2d)],
               "V2d", METRICS_CSV_FR)

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*68}")
print("  PIPELINE SUMMARY  (fraud + rare-variant only)")
print(f"{'='*68}")
import pandas as pd
df = pd.read_csv(METRICS_CSV_FR)
cols = ["version", "cases_log", "pct_removed_vs_raw",
        "fitness_tbr", "precision_etc", "generalization",
        "simplicity_pm4py", "S1_structural", "S2_adv_structural",
        "places", "transitions", "arcs"]
print(df[cols].to_string(index=False))
print(f"\nAll outputs written to:  {common.OUTPUT_DIR}/")
print("Done.")
