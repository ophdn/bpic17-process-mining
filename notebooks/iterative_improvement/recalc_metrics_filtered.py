"""
recalc_metrics_filtered.py — Recompute all metrics against the FILTERED log
============================================================================
Each model is evaluated against the log it was trained on (not the raw log).

Outputs (overwrite existing):
    output/iteration_metrics.csv           — standard pipeline (V1→V2a→V2b→V2c→V2d)
    output/iteration_metrics_fraud_rare.csv — fraud+rare shortcut pipeline (V1→V2c→V2d)

Run from any directory:
    python iterative_improvement/recalc_metrics_filtered.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import pm4py
import common

OUTPUT_DIR = common.OUTPUT_DIR


def load_pnml(name):
    path = os.path.join(OUTPUT_DIR, f"petri_{name}.pnml")
    print(f"  Loading {path} …")
    return pm4py.read_pnml(path)


def load_pkl(path):
    print(f"  Loading {path} …")
    return common.load_log_pkl(path)


def make_record(version, algo, log_filtered, log_raw, log_eval, net, im, fm):
    n_cases = len(log_filtered)
    n_raw   = len(log_raw)
    m = common.compute_metrics(log_eval, net, im, fm)
    return {
        "version"            : version,
        "algo"               : algo,
        "cases_log"          : n_cases,
        "pct_removed_vs_raw" : round((n_raw - n_cases) / n_raw * 100, 2),
        **m,
    }


def save_csv(records, path):
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"  [saved] {path}")


# ── Load raw log (for pct_removed_vs_raw) and V1 log (evaluation baseline) ───
print("Loading raw log for case-count reference …")
log_raw, n_raw, _, _ = common.load_raw_log()
print(f"Loading V1 log as evaluation baseline …")
log_v1_eval = common.load_log_pkl(common.LOG_V1_PKL)
print(f"  V1 log: {len(log_v1_eval)} cases")

# ═══════════════════════════════════════════════════════════════════════════
# Standard pipeline: V1 → V2a → V2b → V2c → V2d
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*68)
print("  Standard pipeline  (V1 → V2a → V2b → V2c → V2d)")
print("="*68)

STANDARD = [
    ("V1 — lifecycle filter",      "V1_IMf",  common.LOG_V1_PKL),
    ("V2a — merge sequences",      "V2a_IMf", common.LOG_V2A_PKL),
    ("V2b — endpoint filter",      "V2b_IMf", common.LOG_V2B_PKL),
    ("V2c — fraud filter",         "V2c_IMf", common.LOG_V2C_PKL),
    ("V2d — rare variant filter",  "V2d_IMf", common.LOG_V2_PKL),   # log_v2.pkl = V2d output
]

records_std = []
for version, pnml_name, pkl_path in STANDARD:
    print(f"\n--- {version} ---")
    log_filtered = load_pkl(pkl_path)
    net, im, fm  = load_pnml(pnml_name)
    rec = make_record(version, "IMf", log_filtered, log_raw, log_v1_eval, net, im, fm)
    common.print_metrics_table(rec)
    records_std.append(rec)

save_csv(records_std, os.path.join(OUTPUT_DIR, "iteration_metrics_filtered.csv"))

# ═══════════════════════════════════════════════════════════════════════════
# Fraud+rare shortcut: V1 → V2c (direct) → V2d (direct)
# ═══════════════════════════════════════════════════════════════════════════
LOG_FR_V2C_PKL = os.path.join(OUTPUT_DIR, "log_fr_v2c.pkl")
LOG_FR_V2D_PKL = os.path.join(OUTPUT_DIR, "log_fr_v2d.pkl")
METRICS_CSV_FR = os.path.join(OUTPUT_DIR, "iteration_metrics_fraud_rare_filtered.csv")

print("\n" + "="*68)
print("  Fraud+rare pipeline  (V1 → V2c direct → V2d direct)")
print("="*68)

FRAUD_RARE = [
    ("V1 — lifecycle filter",                   "V1_IMf",      common.LOG_V1_PKL),
    ("V2c — fraud filter (direct)",             "FR_V2c_IMf",  LOG_FR_V2C_PKL),
    ("V2d — rare variant filter (direct)",      "FR_V2d_IMf",  LOG_FR_V2D_PKL),
]

records_fr = []
for version, pnml_name, pkl_path in FRAUD_RARE:
    print(f"\n--- {version} ---")
    log_filtered = load_pkl(pkl_path)
    net, im, fm  = load_pnml(pnml_name)
    rec = make_record(version, "IMf", log_filtered, log_raw, log_v1_eval, net, im, fm)
    common.print_metrics_table(rec)
    records_fr.append(rec)

save_csv(records_fr, METRICS_CSV_FR)

print("\nDone. All metrics now evaluated against the respective filtered log.")
