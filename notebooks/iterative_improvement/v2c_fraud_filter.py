"""
v2c_fraud_filter.py — V2c: Remove fraud cases
==============================================
Input:  output/log_v2b.pkl  (produced by v2b_endpoint_filter.py)
        raw XES log         (loaded from common.LOG_PATH for metric evaluation)
Output: output/log_v2c.pkl
        output/petri_V2c_IMf.pnml
        output/petri_V2c_IMf.png   (if Graphviz available)
        output/iteration_metrics.csv  (V2c row upserted)

Run from any directory:
    python iterative_improvement/v2c_fraud_filter.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

# ═════════════════════════════════════════════════════════════════════════════
# LOAD
# ═════════════════════════════════════════════════════════════════════════════
log_raw, n_raw, v_raw, a_raw = common.load_raw_log()

print(f"\nLoading V1 log (evaluation baseline) from {common.LOG_V1_PKL} …")
log_v1 = common.load_log_pkl(common.LOG_V1_PKL)

print(f"\nLoading V2b log from {common.LOG_V2B_PKL} …")
log_v2b = common.load_log_pkl(common.LOG_V2B_PKL)
n_v2b, v_v2b, a_v2b = common.log_summary(log_v2b, "V2b")

# ═════════════════════════════════════════════════════════════════════════════
# V2c: REMOVE FRAUD CASES
# ═════════════════════════════════════════════════════════════════════════════
print(f"\nApplying V2c — remove fraud cases …")
print(f"  Fraud activity: '{common.FRAUD_ACTIVITY}'")

log_v2c = common.remove_fraud_cases(log_v2b)
n_v2c   = len(log_v2c)

print("\n  Removal statistics:")
common.print_removal_stats(n_v2b, n_v2c, n_raw, label_before="V2b")

# ═════════════════════════════════════════════════════════════════════════════
# DISCOVER & EVALUATE
# ═════════════════════════════════════════════════════════════════════════════
print("\nDiscovering IMf on V2c log …")
net, im, fm = common.discover_imf(log_v2c)
m = common.compute_metrics(log_v1, net, im, fm)

print("\n" + "=" * 68)
print("  V2c — Fraud filter  |  QUALITY METRICS  (vs V1 lifecycle-filtered log)")
print("=" * 68)
common.print_metrics_table(m)

# ═════════════════════════════════════════════════════════════════════════════
# SAVE
# ═════════════════════════════════════════════════════════════════════════════
print("\nSaving outputs …")
common.save_log_pkl(log_v2c, common.LOG_V2C_PKL)
common.save_pnml("V2c_IMf", net, im, fm)
common.save_petri_png("V2c_IMf", net, im, fm)

records = [{
    "version"            : "V2c — fraud filter",
    "algo"               : "IMf",
    "cases_log"          : n_v2c,
    "pct_removed_vs_raw" : round((n_raw - n_v2c) / n_raw * 100, 2),
    **m,
}]
common.append_metrics(records, version_prefix="V2c")

print("\nDone.  Next step: python v2d_rare_variants.py")
