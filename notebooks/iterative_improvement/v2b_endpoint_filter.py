"""
v2b_endpoint_filter.py — V2b: Remove invalid endpoint cases
===========================================================
Input:  output/log_v2a.pkl  (produced by v2a_merge_sequences.py)
        raw XES log         (loaded from common.LOG_PATH for metric evaluation)
Output: output/log_v2b.pkl
        output/petri_V2b_IMf.pnml
        output/petri_V2b_IMf.png   (if Graphviz available)
        output/iteration_metrics.csv  (V2b row upserted)

Run from any directory:
    python iterative_improvement/v2b_endpoint_filter.py
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

print(f"\nLoading V2a log from {common.LOG_V2A_PKL} …")
log_v2a = common.load_log_pkl(common.LOG_V2A_PKL)
n_v2a, v_v2a, a_v2a = common.log_summary(log_v2a, "V2a")

# ═════════════════════════════════════════════════════════════════════════════
# V2b: REMOVE INVALID ENDPOINT CASES
# ═════════════════════════════════════════════════════════════════════════════
print(f"\nApplying V2b — remove invalid endpoint cases …")
print(f"  Invalid end activities: {common.INVALID_END_ACTIVITIES}")

log_v2b = common.remove_invalid_endpoint_cases(log_v2a)
n_v2b   = len(log_v2b)

print("\n  Removal statistics:")
common.print_removal_stats(n_v2a, n_v2b, n_raw, label_before="V2a")

# ═════════════════════════════════════════════════════════════════════════════
# DISCOVER & EVALUATE
# ═════════════════════════════════════════════════════════════════════════════
print("\nDiscovering IMf on V2b log …")
net, im, fm = common.discover_imf(log_v2b)
m = common.compute_metrics(log_v1, net, im, fm)

print("\n" + "=" * 68)
print("  V2b — Endpoint filter  |  QUALITY METRICS  (vs V1 lifecycle-filtered log)")
print("=" * 68)
common.print_metrics_table(m)

# ═════════════════════════════════════════════════════════════════════════════
# SAVE
# ═════════════════════════════════════════════════════════════════════════════
print("\nSaving outputs …")
common.save_log_pkl(log_v2b, common.LOG_V2B_PKL)
common.save_pnml("V2b_IMf", net, im, fm)
common.save_petri_png("V2b_IMf", net, im, fm)

records = [{
    "version"            : "V2b — endpoint filter",
    "algo"               : "IMf",
    "cases_log"          : n_v2b,
    "pct_removed_vs_raw" : round((n_raw - n_v2b) / n_raw * 100, 2),
    **m,
}]
common.append_metrics(records, version_prefix="V2b")

print("\nDone.  Next step: python v2c_fraud_filter.py")
