"""
v1_lifecycle.py — V1: Lifecycle filter (COMPLETE events only)
=============================================================
Input:  raw XES log  (path defined in common.LOG_PATH)
Output: output/log_v1.pkl
        output/petri_V1_IMf.pnml
        output/petri_V1_IMf.png   (if Graphviz available)
        output/iteration_metrics.csv  (V1 rows upserted)

Run from any directory:
    python iterative_improvement/v1_lifecycle.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

# ═════════════════════════════════════════════════════════════════════════════
# LOAD RAW LOG
# ═════════════════════════════════════════════════════════════════════════════
log_raw, n_raw, v_raw, a_raw = common.load_raw_log()

# ═════════════════════════════════════════════════════════════════════════════
# V1: LIFECYCLE FILTER
# ═════════════════════════════════════════════════════════════════════════════
print("\nApplying V1 — lifecycle filter (COMPLETE events only) …")
log_v1 = common.filter_complete_events(log_raw)
n_v1, v_v1, a_v1 = common.log_summary(log_v1, "V1")

print("\n  Removal statistics (this step):")
common.print_removal_stats(n_raw, n_v1, n_raw, label_before="raw log")
print(f"  Variants   : {v_raw} -> {v_v1}")
print(f"  Activities : {a_raw} -> {a_v1}")

# ═════════════════════════════════════════════════════════════════════════════
# DISCOVER & EVALUATE
# ═════════════════════════════════════════════════════════════════════════════
print("\nDiscovering IMf on V1 log …")
net, im, fm = common.discover_imf(log_v1)
m = common.compute_metrics(log_v1, net, im, fm)

print("\n" + "=" * 68)
print("  V1 — Lifecycle filter  |  QUALITY METRICS  (vs V1 lifecycle-filtered log)")
print("=" * 68)
common.print_metrics_table(m)

# ═════════════════════════════════════════════════════════════════════════════
# SAVE
# ═════════════════════════════════════════════════════════════════════════════
print("\nSaving outputs …")
common.save_log_pkl(log_v1, common.LOG_V1_PKL)
common.save_pnml("V1_IMf", net, im, fm)
common.save_petri_png("V1_IMf", net, im, fm)

records = [{
    "version"            : "V1 — lifecycle filter",
    "algo"               : "IMf",
    "cases_log"          : n_v1,
    "pct_removed_vs_raw" : round((n_raw - n_v1) / n_raw * 100, 2),
    **m,
}]
common.append_metrics(records, version_prefix="V1")

print("\nDone.  Next step: python v2_rojos.py")
