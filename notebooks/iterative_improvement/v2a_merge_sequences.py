"""
v2a_merge_sequences.py — V2a: Remove near-instant duplicate activities
=======================================================================
Input:  output/log_v1.pkl   (produced by v1_lifecycle.py)
        raw XES log         (loaded from common.LOG_PATH for metric evaluation)
Output: output/log_v2a.pkl
        output/petri_V2a_IMf.pnml
        output/petri_V2a_IMf.png   (if Graphviz available)
        output/iteration_metrics.csv  (V2a row upserted)

Run from any directory:
    python iterative_improvement/v2a_merge_sequences.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import pm4py

# ═════════════════════════════════════════════════════════════════════════════
# LOAD
# ═════════════════════════════════════════════════════════════════════════════
log_raw, n_raw, v_raw, a_raw = common.load_raw_log()

print(f"\nLoading V1 log from {common.LOG_V1_PKL} …")
log_v1 = common.load_log_pkl(common.LOG_V1_PKL)
n_v1, v_v1, a_v1 = common.log_summary(log_v1, "V1")

# ═════════════════════════════════════════════════════════════════════════════
# V2a: MERGE NEAR-INSTANT SEQUENCES
# ═════════════════════════════════════════════════════════════════════════════
print(f"\nApplying V2a — merge near-instant sequences …")
print(f"  Merged activities: {[list(d) for d, _ in common.SEQUENCE_MERGES]}")

log_v2a = common.merge_sequences(log_v1)
n_v2a   = len(log_v2a)
a_v2a   = len(pm4py.get_event_attribute_values(log_v2a, "concept:name"))

print("\n  Removal statistics:")
common.print_removal_stats(n_v1, n_v2a, n_raw, label_before="V1")
print(f"  Activities : {a_v1} -> {a_v2a}")

# ═════════════════════════════════════════════════════════════════════════════
# DISCOVER & EVALUATE
# ═════════════════════════════════════════════════════════════════════════════
print("\nDiscovering IMf on V2a log …")
net, im, fm = common.discover_imf(log_v2a)
m = common.compute_metrics(log_v1, net, im, fm)

print("\n" + "=" * 68)
print("  V2a — Merge sequences  |  QUALITY METRICS  (vs V1 lifecycle-filtered log)")
print("=" * 68)
common.print_metrics_table(m)

# ═════════════════════════════════════════════════════════════════════════════
# SAVE
# ═════════════════════════════════════════════════════════════════════════════
print("\nSaving outputs …")
common.save_log_pkl(log_v2a, common.LOG_V2A_PKL)
common.save_pnml("V2a_IMf", net, im, fm)
common.save_petri_png("V2a_IMf", net, im, fm)

records = [{
    "version"            : "V2a — merge sequences",
    "algo"               : "IMf",
    "cases_log"          : n_v2a,
    "pct_removed_vs_raw" : round((n_raw - n_v2a) / n_raw * 100, 2),
    **m,
}]
common.append_metrics(records, version_prefix="V2a")

print("\nDone.  Next step: python v2b_endpoint_filter.py")
