"""
v2d_rare_variants.py — V2d: Remove rare variants
=================================================
Input:  output/log_v2c.pkl  (produced by v2c_fraud_filter.py)
        raw XES log         (loaded from common.LOG_PATH for metric evaluation)
Output: output/log_v2.pkl   (final V2 log — input for v3_noise_sweep.py)
        output/petri_V2d_IMf.pnml
        output/petri_V2d_IMf.png   (if Graphviz available)
        output/iteration_metrics.csv  (V2d row upserted)

Run from any directory:
    python iterative_improvement/v2d_rare_variants.py
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

print(f"\nLoading V2c log from {common.LOG_V2C_PKL} …")
log_v2c = common.load_log_pkl(common.LOG_V2C_PKL)
n_v2c, v_v2c, a_v2c = common.log_summary(log_v2c, "V2c")

# ═════════════════════════════════════════════════════════════════════════════
# V2d: REMOVE RARE VARIANTS
# ═════════════════════════════════════════════════════════════════════════════
print(f"\nApplying V2d — remove rare variants (threshold < {common.RARE_VARIANT_THRESHOLD}) …")

log_v2  = common.remove_rare_variants(log_v2c, threshold=common.RARE_VARIANT_THRESHOLD)
n_v2, v_v2, a_v2 = common.log_summary(log_v2, "V2d")

print("\n  Removal statistics:")
common.print_removal_stats(n_v2c, n_v2, n_raw, label_before="V2c")
print(f"  Variants   : {v_v2c} -> {v_v2}")
print(f"\n  Total log coverage: {n_v2} / {n_raw} cases retained "
      f"({n_v2 / n_raw:.1%} of raw log)")

# ═════════════════════════════════════════════════════════════════════════════
# DISCOVER & EVALUATE
# ═════════════════════════════════════════════════════════════════════════════
print("\nDiscovering IMf on V2d log …")
net, im, fm = common.discover_imf(log_v2)
m = common.compute_metrics(log_v1, net, im, fm)

print("\n" + "=" * 68)
print("  V2d — Rare variant filter  |  QUALITY METRICS  (vs V1 lifecycle-filtered log)")
print("=" * 68)
common.print_metrics_table(m)

# ═════════════════════════════════════════════════════════════════════════════
# SAVE
# ═════════════════════════════════════════════════════════════════════════════
print("\nSaving outputs …")
common.save_log_pkl(log_v2, common.LOG_V2_PKL)   # final V2 log for V3
common.save_pnml("V2d_IMf", net, im, fm)
common.save_petri_png("V2d_IMf", net, im, fm)

records = [{
    "version"            : "V2d — rare variant filter",
    "algo"               : "IMf",
    "cases_log"          : n_v2,
    "pct_removed_vs_raw" : round((n_raw - n_v2) / n_raw * 100, 2),
    **m,
}]
common.append_metrics(records, version_prefix="V2d")

print("\nDone.  Next step: python v3_noise_sweep.py")
