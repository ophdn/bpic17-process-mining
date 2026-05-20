"""
v3_noise_sweep.py — V3: IMf noise threshold sweep (0.1–0.4)
============================================================
Input:  output/log_v2.pkl   (produced by v2_rojos.py)
        raw XES log         (loaded from common.LOG_PATH for metric evaluation)
Output: output/petri_FINAL_V3_IMf_noise{t}.pnml
        output/petri_FINAL_V3_IMf_noise{t}.png   (if Graphviz available)
        output/bpmn_FINAL_V3_IMf_noise{t}.bpmn
        output/iteration_metrics.csv  (V3 rows upserted)

Selection rule: highest noise_threshold that still achieves fitness >= 0.80
(simplest acceptable model). If none qualify, the highest-fitness threshold
is used and a warning is printed.

Run from any directory:
    python iterative_improvement/v3_noise_sweep.py
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
log_v2a = common.load_log_pkl(common.LOG_V1_PKL)

LOG_FR_V2D_PKL = os.path.join(common.OUTPUT_DIR, "log_fr_v2d.pkl")
print(f"\nLoading V2d log (V1→V2c→V2d, no endpoint filter) from {LOG_FR_V2D_PKL} …")
log_v2 = common.load_log_pkl(LOG_FR_V2D_PKL)
n_v2, v_v2, a_v2 = common.log_summary(log_v2, "V2")

pct_removed = (n_raw - n_v2) / n_raw * 100
print(f"\n  Log entering V3: {n_v2} / {n_raw} cases  "
      f"({n_v2 / n_raw:.1%} retained, {pct_removed:.1f}% removed vs raw log)")

# ═════════════════════════════════════════════════════════════════════════════
# V3: NOISE THRESHOLD SWEEP
# ═════════════════════════════════════════════════════════════════════════════
NOISE_THRESHOLDS = [0.2, 0.3, 0.4,]
v3_candidates = []
records = []

print(f"\nRunning noise threshold sweep over {NOISE_THRESHOLDS} …")
print(f"\n  {'noise':>6}  {'fitness':>8}  {'precision':>10}"
      f"  {'general.':>10}  {'fit_traces':>10}  {'places':>7}  {'trans':>6}  {'status'}")
print(f"  {'-'*84}")

for thresh in NOISE_THRESHOLDS:
    net_t, im_t, fm_t = common.discover_imf(log_v2, noise_threshold=thresh)
    m_t = common.compute_metrics(log_v2a, net_t, im_t, fm_t, log_train=log_v2)
    v3_candidates.append((thresh, net_t, im_t, fm_t, m_t))

    status = "OK (> 79%)" if m_t["perc_fit_traces"] > 79 else "!! below 80%"
    print(f"  {thresh:>6.1f}  {m_t['fitness_tbr']:>8.4f}"
          f"  {m_t['precision_etc']:>10.4f}"
          f"  {m_t['generalization']:>10.4f}"
          f"  {m_t['perc_fit_traces']:>10.4f}"
          f"  {m_t['places']:>7}  {m_t['transitions']:>6}  {status}")

    records.append({
        "version"            : f"V3 noise={thresh}",
        "algo"               : "IMf",
        "cases_log"          : n_v2,
        "pct_removed_vs_raw" : round(pct_removed, 2),
        **m_t,
    })

# ═════════════════════════════════════════════════════════════════════════════
# SELECT BEST THRESHOLD
# ═════════════════════════════════════════════════════════════════════════════
PERC_FIT_TARGET = 79

eligible = [
    (t, n, i, f, m) for t, n, i, f, m in v3_candidates
    if m["perc_fit_traces"] > PERC_FIT_TARGET
]

fitness_warning = ""
if eligible:
    best = max(eligible, key=lambda x: x[0])   # highest noise = simplest model
    best_thresh, net_final, im_final, fm_final, m_final = best
    print(f"\n  Selected noise_threshold={best_thresh} "
          f"(highest with perc_fit_traces > {PERC_FIT_TARGET:.0%}).")
else:
    best = max(v3_candidates, key=lambda x: x[4]["perc_fit_traces"])
    best_thresh, net_final, im_final, fm_final, m_final = best
    fitness_warning = (
        f"\n  [WARNING] No noise_threshold in {NOISE_THRESHOLDS} achieves "
        f"perc_fit_traces > {PERC_FIT_TARGET:.0%}.\n"
        f"  Selected threshold={best_thresh} (best available "
        f"perc_fit_traces={m_final['perc_fit_traces']:.4f}).\n"
        f"  Consider reducing RARE_VARIANT_THRESHOLD or revisiting preprocessing."
    )
    print(fitness_warning)

label_final = f"FINAL_V3_IMf_noise{best_thresh}"

# ═════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print(f"  FINAL MODEL — {label_final}")
print("  QUALITY METRICS  (vs V1 lifecycle-filtered log)")
print("=" * 68)
common.print_metrics_table(m_final)

if fitness_warning:
    print(fitness_warning)

# ═════════════════════════════════════════════════════════════════════════════
# SAVE
# ═════════════════════════════════════════════════════════════════════════════
print("\nSaving outputs …")
common.save_pnml(label_final, net_final, im_final, fm_final)
common.save_petri_png(label_final, net_final, im_final, fm_final)
common.save_bpmn(label_final, net_final, im_final, fm_final)

records.append({
    "version"            : f"V3 FINAL noise={best_thresh}",
    "algo"               : "IMf",
    "cases_log"          : n_v2,
    "pct_removed_vs_raw" : round(pct_removed, 2),
    **m_final,
})

import pandas as pd
METRICS_CSV_FR = os.path.join(common.OUTPUT_DIR, "iteration_metrics_fraud_rare.csv")
new_df = pd.DataFrame(records)
if os.path.exists(METRICS_CSV_FR):
    existing = pd.read_csv(METRICS_CSV_FR)
    existing = existing[~existing["version"].str.startswith("V3")]
    df = pd.concat([existing, new_df], ignore_index=True)
else:
    df = new_df
df.to_csv(METRICS_CSV_FR, index=False)
print(f"  [saved] {METRICS_CSV_FR}")

print("\nDone.")
