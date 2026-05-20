import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

import pm4py
import pandas as pd
from pm4py.objects.petri_net.utils import petri_utils

# ── Load model ────────────────────────────────────────────────────────────
PNML_PATH = r"C:\Users\sophi\OneDrive\Documents\uni\Master\SS26\praktikum\bpic17-process-mining\notebooks\iterative_improvement\output\petri_FINAL_V3_IMf_noise0.2.pnml"
net, im, fm = pm4py.read_pnml(PNML_PATH)

print("Transitions in model:")
for t in sorted(net.transitions, key=lambda x: x.name):
    label = t.label if t.label else "(silent)"
    print(f"  {t.name:20s}  →  {label}")

# ── Decision based on loop analysis results ───────────────────────────────
#
#  Loop 1 (O_Create Offer):   10.99% loss → ⚠️ skip for now, current fitness needed
#  Loop 2 (W_Validate):        5.01% loss → ⚠️ skip for now, current fitness needed
#  Loop 3 (A_Validating):     23.39% loss → ❌ KEEP
#  Loop 4 (O_Refused/skip_34): 0.54% loss → ✅ REMOVE
#  Loop 5 (O_Returned):        0.00% loss → ✅ no repeat exists, nothing to remove

def remove_if_exists(net, transition_name):
    matches = [t for t in net.transitions if t.name == transition_name]
    if matches:
        petri_utils.remove_transition(net, matches[0])
        print(f"  ✅ Removed transition: {transition_name}")
    else:
        print(f"  ⚠️  Transition not found (already removed?): {transition_name}")

print("\nRemoving loop transitions …")

# Loop 4: skip_34 creates the O_Refused repeat back-edge (0.54% fitness loss)
remove_if_exists(net, "skip_34")



# ── Load logs ─────────────────────────────────────────────────────────────
print(f"\nLoading V1 log (evaluation baseline) …")
log_v1 = common.load_log_pkl(common.LOG_V1_PKL)

LOG_FR_V2D_PKL = os.path.join(common.OUTPUT_DIR, "log_fr_v2d.pkl")
print(f"Loading V2d log (case count) …")
log_v2d = common.load_log_pkl(LOG_FR_V2D_PKL)
n_cases = len(log_v2d)
n_raw   = 31509
print(f"  Cases in mining log: {n_cases}")

# ── Compute metrics ───────────────────────────────────────────────────────
print("\nComputing metrics …")
m = common.compute_metrics(log_v1, net, im, fm, log_v2d)
common.print_metrics_table(m)

# ── Save model ────────────────────────────────────────────────────────────
print("\nSaving outputs …")
common.save_pnml("FINAL_V4_loop_removal", net, im, fm)
common.save_petri_png("FINAL_V4_loop_removal", net, im, fm)

# ── Write metrics to CSV ──────────────────────────────────────────────────
METRICS_CSV_FR = os.path.join(common.OUTPUT_DIR, "iteration_metrics_fraud_rare.csv")
record = [{
    "version"            : "V4 — loop removal (skip_34)",
    "algo"               : "IMf",
    "cases_log"          : n_cases,
    "pct_removed_vs_raw" : round((n_raw - n_cases) / n_raw * 100, 2),
    **m,
}]
new_df = pd.DataFrame(record)
if os.path.exists(METRICS_CSV_FR):
    existing = pd.read_csv(METRICS_CSV_FR)
    existing = existing[~existing["version"].str.startswith("V4")]
    df = pd.concat([existing, new_df], ignore_index=True)
else:
    df = new_df
df.to_csv(METRICS_CSV_FR, index=False)
print(f"  [saved] {METRICS_CSV_FR}")