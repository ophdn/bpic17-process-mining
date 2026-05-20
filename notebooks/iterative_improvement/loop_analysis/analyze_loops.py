"""
Loop frequency analysis — updated IMf model (imdf_net_1778963730)
Run: python analyze_loops_v2.py
"""

import pm4py
from collections import Counter
import os
import pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

# ── LOAD LOG ────────────────────────────────────────────────────────────────
LOG_FR_V2D_PKL = os.path.join(common.OUTPUT_DIR, "log_fr_v2d.pkl")
print(f"\nLoading V2d log (final model input) from {LOG_FR_V2D_PKL} …")
log = common.load_log_pkl(LOG_FR_V2D_PKL)
print(f"  {len(log)} cases loaded.\n")

# ── LOOP DEFINITIONS ────────────────────────────────────────────────────────
LOOPS = {
    "Loop 1 — O_Create Offer / O_Created  [skip_6, skip_7]":
        ["O_Create Offer", "O_Created"],

    "Loop 2 — W_Validate application  [init_loop_23, skip_25]":
        ["W_Validate application"],

    "Loop 3 — A_Validating / A_Incomplete  [skip_30]":
        ["A_Validating", "A_Incomplete"],

    "Loop 4 — O_Refused repeat  [skip_34]":
        ["O_Refused"],

    "Loop 5 — O_Returned  [tauSplit_20 branch]":
        ["O_Returned"],
}

# ── HELPERS ──────────────────────────────────────────────────────────────────
def count_activity(case, activity_names):
    return max(
        sum(1 for event in case if event["concept:name"] == act)
        for act in activity_names
    )

# ── ANALYSIS ────────────────────────────────────────────────────────────────
total_cases = len(log)
results = {}

for loop_name, activities in LOOPS.items():
    counter = Counter()
    for case in log:
        n = count_activity(case, activities)
        counter[n] += 1
    results[loop_name] = counter

# ── PER-LOOP REPORT ──────────────────────────────────────────────────────────
print("=" * 72)
print("LOOP FREQUENCY REPORT — updated IMf model")
print("=" * 72)

for loop_name, counter in results.items():
    print(f"\n{loop_name}")
    print("-" * 60)

    rows = []
    cumulative = 0.0
    for count in sorted(counter.keys()):
        n_cases = counter[count]
        pct = 100 * n_cases / total_cases
        cumulative += pct
        rows.append({
            "Occurrences/case": count,
            "# Cases": n_cases,
            "%": round(pct, 2),
            "Cumulative %": round(cumulative, 2),
        })

    print(pd.DataFrame(rows).to_string(index=False))

    pct_le1 = sum(counter[k] for k in counter if k <= 1) / total_cases * 100

    if pct_le1 >= 95:
        verdict = f"✅  SAFE TO REMOVE — {pct_le1:.1f}% of cases have ≤1 occurrence"
    elif pct_le1 >= 85:
        verdict = f"⚠️  CONSIDER BOUNDING — {pct_le1:.1f}% of cases have ≤1 occurrence"
    else:
        verdict = f"❌  KEEP LOOP — only {pct_le1:.1f}% of cases have ≤1 occurrence"

    print(f"  → {verdict}")

# ── FITNESS IMPACT SUMMARY (all loops) ──────────────────────────────────────
print("\n" + "=" * 72)
print("ESTIMATED MAX FITNESS LOSS IF LOOP IS FULLY REMOVED")
print("=" * 72)

summary_rows = []
for loop_name, counter in results.items():
    cases_repeat = sum(counter[k] for k in counter if k > 1)
    pct_loss = round(100 * cases_repeat / total_cases, 2)
    short_name = loop_name.split("—")[1].split("[")[0].strip()
    summary_rows.append({
        "Loop": short_name,
        "Cases repeating": cases_repeat,
        "Max fitness loss %": pct_loss,
        "Safe to remove?": "✅ Yes" if pct_loss < 5
                           else ("⚠️ Maybe" if pct_loss < 15 else "❌ No"),
    })

print(pd.DataFrame(summary_rows).to_string(index=False))
print("""
Decision guide:
  Max fitness loss < 5%  → remove safely (fitness stays well above 80%)
  Max fitness loss 5–15% → remove only if current fitness > 0.90
  Max fitness loss > 15% → keep the loop
""")