"""
Process Discovery & Quality Metrics — BPIC-17
Task 3.3: Process Model Creation and Validation

Algorithms:
  1. Inductive Miner infrequent (IMf)  — library defaults
  2. Heuristics Miner                  — library defaults

Quality metrics:
  - PM4Py built-in : token-based replay fitness, ETC precision, generalization
  - Custom (from [3]): S1 Structural Appropriateness (a_S),
                       S2 Advanced Structural Appropriateness (a'_S)
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import os
import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.heuristics  import algorithm as heuristics_miner
from pm4py.objects.conversion.heuristics_net import converter as hn_converter
from pm4py.algo.evaluation            import algorithm as evaluation_all
from pm4py.visualization.petri_net    import visualizer as pn_vis
from pm4py.visualization.process_tree import visualizer as pt_vis

import warnings
warnings.filterwarnings("ignore")


# ── 1. Configuration ──────────────────────────────────────────────────────────
LOG_PATH   = r"C:\Users\sophi\OneDrive\Documents\uni\Master\SS26\praktikum\bpic17-process-mining\data\BPIChallenge2017.xes.gz"
OUTPUT_DIR = "output_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 2. Load Event Log ─────────────────────────────────────────────────────────
print("Loading event log …")
log = xes_importer.apply(LOG_PATH)
print(f"  Loaded {len(log)} cases.\n")


# ── 3. Discovery (library defaults) ──────────────────────────────────────────
print("[1/2] Inductive Miner infrequent (IMf) — default noise_threshold=0.0 …")
process_tree      = pm4py.discover_process_tree_inductive(log)   # noise_threshold=0.0
net_imf, im_imf, fm_imf = pm4py.convert_to_petri_net(process_tree)
print("  Done.\n")

print("[2/2] Heuristics Miner — library defaults …")
hnet              = heuristics_miner.apply_heu(log)               # all defaults
net_hm, im_hm, fm_hm = hn_converter.apply(hnet)
print("  Done.\n")

models = {
    "IMf": (net_imf, im_imf, fm_imf),
    "HM" : (net_hm,  im_hm,  fm_hm),
}


# ── 4. Custom Simplicity Metrics (from [3]) ───────────────────────────────────
"""
S1 — Structural Appropriateness  (a_S)
──────────────────────────────────────
Measures how compact the model is relative to the number of distinct
observable activities. A model that uses many more transitions than
activities is unnecessarily complex.

    a_S = ( |T_V| / (|T| + 1) )  ×  ( 1 / (|P| - |T_V| + 2) )

where
  |T|   = total number of transitions (visible + silent)
  |T_V| = number of visible (non-silent) transitions
  |P|   = number of places

Boundary cases:
  - Pure sequence with no silent transitions and minimal places → a_S near 1
  - Many silent/duplicate transitions and many places → a_S near 0

Reference: Rozinat & van der Aalst (2008) [3]


S2 — Advanced Structural Appropriateness  (a'_S)
─────────────────────────────────────────────────
Penalises two specific structural anti-patterns:

  T_DA : alternative duplicate tasks
         Visible transitions that share a label with at least one other
         visible transition (i.e. the same activity appears more than once).
  T_IR : redundant invisible tasks
         Silent transitions (label=None / "tau") that have exactly one
         input place and one output place AND whose removal would not
         disconnect the net — heuristic: in-degree = out-degree = 1.

    a'_S = ( |T| - (|T_DA| + |T_IR|) ) / |T|

Boundary cases:
  - No duplicates, no redundant taus → a'_S = 1.0  (perfectly clean)
  - Half the transitions are redundant → a'_S = 0.5

Reference: Rozinat & van der Aalst (2008) [3], Metric 7
"""

def simplicity_structural(net):
    """S1: Structural Appropriateness a_S ∈ (0, 1]. Higher = simpler."""
    T   = len(net.transitions)
    T_V = sum(1 for t in net.transitions if t.label is not None)
    P   = len(net.places)

    if T == 0 or P == 0:
        return 0.0

    a_s = (T_V / (T + 1)) * (1 / (P - T_V + 2))
    return round(a_s, 4)


def simplicity_advanced_structural(net):
    """S2: Advanced Structural Appropriateness a'_S ∈ [0, 1]. Higher = simpler."""
    T = len(net.transitions)
    if T == 0:
        return 0.0

    # T_DA: visible transitions whose label appears more than once
    from collections import Counter
    label_counts = Counter(
        t.label for t in net.transitions if t.label is not None
    )
    T_DA = sum(1 for t in net.transitions
               if t.label is not None and label_counts[t.label] > 1)

    # T_IR: silent transitions with exactly 1 input arc and 1 output arc
    T_IR = sum(
        1 for t in net.transitions
        if t.label is None
        and sum(1 for a in net.arcs if a.target == t) == 1
        and sum(1 for a in net.arcs if a.source == t) == 1
    )

    a_prime_s = (T - (T_DA + T_IR)) / T
    return round(a_prime_s, 4)


def custom_simplicity(net, name):
    n_p  = len(net.places)
    n_t  = len(net.transitions)
    n_a  = len(net.arcs)
    s1   = simplicity_structural(net)
    s2   = simplicity_advanced_structural(net)

    print(f"  Custom Simplicity — {name}")
    print(f"    Places / Transitions / Arcs    : {n_p} / {n_t} / {n_a}")
    print(f"    S1 Structural Appropriateness  : {s1}   (1 = simplest)")
    print(f"    S2 Adv. Structural Appropr.    : {s2}   (1 = simplest)\n")

    return {"model": name, "places": n_p, "transitions": n_t, "arcs": n_a,
            "S1_structural": s1, "S2_adv_structural": s2}


# ── 5. PM4Py Quality Metrics ──────────────────────────────────────────────────
def pm4py_metrics(log, net, im, fm, name):
    print(f"{'='*55}")
    print(f"  PM4Py Metrics — {name}")
    print(f"{'='*55}")

    metrics = evaluation_all.apply(log, net, im, fm)
    fit     = metrics["fitness"]["log_fitness"]
    prec    = metrics["precision"]
    gen     = metrics["generalization"]
    simp    = metrics["simplicity"]

    print(f"    Token-Based Fitness : {fit:.4f}")
    print(f"    ETC Precision       : {prec:.4f}")
    print(f"    Generalization      : {gen:.4f}")
    print(f"    Simplicity (PM4Py)  : {simp:.4f}\n")

    return {"model": name, "fitness_tbr": round(fit, 4),
            "precision_etc": round(prec, 4),
            "generalization": round(gen, 4),
            "simplicity_pm4py": round(simp, 4)}


# ── 6. Run All Evaluations ────────────────────────────────────────────────────
pm4py_results, custom_results = [], []

for name, (net, im, fm) in models.items():
    pm4py_results.append(pm4py_metrics(log, net, im, fm, name))
    custom_results.append(custom_simplicity(net, name))


# ── 7. Summary Tables ─────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  SUMMARY — PM4Py Quality Metrics")
print("="*55)
df_pm = pd.DataFrame(pm4py_results).set_index("model")
print(df_pm.to_string())

print("\n" + "="*55)
print("  SUMMARY — Custom Simplicity Metrics")
print("="*55)
df_cust = pd.DataFrame(custom_results).set_index("model")
print(df_cust.to_string())


# ── 8. Save Outputs ───────────────────────────────────────────────────────────

# Petri nets → PNML (tool-independent, always works)
for name, (net, im, fm) in models.items():
    pnml_path = os.path.join(OUTPUT_DIR, f"petri_{name}.pnml")
    pm4py.write_pnml(net, im, fm, pnml_path)
    print(f"  Saved PNML : {pnml_path}")

# IMf process tree → PNML of underlying net (tree PNG needs Graphviz)
try:
    tree_png = os.path.join(OUTPUT_DIR, "process_tree_IMf.png")
    gviz     = pt_vis.apply(process_tree)
    pt_vis.save(gviz, tree_png)
    print(f"  Saved tree : {tree_png}")
except Exception:
    print("  (Graphviz not found — skipping process tree PNG)")

# CSVs for report tables
df_pm.to_csv(os.path.join(OUTPUT_DIR,   "pm4py_metrics.csv"))
df_cust.to_csv(os.path.join(OUTPUT_DIR, "simplicity_metrics.csv"))
print(f"\n  CSVs written to {OUTPUT_DIR}/")
print("\nDone.")