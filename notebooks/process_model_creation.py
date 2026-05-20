"""
Process Discovery & Quality Metrics — BPIC-17
Task 3.3: Process Model Creation and Validation

Algorithms:
  1. Inductive Miner infrequent (IMf)  — primary / final model candidate
  2. Heuristics Miner                  — secondary / appendix comparison

Quality metrics:
  - PM4Py built-in : token-based replay fitness, alignment-based fitness (opt.),
                     ETC precision, generalization
  - Custom (scratch): S1 Arc-Degree Simplicity, S2 CFC Simplicity
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import os
import warnings
import shutil
import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

# Discovery
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner

# Conversions
from pm4py.objects.conversion.heuristics_net import converter as hn_converter

# Conformance & evaluation
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation import algorithm as evaluation_all

# Visualisation
from pm4py.visualization.petri_net    import visualizer as pn_vis
from pm4py.visualization.process_tree import visualizer as pt_vis

warnings.filterwarnings("ignore")


# ── 1. Configuration ──────────────────────────────────────────────────────────
LOG_PATH = r"C:\Users\sophi\OneDrive\Documents\uni\Master\SS26\praktikum\bpic17-process-mining\data\BPIChallenge2017.xes.gz"   # ← adjust to your local path

# Set True to also compute (slow) alignment-based fitness
RUN_ALIGNMENTS = False

OUTPUT_DIR = "output_models"


def ensure_graphviz_in_path():
    """Try to make Graphviz available for PM4Py visualizers on Windows."""
    if shutil.which("dot"):
        return True

    candidate = r"C:\Program Files\Graphviz\bin"
    dot_exe = os.path.join(candidate, "dot.exe")
    if os.path.exists(dot_exe):
        os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
        return shutil.which("dot") is not None

    return False


GRAPHVIZ_AVAILABLE = ensure_graphviz_in_path()


# ── 2. Load Event Log ─────────────────────────────────────────────────────────
print("Loading event log …")
log = xes_importer.apply(LOG_PATH)
print(f"  Loaded {len(log)} cases.")

# Filter to COMPLETE lifecycle events only
from pm4py.objects.log.obj import EventLog, Trace
log_filtered = EventLog()
log_filtered.attributes.update(log.attributes)
for trace in log:
    new_trace = Trace()
    new_trace.attributes.update(trace.attributes)
    for event in trace:
        if event.get("lifecycle:transition", "complete").upper() == "COMPLETE":
            new_trace.append(event)
    if len(new_trace) > 0:
        log_filtered.append(new_trace)
log = log_filtered
print(f"  After COMPLETE filter: {len(log)} cases.\n")


# ── 3. Discovery ──────────────────────────────────────────────────────────────

def discover_models(log):
    models = {}

    # 3a. Inductive Miner infrequent (IMf) — high-level API (version-stable)
    print("[1/2] Inductive Miner infrequent  (library defaults) ...")
    process_tree = pm4py.discover_process_tree_inductive(log)
    net_imf, im_imf, fm_imf = pm4py.convert_to_petri_net(process_tree)
    models["IMf"] = (net_imf, im_imf, fm_imf, process_tree)
    print("  Done. Petri net derived from process tree — soundness guaranteed.\n")

    # 3b. Heuristics Miner
    print("[2/2] Heuristics Miner  (library defaults) ...")
    hnet = heuristics_miner.apply_heu(log)
    net_hm, im_hm, fm_hm = hn_converter.apply(hnet)
    models["HM"] = (net_hm, im_hm, fm_hm, hnet)
    print("  Done. Petri net derived from heuristics net — soundness not guaranteed.\n")

    return models


models = discover_models(log)


# ── 4. PM4Py Quality Metrics ──────────────────────────────────────────────────

def pm4py_metrics(log, net, im, fm, name):
    print(f"{'='*58}")
    print(f"  PM4Py Metrics — {name}")
    print(f"{'='*58}")

    # One-call PM4Py evaluation (token-based): fitness, precision, generalization, simplicity
    print("  [metrics]   Unified token-based evaluation ...")
    metrics = evaluation_all.apply(log, net, im, fm)
    fit_tbr = metrics["fitness"]["log_fitness"]
    prec    = metrics["precision"]
    gen     = metrics["generalization"]
    simp_pm = metrics["simplicity"]
    print(f"              Token-Based Fitness  : {fit_tbr:.4f}")
    print(f"              ETC Precision       : {prec:.4f}")
    print(f"              Generalization      : {gen:.4f}")
    print(f"              Simplicity (PM4Py)  : {simp_pm:.4f}")

    # Alignment-Based Fitness (optional — very slow on large logs)
    fit_align = None
    if RUN_ALIGNMENTS:
        print("  [fitness]   Alignment-based (slow) ...")
        aligned   = alignments.apply_log(log, net, im, fm)
        fit_align = sum(a["fitness"] for a in aligned) / len(aligned)
        print(f"              Alignment Fitness   : {fit_align:.4f}")
    else:
        print("              Alignment Fitness   : SKIPPED  (set RUN_ALIGNMENTS=True)")

    print()
    return {
        "model"         : name,
        "fitness_tbr"   : round(fit_tbr, 4),
        "fitness_align" : round(fit_align, 4) if fit_align is not None else "N/A",
        "precision_etc" : round(prec, 4)      if prec      is not None else "N/A",
        "generalization": round(gen, 4)        if gen       is not None else "N/A",
        "simplicity_pm4py": round(simp_pm, 4) if simp_pm   is not None else "N/A",
    }



def simplicity_s1(net):
    """S1: Simple Structural Appropriateness (Metric 6, Carmona et al.).
    a_s = (|T| + 2) / (|P| + |T| + |F|)
    """
    n_t = len(net.transitions)
    n_p = len(net.places)
    n_f = len(net.arcs)
    denominator = n_p + n_t + n_f
    if denominator == 0:
        return 0.0
    return round((n_t + 2) / denominator, 4)


def simplicity_s2(net):
    """S2: Advanced Structural Appropriateness (Metric 7, Carmona et al.).
    a'_s = (|T| - (|T_DA| + |T_IR|)) / |T|

    T_DA: transitions sharing a label with at least one other transition
    T_IR: silent transitions whose in/out arc sets are identical
          to those of another transition (truly redundant)
    """
    from collections import Counter

    n_t = len(net.transitions)
    if n_t == 0:
        return 0.0

    # T_DA: duplicate labeled transitions
    label_counts = Counter(
        t.label for t in net.transitions if t.label is not None
    )
    t_da = {t for t in net.transitions
            if t.label is not None and label_counts[t.label] > 1}

    # T_IR: silent transitions with identical in/out arc sets as another transition
    def arc_signature(t):
        ins  = frozenset(a.source.name for a in net.arcs if a.target == t)
        outs = frozenset(a.target.name for a in net.arcs if a.source == t)
        return (ins, outs)

    silent = [t for t in net.transitions if t.label is None]
    sig_counts = Counter(arc_signature(t) for t in silent)
    t_ir = {t for t in silent if sig_counts[arc_signature(t)] > 1}

    return round((n_t - (len(t_da) + len(t_ir))) / n_t, 4)


def custom_simplicity(net, name):
    n_p = len(net.places)
    n_t = len(net.transitions)
    n_a = len(net.arcs)
    s1  = simplicity_s1(net)
    s2  = simplicity_s2(net)

    print(f"  Custom Simplicity — {name}")
    print(f"    Places / Transitions / Arcs : {n_p} / {n_t} / {n_a}")
    print(f"    S1 Simple Structural        : {s1}   (1 = simplest)")
    print(f"    S2 Advanced Structural      : {s2}   (1 = simplest)\n")

    return {
        "model"        : name,
        "places"       : n_p,
        "transitions"  : n_t,
        "arcs"         : n_a,
        "S1_arc_degree": s1,
        "S2_cfc"       : s2,
    }


# ── 6. Run All Evaluations ────────────────────────────────────────────────────

pm4py_results  = []
custom_results = []

for name, (net, im, fm, _) in models.items():
    pm4py_results.append(pm4py_metrics(log, net, im, fm, name))
    custom_results.append(custom_simplicity(net, name))


# ── 7. Summary Tables ─────────────────────────────────────────────────────────

print("\n" + "="*58)
print("  SUMMARY — PM4Py Quality Metrics")
print("="*58)
df_pm = pd.DataFrame(pm4py_results).set_index("model")
print(df_pm.to_string())

print("\n" + "="*58)
print("  SUMMARY — Custom Simplicity Metrics")
print("="*58)
df_cust = pd.DataFrame(custom_results).set_index("model")
print(df_cust.to_string())


# ── 8. Save Visualisations & CSVs ─────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_petri_outputs(name, net, im, fm):
    """Save Petri net using PNG when possible, otherwise fallback to portable formats."""
    saved_any = False

    # Preferred: rendered PNG for the report.
    if GRAPHVIZ_AVAILABLE:
        try:
            gviz = pn_vis.apply(net, im, fm)
            out_png = os.path.join(OUTPUT_DIR, f"petri_{name}.png")
            pn_vis.save(gviz, out_png)
            print(f"  Saved: {out_png}")
            saved_any = True
        except Exception as e:
            print(f"  PNG export failed for {name}: {e}")

    # Fallback 1: save Graphviz source (can be rendered later with dot).
    try:
        gviz = pn_vis.apply(net, im, fm)
        out_dot = os.path.join(OUTPUT_DIR, f"petri_{name}.dot")
        gviz.save(filename=out_dot)
        print(f"  Saved fallback DOT: {out_dot}")
        saved_any = True
    except Exception as e:
        print(f"  Could not save DOT for {name}: {e}")

    # Fallback 2: save structural Petri net format (tool-independent).
    try:
        out_pnml = os.path.join(OUTPUT_DIR, f"petri_{name}.pnml")
        pm4py.write_pnml(net, im, fm, out_pnml)
        print(f"  Saved fallback PNML: {out_pnml}")
        saved_any = True
    except Exception as e:
        print(f"  Could not save PNML for {name}: {e}")

    if not saved_any:
        print(f"  No export succeeded for {name}.")

# Petri net PNGs for both models
for name, (net, im, fm, _) in models.items():
    save_petri_outputs(name, net, im, fm)

# IMf process tree (useful for appendix — shows recursive structure)
try:
    tree      = models["IMf"][3]
    gviz_tree = pt_vis.apply(tree)
    out_tree  = os.path.join(OUTPUT_DIR, "process_tree_IMf.png")
    pt_vis.save(gviz_tree, out_tree)
    print(f"  Saved: {out_tree}")
except Exception as e:
    print(f"  Could not save IMf process tree: {e}")

# CSV exports for report tables
df_pm.to_csv(os.path.join(OUTPUT_DIR, "pm4py_metrics.csv"))
df_cust.to_csv(os.path.join(OUTPUT_DIR, "simplicity_metrics.csv"))
print(f"\n  CSVs written to {OUTPUT_DIR}/")
print("\nDone.")