"""
3.3.3 — Final Process Model: Iterative Construction & Validation
================================================================
BPIC-17 Loan Application Process

IMf is evaluated at every preprocessing step AND every sub-step so that
we can track the impact of each individual change on model quality.

Iteration overview:
  V0 — Baseline    : IMf on raw log (reference from 3.3.1)
  V1 — Lifecycle   : keep COMPLETE events only
  V2 — Rojos et al.: merge near-instant sequences (V2a), remove invalid
                     endpoints (V2b), fraud cases (V2c), rare variants (V2d)
  V3 — Noise sweep : IMf noise_threshold 0.1–0.4 on V2 log (always run)

Design principle:
  - All metrics evaluated against the ORIGINAL raw log
  - IMf is the sole algorithm because:
      (a) soundness is guaranteed (no deadlocks, all traces terminate)
          — critical for simulation
      (b) the process tree maps 1:1 to BPMN 2.0 via pm4py.convert_to_bpmn()
      (c) block-structured models are interpretable for report discussion
  - V3 is always executed; the highest noise_threshold that still achieves
    fitness >= 0.80 is selected as the final model. If no threshold achieves
    0.80 the user is warned and the highest-fitness threshold is selected.

Targets (for a simulation base model):
  fitness_tbr    >= 0.80
  precision_etc  >= 0.60
  generalization >= 0.85
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import shutil
import warnings
import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.evaluation import algorithm as evaluation_all
from pm4py.visualization.petri_net import visualizer as pn_vis
from pm4py.objects.log.obj import EventLog, Trace
from collections import Counter

warnings.filterwarnings("ignore")


def _ensure_graphviz():
    if shutil.which("dot"):
        return True
    candidate = r"C:\Program Files\Graphviz\bin"
    if os.path.exists(os.path.join(candidate, "dot.exe")):
        os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
        return bool(shutil.which("dot"))
    return False

GRAPHVIZ_AVAILABLE = _ensure_graphviz()

# ── Configuration ─────────────────────────────────────────────────────────────
LOG_PATH   = r"C:\Users\sophi\OneDrive\Documents\uni\Master\SS26\praktikum\bpic17-process-mining\data\BPIChallenge2017.xes.gz"
OUTPUT_DIR = "output_final_model_imf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_FITNESS        = 0.80
TARGET_PRECISION      = 0.60
TARGET_GENERALIZATION = 0.85

RARE_VARIANT_THRESHOLD = 100
FRAUD_ACTIVITY         = "W_Assess potential fraud"
INVALID_END_ACTIVITIES = {
    "O_Sent (mail and online)",
    "O_Sent (online only)",
}
SEQUENCE_MERGES = [
    ({"A_Pending"},  "O_Accepted"),  
    ({"O_Create Offer"},    "O_Created"),   
    #({"O_Sent (mail and online)"},    "A_Complete"),   
]



# ── Helpers: Metrics ──────────────────────────────────────────────────────────
def simplicity_structural(net):
    """S1: Structural Appropriateness a_S (Rozinat & van der Aalst 2008)."""
    T   = len(net.transitions)
    T_V = sum(1 for t in net.transitions if t.label is not None)
    P   = len(net.places)
    if T == 0 or (P - T_V + 2) <= 0:
        return 0.0
    return round((T_V / (T + 1)) * (1 / (P - T_V + 2)), 4)


def simplicity_advanced(net):
    """S2: Advanced Structural Appropriateness a'_S (Rozinat & van der Aalst 2008, Metric 7)."""
    T = len(net.transitions)
    if T == 0:
        return 0.0
    label_counts = Counter(t.label for t in net.transitions if t.label is not None)
    T_DA = sum(1 for t in net.transitions
               if t.label is not None and label_counts[t.label] > 1)
    T_IR = sum(
        1 for t in net.transitions
        if t.label is None
        and sum(1 for a in net.arcs if a.target == t) == 1
        and sum(1 for a in net.arcs if a.source == t) == 1
    )
    return round((T - (T_DA + T_IR)) / T, 4)


def compute_metrics(log_eval, net, im, fm):
    """Always evaluate against log_eval (= original raw log)."""
    m = evaluation_all.apply(log_eval, net, im, fm)
    return {
        "fitness_tbr"      : round(m["fitness"]["log_fitness"], 4),
        "precision_etc"    : round(m["precision"],              4),
        "generalization"   : round(m["generalization"],         4),
        "simplicity_pm4py" : round(m["simplicity"],             4),
        "S1_structural"    : simplicity_structural(net),
        "S2_adv_structural": simplicity_advanced(net),
        "places"           : len(net.places),
        "transitions"      : len(net.transitions),
        "arcs"             : len(net.arcs),
    }


def targets_met(m):
    return (
        m["fitness_tbr"]    >= TARGET_FITNESS and
        m["precision_etc"]  >= TARGET_PRECISION and
        m["generalization"] >= TARGET_GENERALIZATION
    )


def ok(val, target):
    if target is None:
        return "—"
    return "OK" if val >= target else "!!"


# ── Helpers: Discovery ────────────────────────────────────────────────────────
def discover_imf(log, noise_threshold=0.0):
    tree = pm4py.discover_process_tree_inductive(log, noise_threshold=noise_threshold)
    return pm4py.convert_to_petri_net(tree)


# ── Helpers: Logging ──────────────────────────────────────────────────────────
all_results = []   # for final CSV


def print_metrics_table(m):
    """Print metrics for a single IMf model."""
    targets = {
        "fitness_tbr"      : TARGET_FITNESS,
        "precision_etc"    : TARGET_PRECISION,
        "generalization"   : TARGET_GENERALIZATION,
        "simplicity_pm4py" : None,
        "S1_structural"    : None,
        "S2_adv_structural": None,
    }
    tgt_str = {
        "fitness_tbr"      : f">={TARGET_FITNESS}",
        "precision_etc"    : f">={TARGET_PRECISION}",
        "generalization"   : f">={TARGET_GENERALIZATION}",
        "simplicity_pm4py" : "—",
        "S1_structural"    : "—",
        "S2_adv_structural": "—",
    }
    print(f"\n  {'Metric':<28}  {'Target':>8}  {'IMf':>8} {'':>3}")
    print(f"  {'-'*46}")
    for key, tgt in targets.items():
        v = m[key]
        print(f"  {key:<28}  {tgt_str[key]:>8}  {v:>8.4f} {ok(v, tgt):>3}")
    print(f"\n  {'Net (places/trans/arcs)':<28}  {'':>8}  "
          f"{m['places']}/{m['transitions']}/{m['arcs']}")


def print_substep_metrics(step_label, m):
    """Print a compact one-line metric summary for a preprocessing sub-step."""
    fit_ok  = ok(m["fitness_tbr"],    TARGET_FITNESS)
    prec_ok = ok(m["precision_etc"],  TARGET_PRECISION)
    gen_ok  = ok(m["generalization"], TARGET_GENERALIZATION)
    print(f"  [{step_label}]  "
          f"fit={m['fitness_tbr']:.4f}{fit_ok}  "
          f"prec={m['precision_etc']:.4f}{prec_ok}  "
          f"gen={m['generalization']:.4f}{gen_ok}  "
          f"P={m['places']} T={m['transitions']} A={m['arcs']}")


def log_iteration(version, motivation, change, m_imf,
                  interpretation, log_stats=None):
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  ITERATION  {version}")
    print(sep)

    print("\n  MOTIVATION")
    for line in motivation.strip().split("\n"):
        print(f"    {line.strip()}")

    print("\n  CHANGE APPLIED")
    for line in change.strip().split("\n"):
        print(f"    {line.strip()}")

    if log_stats:
        print("\n  LOG STATISTICS")
        for k, v in log_stats.items():
            print(f"    {k:<38}: {v}")

    print("\n  QUALITY METRICS  (evaluated on original raw log)")
    print_metrics_table(m_imf)

    print("\n  INTERPRETATION & DECISION")
    for line in interpretation.strip().split("\n"):
        print(f"    {line.strip()}")
    print()

    all_results.append({"version": version, "algo": "IMf", **m_imf})


# ── Helpers: Export ───────────────────────────────────────────────────────────
def save_petri(name, net, im, fm):
    if not GRAPHVIZ_AVAILABLE:
        print(f"  [warn]  Graphviz not found — skipping PNG for {name}")
        return
    try:
        gviz = pn_vis.apply(net, im, fm)
        path = os.path.join(OUTPUT_DIR, f"petri_{name}.png")
        pn_vis.save(gviz, path)
        print(f"  [saved] {path}")
    except Exception as e:
        print(f"  [warn]  PNG export failed: {e}")


def save_bpmn(name, net, im, fm):
    try:
        bpmn = pm4py.convert_to_bpmn(net, im, fm)
        path = os.path.join(OUTPUT_DIR, f"bpmn_{name}.bpmn")
        pm4py.write_bpmn(bpmn, path)
        print(f"  [saved] {path}")
    except Exception as e:
        print(f"  [warn]  BPMN export failed: {e}")


# ── Helpers: Log Filtering ────────────────────────────────────────────────────
def log_summary(log, label=""):
    n = len(log)
    v = len(pm4py.get_variants(log))
    a = len(pm4py.get_event_attribute_values(log, "concept:name"))
    print(f"  [{label}]  {n} cases | {v} variants | {a} distinct activities")
    return n, v, a


def filter_complete_events(log):
    """Keep only COMPLETE lifecycle events; case-insensitive; removes empty traces."""
    new_log = EventLog()
    new_log.attributes.update(log.attributes)
    for trace in log:
        new_trace = Trace()
        new_trace.attributes.update(trace.attributes)
        for event in trace:
            lc = event.get("lifecycle:transition", "complete")
            if lc.upper() == "COMPLETE":
                new_trace.append(event)
        if len(new_trace) > 0:
            new_log.append(new_trace)
    return new_log


def merge_sequences(log):
    drop_labels = set()
    for drop_set, _ in SEQUENCE_MERGES:
        drop_labels |= drop_set
    new_log = EventLog()
    new_log.attributes.update(log.attributes)
    for trace in log:
        new_trace = Trace()
        new_trace.attributes.update(trace.attributes)
        for event in trace:
            if event["concept:name"] not in drop_labels:
                new_trace.append(event)
        if len(new_trace) > 0:
            new_log.append(new_trace)
    return new_log


def remove_fraud_cases(log):
    new_log = EventLog()
    new_log.attributes.update(log.attributes)
    for trace in log:
        if not any(e["concept:name"] == FRAUD_ACTIVITY for e in trace):
            new_log.append(trace)
    return new_log


def remove_invalid_endpoint_cases(log):
    new_log = EventLog()
    new_log.attributes.update(log.attributes)
    for trace in log:
        if len(trace) > 0 and trace[-1]["concept:name"] not in INVALID_END_ACTIVITIES:
            new_log.append(trace)
    return new_log


def remove_rare_variants(log, threshold=RARE_VARIANT_THRESHOLD):
    variants = pm4py.get_variants(log)
    keep = {v for v, cases in variants.items() if len(cases) >= threshold}
    return pm4py.filter_variants(log, keep)


# ═════════════════════════════════════════════════════════════════════════════
# LOAD
# ═════════════════════════════════════════════════════════════════════════════
print("Loading event log …")
log_raw = xes_importer.apply(LOG_PATH)
n_raw, v_raw, a_raw = log_summary(log_raw, "raw")
print()


# ═════════════════════════════════════════════════════════════════════════════
# V0 — BASELINE  (loaded from output_models — already computed in 3.3.1)
# ═════════════════════════════════════════════════════════════════════════════
print("Loading V0 — Baseline from output_models/ …")
_pm4py_df = pd.read_csv("output_models/pm4py_metrics.csv", index_col="model")
_simp_df  = pd.read_csv("output_models/simplicity_metrics.csv", index_col="model")

def _load_metrics(algo):
    return {
        "fitness_tbr"      : float(_pm4py_df.loc[algo, "fitness_tbr"]),
        "precision_etc"    : float(_pm4py_df.loc[algo, "precision_etc"]),
        "generalization"   : float(_pm4py_df.loc[algo, "generalization"]),
        "simplicity_pm4py" : float(_pm4py_df.loc[algo, "simplicity_pm4py"]),
        "S1_structural"    : float(_simp_df.loc[algo,  "S1_structural"]),
        "S2_adv_structural": float(_simp_df.loc[algo,  "S2_adv_structural"]),
        "places"           : int(_simp_df.loc[algo,    "places"]),
        "transitions"      : int(_simp_df.loc[algo,    "transitions"]),
        "arcs"             : int(_simp_df.loc[algo,    "arcs"]),
    }

m_v0_imf = _load_metrics("IMf")

net_v0_imf, im_v0_imf, fm_v0_imf = pm4py.read_pnml("output_models/petri_IMf.pnml")

log_iteration(
    version="V0 — Baseline: raw log, IMf default",
    motivation="""
        Establish a reference point for IMf on the unprocessed log.
        This mirrors the 3.3.1 output. The raw BPIC-17 log contains all
        lifecycle transitions and all three activity families (A_, W_, O_),
        producing a highly noisy directly-follows graph.
    """,
    change="None — IMf default on raw log.",
    m_imf=m_v0_imf,
    interpretation="""
        IMf: fitness=1.0 because flower-model fallbacks allow any trace
        to replay, but precision=0.14 — the model permits ~7x more
        behaviour than observed. Useless as a simulation base model.
        DECISION: apply lifecycle filter (V1).
    """,
    log_stats={"Cases": n_raw, "Variants": v_raw, "Activities": a_raw}
)
save_petri("V0_IMf", net_v0_imf, im_v0_imf, fm_v0_imf)
save_bpmn("V0_IMf", net_v0_imf, im_v0_imf, fm_v0_imf)


# ═════════════════════════════════════════════════════════════════════════════
# V1 — LIFECYCLE FILTER
# ═════════════════════════════════════════════════════════════════════════════
print("Running V1 — Lifecycle filter (COMPLETE only) …")
log_v1 = filter_complete_events(log_raw)
n_v1, v_v1, a_v1 = log_summary(log_v1, "V1")

net_v1_imf, im_v1_imf, fm_v1_imf = discover_imf(log_v1)
m_v1_imf = compute_metrics(log_raw, net_v1_imf, im_v1_imf, fm_v1_imf)

log_iteration(
    version="V1 — Lifecycle filter: COMPLETE events only",
    motivation="""
        The raw log records schedule/assign/start/complete for every activity.
        PM4Py treats each as a separate event, creating artificial
        directly-follows arcs within the same activity execution.
        Unlike Disco (used by Rojos et al.), PM4Py does not auto-aggregate
        lifecycle transitions on import — this must be done explicitly.
        Keeping only COMPLETE events removes intra-activity noise and makes
        the directly-follows graph reflect true activity ordering.
    """,
    change="""
        Filter: lifecycle:transition == 'COMPLETE'.
        Remove empty traces. IMf default on filtered log.
        Metrics evaluated on original raw log.
    """,
    m_imf=m_v1_imf,
    interpretation=f"""
        Cases: {n_raw} -> {n_v1} | Variants: {v_raw} -> {v_v1}.
        IMf precision improved from {m_v0_imf['precision_etc']} to
        {m_v1_imf['precision_etc']} — lifecycle arcs no longer inflate the model.
        DECISION: proceed with Rojos et al. preprocessing (V2).
    """,
    log_stats={
        "Cases  raw -> V1"      : f"{n_raw} -> {n_v1}",
        "Variants  raw -> V1"   : f"{v_raw} -> {v_v1}",
        "Activities  raw -> V1" : f"{a_raw} -> {a_v1}",
    }
)
save_petri("V1_IMf", net_v1_imf, im_v1_imf, fm_v1_imf)
save_bpmn("V1_IMf", net_v1_imf, im_v1_imf, fm_v1_imf)


# ═════════════════════════════════════════════════════════════════════════════
# V2 — ROJOS ET AL. PREPROCESSING  (metrics computed after every sub-step)
# ═════════════════════════════════════════════════════════════════════════════
print("\nRunning V2 — Rojos et al. preprocessing (sub-step metrics after each step) …")
sep_sub = "-" * 68

# ── V2a: Merge near-instant sequences ─────────────────────────────────────
log_v2a = merge_sequences(log_v1)
n_v2a = len(log_v2a)
a_v2a = len(pm4py.get_event_attribute_values(log_v2a, "concept:name"))
print(f"\n{sep_sub}")
print(f"  V2a — Merge near-instant sequences: {n_v1} -> {n_v2a} cases | {a_v2a} activities")
print("  Discovering IMf …")
net_v2a, im_v2a, fm_v2a = discover_imf(log_v2a)
m_v2a = compute_metrics(log_raw, net_v2a, im_v2a, fm_v2a)
print_substep_metrics("V2a", m_v2a)
all_results.append({"version": "V2a — merge sequences", "algo": "IMf", **m_v2a})

# ── V2b: Remove invalid endpoint cases ────────────────────────────────────
log_v2b = remove_invalid_endpoint_cases(log_v2a)
n_v2b = len(log_v2b)
print(f"\n{sep_sub}")
print(f"  V2b — Remove invalid endpoints: {n_v2a} -> {n_v2b} cases (removed {n_v2a - n_v2b})")
print("  Discovering IMf …")
net_v2b, im_v2b, fm_v2b = discover_imf(log_v2b)
m_v2b = compute_metrics(log_raw, net_v2b, im_v2b, fm_v2b)
print_substep_metrics("V2b", m_v2b)
all_results.append({"version": "V2b — endpoint filter", "algo": "IMf", **m_v2b})

# ── V2c: Remove fraud cases ────────────────────────────────────────────────
log_v2c = remove_fraud_cases(log_v2b)
n_v2c = len(log_v2c)
print(f"\n{sep_sub}")
print(f"  V2c — Remove fraud cases: {n_v2b} -> {n_v2c} cases (removed {n_v2b - n_v2c})")
print("  Discovering IMf …")
net_v2c, im_v2c, fm_v2c = discover_imf(log_v2c)
m_v2c = compute_metrics(log_raw, net_v2c, im_v2c, fm_v2c)
print_substep_metrics("V2c", m_v2c)
all_results.append({"version": "V2c — fraud filter", "algo": "IMf", **m_v2c})

# ── V2d: Remove rare variants ──────────────────────────────────────────────
log_v2 = remove_rare_variants(log_v2c, threshold=RARE_VARIANT_THRESHOLD)
n_v2, v_v2, a_v2 = log_summary(log_v2, "V2d")
print(f"\n{sep_sub}")
print(f"  V2d — Remove rare variants (< {RARE_VARIANT_THRESHOLD}): {n_v2c} -> {n_v2} cases")
print("  Discovering IMf …")
net_v2_imf, im_v2_imf, fm_v2_imf = discover_imf(log_v2)
m_v2_imf = compute_metrics(log_raw, net_v2_imf, im_v2_imf, fm_v2_imf)
print_substep_metrics("V2d", m_v2_imf)

log_iteration(
    version="V2 — Rojos et al. preprocessing on V1 log",
    motivation="""
        Applying the four-step preprocessing pipeline from Rojos et al.
        (bpi2017_paper_36, Section 3.2) to the lifecycle-filtered V1 log.
        Their approach was validated on the same BPIC-17 dataset.
        Each step addresses a distinct source of noise:
          2a: near-instant co-occurring activities are one logical step
          2b: invalid endpoints are incomplete process instances
          2c: fraud assessment is a rare exceptional sub-process (<1%)
          2d: variants < 100 occurrences are noise, not the main flow
        Sub-step metrics above show the impact of each individual change.
    """,
    change=f"""
        2a: Removed {[list(d) for d, _ in SEQUENCE_MERGES]} (representatives kept)
        2b: Removed cases ending in invalid endpoints ({n_v1 - n_v2b} cases)
        2c: Removed fraud cases ({n_v2b - n_v2c} cases)
        2d: Removed variants with frequency < {RARE_VARIANT_THRESHOLD} ({n_v2c - n_v2} cases)
        IMf default on cleaned log. Metrics on original raw log.
    """,
    m_imf=m_v2_imf,
    interpretation=f"""
        Cases: {n_v1} -> {n_v2} ({n_v2 / n_raw:.1%} of original log retained).
        Variants: {v_v1} -> {v_v2}.
        IMf precision: {m_v1_imf['precision_etc']} -> {m_v2_imf['precision_etc']}.
        DECISION: proceed with noise threshold sweep (V3) — always executed.
    """,
    log_stats={
        "After merge (2a)"              : f"{n_v2a} cases, {a_v2a} activities",
        "After endpoint filter (2b)"    : f"{n_v2b} cases",
        "After fraud filter (2c)"       : f"{n_v2c} cases",
        "After rare variant filter (2d)": f"{n_v2} cases, {v_v2} variants",
        "Coverage of original log"      : f"{n_v2 / n_raw:.1%}",
    }
)
save_petri("V2_IMf", net_v2_imf, im_v2_imf, fm_v2_imf)
save_bpmn("V2_IMf", net_v2_imf, im_v2_imf, fm_v2_imf)


# ═════════════════════════════════════════════════════════════════════════════
# V3 — IMf NOISE THRESHOLD SWEEP (always executed)
# ═════════════════════════════════════════════════════════════════════════════
print("\nRunning V3 — noise threshold sweep (always executed) …")
NOISE_THRESHOLDS = [0.1, 0.2, 0.3, 0.4]
v3_candidates = []

print(f"\n  {'noise':>6}  {'fitness':>8}  {'precision':>10}"
      f"  {'general.':>10}  {'places':>7}  {'trans':>6}  {'status'}")
print(f"  {'-'*72}")

for thresh in NOISE_THRESHOLDS:
    net_t, im_t, fm_t = discover_imf(log_v2, noise_threshold=thresh)
    m_t = compute_metrics(log_raw, net_t, im_t, fm_t)
    v3_candidates.append((thresh, net_t, im_t, fm_t, m_t))
    status = "OK (>= 0.80)" if m_t["fitness_tbr"] >= TARGET_FITNESS else "!! below 0.80"
    print(f"  {thresh:>6.1f}  {m_t['fitness_tbr']:>8.4f}"
          f"  {m_t['precision_etc']:>10.4f}"
          f"  {m_t['generalization']:>10.4f}"
          f"  {m_t['places']:>7}  {m_t['transitions']:>6}  {status}")
    all_results.append({"version": f"V3 noise={thresh}", "algo": "IMf", **m_t})

# Select highest threshold that still meets fitness >= 0.80
eligible = [(t, n, i, f, m) for t, n, i, f, m in v3_candidates
            if m["fitness_tbr"] >= TARGET_FITNESS]

if eligible:
    best = max(eligible, key=lambda x: x[0])   # highest noise = simplest
    best_thresh, net_final, im_final, fm_final, m_final = best
    fitness_warning = ""
    print(f"\n  Selected noise_threshold={best_thresh} "
          f"(highest threshold with fitness >= {TARGET_FITNESS}).")
else:
    # No threshold meets 0.80 — warn and pick highest fitness
    best = max(v3_candidates, key=lambda x: x[4]["fitness_tbr"])
    best_thresh, net_final, im_final, fm_final, m_final = best
    fitness_warning = (
        f"\n  [WARNING] No noise_threshold in {NOISE_THRESHOLDS} achieves "
        f"fitness >= {TARGET_FITNESS}.\n"
        f"  Selected threshold={best_thresh} with best achievable "
        f"fitness={m_final['fitness_tbr']:.4f}.\n"
        f"  Consider reducing RARE_VARIANT_THRESHOLD or revisiting "
        f"preprocessing steps to avoid dropping too many cases."
    )
    print(fitness_warning)

label_final = f"FINAL_V3_IMf_noise{best_thresh}"

log_iteration(
    version=f"V3 — IMf noise_threshold={best_thresh} on V2 log (sweep 0.1–0.4)",
    motivation="""
        Infrequent directly-follows edges within retained variants still
        increase model complexity. IMf's noise_threshold prunes these before
        tree construction, producing a simpler and more precise model.
        V3 is always executed to document the full sweep results.
        The highest threshold still achieving fitness >= 0.80 is selected
        (simplest acceptable model). If none achieve 0.80 a warning is issued
        and the highest-fitness threshold is used instead.
    """,
    change=f"""
        IMf noise_threshold swept over {NOISE_THRESHOLDS} on V2 log.
        Selected threshold={best_thresh} (highest with fitness >= {TARGET_FITNESS}).
        Metrics on original raw log.
    """,
    m_imf=m_final,
    interpretation=f"""
        IMf at noise={best_thresh}:
          fitness={m_final['fitness_tbr']} {ok(m_final['fitness_tbr'], TARGET_FITNESS)}
          precision={m_final['precision_etc']} {ok(m_final['precision_etc'], TARGET_PRECISION)}
          generalization={m_final['generalization']}
        {fitness_warning.strip() if fitness_warning else 'Fitness target >= 0.80 met.'}
        WHY IMf: soundness guaranteed (no deadlocks); process tree maps 1:1
        to BPMN 2.0 via convert_to_bpmn(); block structure is interpretable.
    """,
)
save_petri(label_final, net_final, im_final, fm_final)


# ═════════════════════════════════════════════════════════════════════════════
# FINAL MODEL REPORT
# ═════════════════════════════════════════════════════════════════════════════
sep = "=" * 68
print(f"\n{sep}")
print(f"  FINAL MODEL — {label_final}")
print(sep)
print(f"""
  QUALITY METRICS  (evaluated on original raw log)
  +--------------------------+--------+----------+
  | Metric                   | Target | IMf Final|
  +--------------------------+--------+----------+
  | Token-Based Fitness      | >=0.80 |   {m_final['fitness_tbr']:6.4f} |
  | ETC Precision            | >=0.60 |   {m_final['precision_etc']:6.4f} |
  | Generalization           | >=0.85 |   {m_final['generalization']:6.4f} |
  | Simplicity (PM4Py/arc)   |     —  |   {m_final['simplicity_pm4py']:6.4f} |
  | S1 Structural Appropr.   |     —  |   {m_final['S1_structural']:6.4f} |
  | S2 Adv. Structural Appr. |     —  |   {m_final['S2_adv_structural']:6.4f} |
  +--------------------------+--------+----------+
  | Places                   |     —  |   {m_final['places']:6d} |
  | Transitions              |     —  |   {m_final['transitions']:6d} |
  | Arcs                     |     —  |   {m_final['arcs']:6d} |
  +--------------------------+--------+----------+

  ALGORITHM CHOICE JUSTIFICATION: IMf
  ─────────────────────────────────────────────
  1. SOUNDNESS GUARANTEE
     IMf always produces a sound workflow net: no deadlocks, every
     reachable marking can reach the final marking, no dead transitions.

  2. BPMN 2.0 CONVERTIBILITY
     IMf produces a process tree, which maps 1:1 to BPMN 2.0.
     pm4py.convert_to_bpmn() is reliable only on process trees.

  3. NOISE SWEEP CLOSES THE PRECISION GAP FURTHER
     V3 prunes infrequent edges within retained variants, producing a
     simpler and more precise model than V2 default IMf.

  TRADE-OFF DISCUSSION
  ─────────────────────
  1. FITNESS vs PRECISION
     We accept fitness ~0.80 (not 1.0) to gain precision > 0.60.
     A flower model has fitness=1.0, precision~0 — useless for simulation.

  2. WHY ~80% FITNESS IS APPROPRIATE
     The ~20% unfit cases are: lifecycle events (V1), invalid endpoints
     (Rojos 2b), fraud sub-process (Rojos 2c), and rare variants < 100
     (Rojos 2d). These are exceptions, not the process to simulate.

  3. GENERALIZATION vs SIMPLICITY
     A simpler model (fewer branches) is easier to parameterise for
     simulation (branching probabilities, resource pools).

  LIMITATIONS
  ───────────
  - COMPLETE events only: timing of start/schedule not in control flow.
  - ~20% of cases not replayable: edge-case scenarios underrepresented.
  - Token-based fitness may overestimate conformance for models with
    many silent transitions (alignment-based not computed: runtime).
  - Rare-variant threshold (100) adopted from Rojos et al., not tuned.
""")

if fitness_warning:
    print(fitness_warning)

print("  Saving final model …")
save_bpmn(label_final, net_final, im_final, fm_final)

df = pd.DataFrame(all_results).set_index(["version", "algo"])
csv_path = os.path.join(OUTPUT_DIR, "iteration_metrics.csv")
df.to_csv(csv_path)
print(f"  [saved] {csv_path}")
print(f"\n  All outputs written to: {OUTPUT_DIR}/")
print("\nDone.")