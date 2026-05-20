"""
common.py — Shared configuration, helpers, and log-loading utilities
for the iterative IMf preprocessing pipeline (V1 → V2 → V3).

All metrics are always evaluated against the original raw log.
"""

import os
import pickle
import shutil
import warnings
import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.evaluation import algorithm as evaluation_all
from pm4py.visualization.petri_net import visualizer as pn_vis
from pm4py.objects.log.obj import EventLog, Trace, Event
from collections import Counter

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS_CSV = os.path.join(OUTPUT_DIR, "iteration_metrics.csv")

LOG_PATH    = r"C:\Users\sophi\OneDrive\Documents\uni\Master\SS26\praktikum\bpic17-process-mining\data\BPIChallenge2017.xes.gz"
LOG_V1_PKL  = os.path.join(OUTPUT_DIR, "log_v1.pkl")
LOG_V2A_PKL = os.path.join(OUTPUT_DIR, "log_v2a.pkl")
LOG_V2B_PKL = os.path.join(OUTPUT_DIR, "log_v2b.pkl")
LOG_V2C_PKL = os.path.join(OUTPUT_DIR, "log_v2c.pkl")
LOG_V2_PKL  = os.path.join(OUTPUT_DIR, "log_v2.pkl")

# ── Targets ───────────────────────────────────────────────────────────────────
TARGET_FITNESS        = 0.80
TARGET_PRECISION      = 0.60
TARGET_GENERALIZATION = 0.85

# ── Preprocessing config ──────────────────────────────────────────────────────
RARE_VARIANT_THRESHOLD = 20
FRAUD_ACTIVITY         = "W_Assess potential fraud"
INVALID_END_ACTIVITIES = {
    "O_Sent (mail and online)",
    "O_Sent (online only)",
}
SEQUENCE_MERGES = [
    ({"O_Accepted"}, "A_Pending"),
    ({"O_Create Offer"}, "O_Created"),
]

# ── Graphviz ──────────────────────────────────────────────────────────────────
def _ensure_graphviz():
    if shutil.which("dot"):
        return True
    candidate = r"C:\Program Files\Graphviz\bin"
    if os.path.exists(os.path.join(candidate, "dot.exe")):
        os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
        return bool(shutil.which("dot"))
    return False

GRAPHVIZ_AVAILABLE = _ensure_graphviz()


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def simplicity_structural(net):
    """S1: Structural Appropriateness 
    a_s = (|T| + 2) / (|P| + |T| + |F|)
    """
    n_t = len(net.transitions)
    n_p = len(net.places)
    n_f = len(net.arcs)
    denominator = n_p + n_t + n_f
    if denominator == 0:
        return 0.0
    return round((n_t + 2) / denominator, 4)


def simplicity_advanced(net):
    """S2: Advanced Structural Appropriateness a'_S (Rozinat & van der Aalst 2008, Metric 7)."""
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


def compute_metrics(log_eval, net, im, fm, log_train=None):
    """
    Evaluate model quality.
    - fitness_tbr / precision / generalization / simplicity: against log_eval (= log_v1, for comparability)
    - perc_fit_traces: against log_train (= step-specific log) if provided, else log_eval
    """
    m = evaluation_all.apply(log_eval, net, im, fm)
    if log_train is not None:
        m_train = evaluation_all.apply(log_train, net, im, fm)
        perc_fit = round(m_train["fitness"]["perc_fit_traces"], 4)
    else:
        perc_fit = round(m["fitness"]["perc_fit_traces"], 4)
    return {
        "fitness_tbr"      : round(m["fitness"]["log_fitness"],      4),
        "perc_fit_traces"  : perc_fit,
        "precision_etc"    : round(m["precision"],                   4),
        "generalization"   : round(m["generalization"],              4),
        "simplicity_pm4py" : round(m["simplicity"],                  4),
        "S1_structural"    : simplicity_structural(net),
        "S2_adv_structural": simplicity_advanced(net),
        "places"           : len(net.places),
        "transitions"      : len(net.transitions),
        "arcs"             : len(net.arcs),
    }


def ok(val, target):
    if target is None:
        return "—"
    return "OK" if val >= target else "!!"


# ─────────────────────────────────────────────────────────────────────────────
# DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
def discover_imf(log, noise_threshold=0.0):
    tree = pm4py.discover_process_tree_inductive(log, noise_threshold=noise_threshold)
    return pm4py.convert_to_petri_net(tree)


# ─────────────────────────────────────────────────────────────────────────────
# PRINTING
# ─────────────────────────────────────────────────────────────────────────────
def print_metrics_table(m):
    targets = {
        "fitness_tbr"      : TARGET_FITNESS,
        "perc_fit_traces"  : 0.80,
        "precision_etc"    : TARGET_PRECISION,
        "generalization"   : TARGET_GENERALIZATION,
        "simplicity_pm4py" : None,
        "S1_structural"    : None,
        "S2_adv_structural": None,
    }
    tgt_str = {
        "fitness_tbr"      : f">={TARGET_FITNESS}",
        "perc_fit_traces"  : ">=0.80",
        "precision_etc"    : f">={TARGET_PRECISION}",
        "generalization"   : f">={TARGET_GENERALIZATION}",
        "simplicity_pm4py" : "—",
        "S1_structural"    : "—",
        "S2_adv_structural": "—",
    }
    print(f"\n  {'Metric':<28}  {'Target':>8}  {'Value':>8} {'':>3}")
    print(f"  {'-'*46}")
    for key, tgt in targets.items():
        v = m[key]
        print(f"  {key:<28}  {tgt_str[key]:>8}  {v:>8.4f} {ok(v, tgt):>3}")
    print(f"\n  {'Net (places/trans/arcs)':<28}  {'':>8}  "
          f"{m['places']}/{m['transitions']}/{m['arcs']}")


def print_substep_metrics(step_label, m):
    """Compact one-liner for a sub-step."""
    fit_ok  = ok(m["fitness_tbr"],    TARGET_FITNESS)
    prec_ok = ok(m["precision_etc"],  TARGET_PRECISION)
    gen_ok  = ok(m["generalization"], TARGET_GENERALIZATION)
    print(f"  [{step_label}]  "
          f"fit={m['fitness_tbr']:.4f}{fit_ok}  "
          f"prec={m['precision_etc']:.4f}{prec_ok}  "
          f"gen={m['generalization']:.4f}{gen_ok}  "
          f"P={m['places']} T={m['transitions']} A={m['arcs']}")


def print_removal_stats(n_before, n_after, n_raw, label_before="previous step"):
    """Print how many cases were removed, both vs the previous step and vs raw."""
    removed  = n_before - n_after
    pct_step = removed / n_before * 100 if n_before > 0 else 0.0
    pct_raw  = (n_raw - n_after) / n_raw * 100 if n_raw > 0 else 0.0
    print(f"  Cases: {n_before} -> {n_after}  "
          f"(removed {removed} = {pct_step:.1f}% of {label_before}; "
          f"{pct_raw:.1f}% of raw log removed in total)")


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def save_pnml(name, net, im, fm):
    path = os.path.join(OUTPUT_DIR, f"petri_{name}.pnml")
    pm4py.write_pnml(net, im, fm, path)
    print(f"  [saved] {path}")


def save_petri_png(name, net, im, fm):
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


# ─────────────────────────────────────────────────────────────────────────────
# SHARED METRICS CSV
# ─────────────────────────────────────────────────────────────────────────────
def append_metrics(records, version_prefix):
    """
    Upsert records into the shared iteration_metrics.csv.
    Existing rows whose 'version' starts with version_prefix are replaced,
    so re-running a script always produces a clean result.
    """
    new_df = pd.DataFrame(records)
    if os.path.exists(METRICS_CSV):
        existing = pd.read_csv(METRICS_CSV)
        existing = existing[~existing["version"].str.startswith(version_prefix)]
        df = pd.concat([existing, new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(METRICS_CSV, index=False)
    print(f"  [saved] {METRICS_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
# LOG SERIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def save_log_pkl(log, path):
    with open(path, "wb") as f:
        pickle.dump(log, f)
    print(f"  [saved] {path}")


def load_log_pkl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required input log not found: {path}\n"
            f"Run the preceding step first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# LOG LOADING & STATS
# ─────────────────────────────────────────────────────────────────────────────
def load_raw_log():
    """Load the raw XES log and print a summary. Returns (log, n, v, a)."""
    print("Loading raw event log …")
    log = xes_importer.apply(LOG_PATH)
    n = len(log)
    v = len(pm4py.get_variants(log))
    a = len(pm4py.get_event_attribute_values(log, "concept:name"))
    print(f"  [raw]  {n} cases | {v} variants | {a} distinct activities")
    return log, n, v, a


def log_summary(log, label=""):
    n = len(log)
    v = len(pm4py.get_variants(log))
    a = len(pm4py.get_event_attribute_values(log, "concept:name"))
    print(f"  [{label}]  {n} cases | {v} variants | {a} distinct activities")
    return n, v, a


# ─────────────────────────────────────────────────────────────────────────────
# LOG FILTERING
# ─────────────────────────────────────────────────────────────────────────────
def filter_complete_events(log):
    """Keep only COMPLETE lifecycle events; removes empty traces."""
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
    """Merge adjacent activity pairs defined in SEQUENCE_MERGES (undirected).

    For each (drop_set, keep_label): wherever a drop_set activity and
    keep_label appear as direct neighbours (in either order), replace both
    with a single keep_label event whose timestamp is the later of the two
    (so the merged event's duration implicitly covers both activities).
    Repeated until no more merges are possible in the trace.
    """
    new_log = EventLog()
    new_log.attributes.update(log.attributes)

    for trace in log:
        events = list(trace)

        # Repeat until a full pass produces no merge (handles cascades)
        changed = True
        while changed:
            changed = False
            new_events = []
            i = 0
            while i < len(events):
                if i + 1 < len(events):
                    a_name = events[i]["concept:name"]
                    b_name = events[i + 1]["concept:name"]
                    for drop_set, keep_label in SEQUENCE_MERGES:
                        if (a_name in drop_set and b_name == keep_label) or \
                           (b_name in drop_set and a_name == keep_label):
                            # Determine which of the two events carries keep_label
                            keep_src = events[i] if a_name == keep_label else events[i + 1]
                            later_ts = max(
                                events[i]["time:timestamp"],
                                events[i + 1]["time:timestamp"],
                            )
                            merged_event = Event(keep_src)
                            merged_event["time:timestamp"] = later_ts
                            new_events.append(merged_event)
                            i += 2
                            changed = True
                            break
                    else:
                        new_events.append(events[i])
                        i += 1
                else:
                    new_events.append(events[i])
                    i += 1
            events = new_events

        new_trace = Trace()
        new_trace.attributes.update(trace.attributes)
        for e in events:
            new_trace.append(e)
        if len(new_trace) > 0:
            new_log.append(new_trace)

    return new_log


def remove_fraud_cases(log):
    """Remove all cases that contain the fraud assessment activity."""
    new_log = EventLog()
    new_log.attributes.update(log.attributes)
    for trace in log:
        if not any(e["concept:name"] == FRAUD_ACTIVITY for e in trace):
            new_log.append(trace)
    return new_log


def remove_invalid_endpoint_cases(log):
    """Remove cases whose last event is an invalid endpoint activity."""
    new_log = EventLog()
    new_log.attributes.update(log.attributes)
    for trace in log:
        if len(trace) > 0 and trace[-1]["concept:name"] not in INVALID_END_ACTIVITIES:
            new_log.append(trace)
    return new_log


def remove_rare_variants(log, threshold=RARE_VARIANT_THRESHOLD):
    """Remove variants with fewer than threshold occurrences."""
    variants = pm4py.get_variants(log)
    keep = {v for v, cases in variants.items() if len(cases) >= threshold}
    return pm4py.filter_variants(log, keep)
