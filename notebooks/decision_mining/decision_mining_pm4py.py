"""
Decision Mining mit pm4py — BPIC-17
=====================================
Nutzt pm4py's eingebautes Decision Mining (Reimplementierung des ProM Plugins
aus de Leoni & van der Aalst, 2013).

Fixes in dieser Version
-----------------------
* Leakage-Fix: Nur Case-Attribute werden als Features verwendet.
  Event-Attribute (Action, EventID, org:resource etc.) werden automatisch
  erkannt und ausgeschlossen.
* detect_case_attributes() liest Case-Attribute direkt aus trace.attributes
  und ergaenzt bekannte BPIC-17 Case-Attribute als Fallback.

Usage:
  python decision_mining_pm4py.py --log ../data/BPIChallenge2017.xes.gz \\
                                   --pnml model.pnml

  # oder mit BPMN:
  python decision_mining_pm4py.py --log ../data/BPIChallenge2017.xes.gz \\
                                   --bpmn model.bpmn

Dependencies:
  pip install pm4py scikit-learn matplotlib
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import pm4py

warnings.filterwarnings("ignore")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="pm4py Decision Mining — BPIC-17")
    p.add_argument("--log",  required=True)
    p.add_argument("--bpmn", default=None, help="BPMN-Datei (wird zu Petri-Netz konvertiert)")
    p.add_argument("--pnml", default=None, help="PNML-Datei (direkt einlesen, bevorzugt)")
    p.add_argument("--output-dir",        default="decision_mining_output")
    p.add_argument("--max-depth",         type=int, default=3)
    p.add_argument("--min-samples-leaf",  type=int, default=30)
    p.add_argument("--alignment-timeout", type=int, default=30,
                   help="Timeout pro Trace in Sekunden (default: 30)")
    p.add_argument("--case-col",     default="case:concept:name")
    p.add_argument("--activity-col", default="concept:name")
    p.add_argument("--time-col",     default="time:timestamp")
    return p.parse_args()


# ===========================================================================
# 1. Load log
# ===========================================================================

def load_log(log_path, args):
    import pm4py

    print(f"[1/4] Lade Event Log: {log_path}")
    name = Path(log_path).name.lower()

    if name.endswith(".xes.gz") or name.endswith(".xes"):
        raw = pm4py.read_xes(log_path)
    elif name.endswith(".csv"):
        df = pd.read_csv(log_path)
        df[args.time_col] = pd.to_datetime(df[args.time_col], utc=True)
        raw = pm4py.format_dataframe(
            df,
            case_id=args.case_col,
            activity_key=args.activity_col,
            timestamp_key=args.time_col,
        )
    else:
        raise ValueError(f"Unbekanntes Format: {log_path}")

    if isinstance(raw, pd.DataFrame):
        raw = pm4py.convert_to_event_log(raw)

    n_cases  = len(raw)
    n_events = sum(len(t) for t in raw)
    print(f"    {n_cases} Cases, {n_events} Events geladen.")
    return raw


# ===========================================================================
# 2. Load model -> Petri net
# ===========================================================================

def load_petri_net(args):
    import pm4py

    if args.pnml:
        print(f"[2/4] Lade Petri-Netz direkt aus PNML: {args.pnml}")
        from pm4py.objects.petri_net.importer import importer as pnml_importer
        net, im, fm = pnml_importer.apply(args.pnml)
    elif args.bpmn:
        print(f"[2/4] Lade BPMN und konvertiere zu Petri-Netz: {args.bpmn}")
        bpmn = pm4py.read_bpmn(args.bpmn)
        net, im, fm = pm4py.convert_to_petri_net(bpmn)
    else:
        raise ValueError("Entweder --pnml oder --bpmn muss angegeben werden.")

    n_dp = sum(1 for p in net.places if len(p.out_arcs) > 1)
    print(f"    {len(net.places)} Places, {len(net.transitions)} Transitions")
    print(f"    {n_dp} Decision Points (Places mit >1 ausgehenden Arcs)")
    return net, im, fm


# ===========================================================================
# 3. Discover decision points
# ===========================================================================

def get_decision_points(net):
    print("[3/4] Ermittle Decision Points ...")

    decision_points = []
    for place in net.places:
        out_transitions = [arc.target for arc in place.out_arcs]
        if len(out_transitions) > 1:
            visible = [t for t in out_transitions if t.label is not None]
            if len(visible) >= 1:
                decision_points.append(place)

    print(f"    {len(decision_points)} Decision Points gefunden:")
    for dp in decision_points:
        out_labels = [arc.target.label or f"tau({arc.target.name})"
                      for arc in dp.out_arcs]
        print(f"    '{dp.name}' -> {out_labels}")

    return decision_points


# ===========================================================================
# Leakage-Fix: automatisch Case-Attribute erkennen
# ===========================================================================

# Attribute die immer Event-Attribute oder bedeutungslos sind — nie als Feature
ALWAYS_EXCLUDE = {
    "concept:name", "time:timestamp", "lifecycle:transition",
    # Event-Attribute (aendern sich pro Event)
    "Action",        # Workitem-Status (Created/Obtained/Released...)
    "EventOrigin",   # Woher kommt das Event (Workflow/Offer/Application)
    "org:resource",  # Welcher User hat das Event ausgefuehrt
    "EventID",       # Interne Workitem-ID — bedeutungslos
    # Offer-Attribute (erst nach Angebotserstellung bekannt)
    "OfferID",
    "OfferedAmount",
    "MonthlyCost",
    "NumberOfTerms",
    "FirstWithdrawalAmount",
    # 0% non-null in BPIC-17 — nutzlos
    "Accepted",
    "Selected",
    "CreditScore",
}

# Bekannte BPIC-17 Case-Attribute als Fallback falls trace.attributes leer ist
KNOWN_BPIC17_CASE_ATTRS = [
    "RequestedAmount",
    "ApplicationType",
    "LoanGoal",
]


def detect_case_attributes(log) -> list:
    """
    Liest Case-Attribute direkt aus trace.attributes (pm4py speichert echte
    Case-Attribute dort, Event-Attribute in den einzelnen Events).

    Fallback: bekannte BPIC-17 Case-Attribute wenn trace.attributes leer ist.
    Alle Attribute in ALWAYS_EXCLUDE werden immer entfernt.
    """
    case_level = set()

    # Sample der ersten 200 Traces um Case-Attribute zu erkennen
    for trace in log[:200]:
        if hasattr(trace, "attributes"):
            for k in trace.attributes:
                if k not in ALWAYS_EXCLUDE:
                    case_level.add(k)

    # Fallback: bekannte BPIC-17 Attribute hinzufuegen
    for attr in KNOWN_BPIC17_CASE_ATTRS:
        case_level.add(attr)

    # Nochmals bereinigen
    case_level -= ALWAYS_EXCLUDE

    result = sorted(case_level)
    return result


# ===========================================================================
# 4. Train decision trees (mit Leakage-Fix)
# ===========================================================================

def train_trees(log, net, im, fm, decision_points, args, output_dir):
    from pm4py.algo.conformance.alignments.petri_net import algorithm as align_algo
    from pm4py.objects.petri_net.obj import PetriNet
    from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from collections import defaultdict
    import numpy as np

    SKIP = ">>"

    # --- Erkenne Case-Attribute (Leakage-Fix) ---
    case_attributes = detect_case_attributes(log)
    print(f"\n    Case-Attribute (kein Leakage): {case_attributes}")
    print(f"    Alle anderen Attribute werden als Feature AUSGESCHLOSSEN.\n")

    # ------------------------------------------------------------------
    # Tau-Chain Mapping: dp_place -> branch_key fuer tau-Pfade
    # ------------------------------------------------------------------
    dp_set = set(decision_points)
    place_to_dp_branch = {}

    for dp in decision_points:
        for out_arc in dp.out_arcs:
            branch_trans = out_arc.target
            branch_key   = branch_trans.label or f"tau_{branch_trans.name}"

            queue   = [branch_trans]
            visited = set()
            while queue:
                node = queue.pop(0)
                nid  = id(node)
                if nid in visited:
                    continue
                visited.add(nid)

                if isinstance(node, PetriNet.Transition):
                    if node.label is None:
                        for arc in node.out_arcs:
                            queue.append(arc.target)
                elif isinstance(node, PetriNet.Place):
                    if node not in dp_set:
                        place_to_dp_branch[node] = (dp.name, branch_key)
                        for arc in node.out_arcs:
                            queue.append(arc.target)

    label_to_trans     = defaultdict(list)
    trans_input_places = {}
    for t in net.transitions:
        if t.label:
            label_to_trans[t.label].append(t)
        trans_input_places[t] = [arc.source for arc in t.in_arcs]

    # ------------------------------------------------------------------
    # Alignments berechnen
    # ------------------------------------------------------------------
    print(f"  Berechne Alignments (timeout={args.alignment_timeout}s/Trace) ...")
    parameters = {
        align_algo.Variants.VERSION_STATE_EQUATION_A_STAR.value.Parameters
        .PARAM_MAX_ALIGN_TIME_TRACE: args.alignment_timeout,
    }

    # Keep only the top variants covering 50% of cases — fast to align, robust for training
    from pm4py.algo.filtering.log.variants import variants_filter as _vf
    log_filtered = _vf.filter_log_variants_percentage(log, percentage=0.5)
    print(f"Filtered log: {len(log_filtered)} traces (from {len(log)})")

    # Then run alignments on log_filtered instead of log
    aligned_traces = align_algo.apply(log, net, im, fm, parameters=parameters)
    n_ok = sum(1 for a in aligned_traces if a is not None and a.get("alignment"))
    print(f"  {n_ok}/{len(log)} Traces erfolgreich aligniert.")

    # ------------------------------------------------------------------
    # Replay: Observations sammeln
    # Leakage-Fix: current_vars wird NUR mit Case-Attributen initialisiert,
    # Event-Attribute werden beim Update gefiltert.
    # ------------------------------------------------------------------
    datasets = defaultdict(list)

    for trace, alignment_result in zip(log, aligned_traces):
        if alignment_result is None or not alignment_result.get("alignment"):
            continue

        # NUR Case-Attribute als initiale Variable — kein Event-Leakage
        current_vars = {
            k: v for k, v in trace.attributes.items()
            if k in case_attributes
        }

        event_idx = 0

        for (log_move, model_move) in alignment_result["alignment"]:
            log_label   = log_move   if log_move   != SKIP else None
            model_label = model_move if model_move != SKIP else None

            is_log   = log_label is not None and model_label is None
            is_model = log_label is None     and model_label is not None
            is_both  = log_label is not None and model_label is not None

            if (is_model or is_both) and model_label is not None:
                for fired in label_to_trans.get(model_label, []):
                    for in_place in trans_input_places.get(fired, []):
                        if in_place in dp_set:
                            branch = fired.label
                            datasets[in_place.name].append(
                                (dict(current_vars), branch))
                        elif in_place in place_to_dp_branch:
                            dp_name, branch_key = place_to_dp_branch[in_place]
                            datasets[dp_name].append(
                                (dict(current_vars), branch_key))

            # Update: NUR Case-Attribute aus Event uebernehmen (kein Leakage)
            if is_both or is_log:
                if event_idx < len(trace):
                    ev = trace[event_idx]
                    current_vars.update({
                        k: v for k, v in ev.items()
                        if k in case_attributes  # NUR Case-Attribute!
                    })
                event_idx += 1

    # ------------------------------------------------------------------
    # Decision Trees trainieren
    # ------------------------------------------------------------------
    results = []
    print(f"\n[4/4] Trainiere Decision Trees ...")

    for dp in decision_points:
        dp_name      = dp.name
        out_labels   = [arc.target.label or f"tau({arc.target.name})"
                        for arc in dp.out_arcs]
        observations = datasets.get(dp_name, [])

        print(f"\n  Decision Point: '{dp_name}'")
        print(f"  Branches: {out_labels}")
        print(f"  Observations: {len(observations)}")

        if len(observations) < 10:
            print(f"    SKIP: Zu wenige Observations")
            continue

        y_raw      = pd.Series([lbl for _, lbl in observations], name="target")
        class_dist = y_raw.value_counts().to_dict()
        print(f"  Verteilung: {class_dist}")

        if y_raw.nunique() < 2:
            print(f"    SKIP: Nur eine Klasse beobachtet")
            continue

        X = pd.DataFrame([obs for obs, _ in observations])
        X = X.dropna(axis=1, how="all")
        X = X.loc[:, X.nunique() > 1]

        for col in list(X.columns):
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X[col] = X[col].astype("int64") // 10**9
            try:
                X[col] = pd.to_numeric(X[col])
            except (ValueError, TypeError):
                pass
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].astype(str).fillna("MISSING")
                cats    = sorted(X[col].unique())
                X[col]  = X[col].map({c: i for i, c in enumerate(cats)})

        X = X.loc[:, X.nunique() > 1]
        if X.empty:
            print(f"    SKIP: Keine nutzbaren Features nach Leakage-Filter")
            continue

        print(f"  Features (nach Leakage-Filter): {list(X.columns)}")

        le       = LabelEncoder()
        y_enc    = le.fit_transform(y_raw)
        baseline = y_raw.value_counts().iloc[0] / len(y_raw)

        clf = DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42,
        )
        clf.fit(X, y_enc)

        n_folds   = min(5, len(X))
        cv_scores = cross_val_score(clf, X, y_enc, cv=n_folds, scoring="accuracy")
        cv_acc    = cv_scores.mean()
        lift      = cv_acc - baseline
        quality   = "GOOD" if lift > 0.10 else ("WEAK" if lift > 0 else "NO IMPROVEMENT")

        print(f"    CV Accuracy: {cv_acc:.3f} | Baseline: {baseline:.3f} | "
              f"Lift: {lift:+.3f} [{quality}]")

        rules = export_text(clf, feature_names=list(X.columns))
        for i, cls in enumerate(le.classes_):
            rules = rules.replace(f"class: {i}", f"-> {cls}")

        safe_name = dp_name.replace("/", "_").replace(" ", "_")[:60]
        tree_path = os.path.join(output_dir, f"pm4py_tree_{safe_name}.png")

        class_names = list(le.classes_)
        n_cls       = len(class_names)
        cmap        = plt.cm.get_cmap("tab10", n_cls)

        fig, ax = plt.subplots(
            figsize=(max(14, clf.get_depth() * 5),
                     max(7,  clf.get_depth() * 3))
        )
        plot_tree(clf, feature_names=list(X.columns), class_names=class_names,
                  filled=True, rounded=True, ax=ax, fontsize=9)
        patches = [mpatches.Patch(color=cmap(i), label=class_names[i])
                   for i in range(n_cls)]
        ax.legend(handles=patches, loc="upper right", fontsize=8)
        ax.set_title(
            f"Decision Tree — {dp_name}\n"
            f"Features: {list(X.columns)}\n"
            f"n={len(observations)} | acc={cv_acc:.3f} | "
            f"baseline={baseline:.3f} | lift={lift:+.3f} [{quality}]",
            fontsize=9, fontweight="bold"
        )
        fig.savefig(tree_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Gespeichert: {tree_path}")

        results.append({
            "dp_name":    dp_name,
            "branches":   out_labels,
            "n_obs":      len(observations),
            "class_dist": class_dist,
            "cv_acc":     cv_acc,
            "baseline":   baseline,
            "lift":       lift,
            "quality":    quality,
            "rules":      rules,
            "features":   list(X.columns),
            "tree_path":  tree_path,
        })

    return results


# ===========================================================================
# 5. Save outputs
# ===========================================================================

def save_outputs(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if not results:
        print("\n[!] Keine Ergebnisse zum Speichern.")
        return

    rules_path = os.path.join(output_dir, "pm4py_decision_rules.txt")
    with open(rules_path, "w", encoding="utf-8") as f:
        f.write("DECISION MINING ERGEBNISSE (pm4py) — BPIC-17\n")
        f.write("=" * 60 + "\n")
        f.write("Methode: Alignment-basiertes Decision Mining\n")
        f.write("Referenz: de Leoni & van der Aalst (2013)\n")
        f.write("Leakage-Fix: Nur Case-Attribute als Features\n\n")
        for r in results:
            f.write(f"\n{'='*60}\n")
            f.write(f"Decision Point : {r['dp_name']}\n")
            f.write(f"Branches       : {r['branches']}\n")
            f.write(f"Observations   : {r['n_obs']}\n")
            f.write(f"Verteilung     : {r['class_dist']}\n")
            f.write(f"Features       : {r['features']}\n")
            f.write(f"CV Accuracy    : {r['cv_acc']:.3f} "
                    f"(Baseline {r['baseline']:.3f}, "
                    f"Lift {r['lift']:+.3f}) [{r['quality']}]\n\n")
            f.write("Decision Rules:\n")
            f.write(r["rules"] + "\n")
    print(f"\nGespeichert: {rules_path}")

    md_path = os.path.join(output_dir, "pm4py_decision_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Decision Mining Report — BPIC-17\n\n")
        f.write("**Methode:** Alignment-basiertes Decision Mining ")
        f.write("(de Leoni & van der Aalst, 2013).\n")
        f.write("**Leakage-Fix:** Nur Case-Attribute als Features ")
        f.write("(RequestedAmount, ApplicationType, LoanGoal).\n\n")
        f.write("## Zusammenfassung\n\n")
        f.write("| Decision Point | N | CV Acc | Baseline | Lift | Quality | Branches |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            branches = ", ".join(str(b) for b in r["branches"])
            f.write(f"| `{r['dp_name']}` | {r['n_obs']} | "
                    f"{r['cv_acc']:.3f} | {r['baseline']:.3f} | "
                    f"{r['lift']:+.3f} | {r['quality']} | {branches} |\n")
        f.write("\n## Decision Rules\n\n")
        for r in results:
            f.write(f"### {r['dp_name']}\n\n")
            f.write(f"- **Observations:** {r['n_obs']}\n")
            f.write(f"- **Branches:** {r['branches']}\n")
            f.write(f"- **Features:** {r['features']}\n")
            f.write(f"- **CV Accuracy:** {r['cv_acc']:.3f} "
                    f"(Baseline {r['baseline']:.3f}, Lift {r['lift']:+.3f})\n")
            f.write(f"- **[{r['quality']}]**\n\n")
            f.write("```\n" + r["rules"] + "```\n\n")
            if r.get("tree_path"):
                f.write(f"![Tree]({os.path.basename(r['tree_path'])})\n\n")
    print(f"Gespeichert: {md_path}")
    print(f"\nAlle Outputs in: {output_dir}/")


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log            = load_log(args.log, args)
    net, im, fm    = load_petri_net(args)

    for p in net.places:
        if len(p.out_arcs) > 1:
            out = [(a.target.name, a.target.label) for a in p.out_arcs]
            print(f"Place '{p.name}' ({len(p.out_arcs)} out arcs): {out}")
    decision_points = get_decision_points(net)
    results        = train_trees(log, net, im, fm, decision_points, args, args.output_dir)
    save_outputs(results, args.output_dir)

    if results:
        print(f"\nFertig. {len(results)} Decision Trees trainiert.")
        print("  pm4py_decision_rules.txt  — Regeln als Text")
        print("  pm4py_decision_report.md  — Report fuer die Abgabe")
        print("  pm4py_tree_*.png          — Decision Tree Plots")
    else:
        print("\n[!] Keine Trees trainiert.")
        print("    Tipp: Pruefe ob Case-Attribute in trace.attributes vorhanden sind.")


if __name__ == "__main__":
    main()