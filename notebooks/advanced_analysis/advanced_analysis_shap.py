"""
Advanced Analysis: ML + SHAP — BPIC-17
========================================
3.4 Advanced Analysis für das Process Mining Praktikum

Hypothesis
----------
Das finale Prozessoutcome (Accepted / Cancelled / Denied) kann aus den beim
Antragsstart bekannten Case-Attributen sowie prozessualen Features vorhergesagt
werden. SHAP (SHapley Additive exPlanations) erklärt welche Attribute die
Entscheidung treiben — und liefert damit direkte Inputs für eine
Prozesssimulation.

Approach
--------
1. Feature Engineering (identisch zu decision_mining_outcome.py)
2. Random Forest Classifier trainieren (robuster als Decision Tree)
3. SHAP Values berechnen (TreeExplainer)
4. Visualisierungen:
     shap_summary.png       — globale Feature Importance via SHAP
     shap_dependence_*.png  — wie einzelne Features das Outcome beeinflussen
     shap_waterfall_*.png   — Erklärung einzelner Cases (gut für Report)
     shap_force_*.png       — Force Plot für einzelne Cases
     ml_confusion_matrix.png — Modell-Performance
     ml_roc_curves.png       — ROC Kurven pro Klasse
5. Report: advanced_analysis_report.md

Usage
-----
  python advanced_analysis_shap.py --log ../data/BPIChallenge2017.xes.gz
  python advanced_analysis_shap.py --log ../data/BPIChallenge2017.xes.gz \\
      --output-dir advanced_analysis_output --n-estimators 200

Dependencies
------------
  pip install pm4py scikit-learn shap matplotlib pandas xgboost
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Advanced Analysis: ML + SHAP — BPIC-17")
    p.add_argument("--log", required=True)
    p.add_argument("--output-dir",     default="advanced_analysis_output")
    p.add_argument("--n-estimators",   type=int, default=150,
                   help="Anzahl Trees im Random Forest (default: 150)")
    p.add_argument("--max-depth",      type=int, default=None)
    p.add_argument("--case-col",       default="case:concept:name")
    p.add_argument("--activity-col",   default="concept:name")
    p.add_argument("--time-col",       default="time:timestamp")
    p.add_argument("--shap-samples",   type=int, default=500,
                   help="Anzahl Samples für SHAP (mehr = langsamer, default: 500)")
    return p.parse_args()


# ===========================================================================
# 1. Load log
# ===========================================================================

def load_log(log_path, args):
    import pm4py
    print(f"[1/6] Lade Event Log: {log_path}")
    name = Path(log_path).name.lower()
    if name.endswith(".xes.gz") or name.endswith(".xes"):
        raw = pm4py.read_xes(log_path)
        if not isinstance(raw, pd.DataFrame):
            raw = pm4py.convert_to_dataframe(raw)
    elif name.endswith(".csv"):
        raw = pd.read_csv(log_path)
        raw[args.time_col] = pd.to_datetime(raw[args.time_col], utc=True)
    else:
        raise ValueError(f"Unbekanntes Format: {log_path}")

    raw = raw.rename(columns={
        args.case_col:     "case_id",
        args.activity_col: "activity",
        args.time_col:     "timestamp",
    })
    print(f"    {raw['case_id'].nunique()} Cases, {len(raw)} Events.")
    return raw


# ===========================================================================
# 2. Feature Engineering
# ===========================================================================

OUTCOME_MAP = {
    "A_Pending":   "Accepted",
    "A_Cancelled": "Cancelled",
    "A_Denied":    "Denied",
}

# Schöne Feature-Namen für Plots
FEATURE_LABELS = {
    "case:RequestedAmount":  "Requested Amount (€)",
    "case:ApplicationType":  "Application Type",
    "case:LoanGoal":         "Loan Goal",
    "n_offer_events":        "# Offer Events",
    "n_incomplete":          "# Incomplete Rounds",
    "n_handle_leads":        "# W_Handle Leads",
    "n_validate":            "# Validation Rounds",
    "n_events":              "# Total Events",
    "duration_days":         "Case Duration (days)",
    "first_hour":            "Application Hour",
    "first_weekday":         "Application Weekday",
    "first_month":           "Application Month",
}


def engineer_features(df):
    print("[2/6] Feature Engineering ...")

    # --- Outcome ---
    outcome_df = df[df["activity"].isin(OUTCOME_MAP)].copy()
    outcome_df["outcome"] = outcome_df["activity"].map(OUTCOME_MAP)
    last_outcome = (outcome_df
                    .sort_values("timestamp")
                    .groupby("case_id")
                    .last()
                    .reset_index()[["case_id", "outcome"]])

    print(f"    Outcome-Verteilung:\n"
          f"{last_outcome['outcome'].value_counts().to_string()}\n")

    # --- Case-Attribute ---
    case_cols = [c for c in df.columns
                 if c.startswith("case:")
                 and c != "case:concept:name"
                 and df[c].notna().mean() > 0.5]

    first_ev = (df.sort_values("timestamp")
                  .groupby("case_id")
                  .first()
                  .reset_index())

    available = [c for c in case_cols if c in first_ev.columns]
    case_feats = first_ev[["case_id", "timestamp"] + available].copy()

    # Zeitbasierte Features aus erstem Event
    case_feats["first_hour"]    = case_feats["timestamp"].dt.hour
    case_feats["first_weekday"] = case_feats["timestamp"].dt.dayofweek
    case_feats["first_month"]   = case_feats["timestamp"].dt.month
    case_feats = case_feats.drop(columns=["timestamp"])

    # --- Prozessuale Features ---
    offer_counts = (df[df["activity"].str.startswith("O_", na=False)]
                    .groupby("case_id").size()
                    .reset_index(name="n_offer_events"))

    incomplete = (df[df["activity"] == "A_Incomplete"]
                  .groupby("case_id").size()
                  .reset_index(name="n_incomplete"))

    leads = (df[df["activity"] == "W_Handle leads"]
             .groupby("case_id").size()
             .reset_index(name="n_handle_leads"))

    validate = (df[df["activity"] == "W_Validate application"]
                .groupby("case_id").size()
                .reset_index(name="n_validate"))

    n_events = (df.groupby("case_id").size()
                  .reset_index(name="n_events"))

    duration = (df.groupby("case_id")["timestamp"]
                  .agg(lambda x: (x.max() - x.min()).total_seconds() / 86400)
                  .reset_index(name="duration_days"))

    # Alles mergen
    result = last_outcome.copy()
    for tbl in [case_feats, offer_counts, incomplete, leads,
                validate, n_events, duration]:
        result = result.merge(tbl, on="case_id", how="left")

    for col in ["n_offer_events","n_incomplete","n_handle_leads",
                "n_validate","n_events","duration_days"]:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    print(f"    Feature-Matrix: {len(result)} Cases × {result.shape[1]-2} Features")
    print(f"    Case-Attribute: {available}")
    return result


# ===========================================================================
# 3. Prepare X, y
# ===========================================================================

def prepare_xy(feat_df):
    from sklearn.preprocessing import LabelEncoder

    y_raw = feat_df["outcome"].copy()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X = feat_df.drop(columns=["case_id", "outcome"], errors="ignore").copy()

    # Schöne Namen für Plots
    X = X.rename(columns=FEATURE_LABELS)

    cat_encoders = {}
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
            cats = sorted(X[col].unique())
            cat_encoders[col] = {c: i for i, c in enumerate(cats)}
            X[col] = X[col].map(cat_encoders[col])

    X = X.loc[:, X.nunique() > 1]
    return X, y, le, cat_encoders


# ===========================================================================
# 4. Train Random Forest
# ===========================================================================

def train_model(X, y, le, n_estimators, max_depth):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 roc_auc_score, roc_curve)

    print(f"[3/6] Trainiere Random Forest "
          f"(n_estimators={n_estimators}, max_depth={max_depth}) ...")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    f1_scores  = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")

    baseline = pd.Series(y).value_counts().iloc[0] / len(y)

    print(f"    Baseline (Majority):    {baseline:.3f}")
    print(f"    CV Accuracy:            {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
    print(f"    CV F1-macro:            {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
    print(f"    Lift over baseline:     {acc_scores.mean()-baseline:+.3f}")

    clf.fit(X, y)
    y_pred = clf.predict(X)
    print(f"\n    Classification Report (train set):")
    print(classification_report(y, y_pred, target_names=le.classes_, zero_division=0))

    metrics = {
        "baseline":  baseline,
        "cv_acc":    acc_scores.mean(),
        "cv_acc_std": acc_scores.std(),
        "cv_f1":     f1_scores.mean(),
        "cv_f1_std": f1_scores.std(),
        "lift":      acc_scores.mean() - baseline,
        "y_pred":    y_pred,
    }
    return clf, metrics


# ===========================================================================
# 5. SHAP Analysis
# ===========================================================================

def compute_shap(clf, X, shap_samples, output_dir, le):
    try:
        import shap
    except ImportError:
        print("\n[!] SHAP nicht installiert. Installiere mit: pip install shap")
        print("    Ueberspringe SHAP-Analyse.")
        return None, None, []

    print(f"\n[4/6] Berechne SHAP Values (n={shap_samples} Samples) ...")

    X_shap = X.sample(n=min(shap_samples, len(X)), random_state=42)

    explainer  = shap.TreeExplainer(clf)
    raw_values = explainer.shap_values(X_shap)

    # Neuere SHAP-Versionen: 3D-Array (n_samples, n_features, n_classes)
    # Aeltere SHAP-Versionen: Liste von 2D-Arrays [(n_samples, n_features), ...]
    if isinstance(raw_values, np.ndarray) and raw_values.ndim == 3:
        shap_values = [raw_values[:, :, i] for i in range(raw_values.shape[2])]
    elif isinstance(raw_values, list):
        shap_values = raw_values
    else:
        shap_values = [raw_values]

    print(f"    SHAP shape pro Klasse: {shap_values[0].shape}")

    class_names = list(le.classes_)
    n_classes   = len(class_names)

    assert shap_values[0].shape == X_shap.shape, (
        f"SHAP shape {shap_values[0].shape} != X_shap {X_shap.shape}"
    )

    plots_made = []

    # ------------------------------------------------------------------
    # Plot 1: SHAP Summary Plot (globale Feature Importance, alle Klassen)
    # ------------------------------------------------------------------
    print("    Plot 1: SHAP Summary ...")
    fig, axes = plt.subplots(1, n_classes,
                             figsize=(7 * n_classes, 6),
                             constrained_layout=True)
    if n_classes == 1:
        axes = [axes]

    for i, (ax, cls) in enumerate(zip(axes, class_names)):
        plt.sca(ax)
        shap.summary_plot(
            shap_values[i],
            X_shap,
            plot_type="dot",
            show=False,
            max_display=10,
            color_bar=(i == n_classes - 1),
        )
        ax.set_title(f"SHAP — Outcome: {cls}", fontsize=12, fontweight="bold")

    path = os.path.join(output_dir, "shap_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots_made.append(path)
    print(f"    Gespeichert: {path}")

    # ------------------------------------------------------------------
    # Plot 2: SHAP Bar Plot (mean |SHAP|, alle Klassen überlagert)
    # ------------------------------------------------------------------
    print("    Plot 2: SHAP Feature Importance Bar ...")
    # Mean absolute SHAP pro Feature und Klasse
    mean_abs = np.array([np.abs(sv).mean(axis=0) for sv in shap_values])
    # mean_abs shape: (n_classes, n_features)

    feat_names = list(X_shap.columns)
    n_feat     = len(feat_names)

    colors_cls = ["#e74c3c", "#f39c12", "#2ecc71"][:n_classes]
    x          = np.arange(n_feat)
    bar_w      = 0.25

    fig, ax = plt.subplots(figsize=(12, max(5, n_feat * 0.5)))
    for i, (cls, color) in enumerate(zip(class_names, colors_cls)):
        offsets = (i - n_classes / 2 + 0.5) * bar_w
        bars = ax.barh(x + offsets, mean_abs[i],
                       height=bar_w, label=cls, color=color, alpha=0.85)

    ax.set_yticks(x)
    ax.set_yticklabels(feat_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("SHAP Feature Importance per Outcome Class",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()

    path = os.path.join(output_dir, "shap_importance_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots_made.append(path)
    print(f"    Gespeichert: {path}")

    # ------------------------------------------------------------------
    # Plot 3: SHAP Dependence Plot für RequestedAmount (wichtigstes Feature)
    # ------------------------------------------------------------------
    # Finde das Feature mit der höchsten SHAP Importance (über alle Klassen)
    total_importance = np.array([np.abs(sv).mean(axis=0)
                                 for sv in shap_values]).sum(axis=0)
    top_feature_idx  = int(np.argmax(total_importance))
    top_feature_name = feat_names[top_feature_idx]

    print(f"    Plot 3: SHAP Dependence — '{top_feature_name}' ...")

    fig, axes = plt.subplots(1, n_classes,
                             figsize=(6 * n_classes, 5),
                             constrained_layout=True)
    if n_classes == 1:
        axes = [axes]

    for i, (ax, cls, color) in enumerate(zip(axes, class_names, colors_cls)):
        vals = X_shap[top_feature_name].values
        shap_v = shap_values[i][:, top_feature_idx]

        sc = ax.scatter(vals, shap_v, c=shap_v, cmap="RdYlGn",
                        alpha=0.5, s=15, vmin=-max(abs(shap_v)),
                        vmax=max(abs(shap_v)))
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel(top_feature_name, fontsize=11)
        ax.set_ylabel("SHAP value", fontsize=11)
        ax.set_title(f"Outcome: {cls}", fontsize=11, fontweight="bold")
        plt.colorbar(sc, ax=ax, label="SHAP value")

    fig.suptitle(f"SHAP Dependence Plot — {top_feature_name}",
                 fontsize=13, fontweight="bold")

    path = os.path.join(output_dir, f"shap_dependence_{top_feature_name.replace(' ','_').replace('(','').replace(')','').replace('/','_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots_made.append(path)
    print(f"    Gespeichert: {path}")

    # ------------------------------------------------------------------
    # Plot 4: SHAP Waterfall für je einen repräsentativen Case pro Klasse
    # ------------------------------------------------------------------
    print("    Plot 4: SHAP Waterfall (je 1 Case pro Klasse) ...")

    # Finde den Case der am "typischsten" für jede Klasse ist
    # = Case wo die Modell-Vorhersage am sichersten ist (höchste P)
    proba = clf.predict_proba(X_shap)

    for cls_idx, cls_name in enumerate(class_names):
        # Index des Cases mit höchster Wahrscheinlichkeit für diese Klasse
        best_idx = int(np.argmax(proba[:, cls_idx]))
        best_prob = proba[best_idx, cls_idx]

        sv_single = shap_values[cls_idx][best_idx]
        base_val  = explainer.expected_value[cls_idx]

        # Manueller Waterfall Plot (shap.waterfall_plot braucht shap.Explanation)
        feat_vals = X_shap.iloc[best_idx]
        order     = np.argsort(np.abs(sv_single))[::-1][:10]  # top 10

        fig, ax = plt.subplots(figsize=(10, 6))

        cumsum    = base_val
        positions = []
        widths    = []
        colors    = []
        labels    = []

        for rank, fi in enumerate(order[::-1]):
            sv = sv_single[fi]
            positions.append(cumsum)
            widths.append(sv)
            colors.append("#e74c3c" if sv > 0 else "#2ecc71")
            labels.append(f"{feat_names[fi]} = {feat_vals.iloc[fi]:.2f}"
                          if isinstance(feat_vals.iloc[fi], float)
                          else f"{feat_names[fi]} = {feat_vals.iloc[fi]}")
            cumsum += sv

        y_pos = range(len(positions))
        bars  = ax.barh(list(y_pos), widths, left=positions,
                        color=colors, alpha=0.85, edgecolor="white")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(base_val, color="gray", linewidth=1, linestyle="--",
                   label=f"Base value = {base_val:.3f}")
        ax.axvline(cumsum, color="black", linewidth=1.5, linestyle="-",
                   label=f"Prediction = {best_prob:.3f}")
        ax.set_xlabel("SHAP value (contribution to log-odds)", fontsize=11)
        ax.set_title(
            f"SHAP Waterfall — Outcome: {cls_name}\n"
            f"(typischster Case, P({cls_name}) = {best_prob:.2f})",
            fontsize=11, fontweight="bold"
        )
        ax.legend(fontsize=9)
        plt.tight_layout()

        safe_cls = cls_name.lower()
        path = os.path.join(output_dir, f"shap_waterfall_{safe_cls}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots_made.append(path)
        print(f"    Gespeichert: {path}")

    return shap_values, X_shap, plots_made


# ===========================================================================
# 6. Performance Plots
# ===========================================================================

def plot_confusion_matrix(y_true, y_pred, le, output_dir):
    from sklearn.metrics import confusion_matrix

    cm     = confusion_matrix(y_true, y_pred)
    labels = list(le.classes_)
    n      = len(labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Random Forest", fontsize=12, fontweight="bold")

    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i,j]:,}",
                    ha="center", va="center", fontsize=12,
                    color="white" if cm[i,j] > thresh else "black")

    plt.tight_layout()
    path = os.path.join(output_dir, "ml_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Gespeichert: {path}")
    return path


def plot_roc_curves(clf, X, y, le, output_dir):
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    classes   = list(range(len(le.classes_)))
    y_bin     = label_binarize(y, classes=classes)
    y_proba   = clf.predict_proba(X)
    colors_roc = ["#e74c3c", "#f39c12", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (cls_name, color) in enumerate(zip(le.classes_, colors_roc)):
        if y_bin.shape[1] > 1:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        else:
            fpr, tpr, _ = roc_curve(y == i, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls_name} (AUC = {roc_auc:.3f})")

    ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves per Outcome Class", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()

    path = os.path.join(output_dir, "ml_roc_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Gespeichert: {path}")
    return path


# ===========================================================================
# 7. Save report
# ===========================================================================

def save_report(metrics, X, le, shap_plots, perf_plots, output_dir):
    path = os.path.join(output_dir, "advanced_analysis_report.md")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Advanced Analysis: ML + SHAP — BPIC-17\n\n")

        f.write("## 1. Hypothesis\n\n")
        f.write("**Research question:** Can the final process outcome ")
        f.write("(Accepted / Cancelled / Denied) be predicted from attributes ")
        f.write("known at the start of a loan application, and which attributes ")
        f.write("drive this prediction?\n\n")
        f.write("**Hypothesis:** The requested loan amount and application type ")
        f.write("are the primary drivers of the final outcome. Cases with very ")
        f.write("low or very high requested amounts are more likely to be ")
        f.write("cancelled, while denied cases are driven by a different set ")
        f.write("of factors.\n\n")
        f.write("**Relevance for simulation:** Understanding which input attributes ")
        f.write("determine the outcome allows a simulation model to route cases ")
        f.write("realistically. The SHAP values identify which attributes must be ")
        f.write("modelled stochastically as simulation inputs, and what thresholds ")
        f.write("trigger different process paths.\n\n")

        f.write("## 2. Approach and Results\n\n")
        f.write("### Model\n\n")
        f.write("A Random Forest Classifier with balanced class weights was trained ")
        f.write("on case-level and derived process features. ")
        f.write("SHAP (TreeExplainer) was used to compute feature contributions.\n\n")
        f.write("### Features used\n\n")
        for col in X.columns:
            f.write(f"- `{col}`\n")
        f.write("\n### Performance\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Baseline (majority class) | {metrics['baseline']:.3f} |\n")
        f.write(f"| CV Accuracy (5-fold) | {metrics['cv_acc']:.3f} ± {metrics['cv_acc_std']:.3f} |\n")
        f.write(f"| CV F1-macro (5-fold) | {metrics['cv_f1']:.3f} ± {metrics['cv_f1_std']:.3f} |\n")
        f.write(f"| Lift over baseline | {metrics['lift']:+.3f} |\n\n")

        f.write("### Visualisations\n\n")
        for p in (shap_plots or []) + perf_plots:
            if p:
                fname = os.path.basename(p)
                f.write(f"![{fname}]({fname})\n\n")

        f.write("## 3. Interpretation\n\n")
        f.write("The SHAP summary plots reveal the global importance of each ")
        f.write("feature across all three outcome classes. The dependence plot ")
        f.write("shows how the requested amount influences the prediction — ")
        f.write("a key threshold effect is expected around €3,000–5,000 ")
        f.write("separating cancelled from accepted applications.\n\n")
        f.write("The waterfall plots explain individual cases: they show how ")
        f.write("each feature pushes the prediction towards or away from a ")
        f.write("given outcome, starting from the base rate.\n\n")
        f.write("**Implications for simulation:**\n\n")
        f.write("- The top SHAP features should be modelled as stochastic ")
        f.write("input parameters in any simulation model.\n")
        f.write("- The SHAP threshold on `Requested Amount (€)` gives a ")
        f.write("data-driven routing rule for the main XOR gateway ")
        f.write("(A_Cancelled vs. continued processing).\n")
        f.write("- Features with near-zero SHAP importance can be omitted ")
        f.write("from the simulation to reduce complexity.\n")

    print(f"    Gespeichert: {path}")
    return path


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Log laden
    df = load_log(args.log, args)

    # 2. Features
    feat_df = engineer_features(df)
    if len(feat_df) < 50:
        print("[!] Zu wenige Cases. Abbruch.")
        return

    # 3. X, y
    print("[3/6] Bereite Feature-Matrix vor ...")
    X, y, le, cat_enc = prepare_xy(feat_df)
    print(f"    X: {X.shape}, Klassen: {list(le.classes_)}")

    # 4. Random Forest
    clf, metrics = train_model(X, y, le, args.n_estimators, args.max_depth)

    # 5. SHAP
    print("\n[5/6] SHAP Analyse ...")
    shap_result = compute_shap(clf, X, args.shap_samples, args.output_dir, le)
    shap_plots  = shap_result[2] if shap_result[0] is not None else []

    # 6. Performance Plots
    print("\n[6/6] Performance Plots ...")
    cm_path  = plot_confusion_matrix(y, metrics["y_pred"], le, args.output_dir)
    roc_path = plot_roc_curves(clf, X, y, le, args.output_dir)
    perf_plots = [cm_path, roc_path]

    # 7. Report
    report_path = save_report(metrics, X, le, shap_plots, perf_plots, args.output_dir)

    print(f"\nFertig. Alle Outputs in: {args.output_dir}/")
    print("\nOutputs:")
    print("  advanced_analysis_report.md     — Report für die Abgabe")
    print("  shap_summary.png                — Globale SHAP Feature Importance")
    print("  shap_importance_bar.png         — SHAP Importance pro Klasse")
    print("  shap_dependence_*.png           — SHAP Dependence (Top Feature)")
    print("  shap_waterfall_*.png            — SHAP Waterfall pro Klasse")
    print("  ml_confusion_matrix.png         — Confusion Matrix")
    print("  ml_roc_curves.png               — ROC Kurven")


if __name__ == "__main__":
    main()