"""
exp_classification.py
----------------------
Experimental study 2 ("Classification performance under different
thresholds"): compares HKG-Rec, UHKG-Rec (theta=0.6), and UHKG-Rec
(theta=0.8) using five classifiers (RF, XGBoost, SVM, KNN, LR) on
four metrics (Accuracy, Precision, Specificity, F1-Score).

HKG-Rec is trained at theta=0.6 (the more data-rich of the two
thresholds compared for UHKG-Rec), since its own graph structure
(which Cause/Treatment/Match facts survive filtering) still depends on
theta even though its embedding does not use confidence weighting.

Outputs:
  Experiments/results/classification_results.json
  Experiments/results/fig_classification.png
"""

import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from Embedding.train import train_embedding
from Embedding.classifiers import evaluate_classifiers
from HKG.build_graph import load_graph
from Experiments.plot_style import style_axis, method_color, METHOD_HATCHES

EXP_EPOCHS = 2
SEEDS = cfg.SEEDS[:3]

METHODS = [
    ("HKG-Rec", dict(theta=0.6, use_confidence=False)),
    ("UHKG-Rec (theta=0.6)", dict(theta=0.6, use_confidence=True)),
    ("UHKG-Rec (theta=0.8)", dict(theta=0.8, use_confidence=True)),
]
CLASSIFIERS = ["RF", "XGBoost", "SVM", "KNN", "LR"]
METRICS = ["accuracy", "precision", "specificity", "f1"]
METRIC_TITLES = {"accuracy": "Accuracy", "precision": "Precision",
                  "specificity": "Specificity", "f1": "F1-Score"}


def _labeled_embeddings(result, g):
    id2idx = {n: i for i, n in enumerate(result["node_ids"])}
    X, y = [], []
    for n, d in g.nodes(data=True):
        if d["type"] == "H" and n in result["labels"]:
            X.append(result["mu"][id2idx[n]])
            y.append(result["labels"][n])
    return np.array(X), np.array(y)


def run():
    per_method = {name: {c: {m: [] for m in METRICS} for c in CLASSIFIERS} for name, _ in METHODS}

    for name, kwargs in METHODS:
        g = load_graph(os.path.join(cfg.DATA_DIR, f"uhkg_theta_{kwargs['theta']}.json"))
        for seed in SEEDS:
            print(f"[exp_classification] method={name} seed={seed}")
            result = train_embedding(seed=seed, epochs=EXP_EPOCHS, verbose=False, **kwargs)
            X, y = _labeled_embeddings(result, g)
            if len(X) < 6 or len(set(y)) < 2:
                print(f"  (skipped: too few samples for method={name} seed={seed})")
                continue
            metrics = evaluate_classifiers(X, y, seed=seed)
            if not metrics:
                print(f"  (skipped: too few labeled classes for method={name} seed={seed})")
                continue
            for clf_name, m in metrics.items():
                for metric in METRICS:
                    per_method[name][clf_name][metric].append(m[metric])

    summary = {}
    for name in per_method:
        summary[name] = {}
        for clf in CLASSIFIERS:
            summary[name][clf] = {}
            for metric in METRICS:
                vals = per_method[name][clf][metric]
                summary[name][clf][metric] = {
                    "mean": float(np.mean(vals)) if vals else 0.0,
                    "std": float(np.std(vals)) if vals else 0.0,
                    "n_runs": len(vals),
                }

    significance = {}
    for clf in CLASSIFIERS:
        a = per_method["UHKG-Rec (theta=0.8)"][clf]["f1"]
        b = per_method["HKG-Rec"][clf]["f1"]
        n = min(len(a), len(b))
        if n >= 2:
            t_stat, p_val = stats.ttest_rel(a[:n], b[:n])
            if not np.isfinite(t_stat) or not np.isfinite(p_val):
                significance[clf] = {"p_value": None, "note": "zero variance across seeds"}
            else:
                significance[clf] = {"t_stat": float(t_stat), "p_value": float(p_val),
                                      "significant_at_0.05": bool(p_val < cfg.SIG_ALPHA)}
        else:
            significance[clf] = None

    result = {"methods": summary, "significance_UHKGRec08_vs_HKGRec": significance}
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(cfg.RESULTS_DIR, "classification_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {cfg.RESULTS_DIR}/classification_results.json")

    _plot(summary)
    return result


def _plot(summary):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    x = np.arange(len(CLASSIFIERS))
    n_methods = len(METHODS)
    width = 0.8 / n_methods

    for ax, metric in zip(axes, METRICS):
        for i, (name, _) in enumerate(METHODS):
            means = [summary[name][c][metric]["mean"] for c in CLASSIFIERS]
            stds = [summary[name][c][metric]["std"] for c in CLASSIFIERS]
            offset = (i - (n_methods - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                   label=name, color=method_color(name, i),
                   hatch=METHOD_HATCHES.get(name, ""), edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSIFIERS)
        ax.set_ylabel(METRIC_TITLES[metric])
        ax.set_title(METRIC_TITLES[metric])
        ax.set_ylim(0, 1.15)
        style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08),
               frameon=False, fontsize=10)
    fig.suptitle("Classification performance under different thresholds", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(cfg.RESULTS_DIR, "fig_classification.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    run()
