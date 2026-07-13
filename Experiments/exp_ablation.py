"""
exp_ablation.py
----------------
Experimental study 5 (ablation study): compares the three variants of
Table "Functional differences between UHKG-Rec variants" - HKG-Rec,
UHKG-Rec_NM, and UHKG-Rec - each now trained with its own distinct
embedding (UHKG-Rec_NM's meta-path walker excludes Match-relation
schemes, see HKG/metapaths.py), so the three variants are no longer
identical in the classification comparison either.

Produces:
  (a) classification performance (best classifier, theta=0.8)
  (b) Top-5 recommendation quality (theta=0.8)
  (c) a Precision/Recall/F1/MRR vs K line-chart figure comparing all
      three variants, reusing the same evaluation harness as studies 3/4

Outputs:
  Experiments/results/ablation_classification.json
  Experiments/results/ablation_recommendation.json
  Experiments/results/ablation_topk_results.json
  Experiments/results/fig_ablation_topk.png
  Experiments/results/fig_ablation_summary.png
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from HKG.build_graph import load_graph
from Embedding.train import train_embedding
from Embedding.classifiers import evaluate_classifiers
from Experiments.eval_utils import load_patient_ground_truth, evaluate_method
from Experiments.recommenders import uhkgrec_family_recommender
from Experiments.plot_style import style_axis, method_color

THETA = 0.8
K_FOR_RECOMMENDATION = 5
K_VALUES = [2, 4, 6, 8, 10]
SEEDS = cfg.SEEDS[:3]
EPOCHS = 2

VARIANTS = [
    ("HKG-Rec", dict(use_confidence=False, use_pattern_matching=True)),
    ("UHKG-Rec_NM", dict(use_confidence=True, use_pattern_matching=False)),
    ("UHKG-Rec", dict(use_confidence=True, use_pattern_matching=True)),
]
METRICS = ["precision", "recall", "f1", "mrr"]
METRIC_TITLES = {"precision": "Precision@K", "recall": "Recall@K",
                  "f1": "F1@K", "mrr": "MRR@K"}


def _labeled_embeddings(result, g):
    id2idx = {n: i for i, n in enumerate(result["node_ids"])}
    X, y = [], []
    for n, d in g.nodes(data=True):
        if d["type"] == "H" and n in result["labels"]:
            X.append(result["mu"][id2idx[n]])
            y.append(result["labels"][n])
    return np.array(X), np.array(y)


def run_classification_ablation():
    g = load_graph(os.path.join(cfg.DATA_DIR, f"uhkg_theta_{THETA}.json"))
    per_variant = {name: [] for name, _ in VARIANTS}

    for name, kwargs in VARIANTS:
        for seed in SEEDS:
            result = train_embedding(theta=THETA, seed=seed, epochs=EPOCHS,
                                      verbose=False, **kwargs)
            X, y = _labeled_embeddings(result, g)
            if len(X) < 6 or len(set(y)) < 2:
                continue
            m = evaluate_classifiers(X, y, seed=seed)
            if m:
                best = max(m.values(), key=lambda r: r["f1"])
                per_variant[name].append(best)

    summary = {}
    for name, runs in per_variant.items():
        if not runs:
            summary[name] = None
            continue
        summary[name] = {
            m: {"mean": float(np.mean([r[m] for r in runs])),
                "std": float(np.std([r[m] for r in runs]))}
            for m in ["accuracy", "precision", "recall", "specificity", "f1"]
        }
    return summary


def run_recommendation_ablation():
    patients = load_patient_ground_truth(max_patients=40)
    per_variant = {name: [] for name, _ in VARIANTS}

    for name, kwargs in VARIANTS:
        for seed in SEEDS:
            print(f"[exp_ablation] recommendation: variant={name} seed={seed}")
            rec_fn = uhkgrec_family_recommender(theta=THETA, seed=seed, **kwargs)
            summary = evaluate_method(rec_fn, patients, [K_FOR_RECOMMENDATION])
            per_variant[name].append(summary[K_FOR_RECOMMENDATION])

    agg = {}
    for name, runs in per_variant.items():
        agg[name] = {
            m: {"mean": float(np.mean([r[m]["mean"] for r in runs])),
                "std": float(np.std([r[m]["mean"] for r in runs]))}
            for m in ["precision", "recall", "f1", "mrr"]
        }
    return agg


def run_topk_ablation():
    patients = load_patient_ground_truth(max_patients=40)
    per_variant_runs = {name: [] for name, _ in VARIANTS}

    for name, kwargs in VARIANTS:
        for seed in SEEDS:
            print(f"[exp_ablation] top-k curve: variant={name} seed={seed}")
            rec_fn = uhkgrec_family_recommender(theta=THETA, seed=seed, **kwargs)
            summary = evaluate_method(rec_fn, patients, K_VALUES)
            per_variant_runs[name].append(summary)

    agg = {}
    for name, runs in per_variant_runs.items():
        agg[name] = {}
        for k in K_VALUES:
            agg[name][k] = {}
            for m in METRICS:
                vals = [r[k][m]["mean"] for r in runs]
                agg[name][k][m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def _plot_topk(agg):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, metric in zip(axes, METRICS):
        for i, (name, _) in enumerate(VARIANTS):
            means = np.array([agg[name][k][metric]["mean"] for k in K_VALUES])
            stds = np.array([agg[name][k][metric]["std"] for k in K_VALUES])
            color = method_color(name, i)
            ax.plot(K_VALUES, means, marker="o", color=color, label=name, linewidth=2)
            ax.fill_between(K_VALUES, means - stds, means + stds, color=color, alpha=0.15)
        ax.set_xlabel("Recommendation list size (K)")
        ax.set_ylabel(METRIC_TITLES[metric])
        ax.set_xticks(K_VALUES)
        ax.set_ylim(0, 1.05)
        ax.set_title(METRIC_TITLES[metric])
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06),
               frameon=False, fontsize=10)
    fig.suptitle("Ablation study: recommendation accuracy vs top-k "
                 "(HKG-Rec vs UHKG-Rec_NM vs UHKG-Rec)", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(cfg.RESULTS_DIR, "fig_ablation_topk.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved figure to {out}")


def _plot_summary(cls_summary, rec_summary):
    """A single bar chart summarizing classification F1 and recommendation
    F1@5 side by side for the three variants - the requested 'figure
    comparing the three variants' behavior and performance'."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = [n for n, _ in VARIANTS]

    ax = axes[0]
    means = [cls_summary[n]["f1"]["mean"] if cls_summary.get(n) else 0 for n in names]
    stds = [cls_summary[n]["f1"]["std"] if cls_summary.get(n) else 0 for n in names]
    colors = [method_color(n, i) for i, n in enumerate(names)]
    ax.bar(names, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("F1-score (best classifier)")
    ax.set_title("Classification performance (theta=0.8)")
    ax.set_ylim(0, 1.15)
    style_axis(ax)

    ax = axes[1]
    means = [rec_summary[n]["f1"]["mean"] for n in names]
    stds = [rec_summary[n]["f1"]["std"] for n in names]
    ax.bar(names, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("F1-Score@5")
    ax.set_title("Recommendation quality (theta=0.8, K=5)")
    ax.set_ylim(0, 1.15)
    style_axis(ax)

    fig.suptitle("Ablation study: HKG-Rec vs UHKG-Rec_NM vs UHKG-Rec", fontsize=13)
    fig.tight_layout()
    out = os.path.join(cfg.RESULTS_DIR, "fig_ablation_summary.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved figure to {out}")


def run():
    print("=== Ablation: classification performance (theta=0.8, best classifier) ===")
    cls_summary = run_classification_ablation()
    print(json.dumps(cls_summary, indent=2))

    print("\n=== Ablation: Top-5 recommendation quality (theta=0.8) ===")
    rec_summary = run_recommendation_ablation()
    print(json.dumps(rec_summary, indent=2))

    print("\n=== Ablation: recommendation accuracy vs top-k ===")
    topk_summary = run_topk_ablation()

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(cfg.RESULTS_DIR, "ablation_classification.json"), "w") as f:
        json.dump(cls_summary, f, indent=2)
    with open(os.path.join(cfg.RESULTS_DIR, "ablation_recommendation.json"), "w") as f:
        json.dump(rec_summary, f, indent=2)
    with open(os.path.join(cfg.RESULTS_DIR, "ablation_topk_results.json"), "w") as f:
        json.dump({"k_values": K_VALUES, "results": topk_summary}, f, indent=2)
    print(f"\nSaved ablation results to {cfg.RESULTS_DIR}/ablation_*.json")

    _plot_topk(topk_summary)
    _plot_summary(cls_summary, rec_summary)
    return cls_summary, rec_summary, topk_summary


if __name__ == "__main__":
    run()
