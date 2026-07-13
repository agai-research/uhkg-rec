"""
exp_topk_baselines.py
-----------------------
Experimental study 4 (Section "Baselines"): compares UHKG-Rec (theta=0.6,
theta=0.8) against the two baselines, CAGE_Rec and GAT_Rec (KGAT), as
K varies over {2, 4, 6, 8, 10}, using Precision@K, Recall@K, F1@K, MRR@K.

Marker convention (as specified for this figure):
  UHKG-Rec (theta=0.6): cross (x)
  UHKG-Rec (theta=0.8): circle (o)
  CAGE_Rec:             triangle (^)
  GAT_Rec (KGAT):       square (s)

Outputs:
  Experiments/results/topk_baselines_results.json
  Experiments/results/fig_topk_baselines.png
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from Experiments.eval_utils import load_patient_ground_truth, evaluate_method, paired_test
from Experiments.recommenders import uhkgrec_family_recommender, kgat_recommender, cage_recommender
from Experiments.plot_style import style_axis, method_color, METHOD_MARKERS

K_VALUES = [2, 4, 6, 8, 10]
SEEDS = cfg.SEEDS[:3]
METRICS = ["precision", "recall", "f1", "mrr"]
METRIC_TITLES = {"precision": "Precision@K", "recall": "Recall@K",
                  "f1": "F1@K", "mrr": "MRR@K"}

METHOD_NAMES = ["UHKG-Rec (theta=0.6)", "UHKG-Rec (theta=0.8)", "CAGE_Rec", "GAT_Rec"]


def _rec_fn_for(name, seed):
    if name == "UHKG-Rec (theta=0.6)":
        return uhkgrec_family_recommender(theta=0.6, use_confidence=True,
                                           use_pattern_matching=True, seed=seed)
    if name == "UHKG-Rec (theta=0.8)":
        return uhkgrec_family_recommender(theta=0.8, use_confidence=True,
                                           use_pattern_matching=True, seed=seed)
    if name == "CAGE_Rec":
        return cage_recommender(theta=0.7, seed=seed)
    if name == "GAT_Rec":
        return kgat_recommender(theta=0.7, seed=seed)
    raise ValueError(name)


def run():
    patients = load_patient_ground_truth(max_patients=60)
    per_method_runs = {name: [] for name in METHOD_NAMES}

    for name in METHOD_NAMES:
        for seed in SEEDS:
            print(f"[exp_topk_baselines] method={name} seed={seed}")
            rec_fn = _rec_fn_for(name, seed)
            summary = evaluate_method(rec_fn, patients, K_VALUES)
            per_method_runs[name].append(summary)

    agg = {}
    for name, runs in per_method_runs.items():
        agg[name] = {}
        for k in K_VALUES:
            agg[name][k] = {}
            for m in METRICS:
                vals = [r[k][m]["mean"] for r in runs]
                agg[name][k][m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    k_ref = 6
    ref_name = "UHKG-Rec (theta=0.8)"
    ref_vals = [r[k_ref]["f1"]["mean"] for r in per_method_runs[ref_name]]
    significance = {}
    for name, runs in per_method_runs.items():
        if name == ref_name:
            continue
        other_vals = [r[k_ref]["f1"]["mean"] for r in runs]
        significance[f"{ref_name}_vs_{name}"] = paired_test(ref_vals, other_vals)

    result = {"k_values": K_VALUES, "results": agg, "significance_at_k6": significance}
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(cfg.RESULTS_DIR, "topk_baselines_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {cfg.RESULTS_DIR}/topk_baselines_results.json")

    _plot(agg)
    return result


def _plot(agg):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, metric in zip(axes, METRICS):
        for i, name in enumerate(METHOD_NAMES):
            means = np.array([agg[name][k][metric]["mean"] for k in K_VALUES])
            stds = np.array([agg[name][k][metric]["std"] for k in K_VALUES])
            color = method_color(name, i)
            marker = METHOD_MARKERS.get(name, "o")
            ax.errorbar(K_VALUES, means, yerr=stds, marker=marker, color=color,
                        label=name, linewidth=2, markersize=8, capsize=4)
        ax.set_xlabel("Recommendation list size (K)")
        ax.set_ylabel(METRIC_TITLES[metric])
        ax.set_xticks(K_VALUES)
        ax.set_ylim(0, 1.05)
        ax.set_title(METRIC_TITLES[metric])
        style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06),
               frameon=False, fontsize=10)
    fig.suptitle("Recommendation accuracy: UHKG-Rec vs. baseline methods", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(cfg.RESULTS_DIR, "fig_topk_baselines.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    run()
