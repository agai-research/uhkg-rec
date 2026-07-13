"""
exp_visualization.py
---------------------
Experimental study 1 ("Visualization of embeddings under different
thresholds with various numbers of patterns"): a 2D t-SNE projection
of the service embeddings learned by HKG-Rec, UHKG-Rec_NM, and
UHKG-Rec, all at theta=0.8, colored by biomedical category (Cancer,
Diabetes, Dermatology, Allergy, Respiratory - or the closest
equivalent categories available in the generated dataset, see
Data/categories.py).

Outputs:
  Experiments/results/visualization_results.json (embedding coordinates)
  Experiments/results/fig_visualization.png
"""

import json
import os
import sys

import numpy as np
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from Embedding.train import train_embedding
from HKG.build_graph import load_graph
from Data.categories import build_service_categories
from Experiments.plot_style import category_color

THETA = 0.8
SEED = 0

PANELS = [
    ("HKG-Rec", dict(use_confidence=False, use_pattern_matching=True)),
    ("UHKG-Rec_NM", dict(use_confidence=True, use_pattern_matching=False)),
    ("UHKG-Rec", dict(use_confidence=True, use_pattern_matching=True)),
]


def run():
    if not os.path.exists(os.path.join(cfg.GEN_DIR, "service_categories.json")):
        build_service_categories()
    with open(os.path.join(cfg.GEN_DIR, "service_categories.json")) as f:
        cat_info = json.load(f)
    svc_category = cat_info["service_category"]

    g = load_graph(os.path.join(cfg.DATA_DIR, f"uhkg_theta_{THETA}.json"))

    panel_data = {}
    for name, kwargs in PANELS:
        print(f"[exp_visualization] training embedding for panel: {name}")
        result = train_embedding(theta=THETA, seed=SEED, epochs=4, verbose=False, **kwargs)
        id2idx = {n: i for i, n in enumerate(result["node_ids"])}

        h_nodes = [n for n, d in g.nodes(data=True) if d["type"] == "H"]
        h_nodes = [n for n in h_nodes if n.split("::", 1)[-1] in svc_category]
        if len(h_nodes) < 6:
            print(f"  (too few categorized services at theta={THETA}, "
                  f"falling back to all labeled H nodes for panel {name})")
            h_nodes = [n for n in g.nodes if n in result["labels"]]

        X = np.array([result["mu"][id2idx[n]] for n in h_nodes if n in id2idx])
        used_nodes = [n for n in h_nodes if n in id2idx]
        categories = [svc_category.get(n.split("::", 1)[-1], "Other") for n in used_nodes]

        perplexity = max(2, min(30, max(2, len(used_nodes) // 3)))
        proj = TSNE(n_components=2, perplexity=perplexity, random_state=0,
                    init="pca").fit_transform(X)

        panel_data[name] = {
            "coords": proj.tolist(),
            "categories": categories,
            "services": [n.split("::", 1)[-1] for n in used_nodes],
        }

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(cfg.RESULTS_DIR, "visualization_results.json"), "w") as f:
        json.dump(panel_data, f, indent=2)
    print(f"\nSaved results to {cfg.RESULTS_DIR}/visualization_results.json")

    _plot(panel_data)
    return panel_data


def _plot(panel_data):
    import matplotlib.pyplot as plt

    all_categories = sorted({c for d in panel_data.values() for c in d["categories"]})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (name, _) in zip(axes, PANELS):
        d = panel_data[name]
        coords = np.array(d["coords"])
        cats = d["categories"]
        for i, cat in enumerate(all_categories):
            mask = [c == cat for c in cats]
            if not any(mask):
                continue
            pts = coords[mask]
            ax.scatter(pts[:, 0], pts[:, 1], s=45, color=category_color(cat, i),
                       label=cat, edgecolor="white", linewidth=0.4, alpha=0.9)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Embedding dimension 1")
        ax.set_ylabel("Embedding dimension 2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(all_categories),
               bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=10, title="Disease category")
    fig.suptitle("Visualization of embeddings under different thresholds "
                 "with various numbers of patterns", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(cfg.RESULTS_DIR, "fig_visualization.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    run()
