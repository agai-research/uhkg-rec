"""
main-HKG.py
-----------
Builds the Healthcare Knowledge Graph end to end:
  1) base HKG (P/S/H/O nodes, Have/Require edges) from the dataset
  2) pattern mining (RP/SP nodes, Cause/Treatment/Match edges,
     correlation matrix M^rh)
  3) threshold-filtered UHKG views for theta in {0.6, 0.7, 0.8}

Run this after Data/generate_dataset.py and Data/service_meta.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from HKG.build_graph import build_base_hkg, save_graph, load_graph
from Matching.pattern_mining import mine_patterns, save_theta_views
import json


def main():
    print("Step 1/3 - building base HKG from dataset...")
    base = build_base_hkg()
    base_path = os.path.join(cfg.DATA_DIR, "base_hkg.json")
    save_graph(base, base_path)
    print(f"  {base.number_of_nodes()} nodes, {base.number_of_edges()} edges -> {base_path}")

    print("Step 2/3 - mining requirement/service patterns...")
    uhkg, matrix_info = mine_patterns(base)
    full_path = os.path.join(cfg.DATA_DIR, "uhkg_full.json")
    save_graph(uhkg, full_path)
    with open(os.path.join(cfg.DATA_DIR, "pattern_matrix.json"), "w") as f:
        json.dump(matrix_info, f, indent=2)
    print(f"  RP={matrix_info['n_rp']}  SP={matrix_info['n_sp']} -> {full_path}")

    print("Step 3/3 - writing threshold-filtered UHKG views...")
    save_theta_views(uhkg, cfg.THETAS, cfg.DATA_DIR)

    stats_path = os.path.join(cfg.DATA_DIR, "dataset_stats.json")
    with open(stats_path) as f:
        stats = json.load(f)
    stats["N_rp"] = matrix_info["n_rp"]
    stats["N_sp"] = matrix_info["n_sp"]
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\nHKG construction complete.")


if __name__ == "__main__":
    main()
