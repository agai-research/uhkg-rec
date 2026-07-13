"""
main-exp.py
-----------
Generates the dataset instances needed for the experiments (already
produced by main-HKG.py: base HKG, full UHKG, threshold views) and
runs the four experiments reported in the paper:
  1) classification performance (HKG-Rec vs UHKG-Rec, 5 classifiers)
  2) recommendation accuracy vs top-k (UHKG-Rec, HKG-Rec, KGAT, CAGE)
  3) embedding visualization (t-SNE)
  4) ablation study (HKG-Rec, UHKG-Rec_NM, UHKG-Rec)

Run main-HKG.py first if Data/uhkg_theta_*.json do not exist yet.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def _ensure_data():
    needed = os.path.join(cfg.DATA_DIR, "uhkg_theta_0.7.json")
    if not os.path.exists(needed):
        print("UHKG not found - building it first (main-HKG.py)...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(cfg.ROOT_DIR, "main-HKG.py")], check=True)


def main():
    _ensure_data()

    print("\n########## Study 1: embedding visualization ##########")
    from Experiments import exp_visualization
    exp_visualization.run()

    print("\n########## Study 2: classification performance ##########")
    from Experiments import exp_classification
    exp_classification.run()

    print("\n########## Study 3: recommendation accuracy vs top-k (variants) ##########")
    from Experiments import exp_topk_variants
    exp_topk_variants.run()

    print("\n########## Study 4: recommendation accuracy vs top-k (baselines) ##########")
    from Experiments import exp_topk_baselines
    exp_topk_baselines.run()

    print("\n########## Study 5: ablation study ##########")
    from Experiments import exp_ablation
    exp_ablation.run()

    print("\nAll experiments complete. Results saved under Experiments/results/.")


if __name__ == "__main__":
    main()
