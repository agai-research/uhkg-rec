"""
run.py (HKG-Rec variant)
-------------------------
Deterministic ablation variant: uncertain HKG construction and
probabilistic embedding are both switched off (use_confidence=False -
uniform-weight walks, single point embedding mu_v only). Bilateral
pattern matching (Algorithm 2) is kept. See Appendix A.4.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import config as cfg
from Embedding.train import train_embedding, save_embedding
from importlib import import_module

uhkgrec = import_module("UHKG-Rec")


def train(theta=0.7, seed=0, epochs=cfg.N_EPOCHS):
    result = train_embedding(theta=theta, use_confidence=False, seed=seed, epochs=epochs)
    out = os.path.join(cfg.GEN_DIR, f"embedding_hkgrec_theta{theta}_seed{seed}.npz")
    save_embedding(result, out)
    return result


def recommend(query, theta=0.7):
    query = dict(query)
    query["variant"] = "HKG-Rec"
    query["theta"] = theta
    return uhkgrec.run_query(query, verbose=False)


if __name__ == "__main__":
    print("=== HKG-Rec: training deterministic embedding (theta=0.7) ===")
    train(theta=0.7, seed=0, epochs=3)
    print("\n=== HKG-Rec: sample recommendation ===")
    q = {"symptoms": ["Fatigue", "Headache"], "active_constraints": [], "k": 5}
    print(json.dumps(recommend(q), indent=2))
