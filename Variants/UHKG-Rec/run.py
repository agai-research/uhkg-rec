"""
run.py (UHKG-Rec, full variant)
---------------------------------
The complete proposed framework: uncertain HKG construction,
probabilistic embedding learning, and bilateral pattern matching are
all enabled. See Appendix A.4.
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
    result = train_embedding(theta=theta, use_confidence=True, seed=seed, epochs=epochs)
    out = os.path.join(cfg.GEN_DIR, f"embedding_uhkgrec_theta{theta}_seed{seed}.npz")
    save_embedding(result, out)
    return result


def recommend(query, theta=0.7):
    query = dict(query)
    query["variant"] = "UHKG-Rec"
    query["theta"] = theta
    return uhkgrec.run_query(query, verbose=False)


if __name__ == "__main__":
    print("=== UHKG-Rec (full): training probabilistic embedding (theta=0.7) ===")
    train(theta=0.7, seed=0, epochs=3)
    print("\n=== UHKG-Rec (full): sample recommendation ===")
    q = {"symptoms": ["Fatigue", "Headache"], "active_constraints": [], "k": 5}
    print(json.dumps(recommend(q), indent=2))
