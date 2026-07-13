"""
train.py
--------
Runs Algorithm 1 end-to-end for a given (theta, use_confidence) setting:
  1) confidence-weighted meta-path walks -> Skip-Gram context pairs
  2) joint optimization of the probabilistic embedding + classification
     head (Eq. B.9), via Embedding/model.py
  3) at the end, materializes TREATMENT facts (classifier predictions)
     back into the graph, and saves the mean embeddings mu_v for every
     node, ready to be consumed by the recommendation pipeline and by
     the downstream RF/XGBoost/SVM/KNN/LR classifiers used in the
     experiments.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from HKG.build_graph import load_graph
from HKG.metapaths import generate_walks, context_pairs
from Embedding.model import ProbabilisticEmbedding


def _sp_labels(g):
    """One SP label per H node, taken from its (unique) Treatment edge."""
    sp_ids = sorted({n for n, d in g.nodes(data=True) if d.get("type") == "SP"})
    sp_index = {s: i for i, s in enumerate(sp_ids)}
    labels = {}
    for u, v, d in g.edges(data=True):
        if d.get("relation") == "Treatment" and g.nodes[v]["type"] == "SP":
            labels[u] = sp_index[v]
    return labels, sp_ids


def train_embedding(theta=0.7, use_confidence=True, seed=0, epochs=cfg.N_EPOCHS,
                     verbose=True, use_pattern_matching=True):
    graph_path = os.path.join(cfg.DATA_DIR, f"uhkg_theta_{theta}.json")
    g = load_graph(graph_path)

    node_types = {n: d["type"] for n, d in g.nodes(data=True)}
    node_ids = list(g.nodes)

    walks = generate_walks(g, use_confidence=use_confidence, seed=seed,
                            use_pattern_matching=use_pattern_matching)
    pairs = context_pairs(walks)
    label_map, sp_ids = _sp_labels(g)

    emb = ProbabilisticEmbedding(node_ids, node_types, dim=cfg.EMBED_DIM,
                                  use_confidence=use_confidence, seed=seed)
    rng = np.random.default_rng(seed)
    n_classes = max(1, len(sp_ids))
    W = rng.normal(0, 0.1, size=(cfg.EMBED_DIM, n_classes))
    b = np.zeros(n_classes)

    label_pairs = [(emb.id2idx[h], y) for h, y in label_map.items() if h in emb.id2idx]

    pair_idx = [(emb.id2idx[a], emb.id2idx[b_]) for a, b_ in pairs
                if a in emb.id2idx and b_ in emb.id2idx]

    t0 = time.time()
    for epoch in range(epochs):
        rng.shuffle(pair_idx)
        rng.shuffle(label_pairs)
        losses = []
        bs = cfg.BATCH_SIZE
        n_batches = max(1, len(pair_idx) // bs)
        for i in range(n_batches):
            batch_pairs = pair_idx[i * bs:(i + 1) * bs]
            # spread labeled examples across batches
            lab_bs = max(1, len(label_pairs) // n_batches)
            batch_labels = label_pairs[i * lab_bs:(i + 1) * lab_bs]
            W, b, loss = emb.train_step(batch_pairs, batch_labels, W, b, rng=rng)
            losses.append(loss)
        if verbose:
            print(f"  [theta={theta}, conf={use_confidence}, seed={seed}] "
                  f"epoch {epoch + 1}/{epochs} mean_loss={np.mean(losses):.4f}")
    if verbose:
        print(f"  training done in {time.time() - t0:.1f}s "
              f"({len(node_ids)} nodes, {len(pair_idx)} pairs)")

    # materialize TREATMENT facts (classifier predictions) for every H node
    treatment_facts = []
    for n, t in node_types.items():
        if t != "H":
            continue
        z = emb.mean_embedding(n)
        logits = z @ W + b
        logits -= logits.max()
        q = np.exp(logits)
        q /= q.sum()
        pred = int(np.argmax(q))
        treatment_facts.append({"service": n, "predicted_pattern": sp_ids[pred],
                                 "confidence": float(q[pred])})

    return {
        "node_ids": node_ids,
        "mu": emb.mu,
        "log_sigma2": emb.log_sigma2,
        "sp_ids": sp_ids,
        "labels": label_map,
        "treatment_facts": treatment_facts,
        "W": W, "b": b,
    }


def save_embedding(result, out_path):
    np.savez(out_path,
              node_ids=np.array(result["node_ids"]),
              mu=result["mu"], log_sigma2=result["log_sigma2"],
              sp_ids=np.array(result["sp_ids"]))
    facts_path = out_path.replace(".npz", "_treatment_facts.json")
    with open(facts_path, "w") as f:
        json.dump(result["treatment_facts"], f, indent=2)


if __name__ == "__main__":
    os.makedirs(cfg.GEN_DIR, exist_ok=True)
    result = train_embedding(theta=0.7, use_confidence=True, seed=0, epochs=3)
    out = os.path.join(cfg.GEN_DIR, "embedding_uhkgrec_theta0.7_seed0.npz")
    save_embedding(result, out)
    print(f"Saved embedding to {out}")
    print(f"Example: {len(result['treatment_facts'])} TREATMENT facts materialized")
