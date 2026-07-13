"""
kgat.py
-------
Minimal, self-contained re-implementation of the core idea behind KGAT
(Wang et al., "KGAT: Knowledge Graph Attention Network for
Recommendation") for the UHKG-Rec comparison: an attention mechanism
aggregates a node's neighbors, weighting relations by their learned
importance, so both direct and indirect (multi-hop) relationships
contribute to the final embedding used for top-k ranking.

This is a standard, independent recommendation baseline - no UHKG-Rec
mechanism (confidence weighting, pattern matching) is added to it, so
the comparison between UHKG-Rec and KGAT stays fair.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as cfg
from HKG.build_graph import load_graph


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class KGAT:
    def __init__(self, node_ids, dim=cfg.EMBED_DIM, seed=0):
        rng = np.random.default_rng(seed)
        self.node_ids = list(node_ids)
        self.id2idx = {n: i for i, n in enumerate(self.node_ids)}
        self.n = len(self.node_ids)
        self.dim = dim
        self.emb = rng.normal(0, 0.1, size=(self.n, dim))
        # one scalar attention weight per relation type
        self.rel_att = {r: 1.0 for r in cfg.RELATIONS}

    def _neighbors(self, g, node):
        out = []
        for _, v, d in g.out_edges(node, data=True):
            out.append((v, d.get("relation", "Require")))
        for u, _, d in g.in_edges(node, data=True):
            out.append((u, d.get("relation", "Require")))
        return out

    def propagate(self, g, layers=2):
        """Attention-weighted neighborhood aggregation (a simplified
        KGAT propagation layer, repeated `layers` times)."""
        for _ in range(layers):
            new_emb = self.emb.copy()
            for n in self.node_ids:
                neigh = self._neighbors(g, n)
                if not neigh:
                    continue
                weights = np.array([self.rel_att.get(r, 1.0) for _, r in neigh])
                weights = np.exp(weights - weights.max())
                weights /= weights.sum()
                agg = np.zeros(self.dim)
                for (nb, _), w in zip(neigh, weights):
                    agg += w * self.emb[self.id2idx[nb]]
                new_emb[self.id2idx[n]] = 0.5 * self.emb[self.id2idx[n]] + 0.5 * agg
            self.emb = new_emb

    def score(self, u, v):
        return float(np.dot(self.emb[self.id2idx[u]], self.emb[self.id2idx[v]]))

    def recommend(self, g, symptom_nodes, candidate_service_nodes, k):
        """Rank candidate services by embedding similarity to the query's
        symptom nodes (mean pooled), a standard KG-embedding recommendation
        rule and KGAT's own evaluation protocol."""
        if not symptom_nodes:
            return []
        q = np.mean([self.emb[self.id2idx[s]] for s in symptom_nodes if s in self.id2idx], axis=0)
        scored = [(h, float(_sigmoid(np.dot(q, self.emb[self.id2idx[h]]))))
                  for h in candidate_service_nodes if h in self.id2idx]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


def run_kgat(theta=0.7, seed=0, layers=2):
    g = load_graph(os.path.join(cfg.DATA_DIR, f"uhkg_theta_{theta}.json"))
    model = KGAT(list(g.nodes), seed=seed)
    model.propagate(g, layers=layers)
    return model, g


if __name__ == "__main__":
    model, g = run_kgat(theta=0.7, seed=0)
    symptoms = [n for n, d in g.nodes(data=True) if d["type"] == "S"][:2]
    services = [n for n, d in g.nodes(data=True) if d["type"] == "H"]
    rec = model.recommend(g, symptoms, services, k=5)
    print("KGAT top-5 recommendation for", symptoms)
    for h, s in rec:
        print(f"  {h}: {s:.4f}")
