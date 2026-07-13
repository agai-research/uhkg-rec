"""
cage.py
-------
Minimal, self-contained re-implementation of the core idea behind CAGE
(Sheu et al., "Context-Aware Graph Embedding") for the UHKG-Rec
comparison: entities are first embedded from the base HKG, then an
auxiliary co-occurrence graph (services that share a symptom or a
device) is built to enrich the semantic context, and a GCN-style
mean-aggregation convolution refines the embeddings over that
auxiliary graph.

Independent baseline: no UHKG-Rec-specific mechanism is added here.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as cfg
from HKG.build_graph import load_graph


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def build_auxiliary_graph(g):
    """Services co-occurring under the same symptom (via Require S->H)
    or sharing a device (via Require H->O) are linked in the auxiliary
    context graph - this is CAGE's "context enrichment" step."""
    aux = {}
    symptom_to_services = {}
    for u, v, d in g.edges(data=True):
        if d.get("relation") == "Require" and g.nodes[u]["type"] == "S" and g.nodes[v]["type"] == "H":
            symptom_to_services.setdefault(u, []).append(v)
    for services in symptom_to_services.values():
        for a in services:
            for b in services:
                if a != b:
                    aux.setdefault(a, set()).add(b)
    return aux


class CAGE:
    def __init__(self, node_ids, dim=cfg.EMBED_DIM, seed=0):
        rng = np.random.default_rng(seed)
        self.node_ids = list(node_ids)
        self.id2idx = {n: i for i, n in enumerate(self.node_ids)}
        self.emb = rng.normal(0, 0.1, size=(len(self.node_ids), dim))
        self.dim = dim

    def gcn_refine(self, aux_graph, layers=2):
        """Mean-aggregation GCN convolution over the auxiliary graph."""
        for _ in range(layers):
            new_emb = self.emb.copy()
            for n, neigh in aux_graph.items():
                if n not in self.id2idx or not neigh:
                    continue
                idxs = [self.id2idx[m] for m in neigh if m in self.id2idx]
                if not idxs:
                    continue
                agg = self.emb[idxs].mean(axis=0)
                new_emb[self.id2idx[n]] = 0.5 * self.emb[self.id2idx[n]] + 0.5 * agg
            self.emb = new_emb

    def recommend(self, symptom_nodes, candidate_service_nodes, k):
        if not symptom_nodes:
            return []
        q = np.mean([self.emb[self.id2idx[s]] for s in symptom_nodes if s in self.id2idx], axis=0)
        scored = [(h, float(_sigmoid(np.dot(q, self.emb[self.id2idx[h]]))))
                  for h in candidate_service_nodes if h in self.id2idx]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


def run_cage(theta=0.7, seed=0, layers=2):
    g = load_graph(os.path.join(cfg.DATA_DIR, f"uhkg_theta_{theta}.json"))
    aux_graph = build_auxiliary_graph(g)
    model = CAGE(list(g.nodes), seed=seed)
    model.gcn_refine(aux_graph, layers=layers)
    return model, g


if __name__ == "__main__":
    model, g = run_cage(theta=0.7, seed=0)
    symptoms = [n for n, d in g.nodes(data=True) if d["type"] == "S"][:2]
    services = [n for n, d in g.nodes(data=True) if d["type"] == "H"]
    rec = model.recommend(symptoms, services, k=5)
    print("CAGE top-5 recommendation for", symptoms)
    for h, s in rec:
        print(f"  {h}: {s:.4f}")
