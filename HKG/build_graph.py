"""
build_graph.py
---------------
Builds the base Healthcare Knowledge Graph (HKG) as a typed NetworkX
MultiDiGraph, populated from the dataset produced by
Data/generate_dataset.py.

Node types (Appendix A.1): P (patient), S (symptom), H (healthcare
service), O (IoHT object). RP/SP nodes are added later by
Matching/pattern_mining.py, once requirement and service patterns have
been mined - that step turns this base HKG into the full UHKG.

Each node id is prefixed by its type ("S::Fatigue", "H::Metformin
Therapy", ...) to keep the namespace collision-free across types.

Edges carry a `confidence` attribute (Eq. B.1/B.2). When
`use_confidence=False` (HKG-Rec variant), confidence is still stored
for reference but the embedding/walker module ignores it.
"""

import json
import os
import sys

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


def _load(path):
    with open(path) as f:
        return json.load(f)


def build_base_hkg():
    entities = _load(os.path.join(cfg.RAW_DIR, "entities.json"))
    facts = _load(os.path.join(cfg.RAW_DIR, "facts.json"))
    interactions = _load(os.path.join(cfg.GEN_DIR, "interactions.json"))

    g = nx.MultiDiGraph()

    # --- nodes -------------------------------------------------------
    for s in entities["symptoms"]:
        g.add_node(f"S::{s}", type="S", name=s)
    for h in entities["services"]:
        g.add_node(f"H::{h}", type="H", name=h)
    for o in entities["devices"]:
        g.add_node(f"O::{o}", type="O", name=o)
    for p in entities["patients"]:
        g.add_node(f"P::{p}", type="P", name=p)

    # --- Have edges: Patient -> Symptom, from interaction records -----
    seen_have = set()
    for row in interactions:
        p_id = f"P::{row['patient']}"
        for s in row["symptoms"]:
            s_id = f"S::{s}"
            key = (p_id, s_id)
            if key in seen_have:
                continue
            seen_have.add(key)
            g.add_edge(p_id, s_id, relation="Have", confidence=1.0)  # observed, not uncertain

    # --- Require edges: Symptom -> Service, Service -> Object ---------
    for fact in facts:
        if fact["relation"] != "Require":
            continue
        head, tail, c = fact["head"], fact["tail"], fact["confidence"]
        head_type = "S" if f"S::{head}" in g else ("H" if f"H::{head}" in g else None)
        tail_type = "H" if f"H::{tail}" in g else ("O" if f"O::{tail}" in g else None)
        if head_type is None or tail_type is None:
            continue
        g.add_edge(f"{head_type}::{head}", f"{tail_type}::{tail}",
                    relation="Require", confidence=c)

    return g


def save_graph(g, path):
    """Save the graph structure as a plain JSON edge/node list (portable,
    human-readable file representing the HKG structure)."""
    data = {
        "nodes": [{"id": n, **g.nodes[n]} for n in g.nodes],
        "edges": [{"source": u, "target": v, **d} for u, v, d in g.edges(data=True)],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_graph(path):
    with open(path) as f:
        data = json.load(f)
    g = nx.MultiDiGraph()
    for n in data["nodes"]:
        nid = n.pop("id")
        g.add_node(nid, **n)
    for e in data["edges"]:
        u, v = e.pop("source"), e.pop("target")
        g.add_edge(u, v, **e)
    return g


if __name__ == "__main__":
    graph = build_base_hkg()
    out_path = os.path.join(cfg.DATA_DIR, "base_hkg.json")
    save_graph(graph, out_path)
    print(f"Base HKG built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Saved to {out_path}")
