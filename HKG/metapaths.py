"""
metapaths.py
------------
Confidence-weighted, meta-path-guided random walk generator (a
metapath2vec variant tailored to the UHKG, not the generic library
implementation - see Eq. B.3/B.4 of the implementation appendix and
the 7 schemes in config.METAPATHS / Appendix A.3).

At each step, the walker is at a node of a given type and must move to
the next type prescribed by the current meta-path scheme, following
the scheme's relation (or its inverse, marked with a "-1" suffix). The
next node is drawn from all valid neighbors with probability
proportional to their RBF-transformed confidence (Eq. B.4), i.e. more
reliable facts are preferred but not the only option.

When `use_confidence=False` (HKG-Rec, the deterministic ablation
variant), all valid neighbors get equal weight -> a uniform walk.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from HKG.confidence import rbf_confidence


def _neighbors_by_relation(g, node, relation, reverse):
    """All (neighbor, confidence) pairs reachable from `node` via `relation`,
    walking edges backward if `reverse` is True (the "-1" meta-path step)."""
    out = []
    if not reverse:
        for _, v, d in g.out_edges(node, data=True):
            if d.get("relation") == relation:
                out.append((v, d.get("confidence", 1.0)))
    else:
        for u, _, d in g.in_edges(node, data=True):
            if d.get("relation") == relation:
                out.append((u, d.get("confidence", 1.0)))
    return out


def _pick_next(candidates, use_confidence, sigma2=cfg.RBF_SIGMA2):
    """Eq. B.3: sample the next node with probability proportional to its
    RBF-transformed confidence weight (Eq. B.4); uniform if
    use_confidence is False (HKG-Rec)."""
    if not candidates:
        return None
    if not use_confidence:
        weights = [1.0 for _ in candidates]
    else:
        weights = [rbf_confidence(c, sigma2) for _, c in candidates]
    total = sum(weights)
    if total <= 0:
        return random.choice(candidates)[0]
    r = random.uniform(0, total)
    upto = 0.0
    for (node, _), w in zip(candidates, weights):
        upto += w
        if upto >= r:
            return node
    return candidates[-1][0]


def walk_from(g, start_node, scheme, use_confidence=True, walk_length=cfg.WALK_LENGTH):
    """Generate one walk starting at `start_node`, cycling through the
    meta-path scheme's relation sequence until `walk_length` nodes have
    been visited (or a step has no valid neighbor)."""
    node_types, relations = scheme
    path = [start_node]
    rel_idx = 0
    current = start_node
    while len(path) < walk_length:
        rel = relations[rel_idx % len(relations)]
        reverse = rel.endswith("-1")
        rel_name = rel[:-2] if reverse else rel
        candidates = _neighbors_by_relation(g, current, rel_name, reverse)
        nxt = _pick_next(candidates, use_confidence)
        if nxt is None:
            break
        path.append(nxt)
        current = nxt
        rel_idx += 1
    return path


def generate_walks(g, use_confidence=True, walks_per_node=cfg.WALKS_PER_NODE,
                    walk_length=cfg.WALK_LENGTH, schemes=cfg.METAPATHS, seed=0,
                    use_pattern_matching=True):
    """Run every meta-path scheme starting from each node of the scheme's
    first type, `walks_per_node` times. Returns a list of node-id walks.

    When use_pattern_matching=False (UHKG-Rec_NM), meta-path schemes that
    traverse the Match relation (i.e. that encode bilateral requirement/
    service pattern alignment) are excluded, so the resulting embedding
    does not encode the pattern-matching mechanism being ablated."""
    if not use_pattern_matching:
        schemes = [(nt, rel) for nt, rel in schemes
                   if not any(r.startswith("Match") for r in rel)]

    rng = random.Random(seed)
    random.seed(seed)
    walks = []
    for node_types, relations in schemes:
        start_type = node_types[0]
        starts = [n for n, d in g.nodes(data=True) if d.get("type") == start_type]
        for n in starts:
            for _ in range(walks_per_node):
                w = walk_from(g, n, (node_types, relations), use_confidence, walk_length)
                if len(w) >= 2:
                    walks.append(w)
    rng.shuffle(walks)
    return walks


def context_pairs(walks, window=1):
    """Turn walks into (center, context) Skip-Gram pairs, i.e. the
    (v_t^i, v^{i+1}) pairs used in Eq. B.5's embedding loss."""
    pairs = []
    for w in walks:
        for i in range(len(w) - 1):
            for j in range(1, window + 1):
                if i + j < len(w):
                    pairs.append((w[i], w[i + j]))
    return pairs


if __name__ == "__main__":
    from HKG.build_graph import load_graph

    g = load_graph(os.path.join(cfg.DATA_DIR, "uhkg_theta_0.7.json"))
    walks = generate_walks(g, use_confidence=True, walks_per_node=1, walk_length=5)
    pairs = context_pairs(walks)
    print(f"generated {len(walks)} walks, {len(pairs)} context pairs")
    print("example walk:", walks[0] if walks else None)
