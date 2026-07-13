"""
pattern_mining.py
------------------
Mines requirement patterns (RP) and service patterns (SP) from the base
HKG, following Section "Pattern mining of healthcare requirements and
services":

  - Requirement patterns group symptoms that tend to co-occur in the
    same patient profile (clustering over a symptom co-occurrence
    matrix built from the `Have` edges).
  - Service patterns group healthcare services that address overlapping
    symptom sets (clustering over the service x symptom incidence
    matrix built from the `Require` S->H edges).

For each symptom/service, cluster membership is turned into a
confidence-scored `Cause` / `Treatment` fact (confidence = similarity
to the cluster centroid, i.e. how representative the entity is of its
mined pattern).

The correlation matrix M^rh (Table "Pattern matrices") is computed as
the Jaccard overlap between an RP's symptom set and the symptom set
addressed by an SP's services (PatternMatch(pr_i, ph_j), Section
"Requirement-Service Pattern Matching"), and materialized as `Match`
edges between the best-aligned RP/SP pairs.

Finally, three threshold-filtered views of the resulting UHKG are
written (theta in {0.6, 0.7, 0.8}), keeping only Cause/Treatment/Match
facts whose confidence is >= theta - this is what Appendix A of the
implementation prompt calls "different HKG configurations per
threshold".
"""

import json
import os
import sys

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from HKG.build_graph import load_graph, save_graph


def _symptom_cooccurrence_features(g, symptoms):
    """Row i = symptom i's co-occurrence vector with every other symptom,
    counted across patients that Have both symptoms."""
    idx = {s: i for i, s in enumerate(symptoms)}
    mat = np.zeros((len(symptoms), len(symptoms)))
    patient_symptoms = {}
    for u, v, d in g.edges(data=True):
        if d.get("relation") == "Have":
            patient_symptoms.setdefault(u, []).append(v)
    for p, syms in patient_symptoms.items():
        names = [g.nodes[s]["name"] for s in syms if s in idx or g.nodes[s]["name"] in idx]
        ids = [idx[g.nodes[s]["name"]] for s in syms if g.nodes[s]["name"] in idx]
        for a in ids:
            for b in ids:
                mat[a, b] += 1
    return mat


def _service_symptom_features(g, services, symptoms):
    """Row i = which symptoms service i treats (via Require S->H edges)."""
    s_idx = {s: i for i, s in enumerate(symptoms)}
    h_idx = {h: i for i, h in enumerate(services)}
    mat = np.zeros((len(services), len(symptoms)))
    for u, v, d in g.edges(data=True):
        if d.get("relation") == "Require" and g.nodes[u]["type"] == "S" and g.nodes[v]["type"] == "H":
            s_name, h_name = g.nodes[u]["name"], g.nodes[v]["name"]
            if s_name in s_idx and h_name in h_idx:
                mat[h_idx[h_name], s_idx[s_name]] = 1
    return mat


def _cluster_relative_confidence(labels, sim_to_own_centroid, lo=0.15, hi=0.99):
    """Min-max normalize each entity's similarity to its own cluster
    centroid, within that cluster, then rescale to [lo, hi]. This
    spreads confidence values more informatively across the 0.6-0.8
    threshold range than the raw cosine similarity, which tends to be
    uniformly low for sparse, high-dimensional co-occurrence features."""
    conf = np.zeros(len(labels))
    for k in set(labels):
        idx = np.where(labels == k)[0]
        vals = sim_to_own_centroid[idx]
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < 1e-9:
            conf[idx] = hi
        else:
            conf[idx] = lo + (vals - vmin) / (vmax - vmin) * (hi - lo)
    return conf


def mine_patterns(g):
    symptoms = [n for n in g.nodes if g.nodes[n]["type"] == "S"]
    services = [n for n in g.nodes if g.nodes[n]["type"] == "H"]
    symptom_names = [g.nodes[n]["name"] for n in symptoms]
    service_names = [g.nodes[n]["name"] for n in services]

    # ---- requirement patterns (cluster symptoms) ----------------------
    sym_feats = _symptom_cooccurrence_features(g, symptom_names)
    n_rp = min(cfg.N_REQUIREMENT_PATTERNS, len(symptoms))
    km_s = KMeans(n_clusters=n_rp, n_init=5, random_state=0).fit(sym_feats)
    sym_labels = km_s.labels_
    sym_centroids = km_s.cluster_centers_
    sym_sim = cosine_similarity(sym_feats, sym_centroids)

    # ---- service patterns (cluster services) ---------------------------
    svc_feats = _service_symptom_features(g, service_names, symptom_names)
    n_sp = min(cfg.N_SERVICE_PATTERNS, len(services))
    km_h = KMeans(n_clusters=n_sp, n_init=5, random_state=0).fit(svc_feats)
    svc_labels = km_h.labels_
    svc_centroids = km_h.cluster_centers_
    svc_sim = cosine_similarity(svc_feats, svc_centroids)

    # ---- add RP / SP nodes + Cause / Treatment edges -------------------
    for k in range(n_rp):
        g.add_node(f"RP::RP_{k}", type="RP", name=f"RP_{k}")
    for k in range(n_sp):
        g.add_node(f"SP::SP_{k}", type="SP", name=f"SP_{k}")

    sym_own_sim = sym_sim[np.arange(len(symptoms)), sym_labels]
    sym_conf = _cluster_relative_confidence(sym_labels, sym_own_sim)
    for i, s_node in enumerate(symptoms):
        k = sym_labels[i]
        g.add_edge(s_node, f"RP::RP_{k}", relation="Cause", confidence=round(float(sym_conf[i]), 3))

    svc_own_sim = svc_sim[np.arange(len(services)), svc_labels]
    svc_conf = _cluster_relative_confidence(svc_labels, svc_own_sim)
    for i, h_node in enumerate(services):
        k = svc_labels[i]
        g.add_edge(h_node, f"SP::SP_{k}", relation="Treatment", confidence=round(float(svc_conf[i]), 3))

    # ---- correlation matrix M^rh: PatternMatch(RP_i, SP_j) --------------
    rp_symptom_sets = {k: set() for k in range(n_rp)}
    for i, s_node in enumerate(symptoms):
        rp_symptom_sets[sym_labels[i]].add(symptom_names[i])

    sp_symptom_sets = {k: set() for k in range(n_sp)}
    for i, h_node in enumerate(services):
        treated = {symptom_names[j] for j in range(len(symptom_names)) if svc_feats[i, j] > 0}
        sp_symptom_sets[svc_labels[i]] |= treated

    m_rh = np.zeros((n_rp, n_sp))
    for i in range(n_rp):
        a = rp_symptom_sets[i]
        for j in range(n_sp):
            b = sp_symptom_sets[j]
            inter = len(a & b)
            union = len(a | b) or 1
            m_rh[i, j] = inter / union  # Jaccard = PatternMatch(pr_i, ph_j)

    # ---- Match edges: for each RP, link to its best-matching SP ---------
    row_max = m_rh.max(axis=1)
    row_min = m_rh.min(axis=1)
    for i in range(n_rp):
        j = int(np.argmax(m_rh[i]))
        spread = row_max[i] - row_min[i]
        if spread < 1e-9:
            conf = 0.99
        else:
            conf = 0.15 + (m_rh[i, j] - row_min[i]) / spread * (0.99 - 0.15)
        g.add_edge(f"RP::RP_{i}", f"SP::SP_{j}", relation="Match", confidence=round(float(conf), 3))

    matrix_info = {"n_rp": n_rp, "n_sp": n_sp, "matrix": m_rh.tolist(),
                    "rp_ids": [f"RP_{i}" for i in range(n_rp)],
                    "sp_ids": [f"SP_{j}" for j in range(n_sp)]}
    return g, matrix_info


def save_theta_views(g, theta_list, out_dir):
    """Write one filtered UHKG per confidence threshold theta: keep P/S/H/O
    nodes and Have/Require edges always; keep Cause/Treatment/Match edges
    only if confidence >= theta."""
    for theta in theta_list:
        view = nx.MultiDiGraph()
        for n, d in g.nodes(data=True):
            view.add_node(n, **d)
        for u, v, d in g.edges(data=True):
            if d["relation"] in ("Cause", "Treatment", "Match") and d["confidence"] < theta:
                continue
            view.add_edge(u, v, **d)
        path = os.path.join(out_dir, f"uhkg_theta_{theta}.json")
        save_graph(view, path)
        print(f"  theta={theta}: {view.number_of_nodes()} nodes, "
              f"{view.number_of_edges()} edges -> {path}")


if __name__ == "__main__":
    base = load_graph(os.path.join(cfg.DATA_DIR, "base_hkg.json"))
    uhkg, matrix_info = mine_patterns(base)

    full_path = os.path.join(cfg.DATA_DIR, "uhkg_full.json")
    save_graph(uhkg, full_path)
    with open(os.path.join(cfg.DATA_DIR, "pattern_matrix.json"), "w") as f:
        json.dump(matrix_info, f, indent=2)

    print(f"UHKG mined: {uhkg.number_of_nodes()} nodes, {uhkg.number_of_edges()} edges")
    print(f"  RP={matrix_info['n_rp']}, SP={matrix_info['n_sp']}")
    print(f"Saved full UHKG to {full_path}")
    print("Writing threshold-filtered views:")
    save_theta_views(uhkg, cfg.THETAS, cfg.DATA_DIR)

    # update dataset_stats.json with the mined pattern counts (N_rp, N_sp)
    stats_path = os.path.join(cfg.DATA_DIR, "dataset_stats.json")
    with open(stats_path) as f:
        stats = json.load(f)
    stats["N_rp"] = matrix_info["n_rp"]
    stats["N_sp"] = matrix_info["n_sp"]
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("Updated dataset_stats.json with N_rp / N_sp")
