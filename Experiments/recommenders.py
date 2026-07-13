"""
recommenders.py
----------------
Builds ready-to-call `recommend(symptoms, k) -> [service_name, ...]`
functions for every method compared in the top-k recommendation
experiments: the UHKG-Rec family (HKG-Rec, UHKG-Rec, UHKG-Rec_NM) and
the two baselines (KGAT, CAGE). Shared by exp_topk_variants.py,
exp_topk_baselines.py, and exp_ablation.py so all three experiments
use exactly the same recommendation logic per method.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from HKG.build_graph import load_graph
from Matching.extraction import (requirement_patterns_for_symptoms, match_patterns,
                                  extract_services)
from Scoring.scoring import load_meta, rank_candidates
from Data.lookups import build_conflicts_by_service, build_devices_by_service
from Embedding.train import train_embedding
from Baselines.KGAT.kgat import KGAT
from Baselines.CAGE.cage import build_auxiliary_graph, CAGE


def _inject_predicted_treatment(g, treatment_facts):
    """Add classifier-predicted TREATMENT edges (per-seed, from Algorithm 1)
    on top of the mined ground-truth Treatment edges, so recommendations
    vary across seeds like the baselines do."""
    for fact in treatment_facts:
        h = fact["service"]
        sp = f"SP::{fact['predicted_pattern']}"
        if h in g and sp in g:
            g.add_edge(h, sp, relation="TREATMENT", confidence=fact["confidence"])


def uhkgrec_family_recommender(theta, use_confidence, use_pattern_matching, seed, epochs=2):
    """variant in {HKG-Rec, UHKG-Rec, UHKG-Rec_NM} depending on the flags:
    (use_confidence=False, use_pattern_matching=True)  -> HKG-Rec
    (use_confidence=True,  use_pattern_matching=True)  -> UHKG-Rec
    (use_confidence=True,  use_pattern_matching=False) -> UHKG-Rec_NM
    """
    g = load_graph(os.path.join(cfg.DATA_DIR, f"uhkg_theta_{theta}.json"))
    result = train_embedding(theta=theta, use_confidence=use_confidence, seed=seed,
                              epochs=epochs, verbose=False,
                              use_pattern_matching=use_pattern_matching)
    _inject_predicted_treatment(g, result["treatment_facts"])

    with open(os.path.join(cfg.DATA_DIR, "pattern_matrix.json")) as f:
        matrix_info = json.load(f)
    conflicts_by_service = build_conflicts_by_service()
    devices_by_service = build_devices_by_service()
    service_meta, device_meta = load_meta()

    def recommend(symptoms, k):
        symptom_nodes = [f"S::{s}" for s in symptoms if f"S::{s}" in g]
        s_p = requirement_patterns_for_symptoms(g, symptom_nodes)
        p_h = match_patterns(s_p, matrix_info, None) if use_pattern_matching else set()
        h_p = extract_services(p_h, g, [], conflicts_by_service, devices_by_service,
                                use_pattern_matching=use_pattern_matching,
                                symptom_nodes=symptom_nodes)
        names = sorted({h.split("::", 1)[-1] for cands in h_p.values() for h in cands})
        ranked = rank_candidates(names, devices_by_service, service_meta, device_meta, k)
        return [h for h, _ in ranked]

    return recommend


def kgat_recommender(theta, seed):
    g = load_graph(os.path.join(cfg.DATA_DIR, f"uhkg_theta_{theta}.json"))
    model = KGAT(list(g.nodes), seed=seed)
    model.propagate(g, layers=2)
    services = [n for n, d in g.nodes(data=True) if d["type"] == "H"]

    def recommend(symptoms, k):
        symptom_nodes = [f"S::{s}" for s in symptoms if f"S::{s}" in g]
        rec = model.recommend(g, symptom_nodes, services, k)
        return [h.split("::", 1)[-1] for h, _ in rec]

    return recommend


def cage_recommender(theta, seed):
    g = load_graph(os.path.join(cfg.DATA_DIR, f"uhkg_theta_{theta}.json"))
    aux = build_auxiliary_graph(g)
    model = CAGE(list(g.nodes), seed=seed)
    model.gcn_refine(aux, layers=2)
    services = [n for n, d in g.nodes(data=True) if d["type"] == "H"]

    def recommend(symptoms, k):
        symptom_nodes = [f"S::{s}" for s in symptoms if f"S::{s}" in g]
        rec = model.recommend(symptom_nodes, services, k)
        return [h.split("::", 1)[-1] for h, _ in rec]

    return recommend
