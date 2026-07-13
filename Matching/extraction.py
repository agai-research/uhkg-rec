"""
extraction.py
-------------
Implements Algorithm 2 (Appendix C.2): bilateral requirement/service
pattern matching (`match_patterns`) followed by safety-filtered
candidate-service extraction (`extract_services`).

Implemented iteratively (not recursively), per the implementation
prompt's guidance, to avoid Python recursion-depth issues.

use_pattern_matching=False switches to the UHKG-Rec_NM ablation
variant: instead of using the correlation matrix M^rh to pick the
single best-matching service pattern per requirement pattern, it
retrieves candidate services directly via the symptom->service
`Require` edges already present in the base HKG, bypassing the
RP/SP correlation step entirely - this is what "no bilateral
matching" means in this implementation (see Appendix A.4 of the
implementation prompt for the rationale).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def requirement_patterns_for_symptoms(g, symptom_nodes):
    """S_p: the set of requirement patterns a query's symptoms map to,
    via Cause edges (S -> RP)."""
    rps = set()
    for s in symptom_nodes:
        for _, v, d in g.out_edges(s, data=True):
            if d.get("relation") == "Cause" and g.nodes[v]["type"] == "RP":
                rps.add(v)
    return rps


def match_patterns(s_p, matrix_info, matrix_node_lookup):
    """
    MatchPatterns(S_p, M) - Appendix C.2, iterative version.
    Returns P^h: the set of service-pattern node ids best matching S_p.
    """
    m = matrix_info["matrix"]
    rp_ids = matrix_info["rp_ids"]
    sp_ids = matrix_info["sp_ids"]
    rp_index = {f"RP::{r}": i for i, r in enumerate(rp_ids)}

    p_h = set()
    remaining = list(s_p)
    while remaining:
        p_r = remaining.pop(0)
        if p_r not in rp_index:
            continue
        i = rp_index[p_r]
        row = m[i]
        best_j = max(range(len(row)), key=lambda j: row[j])
        sp_node = f"SP::{sp_ids[best_j]}"
        if sp_node not in p_h:
            p_h.add(sp_node)
    return p_h


def extract_services(p_h, g, active_constraints, conflicts_by_service,
                      devices_by_service, use_pattern_matching=True,
                      symptom_nodes=None):
    """
    ExtractServices(P^h, G, A) - Appendix C.2, iterative version.
    Applies the corrected safety filter (Eq. B.11, logical AND):
        h retained  <=>  A ∩ F^h = empty  AND  A ∩ O^h = empty
    """
    h_p = {}

    if use_pattern_matching:
        candidate_source = p_h
        for sp_node in candidate_source:
            candidates = []
            for u, v, d in g.in_edges(sp_node, data=True):
                if d.get("relation") in ("Treatment", "TREATMENT") and g.nodes[u]["type"] == "H":
                    candidates.append(u)
            h_p[sp_node] = _safety_filter(candidates, active_constraints,
                                           conflicts_by_service, devices_by_service)
    else:
        # UHKG-Rec_NM: bypass RP/SP correlation, use direct S->H Require edges
        candidates = set()
        for s in (symptom_nodes or []):
            for _, v, d in g.out_edges(s, data=True):
                if d.get("relation") == "Require" and g.nodes[v]["type"] == "H":
                    candidates.add(v)
        h_p["direct"] = _safety_filter(list(candidates), active_constraints,
                                        conflicts_by_service, devices_by_service)
    return h_p


def _safety_filter(candidates, active_constraints, conflicts_by_service, devices_by_service):
    kept = []
    a = set(active_constraints or [])
    for h in candidates:
        name = h.split("::", 1)[-1]
        f_h = set(conflicts_by_service.get(name, []))       # contraindicated items
        o_h = set(devices_by_service.get(name, []))          # required IoHT objects
        if (a & f_h) or (a & o_h):
            continue  # unsafe: violates at least one constraint -> excluded
        kept.append(h)
    return kept


if __name__ == "__main__":
    import json
    import config as cfg
    from HKG.build_graph import load_graph

    g = load_graph(os.path.join(cfg.DATA_DIR, "uhkg_theta_0.7.json"))
    with open(os.path.join(cfg.DATA_DIR, "pattern_matrix.json")) as f:
        matrix_info = json.load(f)

    # smoke test: scan symptoms until one with a surviving Cause edge is found
    # (some are filtered out at theta=0.7 if their cluster-membership
    # confidence falls below the threshold - expected behavior)
    for sample_symptom in (n for n, d in g.nodes(data=True) if d["type"] == "S"):
        s_p = requirement_patterns_for_symptoms(g, [sample_symptom])
        if s_p:
            p_h = match_patterns(s_p, matrix_info, None)
            print(f"symptom={sample_symptom}  S_p={s_p}  P^h={p_h}")
            break
    else:
        print("no symptom retained a Cause edge at this threshold")
