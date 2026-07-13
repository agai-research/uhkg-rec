"""
UHKG-Rec.py
-----------
Main entry point: runs the complete UHKG-Rec recommendation pipeline
for a single patient query.

Query format (JSON, see Test/sample_query.json):
{
  "symptoms": ["Fatigue", "Headache"],
  "active_constraints": ["Metformin"],
  "k": 5,
  "theta": 0.7,
  "variant": "UHKG-Rec"      # "UHKG-Rec" | "UHKG-Rec_NM" | "HKG-Rec"
}

Usage:
  python UHKG-Rec.py Test/sample_query.json
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from HKG.build_graph import load_graph
from Matching.extraction import (requirement_patterns_for_symptoms, match_patterns,
                                  extract_services)
from Scoring.scoring import load_meta, rank_candidates
from Data.lookups import build_conflicts_by_service, build_devices_by_service


def run_query(query, verbose=True):
    theta = query.get("theta", 0.7)
    variant = query.get("variant", "UHKG-Rec")
    k = query.get("k", 5)
    use_pattern_matching = variant != "UHKG-Rec_NM"

    g = load_graph(os.path.join(cfg.DATA_DIR, f"uhkg_theta_{theta}.json"))
    with open(os.path.join(cfg.DATA_DIR, "pattern_matrix.json")) as f:
        matrix_info = json.load(f)

    # resolve symptom names -> node ids present in the graph
    symptom_nodes = [f"S::{s}" for s in query["symptoms"] if f"S::{s}" in g]
    if verbose and len(symptom_nodes) < len(query["symptoms"]):
        missing = set(query["symptoms"]) - {n.split("::")[1] for n in symptom_nodes}
        print(f"  (note: symptoms not found in graph, skipped: {sorted(missing)})")

    s_p = requirement_patterns_for_symptoms(g, symptom_nodes)
    p_h = match_patterns(s_p, matrix_info, None) if use_pattern_matching else set()

    conflicts_by_service = build_conflicts_by_service()
    devices_by_service = build_devices_by_service()

    h_p = extract_services(p_h, g, query.get("active_constraints", []),
                            conflicts_by_service, devices_by_service,
                            use_pattern_matching=use_pattern_matching,
                            symptom_nodes=symptom_nodes)

    service_meta, device_meta = load_meta()
    plan = []
    for pattern_id, candidates in h_p.items():
        names = [h.split("::", 1)[-1] for h in candidates]
        ranked = rank_candidates(names, devices_by_service, service_meta, device_meta, k)
        plan.append({"matched_pattern": pattern_id,
                      "recommended_services": [{"service": h, "score": round(sc, 2)}
                                                for h, sc in ranked]})

    result = {"query": query, "variant": variant, "theta": theta,
              "requirement_patterns": sorted(s_p), "matched_service_patterns": sorted(p_h),
              "recommendation": plan}
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            query = json.load(f)
    else:
        query = {"symptoms": ["Fatigue", "Headache"], "active_constraints": [],
                  "k": 5, "theta": 0.7, "variant": "UHKG-Rec"}

    result = run_query(query)
    print(json.dumps(result, indent=2))

    out_path = os.path.join(cfg.ROOT_DIR, "Experiments", "results", "last_query_result.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved result to {out_path}")
