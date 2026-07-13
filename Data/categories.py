"""
categories.py
-------------
Assigns each healthcare service (drug) a single biomedical category,
used purely for coloring the embedding-visualization figure
(Experiments/exp_visualization.py). A service's category is the
specialty of the (first) disease it treats, remapped onto five
canonical labels comparable to the paper's figure (Cancer, Diabetes,
Dermatology, Allergy, Respiratory). If a canonical specialty is
under-represented in the generated dataset, the next most frequent
specialty is used in its place, keeping exactly five categories.
"""

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

CANONICAL_MAP = {
    "Oncology": "Cancer",
    "Endocrinology": "Diabetes",
    "Dermatology": "Dermatology",
    "Immunology": "Allergy",
    "Pulmonology": "Respiratory",
}


def build_service_categories(min_per_category=5):
    with open(os.path.join(cfg.RAW_DIR, "entities.json")) as f:
        entities = json.load(f)
    with open(os.path.join(cfg.GEN_DIR, "drug_disease.json")) as f:
        drug_disease = json.load(f)

    disease_specialty = entities["disease_specialty"]
    svc_specialty = {}
    for drug, diseases in drug_disease.items():
        if diseases:
            svc_specialty[drug] = disease_specialty[diseases[0]]

    counts = Counter(svc_specialty.values())

    # start from the canonical 5, keep those with enough members
    chosen = {sp: label for sp, label in CANONICAL_MAP.items() if counts.get(sp, 0) >= min_per_category}
    # fill any missing slot with the next most frequent specialty not yet used
    remaining_labels = [l for l in CANONICAL_MAP.values() if l not in chosen.values()]
    for sp, _ in counts.most_common():
        if len(chosen) >= 5:
            break
        if sp in chosen or sp in CANONICAL_MAP:
            continue
        if remaining_labels:
            chosen[sp] = remaining_labels.pop(0)

    svc_category = {drug: chosen[sp] for drug, sp in svc_specialty.items() if sp in chosen}

    with open(os.path.join(cfg.GEN_DIR, "service_categories.json"), "w") as f:
        json.dump({"specialty_to_label": chosen, "service_category": svc_category}, f, indent=2)

    print("Category mapping used for the embedding-visualization figure:")
    for sp, label in chosen.items():
        print(f"  {sp} -> {label}  ({counts.get(sp, 0)} services)")
    print(f"{len(svc_category)} services assigned a category "
          f"(out of {len(svc_specialty)} services with a known disease)")
    return chosen, svc_category


if __name__ == "__main__":
    build_service_categories()
