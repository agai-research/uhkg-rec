"""
eval_utils.py
-------------
Shared evaluation harness for the top-k recommendation experiments
(Precision@k, Recall@k, F1-Score@k, MRR@k), using the synthetic
patient-service interactions (Data/generated/interactions.json) as
the recommendation ground truth, and a paired significance test
helper (scipy.stats.ttest_rel).
"""

import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


def load_patient_ground_truth(max_patients=60):
    """Group interactions by patient -> (symptom list, set of positively
    labeled service names). Capped to keep experiment runtime reasonable."""
    with open(os.path.join(cfg.GEN_DIR, "interactions.json")) as f:
        rows = json.load(f)
    by_patient = {}
    for r in rows:
        entry = by_patient.setdefault(r["patient"], {"symptoms": r["symptoms"], "positive": set()})
        if r["label"] == 1:
            entry["positive"].add(r["service"])
    items = [(p, v) for p, v in by_patient.items() if v["positive"]]
    items = items[:max_patients]
    return items


def precision_recall_f1_mrr_at_k(recommended, relevant, k):
    top_k = recommended[:k]
    hits = [1 if h in relevant else 0 for h in top_k]
    precision = sum(hits) / k if k > 0 else 0.0
    recall = sum(hits) / len(relevant) if relevant else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mrr = 0.0
    for rank, h in enumerate(top_k, start=1):
        if h in relevant:
            mrr = 1.0 / rank
            break
    return precision, recall, f1, mrr


def evaluate_method(recommend_fn, patients, k_values):
    """recommend_fn(symptoms, k) -> ordered list of plain service names."""
    per_k = {k: {"precision": [], "recall": [], "f1": [], "mrr": []} for k in k_values}
    for _, entry in patients:
        max_k = max(k_values)
        ranked = recommend_fn(entry["symptoms"], max_k)
        for k in k_values:
            p, r, f1, mrr = precision_recall_f1_mrr_at_k(ranked, entry["positive"], k)
            per_k[k]["precision"].append(p)
            per_k[k]["recall"].append(r)
            per_k[k]["f1"].append(f1)
            per_k[k]["mrr"].append(mrr)
    summary = {}
    for k, metrics in per_k.items():
        summary[k] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                       for m, v in metrics.items()}
        summary[k]["_raw_f1"] = metrics["f1"]  # kept for paired significance tests
    return summary


def paired_test(raw_a, raw_b):
    n = min(len(raw_a), len(raw_b))
    if n < 2:
        return None
    t_stat, p_val = stats.ttest_rel(raw_a[:n], raw_b[:n])
    if not np.isfinite(t_stat) or not np.isfinite(p_val):
        return {"test": "paired t-test (matched patients)", "t_stat": None, "p_value": None,
                "significant_at_0.05": False,
                "note": "identical values across seeds - variance is zero, test undefined"}
    return {"test": "paired t-test (matched patients)", "t_stat": float(t_stat),
            "p_value": float(p_val), "significant_at_0.05": bool(p_val < cfg.SIG_ALPHA)}
