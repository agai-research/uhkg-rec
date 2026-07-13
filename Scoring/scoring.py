"""
scoring.py
----------
Implements Eq. B.10 (score(h), score(W^p)) and the ranking / Top-k
selection step described in Section "Scoring and ranking". Lower
scores are preferred (cost- and time-oriented), so ranking is
ascending.

Note: alpha_time_weight / beta_cost_weight here are distinct from
ALPHA_LOSS (the uncertainty-loss weight of Eq. B.9) - see config.py.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


def load_meta():
    with open(os.path.join(cfg.GEN_DIR, "service_meta.json")) as f:
        service_meta = json.load(f)
    with open(os.path.join(cfg.GEN_DIR, "device_meta.json")) as f:
        device_meta = json.load(f)
    return service_meta, device_meta


def service_score(service_name, devices_by_service, service_meta, device_meta,
                   alpha_w=cfg.ALPHA_TIME_WEIGHT, beta_w=cfg.BETA_COST_WEIGHT):
    """Eq. B.10 - score(h) = alpha*T_h + beta*(cost(h) + sum cost(o))."""
    meta = service_meta.get(service_name, {"recovery_days": 10.0, "cost": 100.0})
    t_h = meta["recovery_days"]
    cost_h = meta["cost"]
    device_cost = sum(device_meta.get(o, {"cost": 50.0})["cost"]
                       for o in devices_by_service.get(service_name, []))
    return alpha_w * t_h + beta_w * (cost_h + device_cost)


def plan_score(service_names, devices_by_service, service_meta, device_meta,
               alpha_w=cfg.ALPHA_TIME_WEIGHT, beta_w=cfg.BETA_COST_WEIGHT):
    """Eq. B.10 - score(W^p), summed over the services of a treatment plan."""
    return sum(service_score(h, devices_by_service, service_meta, device_meta,
                              alpha_w, beta_w) for h in service_names)


def rank_candidates(candidates, devices_by_service, service_meta, device_meta, k):
    """Rank candidate services (plain names, not node ids) ascending by
    score and return the top-k, each with its score."""
    scored = [(h, service_score(h, devices_by_service, service_meta, device_meta))
              for h in candidates]
    scored.sort(key=lambda x: x[1])
    return scored[:k]


if __name__ == "__main__":
    service_meta, device_meta = load_meta()
    with open(os.path.join(cfg.GEN_DIR, "service_devices.json")) as f:
        devices_by_service = json.load(f)
    sample = list(service_meta.keys())[:5]
    ranked = rank_candidates(sample, devices_by_service, service_meta, device_meta, k=3)
    print("Top-3 among 5 sample services:")
    for h, s in ranked:
        print(f"  {h}: score={s:.2f}")
