"""
lookups.py
----------
Builds the two dictionaries Algorithm 2's safety filter needs:
  - conflicts_by_service[h]  -> F^h: services incompatible with h
  - devices_by_service[h]    -> O^h: IoHT objects required by h
Both are keyed by plain service name (no "H::" prefix).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


def build_conflicts_by_service():
    with open(os.path.join(cfg.GEN_DIR, "conflicts.json")) as f:
        conflicts = json.load(f)
    out = {}
    for c in conflicts:
        a, b = c["service_a"], c["service_b"]
        out.setdefault(a, set()).add(b)
        out.setdefault(b, set()).add(a)
    return {k: sorted(v) for k, v in out.items()}


def build_devices_by_service():
    with open(os.path.join(cfg.GEN_DIR, "service_devices.json")) as f:
        return json.load(f)


if __name__ == "__main__":
    conf = build_conflicts_by_service()
    dev = build_devices_by_service()
    print(f"{len(conf)} services have at least one conflict")
    print(f"{len(dev)} services have device requirements")
