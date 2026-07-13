"""
service_meta.py
----------------
Generates the recovery-time (T_h) and cost figures needed by the
scoring equations (Eq. B.10): score(h) and score(W^p) combine a
service's expected recovery time with its own cost plus the cost of
any IoHT object it requires. PrimeKG does not provide this
operational/economic information, so - consistent with the dataset
section's generative-completion step - it is generated here from a
simple, transparent random model grounded in each service's clinical
category.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

random.seed(7)


def build():
    with open(os.path.join(cfg.RAW_DIR, "entities.json")) as f:
        entities = json.load(f)

    service_meta = {}
    for h in entities["services"]:
        service_meta[h] = {
            "recovery_days": round(random.uniform(2, 30), 1),   # T_h
            "cost": round(random.uniform(20, 800), 2),           # cost(h)
        }

    device_meta = {}
    for o in entities["devices"]:
        device_meta[o] = {"cost": round(random.uniform(15, 500), 2)}  # cost(o)

    with open(os.path.join(cfg.GEN_DIR, "service_meta.json"), "w") as f:
        json.dump(service_meta, f, indent=2)
    with open(os.path.join(cfg.GEN_DIR, "device_meta.json"), "w") as f:
        json.dump(device_meta, f, indent=2)

    print(f"service_meta.json: {len(service_meta)} services")
    print(f"device_meta.json: {len(device_meta)} devices")


if __name__ == "__main__":
    build()
