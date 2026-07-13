"""
first_test.py
--------------
Dedicated smoke test for the UHKG-Rec prototype: makes sure the
dataset, HKG, and recommendation pipeline all work together end to
end, using a handful of sample patient queries. Run this first after
setting up the repository, before running the full experiments.

Usage:
  python Test/first_test.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config as cfg

SAMPLE_QUERIES = [
    {"symptoms": ["Fatigue", "Headache"], "active_constraints": [], "k": 5,
     "theta": 0.7, "variant": "UHKG-Rec"},
    {"symptoms": ["Cough", "ShortnessOfBreath"], "active_constraints": [], "k": 3,
     "theta": 0.6, "variant": "UHKG-Rec"},
    {"symptoms": ["JointPain"], "active_constraints": [], "k": 5,
     "theta": 0.7, "variant": "UHKG-Rec_NM"},
    {"symptoms": ["Nausea", "Dizziness"], "active_constraints": [], "k": 5,
     "theta": 0.7, "variant": "HKG-Rec"},
]


def main():
    required_files = [
        os.path.join(cfg.DATA_DIR, "uhkg_theta_0.7.json"),
        os.path.join(cfg.DATA_DIR, "pattern_matrix.json"),
        os.path.join(cfg.GEN_DIR, "conflicts.json"),
        os.path.join(cfg.GEN_DIR, "service_devices.json"),
        os.path.join(cfg.GEN_DIR, "service_meta.json"),
    ]
    missing = [p for p in required_files if not os.path.exists(p)]
    if missing:
        print("Missing data files - run the setup pipeline first:")
        print("  python Data/generate_dataset.py")
        print("  python Data/service_meta.py")
        print("  python main-HKG.py")
        for p in missing:
            print(f"  missing: {p}")
        sys.exit(1)

    from importlib import import_module
    uhkgrec = import_module("UHKG-Rec")

    print(f"Running {len(SAMPLE_QUERIES)} sample queries...\n")
    all_ok = True
    results = []
    for i, q in enumerate(SAMPLE_QUERIES, start=1):
        print(f"--- Query {i}: {q['symptoms']}  (variant={q['variant']}, theta={q['theta']}) ---")
        try:
            result = uhkgrec.run_query(q, verbose=True)
            n_rec = sum(len(p["recommended_services"]) for p in result["recommendation"])
            print(f"  -> {n_rec} recommended service(s) across "
                  f"{len(result['recommendation'])} matched pattern(s)\n")
            results.append(result)
        except Exception as e:
            all_ok = False
            print(f"  FAILED: {e}\n")

    out_path = os.path.join(ROOT, "Test", "first_test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved all query results to {out_path}")

    if all_ok:
        print("\nfirst_test.py: ALL QUERIES RAN SUCCESSFULLY.")
    else:
        print("\nfirst_test.py: SOME QUERIES FAILED, see log above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
