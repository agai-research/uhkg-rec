"""
generate_dataset.py
--------------------
Builds the UHKG-Rec dataset following the two-stage pipeline described
in the paper (Section "Dataset and experimental settings"):

  1) Extraction/abstraction of PrimeKG-like facts (disease-symptom,
     drug-disease indication, drug-effect side effects) -> mapped onto
     the UHKG schema (S, H, O entities; Have/Cause/Treatment/Require
     relations), each with a confidence score derived from simulated
     "source agreement" (Eq. B.1/B.2 of the implementation appendix).
  2) Generative completion of the data PrimeKG does not provide:
     treatment conflicts, patient-service interactions (recommendation
     ground truth) and IoHT device associations.

NOTE ON DATA SOURCE: PrimeKG's servers (zitniklab.hms.harvard.edu) are
not reachable from this sandbox's network allow-list. Following the
prompt's own fallback instruction ("if unable to extract data from
PrimeKG online, generate all/part of the required data based on your
own knowledge"), this script generates a dataset that follows PrimeKG's
schema and scale description (diseases, symptoms, drugs, side effects)
using real, human-recognizable medical vocabulary, rather than fetching
the live graph. Every design choice below (entity/relation types,
confidence formula) still strictly follows the paper.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

random.seed(42)

# ---------------------------------------------------------------------
# Vocabulary pools (kept realistic; combined combinatorially below)
# ---------------------------------------------------------------------

SPECIALTIES = [
    "Cardiology", "Pulmonology", "Dermatology", "Endocrinology", "Neurology",
    "Gastroenterology", "Nephrology", "Rheumatology", "Oncology", "Psychiatry",
    "Orthopedics", "Ophthalmology", "ENT", "Immunology", "Infectious Disease",
    "Sleep Medicine", "Sports Medicine", "Geriatrics", "Urology", "Hematology",
]

DISEASE_STEMS = [
    "Type2Diabetes", "Hypertension", "Asthma", "COPD", "Migraine", "Eczema",
    "Psoriasis", "RheumatoidArthritis", "Osteoarthritis", "GERD", "IBS",
    "ChronicKidneyDisease", "Hyperthyroidism", "Hypothyroidism", "Anxiety",
    "Depression", "Insomnia", "AtrialFibrillation", "HeartFailure", "Anemia",
    "UrinaryTractInfection", "Pneumonia", "Bronchitis", "Sinusitis", "Allergy",
    "Obesity", "Osteoporosis", "Glaucoma", "MacularDegeneration", "Gout",
]

SYMPTOM_STEMS = [
    "Fatigue", "Fever", "Cough", "ShortnessOfBreath", "ChestPain", "Headache",
    "Dizziness", "Nausea", "JointPain", "SkinRash", "Itching", "SwellingLegs",
    "Palpitations", "BlurredVision", "WeightLoss", "WeightGain", "Bloating",
    "Constipation", "Diarrhea", "SoreThroat", "RunnyNose", "Wheezing",
    "LowOxygenSaturation", "IrregularHeartbeat", "HighBloodSugar", "Insomnia",
    "MuscleWeakness", "BackPain", "Anxiety", "PoorAppetite", "NightSweats",
    "FrequentUrination", "DrySkin", "HairLoss", "Numbness", "Tremor",
]

DRUG_STEMS = [
    "Metformin", "Lisinopril", "Albuterol", "Fluticasone", "Sumatriptan",
    "Hydrocortisone", "Methotrexate", "Ibuprofen", "Omeprazole", "Loperamide",
    "Furosemide", "Levothyroxine", "Methimazole", "Sertraline", "Escitalopram",
    "Zolpidem", "Amiodarone", "Bisoprolol", "FerrousSulfate", "Nitrofurantoin",
    "Amoxicillin", "Azithromycin", "Cetirizine", "Orlistat", "Alendronate",
    "Latanoprost", "Ranibizumab", "Allopurinol", "Insulin", "Atorvastatin",
]

SERVICE_KIND = [
    "Therapy", "Monitoring Service", "Consultation", "Management Program",
    "Rehabilitation Service", "Screening Service", "Counseling Service",
]

IOHT_DEVICES = [
    "Pulse Oximeter", "ECG Monitor", "Smartwatch", "Smart Bed", "CGM",
    "Air Quality Sensor", "Thermal Camera", "Defibrillator", "Blood Pressure Cuff",
    "Spirometer", "Sleep Tracker", "Insulin Pump", "Fall Detector",
    "Smart Inhaler", "UV Exposure Sensor", "Hydration Sensor",
]

random.shuffle(DISEASE_STEMS)
random.shuffle(SYMPTOM_STEMS)


def make_named(stems, n, suffix_pool=None, kind_pool=None):
    """Expand a small vocabulary into n distinct plausible entity names."""
    names = []
    i = 0
    while len(names) < n:
        stem = stems[i % len(stems)]
        variant = i // len(stems)
        if variant == 0:
            name = stem
        elif kind_pool:
            name = f"{stem} {kind_pool[variant % len(kind_pool)]}"
        elif suffix_pool:
            name = f"{stem}-{suffix_pool[variant % len(suffix_pool)]}"
        else:
            name = f"{stem}_{variant}"
        if name not in names:
            names.append(name)
        i += 1
    return names


def simulate_source_agreement(strength):
    """
    Simulate PrimeKG's 20-source corroboration count (n_s in Eq. B.1).
    `strength` in [0,1] is a latent reliability level used only to make
    the synthetic data heterogeneous (some facts well supported, some not).
    """
    mean_sources = 2 + strength * 16
    n_s = int(round(random.gauss(mean_sources, 3)))
    return max(1, min(cfg.PRIMEKG_SOURCES, n_s))


def plausibility(n_s):
    return n_s / cfg.PRIMEKG_SOURCES


def logistic_confidence(l, beta=cfg.LOGISTIC_BETA):
    import math
    return 1.0 / (1.0 + math.exp(-beta * (l - 0.5)))


def confidence_from_strength(strength):
    n_s = simulate_source_agreement(strength)
    l = plausibility(n_s)
    return round(logistic_confidence(l), 3), n_s, round(l, 3)


def build():
    diseases = make_named(DISEASE_STEMS, cfg.N_DISEASES)
    symptoms = make_named(SYMPTOM_STEMS, cfg.N_SYMPTOMS)
    services = make_named(DRUG_STEMS, cfg.N_SERVICES, kind_pool=SERVICE_KIND)
    devices = make_named(IOHT_DEVICES, cfg.N_IOHT_DEVICES, suffix_pool=["A", "B", "C"])
    patients = [f"Patient_{i:04d}" for i in range(cfg.N_PATIENTS)]

    # each disease is tagged with a specialty for readability/coherence
    disease_specialty = {d: random.choice(SPECIALTIES) for d in diseases}

    facts = []  # list of (head, relation, tail, confidence, meta) - the raw UHKG facts

    # --- disease-symptom associations (-> Have: Patient-Symptom via disease context,
    #     and the symptom pool a disease can trigger) ---------------------------
    disease_symptoms = {}
    for d in diseases:
        k = random.randint(3, 7)
        assigned = random.sample(symptoms, k)
        disease_symptoms[d] = assigned
        for s in assigned:
            strength = random.uniform(0.3, 0.95)
            c, n_s, l = confidence_from_strength(strength)
            facts.append({"head": d, "relation": "HasSymptom", "tail": s,
                          "confidence": c, "n_sources": n_s, "plausibility": l})

    # --- drug-disease indications (-> Treatment candidate) ---------------------
    drug_disease = {}
    for drug in services:
        k = random.randint(1, 3)
        targets = random.sample(diseases, k)
        drug_disease[drug] = targets
        for d in targets:
            strength = random.uniform(0.4, 0.98)
            c, n_s, l = confidence_from_strength(strength)
            facts.append({"head": drug, "relation": "Treat", "tail": d,
                          "confidence": c, "n_sources": n_s, "plausibility": l})

    # --- drug side effects (used later to build conflicts) ---------------------
    side_effects = {}
    generic_effects = ["Nausea", "Drowsiness", "DrySkin", "Headache", "Dizziness",
                        "GI Upset", "Fatigue", "RapidHeartbeat", "LowBloodPressure"]
    for drug in services:
        side_effects[drug] = random.sample(generic_effects, random.randint(1, 3))

    # --- IoHT device requirements per service -----------------------------------
    service_devices = {}
    for drug in services:
        if random.random() < 0.55:  # not every service needs a device
            k = random.randint(1, 2)
            devs = random.sample(devices, k)
            service_devices[drug] = devs
            for dev in devs:
                strength = random.uniform(0.5, 0.97)
                c, n_s, l = confidence_from_strength(strength)
                facts.append({"head": drug, "relation": "Require", "tail": dev,
                              "confidence": c, "n_sources": n_s, "plausibility": l})
        else:
            service_devices[drug] = []

    # --- symptom -> service (Require relation, S->H) mined from drug-disease
    #     and disease-symptom overlap (a service treats a disease, whose
    #     symptoms it therefore addresses) -----------------------------------
    for drug, ds in drug_disease.items():
        for d in ds:
            for s in disease_symptoms[d]:
                strength = random.uniform(0.35, 0.95)
                c, n_s, l = confidence_from_strength(strength)
                facts.append({"head": s, "relation": "Require", "tail": drug,
                              "confidence": c, "n_sources": n_s, "plausibility": l})

    # cap total facts near the "3000 samples" scale mentioned in the paper
    if len(facts) > cfg.N_SAMPLES_TARGET:
        random.shuffle(facts)
        facts = facts[: cfg.N_SAMPLES_TARGET]

    # ---------------------------------------------------------------------
    # Stage 2: generative completion (conflicts, interactions, IoHT extras)
    # ---------------------------------------------------------------------

    # (a) treatment conflicts: two services sharing an overlapping side-effect
    #     profile are flagged as a plausible incompatibility
    conflicts = []
    drug_list = list(services)
    n_pairs_checked = 0
    for i in range(len(drug_list)):
        for j in range(i + 1, len(drug_list)):
            n_pairs_checked += 1
            a, b = drug_list[i], drug_list[j]
            overlap = set(side_effects[a]) & set(side_effects[b])
            if overlap and random.random() < 0.5:
                strength = random.uniform(0.5, 0.9)
                c, n_s, l = confidence_from_strength(strength)
                conflicts.append({"service_a": a, "service_b": b,
                                   "shared_effects": sorted(overlap),
                                   "confidence": c})
    conflict_density = round(100.0 * len(conflicts) / max(1, n_pairs_checked), 3)

    # (b) patient-service interactions (recommendation ground truth)
    interactions = []
    for p in patients:
        d = random.choice(diseases)
        n_sym = random.randint(1, min(4, len(disease_symptoms[d])))
        p_symptoms = random.sample(disease_symptoms[d], n_sym)
        candidate_positive = [drug for drug, ds in drug_disease.items() if d in ds]
        pos = random.sample(candidate_positive, min(len(candidate_positive), random.randint(1, 2))) \
            if candidate_positive else []
        neg_pool = [s for s in services if s not in pos]
        neg = random.sample(neg_pool, min(len(neg_pool), len(pos) * 2 if pos else 2))
        for drug in pos:
            interactions.append({"patient": p, "disease": d, "symptoms": p_symptoms,
                                  "service": drug, "label": 1})
        for drug in neg:
            interactions.append({"patient": p, "disease": d, "symptoms": p_symptoms,
                                  "service": drug, "label": 0})
    n_pos = sum(1 for r in interactions if r["label"] == 1)
    n_neg = sum(1 for r in interactions if r["label"] == 0)
    interaction_density = round(100.0 * len(interactions) / max(1, len(patients) * len(services)), 3)

    # (c) extra IoHT associations already partially built above (service_devices);
    #     count distinct devices actually used
    used_devices = sorted({dev for devs in service_devices.values() for dev in devs})

    # ---------------------------------------------------------------------
    # Persist everything
    # ---------------------------------------------------------------------
    os.makedirs(cfg.RAW_DIR, exist_ok=True)
    os.makedirs(cfg.GEN_DIR, exist_ok=True)

    with open(os.path.join(cfg.RAW_DIR, "entities.json"), "w") as f:
        json.dump({"diseases": diseases, "symptoms": symptoms, "services": services,
                    "devices": devices, "patients": patients,
                    "disease_specialty": disease_specialty}, f, indent=2)

    with open(os.path.join(cfg.RAW_DIR, "facts.json"), "w") as f:
        json.dump(facts, f, indent=2)

    with open(os.path.join(cfg.GEN_DIR, "conflicts.json"), "w") as f:
        json.dump(conflicts, f, indent=2)

    with open(os.path.join(cfg.GEN_DIR, "interactions.json"), "w") as f:
        json.dump(interactions, f, indent=2)

    with open(os.path.join(cfg.GEN_DIR, "service_devices.json"), "w") as f:
        json.dump(service_devices, f, indent=2)

    with open(os.path.join(cfg.GEN_DIR, "drug_disease.json"), "w") as f:
        json.dump(drug_disease, f, indent=2)

    with open(os.path.join(cfg.GEN_DIR, "disease_symptoms.json"), "w") as f:
        json.dump(disease_symptoms, f, indent=2)

    stats = {
        "N_s": len(symptoms),
        "N_h": len(services),
        "N_devices": len(used_devices),
        "N_diseases": len(diseases),
        "N_patients": len(patients),
        "N_facts": len(facts),
        "d_conf_percent": conflict_density,
        "N_conflicts": len(conflicts),
        "N_int": len(interactions),
        "positive_negative_ratio": round(n_pos / max(1, n_neg), 3),
        "d_int_percent": interaction_density,
        "N_o": len(used_devices),
        "note": "N_rp and N_sp (requirement/service pattern counts) are added "
                "by Matching/pattern_mining.py after mining, and updated here.",
    }
    with open(os.path.join(cfg.DATA_DIR, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("Dataset generated:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return stats


if __name__ == "__main__":
    build()
