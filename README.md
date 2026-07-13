# UHKG-Rec

Prototype implementation of the research paper **UHKG-Rec: Enhancing Service Recommendation in Smart Healthcare Networks via Pattern-driven Probabilistic Representation Learning**.

Authors: **Mouhamed Gaith Ayadi, Haithem Mezni, Shiyam Alalmaei, Hela Elmannai**

---

## 1. What this is

This repository implements the full UHKG-Rec pipeline described in the paper:

1. **HKG construction** from a PrimeKG-style dataset (patients, symptoms, healthcare
   services/drugs, IoHT devices).
2. **Pattern mining** of requirement patterns (RP) and service patterns (SP), and the
   requirement-service correlation matrix.
3. **Confidence-weighted, meta-path-guided probabilistic embedding** (a tailored
   metapath2vec variant with diagonal-Gaussian node embeddings).
4. **Pattern-driven classification** of services into service patterns.
5. **Bilateral pattern matching and safety-filtered candidate extraction**.
6. **Scoring and Top-k ranking** of recommended treatment plans.
7. Two baselines (**KGAT**, **CAGE**) and two ablated variants (**HKG-Rec**,
   **UHKG-Rec_NM**) for comparison.

---

## 2. Repository layout

```
UHKG-Rec/
├── UHKG-Rec.py                  # main entry point: run one recommendation query
├── main-HKG.py                  # builds the HKG/UHKG from the dataset
├── main-exp.py                  # runs all four experiments end to end
├── config.py                    # every hyperparameter, in one place
├── Data/
│   ├── generate_dataset.py       # dataset generation (Section 1 above)
│   ├── service_meta.py           # recovery-time / cost metadata
│   ├── lookups.py                 # conflict / device lookup dictionaries
│   ├── raw/                       # entities.json, facts.json
│   ├── generated/                 # conflicts, interactions, embeddings, ...
│   ├── dataset_stats.json         # N_s, N_h, N_rp, N_sp, ... 
│   ├── base_hkg.json               # base HKG (P/S/H/O + Have/Require)
│   ├── uhkg_full.json              # full UHKG (+ RP/SP + Cause/Treatment/Match)
│   └── uhkg_theta_{0.6,0.7,0.8}.json   # threshold-filtered UHKG views
├── HKG/
│   ├── build_graph.py             # base HKG construction (NetworkX)
│   ├── confidence.py              # confidence scoring
│   └── metapaths.py               # the 7 meta-path schemes + confidence-weighted walker
├── Embedding/
│   ├── model.py                   # diagonal-Gaussian embedding + hand-written Adam
│   ├── train.py                   # Algorithm 1 end to end
│   └── classifiers.py             # RF / XGBoost / SVM / KNN / LR evaluation
├── Matching/
│   ├── pattern_mining.py          # RP/SP mining + correlation matrix M^rh
│   └── extraction.py              # Algorithm 2 (MatchPatterns + ExtractServices)
├── Scoring/
│   └── scoring.py                 # scoring + ranking
├── Baselines/
│   ├── KGAT/kgat.py                # simplified attention-based KG recommendation
│   └── CAGE/cage.py                # simplified context-aware GCN recommendation
├── Variants/
│   ├── HKG-Rec/run.py              # deterministic ablation variant
│   ├── UHKG-Rec_NM/run.py          # "no matching" ablation variant
│   └── UHKG-Rec/run.py             # full proposed framework
├── Experiments/
│   ├── plot_style.py                # shared professional color palette / axis styling
│   ├── recommenders.py               # shared recommend() builders (variants + baselines)
│   ├── eval_utils.py                 # shared Top-k evaluation harness
│   ├── exp_visualization.py          # Study 1: embedding visualization (3-panel, by category)
│   ├── exp_classification.py         # Study 2: classification performance (4 metrics x 3 methods)
│   ├── exp_topk_variants.py          # Study 3: recommendation accuracy vs top-K (variants)
│   ├── exp_topk_baselines.py         # Study 4: recommendation accuracy vs top-K (baselines)
│   ├── exp_ablation.py               # Study 5: ablation study (3 variants)
├── Test/
│   ├── first_test.py               # dedicated smoke test (run this first!)
│   └── sample_query.json           # example query file
├── Notebooks/                      # Colab-ready .ipynb notebooks
    ├── all_experiments_figures.ipynb   # all 5 studies: description + code + figure
    ├── run_all_methods.ipynb
    ├── data_report.ipynb
    ├── kgat.ipynb
    └── cage.ipynb

```

---

## 3. Setup

```bash
pip install numpy networkx scikit-learn scipy matplotlib xgboost
```

No GPU or deep-learning framework (PyTorch/TensorFlow) is required: the probabilistic
embedding is implemented directly in NumPy, including its own small Adam
optimizer, so the prototype has no heavy external dependency.

---

## 4. Running the prototype

### Step 1 - generate the dataset

```bash
python Data/generate_dataset.py
python Data/service_meta.py
```

### Step 2 - build the HKG / UHKG

```bash
python main-HKG.py
```

This produces `Data/base_hkg.json`, `Data/uhkg_full.json`, and the three threshold views
`Data/uhkg_theta_{0.6,0.7,0.8}.json`, plus `Data/pattern_matrix.json` (the correlation
matrix M^rh) and updates `Data/dataset_stats.json` with the mined pattern counts.

### Step 3 - smoke test

```bash
python Test/first_test.py
```

Runs a handful of sample queries end to end and reports success/failure per query.

### Step 4 - run a single query

```bash
python UHKG-Rec.py Test/sample_query.json
```

Query format:

```json
{
  "symptoms": ["Fatigue", "Headache"],
  "active_constraints": ["Metformin"],
  "k": 5,
  "theta": 0.7,
  "variant": "UHKG-Rec"
}
```

`variant` is one of `"UHKG-Rec"`, `"UHKG-Rec_NM"`, `"HKG-Rec"`. `active_constraints` is a
list of item names (services, allergies, etc.) the patient currently has active - any
candidate service that conflicts with one of them, or requires an incompatible object, is
excluded.

### Step 5 - run a single variant or baseline directly

```bash
python "Variants/HKG-Rec/run.py"
python "Variants/UHKG-Rec_NM/run.py"
python "Variants/UHKG-Rec/run.py"
python Baselines/KGAT/kgat.py
python Baselines/CAGE/cage.py
```

### Step 6 - run all experiments

```bash
python main-exp.py
```

Or individually, each reproducing one figure from the revised manuscript:

```bash
python Experiments/exp_visualization.py     # Study 1: embedding visualization (3-panel)
python Experiments/exp_classification.py    # Study 2: classification performance
python Experiments/exp_topk_variants.py     # Study 3: recommendation accuracy vs top-K (variants)
python Experiments/exp_topk_baselines.py    # Study 4: recommendation accuracy vs top-K (baselines)
python Experiments/exp_ablation.py          # Study 5: ablation study
```

---

## 5. Dataset statistics

The counts below were produced by the dataset-generation run included in this
repository (`Data/dataset_stats.json`).

> "This extraction and mapping step results in **260** symptoms, **240** candidate
> healthcare services (drugs), **18** requirement patterns, and **14** service patterns,
> out of the **3000** disease/symptom/treatment samples."

Full table:

| Statistic | Value |
|---|---|
| N_s (symptoms) | 260 |
| N_h (healthcare services / drugs) | 240 |
| N_rp (requirement patterns) | 18 |
| N_sp (service patterns) | 14 |
| N_o (IoHT devices in use) | 40 |
| N_diseases | 220 |
| N_patients | 300 |
| N_facts (extracted+mapped) | 3000 |
| N_conflicts (generated) | 6294 |
| d_conf (conflict density) | 21.95% |
| N_int (patient-service interactions) | 1232 |
| positive/negative ratio | 0.47 |
| d_int (interaction density) | 1.71% |

Re-run `Data/generate_dataset.py` and `main-HKG.py` to regenerate this file if you change
`config.py`'s dataset-size parameters. Note: `N_REQUIREMENT_PATTERNS`/`N_SERVICE_PATTERNS`
were reduced from their original defaults (25/25) to 18/14 so that enough services fall into
each service pattern for multi-class classification to be meaningful at all three confidence
thresholds - see `config.py` and `Matching/pattern_mining.py`.

---

## 7. Reproducibility notes

- All hyperparameters live in `config.py` - nothing is hardcoded elsewhere.
- Every stochastic step (random walks, embedding initialization, KGAT/CAGE
  initialization) accepts an explicit `seed` argument.
- `Experiments/*.py` run every (configuration) cell over multiple seeds and report
  mean +/- std, with a paired t-test (`scipy.stats.ttest_rel`) between UHKG-Rec and each
  baseline/ablated variant.
- Some experiment scripts use a reduced seed count / epoch count relative to
  `config.py`'s defaults (documented at the top of each script) purely to keep full runs
  fast in a constrained environment; increase them for a more rigorous run.
