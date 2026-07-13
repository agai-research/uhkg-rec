"""
config.py
---------
Single place for every hyperparameter used across UHKG-Rec, its variants,
and the baselines. Nothing below should be hardcoded again elsewhere -
scripts import this module instead.

Values follow Table "Parameter settings" of the paper. Two parameters
(sigma2, beta) are not numerically fixed in the paper and are exposed
here as tunable defaults (see Appendix D of the implementation prompt).
"""

import os

# ---- paths -----------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "Data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
GEN_DIR = os.path.join(DATA_DIR, "generated")
RESULTS_DIR = os.path.join(ROOT_DIR, "Experiments", "results")

# ---- dataset scale (kept modest so the whole pipeline runs quickly;
#      counts are written to Data/dataset_stats.json after generation,
#      those are the authoritative numbers - not the values below) ----
N_SAMPLES_TARGET = 3000          # disease/symptom/treatment samples, per paper
N_DISEASES = 220
N_SYMPTOMS = 260
N_SERVICES = 240                 # drugs abstracted as healthcare services
N_IOHT_DEVICES = 40
N_PATIENTS = 300
PRIMEKG_SOURCES = 20             # N_src in Eq. B.1

# ---- confidence scoring (Eq. B.1, B.2) --------------------------------
LOGISTIC_BETA = 6.0              # steepness of the logistic transform

# ---- meta-path / walk / embedding (Table "Parameter settings") -------
WALKS_PER_NODE = 3               # w
WALK_LENGTH = 5                  # l
EMBED_DIM = 128                  # d
N_METAPATHS = 7                  # P
BATCH_SIZE = 50
LEARNING_RATE = 0.005            # gamma
N_EPOCHS = 8                     # kept small for a runnable prototype
ALPHA_LOSS = 0.5                 # alpha, uncertainty loss weight (Eq. B.9)
RBF_SIGMA2 = 0.3                 # sigma^2, RBF confidence bandwidth (Eq. B.4)

# ---- thresholds (theta) ------------------------------------------------
THETAS = [0.6, 0.7, 0.8]

# ---- pattern mining (Section "Pattern mining of healthcare requirements
#      and services") - number of clusters used to mine requirement (RP)
#      and service (SP) patterns from the co-occurrence structure of the
#      generated dataset --------------------------------------------------
N_REQUIREMENT_PATTERNS = 8
N_SERVICE_PATTERNS = 4

# ---- scoring weights (Eq. B.10) - distinct from ALPHA_LOSS above ------
ALPHA_TIME_WEIGHT = 0.5
BETA_COST_WEIGHT = 0.5

# ---- recommendation -----------------------------------------------------
TOPK_RANGE = list(range(2, 11))  # k = 2..10

# ---- experiment protocol -------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]          # >= 5 independent runs
SIG_ALPHA = 0.05                 # significance level for paired tests

# ---- node / relation type vocabularies (Appendix A) --------------------
NODE_TYPES = ["P", "S", "H", "O", "RP", "SP"]
RELATIONS = ["Have", "Cause", "Treatment", "Match", "Require", "TREATMENT"]

# ---- the 7 meta-path schemes (Appendix A.3) -----------------------------
# Each scheme is a tuple of (node_type_sequence, relation_sequence)
METAPATHS = [
    (["P", "S", "P"], ["Have", "Have-1"]),
    (["S", "RP", "S"], ["Cause", "Cause-1"]),
    (["H", "SP", "H"], ["Treatment", "Treatment-1"]),
    (["RP", "SP", "RP"], ["Match", "Match-1"]),
    (["S", "H", "O"], ["Require", "Require"]),
    (["H", "O", "H"], ["Require", "Require-1"]),
    (["P", "S", "RP", "SP", "H"], ["Have", "Cause", "Match", "Treatment-1"]),
]
