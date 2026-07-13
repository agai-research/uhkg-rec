"""
confidence.py
-------------
Implements the confidence-scoring procedure described in the paper's
dataset section (Definition "Uncertain fact" + Eq. B.1/B.2 of the
implementation appendix):

    l(v, v')  = n_s(v, v') / N_src                       (Eq. B.1)
    c = f(g(l)) = 1 / (1 + exp(-beta * (l - 0.5)))        (Eq. B.2)

Facts already carry a precomputed confidence in the generated dataset
(see Data/generate_dataset.py); this module is what the HKG builder
and the two ablation variants call so the formula lives in one place.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


def plausibility(n_sources, n_src=cfg.PRIMEKG_SOURCES):
    """Eq. B.1 - fraction of PrimeKG sources corroborating a fact."""
    return n_sources / n_src


def logistic_confidence(l, beta=cfg.LOGISTIC_BETA):
    """Eq. B.2 - maps plausibility l in [0,1] to a confidence score in [0,1]."""
    return 1.0 / (1.0 + math.exp(-beta * (l - 0.5)))


def confidence(n_sources, n_src=cfg.PRIMEKG_SOURCES, beta=cfg.LOGISTIC_BETA):
    """End-to-end: source count -> confidence score c."""
    return logistic_confidence(plausibility(n_sources, n_src), beta)


def rbf_confidence(p, sigma2=cfg.RBF_SIGMA2):
    """
    Eq. B.4 - Gaussian RBF kernel used as the transition weight c(v,v')
    inside the confidence-weighted random walk (Eq. B.3). Centered at
    full certainty (p=1): higher p -> higher weight.
    """
    return math.exp(-((1.0 - p) ** 2) / sigma2)


if __name__ == "__main__":
    # Quick sanity check: the transform must be monotonically increasing in l,
    # and should roughly separate "well supported" from "weakly supported" facts.
    # (The paper's own worked examples, l=0.70 -> c~0.92 and l=0.10 -> c~0.40,
    # are illustrative rather than derived from a single closed-form beta; the
    # important, verifiable property is monotonicity, checked below.)
    print("l=0.10 ->", round(logistic_confidence(0.10), 3))
    print("l=0.50 ->", round(logistic_confidence(0.50), 3))
    print("l=0.70 ->", round(logistic_confidence(0.70), 3))
    print("l=0.90 ->", round(logistic_confidence(0.90), 3))
    vals = [logistic_confidence(x / 10) for x in range(11)]
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)), "must be monotonic in l"
    print("monotonicity check passed")
