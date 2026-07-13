"""
classifiers.py
--------------
Trains the five downstream classifiers used in the paper's
"Evaluation of classification and recommendation quality" experiment
(RF, XGBoost, SVM, KNN, LR) on a set of node embeddings, to predict
each service's service-pattern (SP) label. This is the classification
step evaluated in Experiments/exp_classification.py - distinct from
the small joint softmax head used only to shape the embedding during
Algorithm 1's training (Embedding/model.py).
"""

import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix)
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)


def macro_specificity(y_true, y_pred, n_classes):
    """Macro-averaged specificity = TN / (TN + FP), one-vs-rest per class,
    then averaged - not provided directly by scikit-learn."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    specs = []
    total = cm.sum()
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - tp - fp - fn
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specs)) if specs else 0.0


def get_classifiers(seed=0):
    return {
        "RF": RandomForestClassifier(n_estimators=100, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=100, eval_metric="mlogloss",
                                  random_state=seed, verbosity=0),
        "SVM": SVC(kernel="rbf", probability=False, random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LR": LogisticRegression(max_iter=500, random_state=seed),
    }


def evaluate_classifiers(X, y, seed=0, test_size=0.3):
    """Fits each of the 5 classifiers on a train split and reports
    accuracy/precision/recall/F1 (macro) on the held-out test split.
    Classes with fewer than 4 members are dropped first (too rare to
    reliably appear in both splits). Labels are then encoded to a
    contiguous 0..n_classes-1 range fitted on the TRAIN split only
    (required by XGBoost); any test example whose class was not seen
    during training is dropped, since it cannot be evaluated anyway.
    Stratification is not used: with up to N_SERVICE_PATTERNS classes
    and a modest number of samples, a stratified split is often
    infeasible (test_size would need to exceed the number of classes)."""
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    y = np.asarray(y)
    vals, counts = np.unique(y, return_counts=True)
    keep_labels = set(vals[counts >= 4])
    mask = np.array([lab in keep_labels for lab in y])
    X, y = np.asarray(X)[mask], y[mask]
    if len(X) < 6 or len(set(y)) < 2:
        return {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    enc = LabelEncoder().fit(y_train)
    test_mask = np.isin(y_test, enc.classes_)
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)

    if len(set(y_train)) < 2 or len(y_test) == 0:
        return {}

    results = {}
    n_classes = len(enc.classes_)
    for name, clf in get_classifiers(seed).items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, average="macro", zero_division=0),
            "recall": recall_score(y_test, pred, average="macro", zero_division=0),
            "specificity": macro_specificity(y_test, pred, n_classes),
            "f1": f1_score(y_test, pred, average="macro", zero_division=0),
        }
    return results
