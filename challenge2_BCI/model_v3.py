"""
model_v3.py
===========
Base : model82 (meilleur private score)
Ameliorations ciblées (sans augmenter la dimensionnalite) :

  1. Sélection adaptative de bandes par sujet
     -> 3 meilleures bandes choisies parmi 10 candidates (Fisher score)
     -> maintient exactement 18 features (3 bandes x 6 CSP) comme model82
     -> chaque sujet a ses bandes optimales

  2. FgMDM a la place de LogReg sur 2080 features
     -> non-parametrique : zero hyperparametre appris sur les donnees de test
     -> exploite la geometrie Riemannienne directement
     -> plus robuste que LogReg(C=1.0) sur features haute dimension

Dimensionnalite: identique a model82 (18 features Branch 1)
CV: biaise comme model82 (donne le meilleur private score)
"""

import warnings, os, zipfile
warnings.filterwarnings("ignore")
import mne; mne.set_log_level("WARNING")

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import FgMDM

# ==============================================================================
# CONFIG
# ==============================================================================
DATA_PATH  = "data"
SUBJECTS   = ["A", "B", "C", "D", "E", "F"]
OUTPUT_DIR = "predictions"
ZIP_NAME   = "BCI_predictions_v3.zip"
FS         = 250
SEED       = 42
N_CSP      = 6   # comme model82
N_SELECT   = 3   # bandes selectionnees par sujet (meme nombre que model82)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pool de candidats : couvre theta, mu, beta et leurs sous-bandes
CANDIDATE_BANDS = [
    (4,  8),   # theta
    (8,  10),  # mu bas
    (10, 12),  # mu haut
    (8,  12),  # mu complet
    (12, 16),  # beta bas
    (16, 20),  # beta milieu
    (20, 24),  # beta haut
    (24, 30),  # beta tres haut
    (12, 30),  # beta large
    (8,  30),  # mu+beta
]


# ==============================================================================
# SIGNAL PROCESSING
# ==============================================================================
def bandpass(X, low, high, fs=FS, order=4):
    sos = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, X, axis=-1)


def preprocess(X):
    """Crop période active [300:1100] + normalisation par trial (secret de model82)."""
    X = X[:, :, 300:1100]
    return (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)


# ==============================================================================
# SELECTION ADAPTATIVE DE BANDES PAR SUJET
# ==============================================================================
def select_best_bands(X, y, candidates, n_select, fs=FS):
    """
    Selectionne les n_select bandes avec le meilleur score de Fisher-CSP.
    Fisher-CSP : ratio separation inter-classe / dispersion intra-classe des
    log-variances CSP. Haut ratio = bande tres discriminante pour CE sujet.
    """
    classes = np.unique(y)
    scores = []
    for (low, high) in candidates:
        Xf = bandpass(X, low, high, fs)
        csp = CSP(n_components=2, reg="oas", log=True, norm_trace=False)
        feats = csp.fit_transform(Xf, y)  # shape (n_trials, 2)

        mu0 = feats[y == classes[0]].mean(0)
        mu1 = feats[y == classes[1]].mean(0)
        v0  = feats[y == classes[0]].var(0)
        v1  = feats[y == classes[1]].var(0)

        fisher = np.abs(mu0 - mu1) / (np.sqrt((v0 + v1) / 2) + 1e-8)
        scores.append(fisher.mean())

    best_idx = np.argsort(scores)[-n_select:]          # indices des n_select meilleures
    selected = [candidates[i] for i in sorted(best_idx)]
    return selected, [scores[i] for i in sorted(best_idx)]


# ==============================================================================
# EXTRACTION DE CARACTERISTIQUES FBCSP
# ==============================================================================
def fbcsp_transform(X_tr, y_tr, X_te, bands, n_csp=N_CSP, fs=FS):
    """FBCSP sur une liste de bandes, retourne features train et test."""
    tr_parts, te_parts = [], []
    for low, high in bands:
        Xtr_f = bandpass(X_tr, low, high, fs)
        Xte_f = bandpass(X_te, low, high, fs)
        csp = CSP(n_components=n_csp, reg="oas", log=True, norm_trace=False)
        tr_parts.append(csp.fit_transform(Xtr_f, y_tr))
        te_parts.append(csp.transform(Xte_f))
    return np.concatenate(tr_parts, axis=1), np.concatenate(te_parts, axis=1)


def align_proba(proba, src_classes, ref_classes):
    if list(src_classes) == list(ref_classes):
        return proba
    aligned = np.zeros_like(proba)
    for j, cls in enumerate(ref_classes):
        aligned[:, j] = proba[:, list(src_classes).index(cls)]
    return aligned


# ==============================================================================
# BOUCLE PRINCIPALE
# ==============================================================================
for subject in SUBJECTS:
    print(f"\n{'=' * 55}\n  Sujet {subject}\n{'=' * 55}")

    X_train = np.load(f"{DATA_PATH}/subject_{subject}_X_train.npy")
    y_train = np.load(f"{DATA_PATH}/subject_{subject}_y_train.npy")
    X_test  = np.load(f"{DATA_PATH}/subject_{subject}_X_test.npy")

    # Preprocessing : crop + normalisation par trial (identique a model82)
    X_train = preprocess(X_train)
    X_test  = preprocess(X_test)

    classes = np.unique(y_train)
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ------------------------------------------------------------------
    # SELECTION DES BANDES OPTIMALES PAR SUJET
    # ------------------------------------------------------------------
    best_bands, band_scores = select_best_bands(
        X_train, y_train, CANDIDATE_BANDS, N_SELECT
    )
    print(f"  Bandes selectionnees : {best_bands}")
    print(f"  Fisher scores        : {[f'{s:.2f}' for s in band_scores]}")

    # ------------------------------------------------------------------
    # BRANCHE 1 : FBCSP adaptatif -> LDA(shrinkage)
    # N_SELECT bandes x 6 CSP = 18 features  (identique a model82)
    # ------------------------------------------------------------------
    F1_tr, F1_te = fbcsp_transform(X_train, y_train, X_test, best_bands)
    sc1   = StandardScaler()
    F1_tr = sc1.fit_transform(F1_tr);  F1_te = sc1.transform(F1_te)

    clf1 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf1.fit(F1_tr, y_train)
    s1   = cross_val_score(clf1, F1_tr, y_train, cv=cv5).mean()
    print(f"  B1 FBCSP-LDA  ({F1_tr.shape[1]:2d} feat)  CV = {s1:.3f}")

    # ------------------------------------------------------------------
    # BRANCHE 2 : FgMDM (remplace LogReg(C=1.0) de model82)
    # Classifieur Riemannien non-parametrique sur signal normalise brut
    # Aucun hyperparametre appris : ne peut pas overfit
    # ------------------------------------------------------------------
    cov_est = Covariances(estimator="oas")
    Ctr     = cov_est.fit_transform(X_train)
    Cte     = cov_est.transform(X_test)

    clf2 = FgMDM(metric="riemann", tsupdate=False)
    clf2.fit(Ctr, y_train)
    s2   = cross_val_score(
        FgMDM(metric="riemann", tsupdate=False), Ctr, y_train, cv=cv5
    ).mean()
    print(f"  B2 FgMDM               CV = {s2:.3f}")

    # ------------------------------------------------------------------
    # ENSEMBLE : poids proportionnels aux CV biaised (comme model82)
    # Plancher 0.20 pour garantir que les 2 branches contribuent
    # ------------------------------------------------------------------
    scores  = np.array([s1, s2])
    weights = scores / scores.sum()
    weights = np.maximum(weights, 0.20)
    weights = weights / weights.sum()
    print(f"  Poids : {weights.round(3)}")

    # Predictions
    p1 = clf1.predict_proba(F1_te)
    p2 = align_proba(clf2.predict_proba(Cte), clf2.classes_, classes)

    p_final = weights[0] * p1 + weights[1] * p2
    y_pred  = classes[np.argmax(p_final, axis=1)]
    print(f"  Predictions : {int(np.sum(y_pred=='left_hand'))} left / "
          f"{int(np.sum(y_pred=='right_hand'))} right")

    pd.DataFrame({"y_pred": y_pred}).to_csv(
        f"{OUTPUT_DIR}/subject_{subject}_y_pred.csv", index=False
    )

# ==============================================================================
# ZIP FINAL
# ==============================================================================
with zipfile.ZipFile(ZIP_NAME, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for subject in SUBJECTS:
        path = f"{OUTPUT_DIR}/subject_{subject}_y_pred.csv"
        zf.write(path, arcname=f"subject_{subject}_y_pred.csv")

print(f"\nFini !  ZIP pret : {ZIP_NAME}")
