
"""
BCI Motor Imagery - Ensemble Pipeline
======================================
4 branches complementaires:
  B1 : FBCSP fin   (8 bandes x 3 fenetres)  --> LDA(shrinkage)
  B2 : FBCSP core  (2 bandes x 2 fenetres)  --> SVM lineaire (C tuned)
  B3 : Riemannian Tangent Space + PCA        --> LogReg (C tuned)
  B4 : FgMDM (Fisher Geodesic, non-param.)

Ensemble : soft vote par scores OOF (Out-Of-Fold) non-biaised par sujet.
  -> chaque branche evalue son pipeline complet en 10-fold CV
  -> le poids est proportionnel a la precision OOF de la branche

Anti-overfitting:
  - LDA shrinkage automatique (Ledoit-Wolf)
  - SVM lineaire (C petit), LogReg (C petit)
  - Covariances OAS (regularisees)
  - FgMDM : zero hyperparametre sur test
  - Poids OOF : pas de leakage CSP/TS dans l'evaluation
"""

import warnings
warnings.filterwarnings("ignore")

import mne
mne.set_log_level("WARNING")

import numpy as np
import pandas as pd
import os
import zipfile

from scipy.signal import butter, sosfiltfilt
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import FgMDM

# ==============================================================================
# CONFIG
# ==============================================================================
DATA_PATH  = "data"
SUBJECTS   = ["A", "B", "C", "D", "E", "F"]
OUTPUT_DIR = "predictions"
ZIP_NAME   = "BCI_predictions_final.zip"
FS         = 256    # Hz  (1537 pts / 256 = 6.004 s par trial)
SEED       = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Banque de filtres : sous-bandes mu + beta
FINE_BANDS = [
    (8,  12),   # mu complet
    (8,  10),   # mu bas
    (10, 12),   # mu haut
    (12, 16),   # beta bas
    (16, 20),   # beta milieu
    (20, 24),   # beta haut
    (24, 30),   # beta tres haut
    (12, 30),   # beta large
]
CORE_BANDS = [(8, 12), (12, 30)]

# Fenetres temporelles (s) -- periode active imagerie motrice
FINE_WINDOWS = [(0.5, 3.5), (1.5, 4.5), (0.5, 4.5)]
CORE_WINDOWS = [(0.5, 3.5), (1.5, 4.5)]

RIEMANN_BAND   = (8, 30)
RIEMANN_WINDOW = (0.5, 4.5)

N_CSP     = 4
CV_FOLDS  = 10


# ==============================================================================
# SIGNAL PROCESSING
# ==============================================================================
def bandpass(X, low, high, fs=FS, order=5):
    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, X, axis=-1)


def car(X):
    return X - X.mean(axis=1, keepdims=True)


def crop(X, t0, t1, fs=FS):
    return X[:, :, int(t0 * fs) : int(t1 * fs)]


# ==============================================================================
# EXTRACTION DE CARACTERISTIQUES
# ==============================================================================
def fbcsp_transform(X_tr, y_tr, X_te, bands, windows, n_csp=N_CSP, fs=FS):
    """Fit CSP sur X_tr/y_tr, transforme X_tr et X_te."""
    tr_parts, te_parts = [], []
    for t0, t1 in windows:
        Xtr_w = crop(X_tr, t0, t1, fs)
        Xte_w = crop(X_te, t0, t1, fs)
        for low, high in bands:
            Xtr_f = bandpass(Xtr_w, low, high, fs)
            Xte_f = bandpass(Xte_w, low, high, fs)
            csp = CSP(n_components=n_csp, reg="ledoit_wolf", log=True, norm_trace=False)
            tr_parts.append(csp.fit_transform(Xtr_f, y_tr))
            te_parts.append(csp.transform(Xte_f))
    return np.concatenate(tr_parts, axis=1), np.concatenate(te_parts, axis=1)


def riemann_transform(X_tr, X_te, band=RIEMANN_BAND, window=RIEMANN_WINDOW, fs=FS):
    """Fit covariances + espace tangent sur X_tr, transforme X_tr et X_te."""
    Xtr = bandpass(crop(X_tr, *window, fs), *band, fs)
    Xte = bandpass(crop(X_te, *window, fs), *band, fs)
    cov = Covariances(estimator="oas")
    ts  = TangentSpace(metric="riemann")
    Ctr = cov.fit_transform(Xtr)
    Cte = cov.transform(Xte)
    Ftr = ts.fit_transform(Ctr)
    Fte = ts.transform(Cte)
    return Ftr, Fte, Ctr, Cte


def riemann_cov(X_tr, X_te, band=RIEMANN_BAND, window=RIEMANN_WINDOW, fs=FS):
    """Matrices de covariance seules (pour FgMDM)."""
    Xtr = bandpass(crop(X_tr, *window, fs), *band, fs)
    Xte = bandpass(crop(X_te, *window, fs), *band, fs)
    cov = Covariances(estimator="oas")
    return cov.fit_transform(Xtr), cov.transform(Xte)


# ==============================================================================
# SCORES OOF (Out-Of-Fold) -- CV propre sans leakage de preprocessing
# ==============================================================================
def oof_score_fbcsp_lda(X_raw, y, bands, windows, n_csp, fs, skf):
    """Accuracy OOF pour pipeline FBCSP + LDA (CSP refit dans chaque fold)."""
    correct = 0
    for tr_idx, val_idx in skf.split(X_raw, y):
        F_tr, F_val = fbcsp_transform(X_raw[tr_idx], y[tr_idx],
                                       X_raw[val_idx], bands, windows, n_csp, fs)
        sc = StandardScaler()
        F_tr = sc.fit_transform(F_tr); F_val = sc.transform(F_val)
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(F_tr, y[tr_idx])
        correct += np.sum(clf.predict(F_val) == y[val_idx])
    return correct / len(y)


def oof_score_fbcsp_svm(X_raw, y, bands, windows, n_csp, fs, skf):
    """Accuracy OOF pour pipeline FBCSP + SVM lineaire (C=0.05)."""
    correct = 0
    for tr_idx, val_idx in skf.split(X_raw, y):
        F_tr, F_val = fbcsp_transform(X_raw[tr_idx], y[tr_idx],
                                       X_raw[val_idx], bands, windows, n_csp, fs)
        sc = StandardScaler()
        F_tr = sc.fit_transform(F_tr); F_val = sc.transform(F_val)
        clf = SVC(kernel="linear", C=0.05, probability=False, random_state=SEED)
        clf.fit(F_tr, y[tr_idx])
        correct += np.sum(clf.predict(F_val) == y[val_idx])
    return correct / len(y)


def oof_score_riemann_lr(X_raw, y, band, window, fs, skf, n_pca=50):
    """Accuracy OOF pour pipeline Riemannien TS + PCA + LogReg (C=0.05)."""
    correct = 0
    for tr_idx, val_idx in skf.split(X_raw, y):
        F_tr, F_val, _, _ = riemann_transform(X_raw[tr_idx], X_raw[val_idx],
                                               band, window, fs)
        sc = StandardScaler()
        F_tr = sc.fit_transform(F_tr); F_val = sc.transform(F_val)
        k = min(n_pca, len(tr_idx) - 1)
        pca = PCA(n_components=k, whiten=True, random_state=SEED)
        F_tr = pca.fit_transform(F_tr); F_val = pca.transform(F_val)
        clf = LogisticRegression(C=0.05, max_iter=3000, solver="lbfgs")
        clf.fit(F_tr, y[tr_idx])
        correct += np.sum(clf.predict(F_val) == y[val_idx])
    return correct / len(y)


def oof_score_fgmdm(X_raw, y, band, window, fs, skf):
    """Accuracy OOF pour FgMDM."""
    correct = 0
    for tr_idx, val_idx in skf.split(X_raw, y):
        Ctr, Cval = riemann_cov(X_raw[tr_idx], X_raw[val_idx], band, window, fs)
        clf = FgMDM(metric="riemann", tsupdate=False)
        clf.fit(Ctr, y[tr_idx])
        correct += np.sum(clf.predict(Cval) == y[val_idx])
    return correct / len(y)


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
    print(f"\n{'=' * 58}")
    print(f"  Sujet {subject}")
    print(f"{'=' * 58}")

    X_train = np.load(f"{DATA_PATH}/subject_{subject}_X_train.npy")
    y_train = np.load(f"{DATA_PATH}/subject_{subject}_y_train.npy")
    X_test  = np.load(f"{DATA_PATH}/subject_{subject}_X_test.npy")

    # Preprocessing : Common Average Reference + mise a l'echelle microvolts
    X_train = car(X_train) * 1e6
    X_test  = car(X_test)  * 1e6

    classes = np.unique(y_train)  # ['left_hand', 'right_hand']
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    # ------------------------------------------------------------------
    # SCORES OOF : evaluation non-biaisee pour le poids de l'ensemble
    # ------------------------------------------------------------------
    print("  Calcul des scores OOF...")
    s1 = oof_score_fbcsp_lda(X_train, y_train, FINE_BANDS, FINE_WINDOWS, N_CSP, FS, skf)
    print(f"  B1 FBCSP fin  + LDA  OOF = {s1:.3f}")

    s2 = oof_score_fbcsp_svm(X_train, y_train, CORE_BANDS, CORE_WINDOWS, N_CSP, FS, skf)
    print(f"  B2 FBCSP core + SVM  OOF = {s2:.3f}")

    s3 = oof_score_riemann_lr(X_train, y_train, RIEMANN_BAND, RIEMANN_WINDOW, FS, skf)
    print(f"  B3 Riemann TS + LR   OOF = {s3:.3f}")

    s4 = oof_score_fgmdm(X_train, y_train, RIEMANN_BAND, RIEMANN_WINDOW, FS, skf)
    print(f"  B4 FgMDM             OOF = {s4:.3f}")

    # Poids : proportionnels aux OOF, plancher a 0.10 pour garantir la diversite
    scores  = np.array([s1, s2, s3, s4])
    weights = scores / scores.sum()
    weights = np.maximum(weights, 0.10)    # plancher : chaque branche contribue
    weights = weights / weights.sum()
    print(f"  Poids final : {weights.round(3)}")

    # ------------------------------------------------------------------
    # ENTRAINEMENT FINAL SUR L'ENSEMBLE DU TRAIN
    # ------------------------------------------------------------------
    print("  Entrainement final...")

    # Branch 1
    F1_tr, F1_te = fbcsp_transform(X_train, y_train, X_test, FINE_BANDS, FINE_WINDOWS)
    sc1 = StandardScaler()
    F1_tr = sc1.fit_transform(F1_tr); F1_te = sc1.transform(F1_te)
    clf1 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf1.fit(F1_tr, y_train)

    # Branch 2 : selectionne C par GridSearch sur le train complet
    F2_tr, F2_te = fbcsp_transform(X_train, y_train, X_test, CORE_BANDS, CORE_WINDOWS)
    sc2 = StandardScaler()
    F2_tr = sc2.fit_transform(F2_tr); F2_te = sc2.transform(F2_te)
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs2 = GridSearchCV(
        SVC(kernel="linear", probability=True, random_state=SEED),
        {"C": [0.01, 0.05, 0.1, 0.5]}, cv=cv5, scoring="accuracy", n_jobs=-1,
    )
    gs2.fit(F2_tr, y_train)
    clf2 = gs2.best_estimator_
    print(f"    B2 meilleur C = {clf2.C}")

    # Branch 3 : selectionne C par GridSearch
    F3_tr, F3_te, Ctr, Cte = riemann_transform(X_train, X_test)
    sc3 = StandardScaler()
    F3_tr = sc3.fit_transform(F3_tr); F3_te = sc3.transform(F3_te)
    n_pca = min(50, X_train.shape[0] - 1)
    pca3 = PCA(n_components=n_pca, whiten=True, random_state=SEED)
    F3_tr = pca3.fit_transform(F3_tr); F3_te = pca3.transform(F3_te)
    gs3 = GridSearchCV(
        LogisticRegression(max_iter=3000, solver="lbfgs", random_state=SEED),
        {"C": [0.005, 0.01, 0.05, 0.1, 0.5]}, cv=cv5, scoring="accuracy", n_jobs=-1,
    )
    gs3.fit(F3_tr, y_train)
    clf3 = gs3.best_estimator_
    print(f"    B3 meilleur C = {clf3.C}")

    # Branch 4
    clf4 = FgMDM(metric="riemann", tsupdate=False)
    clf4.fit(Ctr, y_train)

    # ------------------------------------------------------------------
    # PREDICTION PAR VOTE DOUX PONDERE
    # ------------------------------------------------------------------
    p1 = clf1.predict_proba(F1_te)
    p2 = align_proba(clf2.predict_proba(F2_te), clf2.classes_, classes)
    p3 = align_proba(clf3.predict_proba(F3_te), clf3.classes_, classes)
    p4 = align_proba(clf4.predict_proba(Cte),   clf4.classes_, classes)

    p_final = (weights[0] * p1 + weights[1] * p2
             + weights[2] * p3 + weights[3] * p4)

    y_pred  = classes[np.argmax(p_final, axis=1)]
    n_left  = int(np.sum(y_pred == "left_hand"))
    n_right = int(np.sum(y_pred == "right_hand"))
    print(f"  Predictions : {n_left} left_hand  /  {n_right} right_hand")

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
