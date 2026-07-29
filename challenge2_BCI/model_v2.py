"""
model_v2.py
===========
Base : model82 (meilleur private score) - normalisation par trial conservee
Ameliorations :
  1. Banque de filtres elargie  : 6 bandes vs 3
  2. Deux fenetres temporelles  : early + late du signal actif
  3. Branche SVM lineaire       : mieux regularisee que LDA seul
  4. FgMDM au lieu de LogReg    : classifieur Riemannien non-param.
  5. 3 branches au lieu de 2    : plus de diversite

Anti-overfitting:
  - Normalisation par trial (supprime variabilite d'amplitude)
  - OAS regularisation dans CSP (comme model82)
  - SVM lineaire (C petit)
  - FgMDM : zero hyperparametre appris sur test
  - Minimum weight 0.15 : chaque branche contribue toujours
"""

import warnings, os, zipfile
warnings.filterwarnings("ignore")
import mne; mne.set_log_level("WARNING")

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from pyriemann.estimation import Covariances
from pyriemann.classification import FgMDM

# ==============================================================================
# CONFIG
# ==============================================================================
DATA_PATH  = "data"
SUBJECTS   = ["A", "B", "C", "D", "E", "F"]
OUTPUT_DIR = "predictions"
ZIP_NAME   = "BCI_predictions_v2.zip"
FS         = 250
SEED       = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Banque de filtres elargie : 6 bandes discriminantes (mu + beta fins)
# model82 avait seulement (4,8), (8,12), (12,30)
ALL_BANDS  = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (12, 30)]
CORE_BANDS = [(8, 12), (12, 30)]

# Deux fenetres dans le signal cropé [0:800 samples = 800 pts]
# Fenetre A : debut  (echantillons 0-600 = 1.2s a 3.6s dans le trial original)
# Fenetre B : fin    (echantillons 200-800 = 2.0s a 4.4s dans le trial original)
WIN_A = (0,   600)
WIN_B = (200, 800)
WINDOWS = [WIN_A, WIN_B]

N_CSP = 6   # comme model82 (6 composantes spatiales par bande)


# ==============================================================================
# SIGNAL PROCESSING
# ==============================================================================
def bandpass(X, low, high, fs=FS, order=4):
    sos = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, X, axis=-1)


def preprocess(X):
    """Crop periode active + normalisation par trial (cle de model82)."""
    X = X[:, :, 300:1100]   # 1.2s a 4.4s a 250 Hz  (800 samples)
    return (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)


# ==============================================================================
# EXTRACTION DE CARACTERISTIQUES
# ==============================================================================
def fbcsp_transform(X_tr, y_tr, X_te, bands, windows, n_csp=N_CSP):
    """Filter Bank CSP : bandes x fenetres → log-variance par composante."""
    tr_parts, te_parts = [], []
    for w0, w1 in windows:
        Xtr_w = X_tr[:, :, w0:w1]
        Xte_w = X_te[:, :, w0:w1]
        for low, high in bands:
            Xtr_f = bandpass(Xtr_w, low, high)
            Xte_f = bandpass(Xte_w, low, high)
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

    # Preprocessing : crop + normalisation par trial (comme model82)
    X_train = preprocess(X_train)
    X_test  = preprocess(X_test)

    classes = np.unique(y_train)
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ------------------------------------------------------------------
    # BRANCHE 1 : FBCSP large (6 bandes x 2 fenetres) --> LDA(shrinkage)
    # 6 bandes x 2 fenetres x 6 CSP = 72 features  (ratio 0.51 : safe)
    # ------------------------------------------------------------------
    F1_tr, F1_te = fbcsp_transform(X_train, y_train, X_test, ALL_BANDS, WINDOWS)
    sc1   = StandardScaler()
    F1_tr = sc1.fit_transform(F1_tr);  F1_te = sc1.transform(F1_te)
    clf1  = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf1.fit(F1_tr, y_train)
    s1    = cross_val_score(clf1, F1_tr, y_train, cv=cv5).mean()
    print(f"  B1 FBCSP-LDA  ({F1_tr.shape[1]:3d} feat)  CV = {s1:.3f}")

    # ------------------------------------------------------------------
    # BRANCHE 2 : FBCSP cœur (2 bandes x 2 fenetres) --> SVM lineaire
    # 2 bandes x 2 fenetres x 6 CSP = 24 features  (tres parcimonieux)
    # SVM lineaire : meilleure generalisation que RBF sur petits datasets
    # ------------------------------------------------------------------
    F2_tr, F2_te = fbcsp_transform(X_train, y_train, X_test, CORE_BANDS, WINDOWS)
    sc2   = StandardScaler()
    F2_tr = sc2.fit_transform(F2_tr);  F2_te = sc2.transform(F2_te)
    gs2   = GridSearchCV(
        SVC(kernel="linear", probability=True, random_state=SEED),
        {"C": [0.01, 0.05, 0.1, 0.5, 1.0]},
        cv=cv5, scoring="accuracy", n_jobs=-1,
    )
    gs2.fit(F2_tr, y_train)
    clf2  = gs2.best_estimator_
    s2    = cross_val_score(clf2, F2_tr, y_train, cv=cv5).mean()
    print(f"  B2 FBCSP-SVM  ({F2_tr.shape[1]:3d} feat)  CV = {s2:.3f}  C = {clf2.C}")

    # ------------------------------------------------------------------
    # BRANCHE 3 : FgMDM (Fisher Geodesic MDM)
    # Classifieur Riemannien non-parametrique sur signal normalise brut
    # Remplace LogReg de model82 : pas d'hyperparametre, plus robuste
    # ------------------------------------------------------------------
    cov_est = Covariances(estimator="oas")
    Ctr     = cov_est.fit_transform(X_train)
    Cte     = cov_est.transform(X_test)

    clf3 = FgMDM(metric="riemann", tsupdate=False)
    clf3.fit(Ctr, y_train)
    s3   = cross_val_score(
        FgMDM(metric="riemann", tsupdate=False), Ctr, y_train, cv=cv5
    ).mean()
    print(f"  B3 FgMDM                         CV = {s3:.3f}")

    # ------------------------------------------------------------------
    # ENSEMBLE : poids proportionnels aux CV (comme model82)
    # Plancher 0.15 pour garantir que chaque branche contribue
    # ------------------------------------------------------------------
    scores  = np.array([s1, s2, s3])
    weights = scores / scores.sum()
    weights = np.maximum(weights, 0.15)
    weights = weights / weights.sum()
    print(f"  Poids : {weights.round(3)}")

    # Predictions
    p1 = clf1.predict_proba(F1_te)
    p2 = align_proba(clf2.predict_proba(F2_te), clf2.classes_, classes)
    p3 = align_proba(clf3.predict_proba(Cte),   clf3.classes_, classes)

    p_final = weights[0] * p1 + weights[1] * p2 + weights[2] * p3
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
