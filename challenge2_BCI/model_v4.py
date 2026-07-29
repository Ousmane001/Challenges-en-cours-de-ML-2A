"""
model_v4.py — model82 + fenetre temporelle optimale par sujet
=============================================================
Structure identique a model82 (rapide, meme logique, meme dimensionnalite)
Seule nouveaute : la fenetre de crop [300:1100] est choisie automatiquement
par sujet selon le pic de discriminabilite dans le signal.

Pourquoi ? Le pic d'imagerie motrice ne se produit pas au meme moment
pour tous les sujets. Utiliser la bonne fenetre peut gagner 5-10 points.
"""

import warnings, os, zipfile
warnings.filterwarnings("ignore")
import mne; mne.set_log_level("WARNING")

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# ==============================================================================
# CONFIG  (identique a model82)
# ==============================================================================
DATA_PATH  = "data"
SUBJECTS   = ["A", "B", "C", "D", "E", "F"]
OUTPUT_DIR = "predictions"
ZIP_NAME   = "BCI_predictions_v4.zip"
FS         = 250
SEED       = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

BANDS    = [(4, 8), (8, 12), (12, 30)]   # memes bandes que model82
WIN_LEN  = 800                            # meme longueur que model82
N_CSP    = 6                              # meme nombre que model82


# ==============================================================================
# SIGNAL PROCESSING  (identique a model82)
# ==============================================================================
def bandpass(X, low, high, fs=FS, order=4):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, X, axis=-1)


def normalize(X):
    return (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)


# ==============================================================================
# SELECTION DE FENETRE TEMPORELLE PAR SUJET
# ==============================================================================
def find_best_window(X_raw, y, win_len=WIN_LEN, n_cands=12):
    """
    Cherche la fenetre de win_len echantillons avec la meilleure separabilite
    CSP sur les bandes (8,12) et (12,30). Rapide : 12 fenetres candidates.
    """
    n_total = X_raw.shape[-1]
    # On teste des debuts entre 100 et (n_total - win_len - 50)
    starts  = np.linspace(100, n_total - win_len - 50, n_cands, dtype=int)
    classes = np.unique(y)
    best_score, best_start = -1.0, int(starts[len(starts) // 2])

    for start in starts:
        Xw = normalize(X_raw[:, :, start : start + win_len])
        fisher_sum = 0.0
        for low, high in [(8, 12), (12, 30)]:
            Xf    = bandpass(Xw, low, high)
            csp   = CSP(n_components=2, reg="oas", log=True, norm_trace=False)
            feats = csp.fit_transform(Xf, y)
            mu0   = feats[y == classes[0]].mean(0)
            mu1   = feats[y == classes[1]].mean(0)
            v0    = feats[y == classes[0]].var(0)
            v1    = feats[y == classes[1]].var(0)
            fisher_sum += (np.abs(mu0 - mu1) / (np.sqrt((v0 + v1) / 2) + 1e-8)).mean()
        if fisher_sum > best_score:
            best_score = fisher_sum
            best_start = int(start)

    return best_start, best_start + win_len


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
    print(f"\n{'='*55}\n  Sujet {subject}\n{'='*55}")

    X_train = np.load(f"{DATA_PATH}/subject_{subject}_X_train.npy")
    y_train = np.load(f"{DATA_PATH}/subject_{subject}_y_train.npy")
    X_test  = np.load(f"{DATA_PATH}/subject_{subject}_X_test.npy")

    classes = np.unique(y_train)
    cv5     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # --- Fenetre optimale par sujet ---
    w_start, w_end = find_best_window(X_train, y_train)
    at_250hz = f"{w_start/FS:.2f}s - {w_end/FS:.2f}s"
    print(f"  Fenetre optimale : [{w_start}:{w_end}]  ({at_250hz})")

    # --- Crop + normalisation (identique a model82) ---
    X_train = normalize(X_train[:, :, w_start:w_end])
    X_test  = normalize(X_test[:,  :, w_start:w_end])

    # ==================================================================
    # BRANCHE 1 : FBCSP -> LDA   (identique a model82)
    # ==================================================================
    X_tr_csp, X_te_csp = [], []
    for low, high in BANDS:
        X_tr_f = bandpass(X_train, low, high)
        X_te_f = bandpass(X_test,  low, high)
        csp    = CSP(n_components=N_CSP, reg="oas", log=True)
        X_tr_csp.append(csp.fit_transform(X_tr_f, y_train))
        X_te_csp.append(csp.transform(X_te_f))

    X_tr_csp = np.concatenate(X_tr_csp, axis=1)
    X_te_csp = np.concatenate(X_te_csp, axis=1)

    sc1 = StandardScaler()
    X_tr_csp = sc1.fit_transform(X_tr_csp)
    X_te_csp = sc1.transform(X_te_csp)

    clf_csp = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf_csp.fit(X_tr_csp, y_train)
    score_csp = cross_val_score(clf_csp, X_tr_csp, y_train, cv=cv5).mean()
    print(f"  B1 CSP-LDA  ({X_tr_csp.shape[1]} feat)  CV = {score_csp:.3f}")

    # ==================================================================
    # BRANCHE 2 : Riemannian TS -> LogReg   (identique a model82)
    # ==================================================================
    cov = Covariances(estimator="oas")
    ts  = TangentSpace()

    X_tr_cov = cov.fit_transform(X_train)
    X_te_cov = cov.transform(X_test)
    X_tr_ts  = ts.fit_transform(X_tr_cov)
    X_te_ts  = ts.transform(X_te_cov)

    sc2      = StandardScaler()
    X_tr_ts  = sc2.fit_transform(X_tr_ts)
    X_te_ts  = sc2.transform(X_te_ts)

    clf_r    = LogisticRegression(max_iter=2000, C=1.0)
    clf_r.fit(X_tr_ts, y_train)
    score_r  = cross_val_score(clf_r, X_tr_ts, y_train, cv=cv5).mean()
    print(f"  B2 Riemann  (2080 feat) CV = {score_r:.3f}")

    # ==================================================================
    # ENSEMBLE (identique a model82)
    # ==================================================================
    scores  = np.array([score_csp, score_r])
    weights = scores / scores.sum()
    print(f"  Poids : {weights.round(3)}")

    proba_csp = clf_csp.predict_proba(X_te_csp)
    proba_r   = align_proba(clf_r.predict_proba(X_te_ts), clf_r.classes_, classes)

    p_final = weights[0] * proba_csp + weights[1] * proba_r
    y_pred  = classes[np.argmax(p_final, axis=1)]
    print(f"  Predictions : {int(np.sum(y_pred=='left_hand'))} left / "
          f"{int(np.sum(y_pred=='right_hand'))} right")

    pd.DataFrame({"y_pred": y_pred}).to_csv(
        f"{OUTPUT_DIR}/subject_{subject}_y_pred.csv", index=False
    )

# ==============================================================================
# ZIP
# ==============================================================================
with zipfile.ZipFile(ZIP_NAME, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for subject in SUBJECTS:
        zf.write(f"{OUTPUT_DIR}/subject_{subject}_y_pred.csv",
                 arcname=f"subject_{subject}_y_pred.csv")

print(f"\nFini !  ZIP pret : {ZIP_NAME}")
