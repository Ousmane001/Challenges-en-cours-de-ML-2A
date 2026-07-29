"""
model_v5.py
===========
Reconstruction complete avec FS=512 Hz (CRITIQUE - confirme par l ami 0.92)

Architecture (inspiree de l ami a 0.92) :
  - FS = 512 Hz  -> trial de 3s exactement (1537/512 = 3.002s)
  - CSP custom   -> scipy.linalg.eigh directement (plus simple, sans MNE)
  - FBCSP 5 sous-bandes Mu/Beta : 8-12, 12-16, 16-20, 20-24, 24-30
  - Log-variance features par canal
  - LDA(shrinkage) par sujet
  - Sélection par sujet : meilleure config choisie par CV 5-fold

Zero pyriemann, zero MNE, zero GridSearch -> rapide comme model82
"""

import warnings, os, zipfile
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ==============================================================================
# CONFIG
# ==============================================================================
DATA_PATH  = "data"
SUBJECTS   = ["A", "B", "C", "D", "E", "F"]
OUTPUT_DIR = "predictions"
ZIP_NAME   = "BCI_predictions_v5.zip"

FS   = 512   # Hz — CORRECT (1537 / 512 = 3.00s par trial)
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sous-bandes Mu/Beta (comme l ami)
BETA_BANDS   = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]
BROAD_BANDS  = [(4, 8), (8, 12), (12, 30)]          # bandes model82 comme reference
MI_BAND      = [(8, 30)]                              # imagerie motrice classique
FULL_BAND    = [(4, 40)]                              # large bande

# Fenetre active : trial de 3s, on enleve les 50 premiers et derniers samples
# pour eviter les artefacts de bord des filtres
T0 = 50
T1 = 1487   # 1487-50 = 1437 samples = 2.81s a 512 Hz


# ==============================================================================
# SIGNAL PROCESSING
# ==============================================================================
def bandpass(X, low, high, fs=FS, order=5):
    sos = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, X, axis=-1)


def normalize(X):
    """Normalisation par trial par canal (supprime variabilite d amplitude)."""
    return (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)


# ==============================================================================
# CSP CUSTOM — scipy.linalg.eigh (comme l ami)
# ==============================================================================
def fit_csp(X, y, n_components=4):
    """
    Ajuste les filtres CSP par decomposition aux valeurs propres generalisees.
    Retourne W : (n_channels, n_components).
    """
    classes = np.unique(y)
    # Covariance moyenne par classe (sur l ensemble des trials)
    C0 = np.mean([np.cov(x) for x in X[y == classes[0]]], axis=0)
    C1 = np.mean([np.cov(x) for x in X[y == classes[1]]], axis=0)
    # Stabilisation numerique
    reg = 1e-8 * np.eye(C0.shape[0])
    # Decomposition : C0 * w = lambda * (C0 + C1) * w
    eigenvalues, eigenvectors = eigh(C0 + reg, C0 + C1 + 2 * reg)
    # Selectionner les k premiers et k derniers filtres (plus discriminants)
    k   = n_components // 2
    idx = np.argsort(eigenvalues)[::-1]
    sel = np.concatenate([idx[:k], idx[-k:]])
    return eigenvectors[:, sel]


def apply_csp_logvar(X, W):
    """Applique les filtres W et retourne le log-var de chaque composante."""
    # X : (n_trials, n_channels, n_times)
    # W : (n_channels, n_components)
    Z = np.einsum("ij,njk->nik", W.T, X)        # (n_trials, n_components, n_times)
    return np.log(np.var(Z, axis=-1) + 1e-10)   # (n_trials, n_components)


# ==============================================================================
# EXTRACTION DE CARACTERISTIQUES
# ==============================================================================
def fbcsp_features(X_tr, y_tr, X_te, bands, n_csp=4):
    """Filter Bank CSP : une paire (train, test) de features."""
    tr_parts, te_parts = [], []
    for low, high in bands:
        Xtr_f = bandpass(X_tr, low, high)
        Xte_f = bandpass(X_te, low, high)
        W     = fit_csp(Xtr_f, y_tr, n_csp)
        tr_parts.append(apply_csp_logvar(Xtr_f, W))
        te_parts.append(apply_csp_logvar(Xte_f, W))
    return np.concatenate(tr_parts, axis=1), np.concatenate(te_parts, axis=1)


def logvar_features(X, bands):
    """Log-variance par canal apres filtrage (pas de CSP)."""
    parts = []
    for low, high in bands:
        Xf = bandpass(X, low, high)
        parts.append(np.log(np.var(Xf, axis=-1) + 1e-10))   # (n_trials, n_channels)
    return np.concatenate(parts, axis=1)


# ==============================================================================
# BOUCLE PRINCIPALE
# ==============================================================================
for subject in SUBJECTS:
    print(f"\n{'='*55}\n  Sujet {subject}\n{'='*55}")

    X_train = np.load(f"{DATA_PATH}/subject_{subject}_X_train.npy")
    y_train = np.load(f"{DATA_PATH}/subject_{subject}_y_train.npy")
    X_test  = np.load(f"{DATA_PATH}/subject_{subject}_X_test.npy")

    # Crop + normalisation par trial
    X_train = normalize(X_train[:, :, T0:T1])
    X_test  = normalize(X_test[:,  :, T0:T1])

    classes = np.unique(y_train)
    cv5     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ------------------------------------------------------------------
    # CALCUL DE TOUTES LES CONFIGS (une seule fois par sujet)
    # ------------------------------------------------------------------
    configs = {}

    # FBCSP sur sous-bandes beta (approche de l ami)
    for n in [4, 6, 8]:
        F_tr, F_te = fbcsp_features(X_train, y_train, X_test, BETA_BANDS, n)
        configs[f"fbcsp_beta_n{n}"] = (F_tr, F_te)

    # FBCSP sur bandes model82 (reference)
    F_tr, F_te = fbcsp_features(X_train, y_train, X_test, BROAD_BANDS, 6)
    configs["fbcsp_broad_n6"] = (F_tr, F_te)

    # FBCSP bande MI unique 8-30 Hz
    F_tr, F_te = fbcsp_features(X_train, y_train, X_test, MI_BAND, 8)
    configs["fbcsp_mi_n8"] = (F_tr, F_te)

    # Log-variance par canal (plusieurs bandes)
    configs["logvar_mi"]   = (logvar_features(X_train, MI_BAND),
                               logvar_features(X_test,  MI_BAND))
    configs["logvar_full"] = (logvar_features(X_train, FULL_BAND),
                               logvar_features(X_test,  FULL_BAND))
    configs["logvar_beta"] = (logvar_features(X_train, BETA_BANDS),
                               logvar_features(X_test,  BETA_BANDS))

    # FBCSP beta + log-var MI (combine)
    F_csp_tr, F_csp_te = configs["fbcsp_beta_n4"]
    F_lv_tr  = logvar_features(X_train, MI_BAND)
    F_lv_te  = logvar_features(X_test,  MI_BAND)
    configs["fbcsp4_logvar_mi"] = (
        np.concatenate([F_csp_tr, F_lv_tr], axis=1),
        np.concatenate([F_csp_te, F_lv_te], axis=1),
    )

    # ------------------------------------------------------------------
    # SELECTION DE LA MEILLEURE CONFIG PAR CV
    # ------------------------------------------------------------------
    best_score, best_name = -1.0, ""
    results = {}

    for name, (F_tr, _) in configs.items():
        sc    = StandardScaler()
        F_s   = sc.fit_transform(F_tr)
        clf   = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        score = cross_val_score(clf, F_s, y_train, cv=cv5).mean()
        results[name] = score
        marker = " <--" if score > best_score else ""
        print(f"  {name:22s}  {F_tr.shape[1]:4d} feat  CV={score:.3f}{marker}")
        if score > best_score:
            best_score = score
            best_name  = name

    print(f"\n  --> Config choisie : {best_name}  (CV = {best_score:.3f})")

    # ------------------------------------------------------------------
    # ENTRAINEMENT FINAL ET PREDICTION
    # ------------------------------------------------------------------
    F_tr_best, F_te_best = configs[best_name]
    sc_final  = StandardScaler()
    F_tr_best = sc_final.fit_transform(F_tr_best)
    F_te_best = sc_final.transform(F_te_best)

    clf_final = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf_final.fit(F_tr_best, y_train)
    y_pred = clf_final.predict(F_te_best)

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
