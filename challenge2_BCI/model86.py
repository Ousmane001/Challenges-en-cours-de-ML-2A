import numpy as np
import pandas as pd
import os
import zipfile
from scipy.signal import butter, filtfilt
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =========================
# CONFIGURATION
# =========================
DATA_PATH = "data"
SUBJECTS = ['A', 'B', 'C', 'D', 'E', 'F']
OUTPUT_DIR = "predictions"
ZIP_NAME = "BCI_predictions_99.zip"

os.makedirs(OUTPUT_DIR, exist_ok=True)
FS = 250

# Banque de filtres très fine (Alpha, Mu, et Beta découpés avec précision)
BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]

# Multi-fenêtrage temporel (Capture l'évolution du signal)
# Fenêtre 1 : 1.0s à 3.6s  |  Fenêtre 2 : 2.0s à 4.6s
WINDOWS = [(250, 900), (500, 1150)]

# =========================
# PREPROCESSING BCI AVANCÉ
# =========================
def bandpass(X, low, high, fs=FS, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, X, axis=-1)

def apply_car(X):
    # Common Average Reference : Le secret pour nettoyer le bruit EEG
    return X - np.mean(X, axis=1, keepdims=True)

def preprocess(X):
    X = apply_car(X)
    # Mise à l'échelle (microvolts) pour la stabilité numérique
    return X * 1e6

# =========================
# BOUCLE PRINCIPALE
# =========================
for subject in SUBJECTS:
    print(f"\n🚀 Entraînement Sujet {subject} (Multi-Window FBCSP)...")

    X_train = np.load(f"{DATA_PATH}/subject_{subject}_X_train.npy")
    y_train = np.load(f"{DATA_PATH}/subject_{subject}_y_train.npy")
    X_test  = np.load(f"{DATA_PATH}/subject_{subject}_X_test.npy")

    X_train = preprocess(X_train)
    X_test  = preprocess(X_test)

    X_train_feats = []
    X_test_feats  = []

    # 1. Extraction des caractéristiques Spatio-Temporelles-Fréquentielles
    for w_start, w_end in WINDOWS:
        X_tr_w = X_train[:, :, w_start:w_end]
        X_te_w = X_test[:, :, w_start:w_end]

        for low, high in BANDS:
            X_tr_filt = bandpass(X_tr_w, low, high)
            X_te_filt = bandpass(X_te_w, low, high)

            # CSP robuste avec régularisation
            csp = CSP(n_components=4, reg='oas', log=True, norm_trace=False)
            X_train_feats.append(csp.fit_transform(X_tr_filt, y_train))
            X_test_feats.append(csp.transform(X_te_filt))

    # Fusion (2 fenêtres * 7 bandes * 4 composantes = 56 dimensions très riches)
    X_train_feats = np.concatenate(X_train_feats, axis=1)
    X_test_feats  = np.concatenate(X_test_feats, axis=1)

    # 2. Standardisation des 56 features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feats)
    X_test_scaled  = scaler.transform(X_test_feats)

    # 3. Apprentissage : SVM Non-linéaire + LDA Linéaire
    # Le SVM va trouver les motifs complexes, le LDA va assurer une base stable.
    clf_svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    clf_svm.fit(X_train_scaled, y_train)
    proba_svm = clf_svm.predict_proba(X_test_scaled)

    clf_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_lda.fit(X_train_scaled, y_train)
    proba_lda = clf_lda.predict_proba(X_test_scaled)

    # Soft Voting : L'ensemble parfait
    proba_final = (0.6 * proba_svm) + (0.4 * proba_lda)
    
    y_pred = clf_svm.classes_[np.argmax(proba_final, axis=1)]

    # 4. Sauvegarde
    pd.DataFrame({'y_pred': y_pred}).to_csv(
        f"{OUTPUT_DIR}/subject_{subject}_y_pred.csv", index=False
    )

# =========================
# ZIP FINAL
# =========================
with zipfile.ZipFile(ZIP_NAME, 'w') as zipf:
    for subject in SUBJECTS:
        zipf.write(
            f"{OUTPUT_DIR}/subject_{subject}_y_pred.csv",
            arcname=f"subject_{subject}_y_pred.csv"
        )

print(f"\n🏆 ZIP ULTIME READY : {ZIP_NAME}")