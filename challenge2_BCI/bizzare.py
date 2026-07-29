import numpy as np
import pandas as pd
import os
import zipfile

from scipy.signal import butter, filtfilt
from mne.decoding import CSP

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# =========================
# CONFIG
# =========================
DATA_PATH = "data"
SUBJECTS = ['A', 'B', 'C', 'D', 'E', 'F']
OUTPUT_DIR = "predictions"
ZIP_NAME = "BCI_predictions.zip"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 250

# 🔥 OPTIMISÉ BCI MOTOR IMAGERY
BANDS = [
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 26),
    (26, 32)
]

# =========================
# PREPROCESS
# =========================
def bandpass(X, low, high, fs=FS, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, X, axis=-1)

def crop(X):
    return X[:, :, 300:1100]

def normalize(X):
    return (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)

# =========================
# MAIN
# =========================
for subject in SUBJECTS:
    print(f"\n🚀 Processing subject {subject}")

    X_train = np.load(f"{DATA_PATH}/subject_{subject}_X_train.npy")
    y_train = np.load(f"{DATA_PATH}/subject_{subject}_y_train.npy")
    X_test  = np.load(f"{DATA_PATH}/subject_{subject}_X_test.npy")

    X_train = normalize(crop(X_train))
    X_test = normalize(crop(X_test))

    train_features = []
    test_features = []
    band_scores = []

    # =========================
    # FILTER BANK CSP + SCORE
    # =========================
    for band in BANDS:
        X_tr = bandpass(X_train, band[0], band[1])
        X_te = bandpass(X_test, band[0], band[1])

        csp = CSP(n_components=6, reg='oas', log=True)

        Xtr = csp.fit_transform(X_tr, y_train)
        Xte = csp.transform(X_te)

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf = CalibratedClassifierCV(clf, method='sigmoid', cv=3)

        clf.fit(Xtr, y_train)

        proba_train = clf.predict_proba(Xtr)
        score = np.mean(np.argmax(proba_train, axis=1) == y_train)

        band_scores.append(score)
        train_features.append((clf, scaler, csp))
        test_features.append(Xte)

    # =========================
    # NORMALISATION DES POIDS
    # =========================
    band_scores = np.array(band_scores)
    weights = band_scores / band_scores.sum()

    print("Weights:", weights)

    # =========================
    # ENSEMBLE PROBABILISTE
    # =========================
    final_proba = None

    for i, (clf_pack, Xte) in enumerate(zip(train_features, test_features)):
        clf, scaler, csp = clf_pack

        proba = clf.predict_proba(Xte)

        if final_proba is None:
            final_proba = weights[i] * proba
        else:
            final_proba += weights[i] * proba

    y_pred = np.argmax(final_proba, axis=1)

    # =========================
    # SAVE
    # =========================
    pd.DataFrame({'y_pred': y_pred}).to_csv(
        f"{OUTPUT_DIR}/subject_{subject}_y_pred.csv",
        index=False
    )

# =========================
# ZIP
# =========================
with zipfile.ZipFile(ZIP_NAME, 'w') as zipf:
    for subject in SUBJECTS:
        zipf.write(
            f"{OUTPUT_DIR}/subject_{subject}_y_pred.csv",
            arcname=f"subject_{subject}_y_pred.csv"
        )

print("\n🏆 ZIP READY:", ZIP_NAME)