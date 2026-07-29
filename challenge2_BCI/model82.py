import numpy as np
import pandas as pd
import os
import zipfile

from scipy.signal import butter, filtfilt
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression

# =========================
# CONFIG
# =========================
DATA_PATH = "data"
SUBJECTS = ['A', 'B', 'C', 'D', 'E', 'F']
OUTPUT_DIR = "predictions"
ZIP_NAME = "BCI_predictions.zip"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 250

# RESTORE GOOD FILTER BANK
BANDS = [
    (4, 8),
    (8, 12),
    (12, 30)
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

    # =========================
    # CSP (IMPORTANT: SIMPLE & STABLE)
    # =========================
    X_train_csp, X_test_csp = [], []

    for band in BANDS:
        X_tr = bandpass(X_train, band[0], band[1])
        X_te = bandpass(X_test, band[0], band[1])

        csp = CSP(n_components=6, reg='oas', log=True)

        X_train_csp.append(csp.fit_transform(X_tr, y_train))
        X_test_csp.append(csp.transform(X_te))

    X_train_csp = np.concatenate(X_train_csp, axis=1)
    X_test_csp = np.concatenate(X_test_csp, axis=1)

    scaler = StandardScaler()
    X_train_csp = scaler.fit_transform(X_train_csp)
    X_test_csp = scaler.transform(X_test_csp)

    clf_csp = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_csp.fit(X_train_csp, y_train)
    proba_csp = clf_csp.predict_proba(X_test_csp)

    # =========================
    # RIEMANN (STABLE VERSION)
    # =========================
    cov = Covariances(estimator='oas')
    ts = TangentSpace()

    X_train_cov = cov.fit_transform(X_train)
    X_test_cov = cov.transform(X_test)

    X_train_ts = ts.fit_transform(X_train_cov)
    X_test_ts = ts.transform(X_test_cov)

    scaler2 = StandardScaler()
    X_train_ts = scaler2.fit_transform(X_train_ts)
    X_test_ts = scaler2.transform(X_test_ts)

    clf_r = LogisticRegression(max_iter=2000, C=1.0)
    clf_r.fit(X_train_ts, y_train)

    proba_r = clf_r.predict_proba(X_test_ts)

    # =========================
    # WEIGHTING (SAFE)
    # =========================
    score_csp = cross_val_score(clf_csp, X_train_csp, y_train, cv=5).mean()
    score_r   = cross_val_score(clf_r, X_train_ts, y_train, cv=5).mean()

    scores = np.array([score_csp, score_r])
    weights = scores / scores.sum()

    print("Weights:", weights)

    # =========================
    # ENSEMBLE FINAL
    # =========================
    proba_final = (
        weights[0] * proba_csp +
        weights[1] * proba_r
    )

    y_pred = clf_csp.classes_[np.argmax(proba_final, axis=1)]

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