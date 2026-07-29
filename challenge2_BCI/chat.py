import numpy as np
import pandas as pd
import os
import zipfile

from scipy.signal import butter, sosfiltfilt
from scipy.linalg import eigh

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# =========================
# CONFIG
# =========================
DATA_PATH = "data"
SUBJECTS = ['A','B','C','D','E','F']
OUTPUT_DIR = "predictions"
ZIP_NAME = "predictions.zip"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 512

FBCSP_BANDS = [(8,12),(12,16),(16,20),(20,24),(24,30)]
GLOBAL_BANDS = [(4,40),(8,30)]

# =========================
# PREPROCESS
# =========================
def normalize(X):
    return (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True)+1e-10)

def bandpass(X, low, high):
    sos = butter(4, [low, high], btype='band', fs=FS, output='sos')
    return sosfiltfilt(sos, X, axis=-1)

# =========================
# CSP
# =========================
def compute_csp(X, y, n_components=3):
    classes = np.unique(y)
    covs = []
    for c in classes:
        Xc = X[y==c]
        cov = np.mean([np.cov(trial)/np.trace(np.cov(trial)) for trial in Xc], axis=0)
        covs.append(cov)
    R = covs[0] + covs[1]
    eigvals, eigvecs = eigh(covs[0], R)
    ix = np.argsort(eigvals)[::-1]
    W = np.concatenate([eigvecs[:,ix[:n_components]], eigvecs[:,ix[-n_components:]]], axis=1)
    return W

def extract_csp(X, W):
    return np.array([np.log(np.var(W.T @ t, axis=1)+1e-10) for t in X])

def extract_logvar(X):
    return np.array([np.log(np.var(t, axis=1)+1e-10) for t in X])

# =========================
# MAIN
# =========================
for subject in SUBJECTS:
    print(f"\n🚀 {subject}")

    X_train = np.load(f"{DATA_PATH}/subject_{subject}_X_train.npy")
    y_train = np.load(f"{DATA_PATH}/subject_{subject}_y_train.npy")
    X_test  = np.load(f"{DATA_PATH}/subject_{subject}_X_test.npy")

    X_train = normalize(X_train[:,:,300:1100])
    X_test  = normalize(X_test[:,:,300:1100])

    # ===== CSP =====
    Xcsp_tr, Xcsp_te = [], []
    for b in FBCSP_BANDS:
        Xtr = bandpass(X_train, *b)
        Xte = bandpass(X_test, *b)
        W = compute_csp(Xtr, y_train)
        Xcsp_tr.append(extract_csp(Xtr, W))
        Xcsp_te.append(extract_csp(Xte, W))

    Xcsp_tr = np.concatenate(Xcsp_tr, axis=1)
    Xcsp_te = np.concatenate(Xcsp_te, axis=1)

    # ===== GLOBAL =====
    Xg_tr, Xg_te = [], []
    for b in GLOBAL_BANDS:
        Xtr = bandpass(X_train, *b)
        Xte = bandpass(X_test, *b)
        Xg_tr.append(extract_logvar(Xtr))
        Xg_te.append(extract_logvar(Xte))

    Xg_tr = np.concatenate(Xg_tr, axis=1)
    Xg_te = np.concatenate(Xg_te, axis=1)

    # ===== FEATURES =====
    X_train_feat = np.concatenate([Xcsp_tr, Xg_tr], axis=1)
    X_test_feat  = np.concatenate([Xcsp_te, Xg_te], axis=1)

    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_test_feat  = scaler.transform(X_test_feat)

    # ===== MODELS =====
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    logreg = LogisticRegression(max_iter=2000)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    s1 = cross_val_score(lda, X_train_feat, y_train, cv=cv).mean()
    s2 = cross_val_score(logreg, X_train_feat, y_train, cv=cv).mean()

    w1, w2 = s1/(s1+s2), s2/(s1+s2)
    print("weights:", w1, w2)

    lda.fit(X_train_feat, y_train)
    logreg.fit(X_train_feat, y_train)

    p1 = lda.predict_proba(X_test_feat)
    p2 = logreg.predict_proba(X_test_feat)

    proba = w1*p1 + w2*p2
    y_pred = lda.classes_[np.argmax(proba, axis=1)]

    pd.DataFrame({'y_pred': y_pred}).to_csv(
        f"{OUTPUT_DIR}/subject_{subject}_y_pred.csv", index=False
    )

# ZIP
with zipfile.ZipFile(ZIP_NAME, 'w') as z:
    for s in SUBJECTS:
        z.write(f"{OUTPUT_DIR}/subject_{s}_y_pred.csv",
                arcname=f"subject_{s}_y_pred.csv")

print("\n🏆 DONE")