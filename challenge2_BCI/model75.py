import numpy as np
import pandas as pd
import os
import zipfile

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP

DATA_PATH = "data"
SUBJECTS = ['A', 'B', 'C', 'D', 'E', 'F']
OUTPUT_DIR = "predictions"
ZIP_NAME = "BCI_predictions.zip"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# PREPROCESSING
# =========================
def crop_signal(X):
    # garder la partie centrale (très important en EEG)
    return X[:, :, 200:1200]


def normalize(X):
    # normalisation par essai
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True) + 1e-10
    return (X - mean) / std


# =========================
# PIPELINE
# =========================
def build_pipeline(n_components):
    return Pipeline([
        ('csp', CSP(n_components=n_components, log=True, reg='ledoit_wolf')),
        ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    ])


# =========================
# MAIN
# =========================
for subject in SUBJECTS:
    print(f"Processing subject {subject}...")

    X_train = np.load(f"{DATA_PATH}/subject_{subject}_X_train.npy")
    y_train = np.load(f"{DATA_PATH}/subject_{subject}_y_train.npy")
    X_test = np.load(f"{DATA_PATH}/subject_{subject}_X_test.npy")

    # preprocessing
    X_train = crop_signal(X_train)
    X_test = crop_signal(X_test)

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    y_train_bin = np.array([0 if y == 'left_hand' else 1 for y in y_train])

    # =========================
    # ENSEMBLE CSP (clé du score)
    # =========================
    preds = []

    for n in [4, 6, 8, 10]:
        model = build_pipeline(n)
        model.fit(X_train, y_train_bin)
        preds.append(model.predict(X_test))

    # vote majoritaire
    preds = np.array(preds)
    y_pred_bin = np.round(np.mean(preds, axis=0)).astype(int)

    y_pred = np.array(['left_hand' if y == 0 else 'right_hand' for y in y_pred_bin])

    pd.DataFrame({'y_pred': y_pred}).to_csv(
        f"{OUTPUT_DIR}/subject_{subject}_y_pred.csv", index=False
    )


# =========================
# ZIP
# =========================
with zipfile.ZipFile(ZIP_NAME, 'w') as zipf:
    for subject in SUBJECTS:
        file_path = f"{OUTPUT_DIR}/subject_{subject}_y_pred.csv"
        zipf.write(file_path, arcname=f"subject_{subject}_y_pred.csv")

print("✅ ZIP ready:", ZIP_NAME)