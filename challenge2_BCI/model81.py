import numpy as np
import pandas as pd
import os
import zipfile

from scipy.signal import butter, filtfilt
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# CONFIG
# =========================
DATA_PATH = "data"
SUBJECTS = ['A', 'B', 'C', 'D', 'E', 'F']
OUTPUT_DIR = "predictions"
ZIP_NAME = "BCI_predictions.zip"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 250

BANDS = [
    (8, 12),
    (12, 30)
]

# =========================
# FILTER
# =========================
def bandpass(X, low, high, fs=FS, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, X, axis=-1)

def crop(X):
    return X[:, :, 400:1200]

def normalize(X):
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True) + 1e-10
    return (X - mean) / std

# =========================
# EEGNET
# =========================
class EEGNet(nn.Module):
    def __init__(self, n_channels=64, n_samples=800):
        super().__init__()

        self.temporal = nn.Conv2d(1, 8, (1, 64), padding=(0,32), bias=False)
        self.spatial = nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False)

        self.pool = nn.AvgPool2d((1, 4))
        self.dropout = nn.Dropout(0.6)

        self.fc = nn.Linear(16 * (n_samples // 4), 2)

    def forward(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = torch.relu(x)

        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        return self.fc(x)

def train_eegnet(X_train, y_train):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X = torch.tensor(X_train[:, None, :, :], dtype=torch.float32).to(device)
    y = torch.tensor((y_train == 'right_hand').astype(int)).long().to(device)

    model = EEGNet(n_channels=64, n_samples=X.shape[-1]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

    return model

def predict_eegnet(model, X_test):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X = torch.tensor(X_test[:, None, :, :], dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(X), dim=1).cpu().numpy()

    return proba

# =========================
# MAIN LOOP
# =========================
for subject in SUBJECTS:
    print(f"\n🚀 Processing subject {subject}")

    X_train = np.load(f"{DATA_PATH}/subject_{subject}_X_train.npy")
    y_train = np.load(f"{DATA_PATH}/subject_{subject}_y_train.npy")
    X_test  = np.load(f"{DATA_PATH}/subject_{subject}_X_test.npy")

    X_train = normalize(crop(X_train))
    X_test  = normalize(crop(X_test))

    # =========================
    # 🔵 CSP
    # =========================
    X_train_csp = []
    X_test_csp = []

    for band in BANDS:
        X_tr = bandpass(X_train, band[0], band[1])
        X_te = bandpass(X_test, band[0], band[1])

        csp = CSP(n_components=2, reg='oas', log=True)
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
    # 🟢 RIEMANN
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

    clf_r = LogisticRegression(max_iter=2000, C=0.1)
    clf_r.fit(X_train_ts, y_train)

    proba_r = clf_r.predict_proba(X_test_ts)

    # =========================
    # 🔴 EEGNET
    # =========================
    model_eeg = train_eegnet(X_train, y_train)
    proba_eeg = predict_eegnet(model_eeg, X_test)

    # =========================
    # 🧠 ENSEMBLE FINAL
    # =========================
    proba_final = (
        0.4 * proba_csp +
        0.3 * proba_r +
        0.3 * proba_eeg
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