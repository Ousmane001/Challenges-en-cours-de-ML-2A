"""
BCI Motor Imagery – model_deep.py
===================================
Cible : >0.90 de précision moyenne sur 6 sujets.

5 branches + pondération OOF (5-fold, plancher 0.10) :

  B1 : FBCSP fin (9 bandes × 3 fenêtres × 4 CSP) → LDA (Ledoit-Wolf)
  B2 : Riemannian multi-bandes (Mu 8-12, Beta 12-30, Large 8-30)
         → 3 TS-LogReg indépendants → vote moyen des probabilités
  B3 : FgMDM broadband (non-paramétrique, zéro hyperparamètre)
  B4 : EEGNet léger (F1=4, kern=32, 640 pts) × 3 seeds + TTA-3
         OOF estimé en 3-fold (15 epochs rapides) pour la pondération
  B5 : FBCSP core (4 bandes × 2 fenêtres × 6 CSP) → SVM-RBF

Temps estimé : ~15-25 min sur CPU (6 sujets).
"""

import warnings
warnings.filterwarnings("ignore")
import os, zipfile, copy

import mne
mne.set_log_level("WARNING")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.signal import butter, sosfiltfilt
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pyriemann.classification import FgMDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# ==============================================================================
# CONFIG
# ==============================================================================
DATA_PATH  = "data"
SUBJECTS   = ["A", "B", "C", "D", "E", "F"]
OUTPUT_DIR = "predictions"
ZIP_NAME   = "BCI_predictions_deep.zip"
FS         = 256
SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FINE_BANDS = [
    (8, 10), (10, 12),
    (12, 16), (16, 20),
    (20, 24), (24, 28),
    (28, 32),
    (8, 12), (12, 30),
]
FINE_WINDOWS = [(0.5, 3.5), (1.5, 4.5), (0.5, 4.5)]

CORE_BANDS   = [(8, 12), (12, 20), (20, 30), (12, 30)]
CORE_WINDOWS = [(0.5, 3.5), (1.5, 4.5)]

RIEMANN_BANDS   = [(8, 12), (12, 30), (8, 30)]
RIEMANN_BAND_B3 = (8, 30)
RIEMANN_WINDOW  = (0.5, 4.5)

# EEGNet léger : fenêtre 2.5s → 640 pts (4× plus rapide)
EEGNET_BAND   = (8, 30)
EEGNET_WINDOW = (1.0, 3.5)   # 2.5s × 256 = 640 points
N_EEGNET      = 3             # modèles dans l'ensemble final
N_EPOCHS_EEG  = 30            # entraînement final
N_EPOCHS_OOF  = 15            # entraînement rapide pour l'OOF EEGNet
N_TTA         = 3
CV_OOF        = 5             # folds pour OOF classique
CV_EEG_OOF    = 3             # folds (rapide) pour l'OOF EEGNet

N_CSP_FINE = 4
N_CSP_CORE = 6


# ==============================================================================
# SIGNAL
# ==============================================================================
def bandpass(X, low, high, fs=FS, order=5):
    sos = butter(order, [low / (fs/2), high / (fs/2)], btype="band", output="sos")
    return sosfiltfilt(sos, X, axis=-1)

def car(X):
    return X - X.mean(axis=1, keepdims=True)

def crop(X, t0, t1, fs=FS):
    return X[:, :, int(t0 * fs): int(t1 * fs)]


# ==============================================================================
# FBCSP
# ==============================================================================
def fbcsp_features(X_tr, y_tr, X_te, bands, windows, n_csp=N_CSP_FINE, fs=FS):
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


# ==============================================================================
# RIEMANNIAN
# ==============================================================================
def riemann_ts(X_tr, X_te, band, window, fs=FS):
    Xtr = bandpass(crop(X_tr, *window, fs), *band, fs)
    Xte = bandpass(crop(X_te, *window, fs), *band, fs)
    cov = Covariances(estimator="oas")
    ts  = TangentSpace(metric="riemann")
    Ctr = cov.fit_transform(Xtr)
    Cte = cov.transform(Xte)
    return ts.fit_transform(Ctr), ts.transform(Cte), Ctr, Cte

def riemann_cov_only(X_tr, X_te, band, window, fs=FS):
    Xtr = bandpass(crop(X_tr, *window, fs), *band, fs)
    Xte = bandpass(crop(X_te, *window, fs), *band, fs)
    cov = Covariances(estimator="oas")
    return cov.fit_transform(Xtr), cov.transform(Xte)


# ==============================================================================
# EEGNET (léger : F1=4, kern=32, F2=8)
# ==============================================================================
class EEGNet(nn.Module):
    def __init__(self, n_channels=64, n_times=640, F1=4, D=2, F2=8,
                 kern_len=32, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, D * F1, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, (1, 16), padding=(0, 8), groups=D * F1, bias=False),
            nn.Conv2d(D * F1, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        t1 = n_times // 4
        t2 = t1 // 8
        self.fc = nn.Linear(F2 * t2, 2)

    def forward(self, x):
        return self.fc(self.block2(self.block1(x)).flatten(1))


def augment_eeg(X, noise_frac=0.08, max_shift=15, ch_drop_p=0.04, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    X = X.copy()
    std = X.std(axis=-1, keepdims=True) + 1e-8
    X += noise_frac * std * rng.standard_normal(X.shape)
    shifts = rng.integers(-max_shift, max_shift, size=X.shape[0])
    for i, s in enumerate(shifts):
        X[i] = np.roll(X[i], int(s), axis=-1)
    X *= (rng.random((X.shape[0], X.shape[1], 1)) > ch_drop_p)
    return X


def eegnet_prepare(X_raw, band=EEGNET_BAND, window=EEGNET_WINDOW, fs=FS):
    Xc = bandpass(crop(X_raw, *window, fs), *band, fs)
    mu  = Xc.mean(axis=-1, keepdims=True)
    sig = Xc.std(axis=-1, keepdims=True) + 1e-8
    return (Xc - mu) / sig


def train_eegnet(X_raw, y_bin, seed=0, n_epochs=N_EPOCHS_EEG):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    X_aug = np.concatenate([X_raw, augment_eeg(X_raw, rng=rng)], axis=0)
    y_aug = np.tile(y_bin, 2)
    perm  = rng.permutation(len(X_aug))
    X_aug, y_aug = X_aug[perm], y_aug[perm]

    Xt = torch.tensor(X_aug[:, None, :, :], dtype=torch.float32).to(DEVICE)
    yt = torch.tensor(y_aug, dtype=torch.long).to(DEVICE)

    n_times = X_raw.shape[-1]
    model   = EEGNet(n_channels=X_raw.shape[1], n_times=n_times).to(DEVICE)
    opt     = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    bsz = 32
    model.train()
    for _ in range(n_epochs):
        perm_b = torch.randperm(len(Xt), device=DEVICE)
        for start in range(0, len(Xt), bsz):
            idx_b = perm_b[start: start + bsz]
            opt.zero_grad()
            loss_fn(model(Xt[idx_b]), yt[idx_b]).backward()
            opt.step()
        sched.step()
    return model


def predict_eegnet(model, X_raw, n_tta=N_TTA):
    model.eval()
    rng = np.random.default_rng(0)
    copies = [X_raw] + [augment_eeg(X_raw, rng=rng) for _ in range(n_tta - 1)]
    all_p = []
    for Xc in copies:
        Xt = torch.tensor(Xc[:, None, :, :], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            all_p.append(torch.softmax(model(Xt), dim=1).cpu().numpy())
    return np.mean(all_p, axis=0)


# ==============================================================================
# FONCTIONS OOF
# ==============================================================================
def oof_fbcsp_lda(X_raw, y, bands, windows, n_csp, fs, skf):
    correct = 0
    for tr_i, va_i in skf.split(X_raw, y):
        F_tr, F_va = fbcsp_features(X_raw[tr_i], y[tr_i], X_raw[va_i],
                                     bands, windows, n_csp, fs)
        sc = StandardScaler()
        F_tr = sc.fit_transform(F_tr); F_va = sc.transform(F_va)
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(F_tr, y[tr_i])
        correct += (clf.predict(F_va) == y[va_i]).sum()
    return correct / len(y)


def oof_riemann_multiband(X_raw, y, bands, window, fs, skf):
    correct = 0
    for tr_i, va_i in skf.split(X_raw, y):
        p_all = []
        for band in bands:
            Ftr, Fva, _, _ = riemann_ts(X_raw[tr_i], X_raw[va_i], band, window, fs)
            sc = StandardScaler()
            Ftr = sc.fit_transform(Ftr); Fva = sc.transform(Fva)
            clf = LogisticRegression(C=0.05, max_iter=3000, solver="lbfgs")
            clf.fit(Ftr, y[tr_i])
            p_all.append(clf.predict_proba(Fva))
        proba = np.mean(p_all, axis=0)
        preds = np.array(["left_hand", "right_hand"])[np.argmax(proba, axis=1)]
        correct += (preds == y[va_i]).sum()
    return correct / len(y)


def oof_fgmdm(X_raw, y, band, window, fs, skf):
    correct = 0
    for tr_i, va_i in skf.split(X_raw, y):
        Ctr, Cva = riemann_cov_only(X_raw[tr_i], X_raw[va_i], band, window, fs)
        clf = FgMDM(metric="riemann", tsupdate=False)
        clf.fit(Ctr, y[tr_i]); correct += (clf.predict(Cva) == y[va_i]).sum()
    return correct / len(y)


def oof_eegnet_fast(X_prep, y_bin, fs, n_epochs_fast=N_EPOCHS_OOF):
    """OOF EEGNet rapide : 3 folds, 1 modèle par fold, epochs réduits."""
    skf_fast = StratifiedKFold(n_splits=CV_EEG_OOF, shuffle=True, random_state=SEED)
    correct = 0
    for seed_f, (tr_i, va_i) in enumerate(skf_fast.split(X_prep, y_bin)):
        model = train_eegnet(X_prep[tr_i], y_bin[tr_i], seed=seed_f,
                              n_epochs=n_epochs_fast)
        p = predict_eegnet(model, X_prep[va_i], n_tta=2)
        correct += (np.argmax(p, axis=1) == y_bin[va_i]).sum()
    return correct / len(y_bin)


def oof_fbcsp_svm(X_raw, y, bands, windows, n_csp, fs, skf, C=0.1):
    correct = 0
    for tr_i, va_i in skf.split(X_raw, y):
        F_tr, F_va = fbcsp_features(X_raw[tr_i], y[tr_i], X_raw[va_i],
                                     bands, windows, n_csp, fs)
        sc = StandardScaler()
        F_tr = sc.fit_transform(F_tr); F_va = sc.transform(F_va)
        clf = SVC(kernel="rbf", C=C, gamma="scale", probability=False)
        clf.fit(F_tr, y[tr_i])
        correct += (clf.predict(F_va) == y[va_i]).sum()
    return correct / len(y)


def align_proba(p, src_cls, ref_cls):
    if list(src_cls) == list(ref_cls):
        return p
    aligned = np.zeros_like(p)
    for j, c in enumerate(ref_cls):
        aligned[:, j] = p[:, list(src_cls).index(c)]
    return aligned


# ==============================================================================
# BOUCLE PRINCIPALE
# ==============================================================================
for subject in SUBJECTS:
    print(f"\n{'=' * 60}")
    print(f"  Sujet {subject}")
    print(f"{'=' * 60}")

    X_train = np.load(f"{DATA_PATH}/subject_{subject}_X_train.npy")
    y_train = np.load(f"{DATA_PATH}/subject_{subject}_y_train.npy")
    X_test  = np.load(f"{DATA_PATH}/subject_{subject}_X_test.npy")

    X_train = car(X_train) * 1e6
    X_test  = car(X_test)  * 1e6

    classes = np.unique(y_train)
    y_bin   = (y_train == "right_hand").astype(int)
    skf     = StratifiedKFold(n_splits=CV_OOF, shuffle=True, random_state=SEED)

    X_eeg_tr = eegnet_prepare(X_train)
    X_eeg_te = eegnet_prepare(X_test)

    # ------------------------------------------------------------------
    # SCORES OOF (pondération)
    # ------------------------------------------------------------------
    print("  Scores OOF...")
    s1 = oof_fbcsp_lda(X_train, y_train, FINE_BANDS, FINE_WINDOWS, N_CSP_FINE, FS, skf)
    print(f"    B1 FBCSP-fin + LDA         OOF = {s1:.3f}")

    s2 = oof_riemann_multiband(X_train, y_train, RIEMANN_BANDS, RIEMANN_WINDOW, FS, skf)
    print(f"    B2 Riemann multi-bandes    OOF = {s2:.3f}")

    s3 = oof_fgmdm(X_train, y_train, RIEMANN_BAND_B3, RIEMANN_WINDOW, FS, skf)
    print(f"    B3 FgMDM                   OOF = {s3:.3f}")

    s4 = oof_eegnet_fast(X_eeg_tr, y_bin, FS)
    print(f"    B4 EEGNet (rapide)         OOF = {s4:.3f}")

    s5 = oof_fbcsp_svm(X_train, y_train, CORE_BANDS, CORE_WINDOWS, N_CSP_CORE, FS, skf)
    print(f"    B5 FBCSP-core + SVM        OOF = {s5:.3f}")

    scores  = np.array([s1, s2, s3, s4, s5])
    weights = scores / scores.sum()
    weights = np.maximum(weights, 0.10)
    weights = weights / weights.sum()
    print(f"  Poids : {weights.round(3)}")

    # ------------------------------------------------------------------
    # ENTRAÎNEMENT FINAL
    # ------------------------------------------------------------------
    print("  Entraînement final...")

    # B1
    F1_tr, F1_te = fbcsp_features(X_train, y_train, X_test, FINE_BANDS, FINE_WINDOWS)
    sc1 = StandardScaler()
    F1_tr = sc1.fit_transform(F1_tr); F1_te = sc1.transform(F1_te)
    clf1 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf1.fit(F1_tr, y_train)

    # B2 : 3 LogReg par bande, vote moyen
    b2_te_parts = []
    for band in RIEMANN_BANDS:
        Ftr, Fte, _, _ = riemann_ts(X_train, X_test, band, RIEMANN_WINDOW)
        sc_b = StandardScaler()
        Ftr = sc_b.fit_transform(Ftr); Fte = sc_b.transform(Fte)
        clf_b = LogisticRegression(C=0.05, max_iter=3000, solver="lbfgs")
        clf_b.fit(Ftr, y_train)
        b2_te_parts.append(clf_b.predict_proba(Fte))

    # B3
    Ctr3, Cte3 = riemann_cov_only(X_train, X_test, RIEMANN_BAND_B3, RIEMANN_WINDOW)
    clf3 = FgMDM(metric="riemann", tsupdate=False)
    clf3.fit(Ctr3, y_train)

    # B4 : EEGNet ensemble
    print(f"    EEGNet : {N_EEGNET} modèles × {N_EPOCHS_EEG} epochs...")
    eeg_probas = []
    for seed in range(N_EEGNET):
        mdl = train_eegnet(X_eeg_tr, y_bin, seed=seed, n_epochs=N_EPOCHS_EEG)
        eeg_probas.append(predict_eegnet(mdl, X_eeg_te, n_tta=N_TTA))
    p4_te = np.mean(eeg_probas, axis=0)   # [P(left), P(right)]

    # B5 : SVM-RBF (GridSearch C)
    F5_tr, F5_te = fbcsp_features(X_train, y_train, X_test,
                                   CORE_BANDS, CORE_WINDOWS, N_CSP_CORE)
    sc5 = StandardScaler()
    F5_tr = sc5.fit_transform(F5_tr); F5_te = sc5.transform(F5_te)
    cv5  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs5  = GridSearchCV(
        SVC(kernel="rbf", probability=True, random_state=SEED),
        {"C": [0.05, 0.2, 0.8], "gamma": ["scale"]},
        cv=cv5, scoring="accuracy", n_jobs=-1,
    )
    gs5.fit(F5_tr, y_train)
    clf5 = gs5.best_estimator_
    print(f"    B5 meilleur C={clf5.C}")

    # ------------------------------------------------------------------
    # PRÉDICTIONS
    # ------------------------------------------------------------------
    p1 = clf1.predict_proba(F1_te)
    p2 = np.mean(b2_te_parts, axis=0)
    p3 = align_proba(clf3.predict_proba(Cte3), clf3.classes_, classes)
    p4 = p4_te        # [P(left_hand), P(right_hand)] par construction
    p5 = align_proba(clf5.predict_proba(F5_te), clf5.classes_, classes)

    p_final = (weights[0] * p1 + weights[1] * p2 + weights[2] * p3
             + weights[3] * p4 + weights[4] * p5)

    y_pred = classes[np.argmax(p_final, axis=1)]
    print(f"  left={int((y_pred=='left_hand').sum())}  right={int((y_pred=='right_hand').sum())}")

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

print(f"\nFini ! ZIP : {ZIP_NAME}")
