import numpy as np
import pandas as pd
import os
import zipfile
from scipy.signal import butter, sosfiltfilt
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =========================
# CONFIGURATION
# =========================
DATA_PATH = "data"
SUBJECTS = ['A', 'B', 'C', 'D', 'E', 'F']
OUTPUT_DIR = "predictions"
ZIP_NAME = "BCI_predictions_512Hz_Optimized.zip"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 512 
FBCSP_BANDS = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]

# =========================
# SIGNAL PROCESSING
# =========================
def bandpass_sos(X, low, high, fs=FS, order=4):
    """
    Filtre passe-bande SOS.
    CRITIQUE : Forcer float64 pour éviter l'instabilité numérique à 512 Hz.
    """
    X_float = X.astype(np.float64) 
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, X_float, axis=-1)

def crop_signal(X, start_sec=1.0, end_sec=3.5, fs=FS):
    """
    Isole la fenêtre temporelle pertinente (l'imagerie motrice pure).
    À ajuster selon ton dataset (ex: le cue visuel disparaît à 1s).
    """
    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)
    return X[:, :, start_idx:end_idx]

# =========================
# CUSTOM SCIKIT-LEARN TRANSFORMERS
# =========================
class LogVarExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return np.log(np.var(X, axis=-1) + 1e-8)

class CustomCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4, reg=1e-5):
        self.n_components = n_components
        self.reg = reg # Régularisation vitale pour le score privé
        
    def fit(self, X, y):
        classes = np.unique(y)
        
        # Calcul des covariances moyennes par classe
        def _get_cov(X_class):
            covs = [np.cov(trial) for trial in X_class]
            mean_cov = np.mean(covs, axis=0)
            # Régularisation de Tikhonov pour stabiliser l'inversion
            return mean_cov + self.reg * np.eye(mean_cov.shape[0])

        cov0 = _get_cov(X[y == classes[0]])
        cov1 = _get_cov(X[y == classes[1]])
        
        # Problème aux valeurs propres généralisées
        evals, evecs = eigh(cov0, cov0 + cov1)
        
        # Tri et sélection des filtres spatiaux extrêmes
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        half = self.n_components // 2
        self.filters_ = np.hstack((evecs[:, :half], evecs[:, -half:]))
        return self
        
    def transform(self, X):
        # Projection spatiale et log-variance
        X_filtered = np.asarray([np.dot(self.filters_.T, trial) for trial in X])
        return np.log(np.var(X_filtered, axis=-1) + 1e-8)

class FBCSP(BaseEstimator, TransformerMixin):
    def __init__(self, bands, n_components=4, fs=FS):
        self.bands = bands
        self.n_components = n_components
        self.fs = fs
        self.csps = [CustomCSP(n_components=n_components) for _ in bands]
        
    def fit(self, X, y):
        for i, band in enumerate(self.bands):
            X_band = bandpass_sos(X, band[0], band[1], self.fs)
            self.csps[i].fit(X_band, y)
        return self
        
    def transform(self, X):
        features = []
        for i, band in enumerate(self.bands):
            X_band = bandpass_sos(X, band[0], band[1], self.fs)
            feat = self.csps[i].transform(X_band)
            features.append(feat)
        return np.hstack(features)

# =========================
# MAIN - SÉLECTION PAR SUJET
# =========================
for subject in SUBJECTS:
    print(f"\n🚀 Sujet {subject}")
    
    # Chargement
    X_train_raw = np.load(f"{DATA_PATH}/subject_{subject}_X_train.npy")
    y_train = np.load(f"{DATA_PATH}/subject_{subject}_y_train.npy")
    X_test_raw = np.load(f"{DATA_PATH}/subject_{subject}_X_test.npy")
    
    # IMPORTANT: On filtre AVANT de croper pour éviter les effets de bord du filtre
    # Prétraitement Broad (4-40Hz)
    X_train_broad = crop_signal(bandpass_sos(X_train_raw, 4, 40))
    X_test_broad  = crop_signal(bandpass_sos(X_test_raw, 4, 40))
    
    # Prétraitement MI Classique (8-30Hz)
    X_train_mi = crop_signal(bandpass_sos(X_train_raw, 8, 30))
    X_test_mi  = crop_signal(bandpass_sos(X_test_raw, 8, 30))
    
    # FBCSP gère ses propres filtres, on lui donne juste les signaux croppés
    X_train_base = crop_signal(X_train_raw)
    X_test_base  = crop_signal(X_test_raw)

    # Dictionnaire de pipelines
    pipelines = {
        "LogVar_4-40Hz_LDA": Pipeline([
            ('logvar', LogVarExtractor()),
            ('scaler', StandardScaler()),
            ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ]),
        "CSP_8-30Hz_LDA": Pipeline([
            ('csp', CustomCSP(n_components=6)), # 6 composantes marchent souvent mieux sur 8-30Hz
            ('scaler', StandardScaler()),
            ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ]),
        "FBCSP_LDA": Pipeline([
            ('fbcsp', FBCSP(bands=FBCSP_BANDS, n_components=4)),
            ('scaler', StandardScaler()),
            ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ]),
        "FBCSP_LogReg": Pipeline([
            ('fbcsp', FBCSP(bands=FBCSP_BANDS, n_components=4)),
            ('scaler', StandardScaler()),
            # LogReg fort régularisé (C=0.05) pour FBCSP qui génère beaucoup de features
            ('clf', LogisticRegression(C=0.05, solver='liblinear', max_iter=2000)) 
        ])
    }
    
    data_train_map = {
        "LogVar_4-40Hz_LDA": X_train_broad,
        "CSP_8-30Hz_LDA": X_train_mi,
        "FBCSP_LDA": X_train_base,
        "FBCSP_LogReg": X_train_base
    }
    
    data_test_map = {
        "LogVar_4-40Hz_LDA": X_test_broad,
        "CSP_8-30Hz_LDA": X_test_mi,
        "FBCSP_LDA": X_test_base,
        "FBCSP_LogReg": X_test_base
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = 0
    best_pipe_name = ""
    
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, data_train_map[name], y_train, cv=cv, scoring='accuracy')
        mean_score = scores.mean()
        print(f" - {name}: {mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_pipe_name = name

    print(f"🏆 Élu: {best_pipe_name} avec {best_score:.4f}")
    
    # Entraînement Final & Prédiction
    best_pipe = pipelines[best_pipe_name]
    best_pipe.fit(data_train_map[best_pipe_name], y_train)
    y_pred = best_pipe.predict(data_test_map[best_pipe_name])
    
    pd.DataFrame({'y_pred': y_pred}).to_csv(
        f"{OUTPUT_DIR}/subject_{subject}_y_pred.csv", index=False
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
print("\n📦 ZIP PRÊT:", ZIP_NAME)