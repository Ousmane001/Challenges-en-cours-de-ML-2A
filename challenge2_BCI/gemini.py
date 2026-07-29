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
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import BaggingClassifier

# =========================
# CONFIGURATION
# =========================
DATA_PATH = "data"
SUBJECTS = ['A', 'B', 'C', 'D', 'E', 'F']
OUTPUT_DIR = "predictions"
ZIP_NAME = "BCI_predictions_512Hz_Bagging_Augmented.zip"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 512 
FBCSP_BANDS = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]

# =========================
# SIGNAL PROCESSING & AUGMENTATION
# =========================
def bandpass_sos(X, low, high, fs=FS, order=4):
    X_float = X.astype(np.float64) 
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, X_float, axis=-1)

def augment_train_data(X, y, base_start=1.0, base_end=3.0, fs=FS):
    shifts = [-0.1, 0.0, 0.1]
    X_aug, y_aug = [], []
    
    # Taille exacte de la fenêtre pour éviter les erreurs de vstack
    window_samples = int((base_end - base_start) * fs)
    max_len = X.shape[2] 
    
    for shift in shifts:
        start_idx = int((base_start + shift) * fs)
        end_idx = start_idx + window_samples
        
        # Sécurité : on recale si on dépasse les bords du signal
        if end_idx > max_len:
            end_idx = max_len
            start_idx = end_idx - window_samples
        if start_idx < 0:
            start_idx = 0
            end_idx = window_samples
            
        X_aug.append(X[:, :, start_idx:end_idx])
        y_aug.append(y)
        
    return np.vstack(X_aug), np.hstack(y_aug)

def crop_signal(X, start_sec, end_sec, fs=FS):
    window_samples = int((end_sec - start_sec) * fs)
    start_idx = int(start_sec * fs)
    end_idx = start_idx + window_samples
    
    max_len = X.shape[2]
    
    if end_idx > max_len:
        end_idx = max_len
        start_idx = end_idx - window_samples
        
    return X[:, :, start_idx:end_idx]

# =========================
# CUSTOM SCIKIT-LEARN TRANSFORMERS
# =========================
class CustomCSP_LedoitWolf(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components
        
    def fit(self, X, y):
        classes = np.unique(y)
        
        def _get_robust_cov(X_class):
            X_concat = np.hstack(X_class)
            lw = LedoitWolf().fit(X_concat.T)
            return lw.covariance_

        cov0 = _get_robust_cov(X[y == classes[0]])
        cov1 = _get_robust_cov(X[y == classes[1]])
        
        evals, evecs = eigh(cov0, cov0 + cov1)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        half = self.n_components // 2
        self.filters_ = np.hstack((evecs[:, :half], evecs[:, -half:]))
        return self
        
    def transform(self, X):
        X_filtered = np.asarray([np.dot(self.filters_.T, trial) for trial in X])
        return np.log(np.var(X_filtered, axis=-1) + 1e-8)

class FBCSP(BaseEstimator, TransformerMixin):
    def __init__(self, bands, n_components=4, fs=FS):
        self.bands = bands
        self.n_components = n_components
        self.fs = fs
        self.csps = [CustomCSP_LedoitWolf(n_components=n_components) for _ in bands]
        
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
    print(f"\n🚀 Entraînement Sujet {subject}")
    
    X_train_raw = np.load(f"{DATA_PATH}/subject_{subject}_X_train.npy")
    y_train = np.load(f"{DATA_PATH}/subject_{subject}_y_train.npy")
    X_test_raw = np.load(f"{DATA_PATH}/subject_{subject}_X_test.npy")
    
    # Augmentation des données Train (3x plus de données)
    X_train_aug, y_train_aug = augment_train_data(X_train_raw, y_train, base_start=1.0, base_end=3.0)
    
    # Coupure normale du Test
    X_test_crop = crop_signal(X_test_raw, 1.0, 3.0)

    # -----------------------------------------------------
    # L'ASTUCE DU BAGGING : On le place APRES le FBCSP
    # -----------------------------------------------------
    
    # 1. Le sous-pipeline 2D qui sera cloné 15 fois
    inner_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feat_sel', SelectKBest(f_classif, k=12)),
        ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    ])
    
    # 2. Le classifieur d'Ensemble
    bagging_clf = BaggingClassifier(
        estimator=inner_pipeline,
        n_estimators=15,
        max_samples=0.8, 
        n_jobs=-1,       
        random_state=42
    )
    
    # 3. Le pipeline final complet (3D in -> 2D -> Bagging)
    final_pipeline = Pipeline([
        ('fbcsp', FBCSP(bands=FBCSP_BANDS, n_components=4)),
        ('bagging', bagging_clf)
    ])
    
    # Évaluation croisée pour suivre l'évolution
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(final_pipeline, X_train_aug, y_train_aug, cv=cv, scoring='accuracy')
    print(f"   - Validation Croisée (FBCSP + Bagging) : {scores.mean():.4f}")
    
    # Entraînement sur tout le dataset augmenté
    final_pipeline.fit(X_train_aug, y_train_aug)
    
    # Prédiction sur le Test
    y_pred = final_pipeline.predict(X_test_crop)
    
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
print("\n🏆 ZIP PRÊT:", ZIP_NAME)