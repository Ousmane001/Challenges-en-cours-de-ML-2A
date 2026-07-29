import numpy as np
import pandas as pd
import os
import zipfile

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

DATA_PATH = "data"
SUBJECTS = ['A', 'B', 'C', 'D', 'E', 'F']
OUTPUT_DIR = "predictions"
ZIP_NAME = "BCI_predictions.zip"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1. RETOUR À TON PREPROCESSING GAGNANT
# =========================
def crop(X):
    return X[:, :, 200:1200]

def normalize(X):
    # Centrage et réduction par canal et par essai
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True) + 1e-10
    return (X - mean) / std

cropper = FunctionTransformer(crop)
normalizer = FunctionTransformer(normalize)

# =========================
# MAIN
# =========================
for subject in SUBJECTS:
    print(f"\n🧠 Processing subject {subject}...")

    X_train = np.load(f"{DATA_PATH}/subject_{subject}_X_train.npy")
    y_train = np.load(f"{DATA_PATH}/subject_{subject}_y_train.npy")
    X_test = np.load(f"{DATA_PATH}/subject_{subject}_X_test.npy")

    y_train_bin = np.array([0 if y == 'left_hand' else 1 for y in y_train])

    # 2. LE PIPELINE PROPRE
    pipeline = Pipeline([
        ('crop', cropper),
        ('norm', normalizer),
        ('cov', Covariances(estimator='oas')), 
        ('ts', TangentSpace(metric='riemann')),
        ('scaler', StandardScaler()),
        # Le solver 'saga' permet d'utiliser la pénalité L1 (Lasso) ou L2
        ('clf', LogisticRegression(solver='saga', max_iter=3000, class_weight='balanced'))
    ])

    # 3. RECHERCHE INTELLIGENTE AUTOUR DE TES VALEURS
    param_grid = {
        'clf__penalty': ['l1', 'l2'], # L1 va supprimer les features inutiles
        'clf__C': [0.01, 0.05, 0.1, 0.5, 1.0] # On explore autour de ton 0.1 d'origine
    }

    # cv=5 pour fiabiliser le score
    grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    
    grid.fit(X_train, y_train_bin)

    print(f"   -> Meilleurs paramètres : {grid.best_params_}")
    print(f"   -> Score interne (CV) estimé : {grid.best_score_:.3f}")

    y_pred_bin = grid.predict(X_test)
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

print("\n✅ ZIP ready:", ZIP_NAME)