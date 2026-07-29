import numpy as np
import pandas as pd
import os
import zipfile
from scipy.signal import butter, filtfilt

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATA_PATH = "data"
SUBJECTS = ['A', 'B', 'C', 'D', 'E', 'F']
OUTPUT_DIR = "predictions"
ZIP_NAME = "BCI_predictions.zip"
# ⚠️ IMPORTANT : Remplace par la vraie fréquence d'échantillonnage de tes données !
SAMPLING_FREQ = 250.0  

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# PREPROCESSING
# =========================
def crop(X):
    # La fenêtre 200:1200 dépend de ta fréquence d'échantillonnage.
    # Assure-toi qu'elle correspond au moment où le sujet "pense" au mouvement.
    return X[:, :, 200:1200]

def bandpass_filter(X, lowcut=8.0, highcut=30.0, fs=250.0, order=4):
    """
    Filtre le signal pour ne garder que les ondes Mu et Beta (8-30 Hz),
    cruciales pour l'imagerie motrice.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # filtfilt applique le filtre dans les deux sens pour éviter le déphasage
    return filtfilt(b, a, X, axis=-1)

# Note: Je retire ta fonction normalize car le filtre passe-bande centre 
# déjà le signal autour de 0, et pyriemann gère très bien l'échelle via la covariance.

# =========================
# MAIN
# =========================
for subject in SUBJECTS:
    print(f"Processing subject {subject}...")

    # Chargement
    X_train = np.load(f"{DATA_PATH}/subject_{subject}_X_train.npy")
    y_train = np.load(f"{DATA_PATH}/subject_{subject}_y_train.npy")
    X_test = np.load(f"{DATA_PATH}/subject_{subject}_X_test.npy")

    # Binarisation
    y_train_bin = np.array([0 if y == 'left_hand' else 1 for y in y_train])

    # 1. Découpage temporel
    X_train = crop(X_train)
    X_test = crop(X_test)

    # 2. Filtrage Fréquentiel (LA clé du succès)
    X_train = bandpass_filter(X_train, fs=SAMPLING_FREQ)
    X_test = bandpass_filter(X_test, fs=SAMPLING_FREQ)

    # Création d'un Pipeline propre
    # LogisticRegressionCV va tester automatiquement plusieurs valeurs de 'C'
    # en validation croisée pour trouver la meilleure régularisation spécifique à CE sujet.
    clf_pipeline = make_pipeline(
        Covariances(estimator='oas'), 
        TangentSpace(),
        StandardScaler(),
        LogisticRegressionCV(
            cv=5,               # Validation croisée à 5 plis en interne
            max_iter=2000,
            scoring='accuracy',
            solver='lbfgs',
            class_weight='balanced', # Au cas où tes classes sont légèrement déséquilibrées
            n_jobs=-1           # Utilise tous les coeurs de ton processeur
        )
    )

    # Entraînement
    clf_pipeline.fit(X_train, y_train_bin)
    
    # Affichage du C optimal trouvé pour le sujet (pour info)
    best_c = clf_pipeline.named_steps['logisticregressioncv'].C_[0]
    print(f"   -> Meilleur hyperparamètre C trouvé : {best_c:.4f}")

    # Prédiction
    y_pred_bin = clf_pipeline.predict(X_test)
    y_pred = np.array(['left_hand' if y == 0 else 'right_hand' for y in y_pred_bin])

    # Sauvegarde
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