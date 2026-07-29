import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings

class Model:
    def __init__(self):
        self.feature_columns_ = None
        self.model_ = None
        self.target_min_ = None
        self.target_max_ = None
        self.random_state = 42

    def _prepare_features(self, X, fit=False):
        """Préparation des features avec gestion robuste des types."""
        X_df = X.copy()

        # Encodage du genre
        if "gender" in X_df.columns:
            X_df["gender"] = X_df["gender"].map({"f": 0.0, "m": 1.0}).fillna(0.0)

        # Forcer le type numérique pour éviter les erreurs sklearn
        X_df = X_df.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0.0)

        if fit:
            self.feature_columns_ = list(X_df.columns)
        else:
            X_df = X_df.reindex(columns=self.feature_columns_, fill_value=0.0)

        return X_df.to_numpy(dtype=np.float64)

    def fit(self, X, y):
        """Entraînement d'un Stacking Ensemblist robuste pour données génomiques."""
        X_np = self._prepare_features(X, fit=True)
        y_np = np.ravel(y).astype(float)
        
        # Mémoriser les bornes pour le clipping
        self.target_min_ = float(np.min(y_np))
        self.target_max_ = float(np.max(y_np))

        n_features = X_np.shape[1]

        # -----------------------------------------------------------------
        # NIVEAU 1 : Modèles de base (Diversité maximale)
        # -----------------------------------------------------------------
        
        # 1. Modèle linéaire classique : ElasticNet (Façon "Horvath Clock")
        pipe_enet = Pipeline([
            ('var', VarianceThreshold()), # Supprime les variables constantes
            ('kbest', SelectKBest(score_func=f_regression, k=min(3000, n_features))),
            ('scaler', StandardScaler()),
            ('enet', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=3000, random_state=self.random_state))
        ])

        # 2. Modèle linéaire robuste : Ridge avec plus de features
        pipe_ridge = Pipeline([
            ('var', VarianceThreshold()),
            ('kbest', SelectKBest(score_func=f_regression, k=min(5000, n_features))),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=15.0, random_state=self.random_state))
        ])

        # 3. Modèle non-linéaire : Gradient Boosting (Capture les interactions complexes)
        # On réduit fortement k pour éviter l'overfitting et les Timeouts sur les arbres
        pipe_hgb = Pipeline([
            ('var', VarianceThreshold()),
            ('kbest', SelectKBest(score_func=f_regression, k=min(800, n_features))),
            ('scaler', StandardScaler()),
            ('hgb', HistGradientBoostingRegressor(
                learning_rate=0.05, 
                max_iter=250, 
                max_depth=5, 
                min_samples_leaf=5,
                l2_regularization=1.0,
                random_state=self.random_state
            ))
        ])

        # -----------------------------------------------------------------
        # NIVEAU 2 : Méta-Modèle
        # -----------------------------------------------------------------
        estimators = [
            ('enet', pipe_enet),
            ('ridge', pipe_ridge),
            ('hgb', pipe_hgb)
        ]
        
        # Le StackingRegressor va entraîner les modèles de base via CV, 
        # puis utiliser un Ridge final pour apprendre à pondérer leurs prédictions.
        self.model_ = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0, random_state=self.random_state),
            cv=KFold(n_splits=5, shuffle=True, random_state=self.random_state),
            n_jobs=-1, # Utilise tous les coeurs disponibles
            passthrough=False # Le méta-modèle ne voit que les prédictions (évite l'overfitting)
        )

        # Silence les warnings de convergence habituels sur ce type de données
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(X_np, y_np)

    def predict(self, X):
        """Prédiction avec écrêtage (clipping) basé sur l'entraînement."""
        X_np = self._prepare_features(X, fit=False)
        preds = np.ravel(self.model_.predict(X_np))
        
        # Le clipping empêche le modèle de prédire des âges impossibles (ex: -5 ans ou 150 ans)
        return np.clip(preds, self.target_min_, self.target_max_)