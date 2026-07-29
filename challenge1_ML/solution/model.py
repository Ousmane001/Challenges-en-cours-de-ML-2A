import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, Ridge, HuberRegressor, BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.model_selection import KFold
import warnings

# Ignorer les warnings de convergence 
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

class Model:
    def __init__(self):
        self.feature_columns_ = None
        self.meta_model_ = BayesianRidge()
        self.base_models_ = []
        self.seeds_ = [0, 42, 99, 123, 2024] # Les 5 seeds pour tuer la variance

    def _prepare_features(self, X, fit=False):
        """Standardisation et encodage sécurisés des données."""
        X_df = X.copy()
        
        # Encodage du genre
        if "gender" in X_df.columns:
            X_df["gender"] = X_df["gender"].map({"f": 0.0, "m": 1.0}).fillna(0.0)
        
        X_df = X_df.astype(float, copy=False)

        if fit:
            self.feature_columns_ = list(X_df.columns)
        else:
            X_df = X_df.reindex(columns=self.feature_columns_, fill_value=0.0)

        return X_df.to_numpy(dtype=np.float64)

    def _get_model_blueprints(self, n_features):
        """Génère les 5 architectures expertes."""
        pipe_pca_ridge = Pipeline([
            ("var", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_regression, k=min(2500, n_features))),
            ("pca", PCA(n_components=120, random_state=42)),
            ("model", Ridge(alpha=15.0, random_state=42))
        ])

        pipe_ridge = Pipeline([
            ("var", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_regression, k=min(3000, n_features))),
            ("model", Ridge(alpha=20.0, random_state=42))
        ])

        pipe_enet = Pipeline([
            ("var", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_regression, k=min(2000, n_features))),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.4, max_iter=10000, random_state=42))
        ])
        
        pipe_huber = Pipeline([
            ("var", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_regression, k=min(1500, n_features))),
            ("model", HuberRegressor(epsilon=1.35, alpha=10.0, max_iter=2000))
        ])

        pipe_pls = Pipeline([
            ("var", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_regression, k=min(2500, n_features))),
            ("model", PLSRegression(n_components=10))
        ])
        
        return [pipe_pca_ridge, pipe_ridge, pipe_enet, pipe_huber, pipe_pls]

    def fit(self, X, y):
        """
        Entraînement via Bagged Stacking :
        1. Boucle OOF (Out-Of-Fold) répétée 5 fois pour entraîner le méta-modèle.
        2. Entraînement final des modèles de base sur 100% des données.
        """
        X_np = self._prepare_features(X, fit=True)
        y_np = np.ravel(y).astype(float)
        
        n_samples, n_features = X_np.shape
        blueprints = self._get_model_blueprints(n_features)
        n_models = len(blueprints)
        
        # Matrice pour stocker les prédictions "Out-Of-Fold" lissées par les 5 seeds
        oof_predictions = np.zeros((n_samples, n_models))

        # 1. GÉNÉRATION DES PRÉDICTIONS OUT-OF-FOLD (Anti-Overfitting)
        for seed in self.seeds_:
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            
            for tr_idx, val_idx in kf.split(X_np):
                X_tr, X_val = X_np[tr_idx], X_np[val_idx]
                y_tr = y_np[tr_idx]
                
                # On clone les modèles pour qu'ils soient vierges à chaque pli
                fold_models = [clone(m) for m in blueprints]
                
                for i, model in enumerate(fold_models):
                    model.fit(X_tr, y_tr)
                    # On accumule les prédictions (divisées par le nombre de seeds pour faire la moyenne)
                    oof_predictions[val_idx, i] += model.predict(X_val) / len(self.seeds_)

        # 2. ENTRAÎNEMENT DU MÉTA-MODÈLE
        # Le BayesianRidge apprend comment pondérer les modèles en se basant sur la matrice OOF
        self.meta_model_.fit(oof_predictions, y_np)

        # 3. ENTRAÎNEMENT FINAL DES MODÈLES DE BASE (Pour la phase de test)
        # On ré-entraîne les 5 modèles sur TOUTE la donnée d'entraînement pour maximiser le savoir
        self.base_models_ = [clone(m) for m in blueprints]
        for model in self.base_models_:
            model.fit(X_np, y_np)

    def predict(self, X):
        """Prédiction : On passe X dans les modèles de base, puis dans le méta-modèle."""
        X_np = self._prepare_features(X, fit=False)
        
        # 1. Récupération des prédictions des 5 modèles experts
        base_preds = np.zeros((X_np.shape[0], len(self.base_models_)))
        for i, model in enumerate(self.base_models_):
            base_preds[:, i] = np.ravel(model.predict(X_np))
            
        # 2. Le méta-modèle combine ces prédictions
        final_preds = self.meta_model_.predict(base_preds)
        
        # 3. On contraint l'âge aux limites biologiques du dataset (18 - 70 ans)
        return np.clip(final_preds, 18.0, 70.0)