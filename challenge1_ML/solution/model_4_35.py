import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor
import warnings

class Model:
    def __init__(self):
        self.feature_columns_ = None
        self.model_ = None
        self.target_min_ = None
        self.target_max_ = None
        self.random_state = 42

    def _prepare_features(self, X, fit=False):
        X_df = X.copy()
        
        if "gender" in X_df.columns:
            X_df["gender"] = X_df["gender"].map({"f": 0.0, "m": 1.0}).fillna(0.0)
        
        X_df = X_df.astype(float, copy=False)

        if fit:
            self.feature_columns_ = list(X_df.columns)
        else:
            X_df = X_df.reindex(columns=self.feature_columns_, fill_value=0.0)

        return X_df.to_numpy(dtype=np.float64)

    def fit(self, X, y):
        X_np = self._prepare_features(X, fit=True)
        y_np = np.ravel(y).astype(float)
        
        self.target_min_ = float(np.min(y_np))
        self.target_max_ = float(np.max(y_np))
        n_features = X_np.shape[1]

        # -----------------------------------------------------------------
        # 1. Ridge avec Auto-Tuning (Trouve l'alpha optimal tout seul)
        # -----------------------------------------------------------------
        pipe_ridge = Pipeline([
            ('var', VarianceThreshold()),
            ('kbest', SelectKBest(score_func=f_regression, k=min(4000, n_features))),
            ('scaler', StandardScaler()),
            # Cherche le meilleur alpha parmi 50 valeurs
            ('ridge', RidgeCV(alphas=np.logspace(-2, 3, 50))) 
        ])

        # -----------------------------------------------------------------
        # 2. PLS Regression (Spécialiste des données ADN très corrélées)
        # -----------------------------------------------------------------
        pipe_pls = Pipeline([
            ('var', VarianceThreshold()),
            ('kbest', SelectKBest(score_func=f_regression, k=min(2000, n_features))),
            ('scaler', StandardScaler()),
            ('pls', PLSRegression(n_components=15))
        ])

        # -----------------------------------------------------------------
        # 3. ElasticNet avec Auto-Tuning
        # -----------------------------------------------------------------
        pipe_enet = Pipeline([
            ('var', VarianceThreshold()),
            ('kbest', SelectKBest(score_func=f_regression, k=min(2000, n_features))),
            ('scaler', StandardScaler()),
            ('enet', ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.9], 
                alphas=np.logspace(-3, 1, 20), 
                max_iter=5000, 
                cv=3, 
                random_state=self.random_state
            ))
        ])

        # -----------------------------------------------------------------
        # Vote final pondéré
        # -----------------------------------------------------------------
        # On fait légèrement plus confiance au Ridge auto-ajusté
        self.model_ = VotingRegressor([
            ('ridge', pipe_ridge),
            ('pls', pipe_pls),
            ('enet', pipe_enet)
        ], weights=[0.4, 0.3, 0.3])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(X_np, y_np)

    def predict(self, X):
        X_np = self._prepare_features(X, fit=False)
        preds = np.ravel(self.model_.predict(X_np))
        
        return np.clip(preds, self.target_min_, self.target_max_)