import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold, SelectFromModel
from sklearn.linear_model import RidgeCV, ElasticNetCV, HuberRegressor
from sklearn.svm import LinearSVR
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
        # L'Extracteur de Features Multivarié (Le secret de cette version)
        # -----------------------------------------------------------------
        # Ce bloc va réduire la dimension tout en supprimant la colinéarité
        def get_feature_selector():
            return Pipeline([
                ('var', VarianceThreshold()),
                ('kbest', SelectKBest(score_func=f_regression, k=min(3000, n_features))),
                ('scaler', StandardScaler()),
                ('meta_select', SelectFromModel(
                    LinearSVR(C=0.05, dual="auto", random_state=self.random_state, max_iter=5000), 
                    max_features=600 # Ne garde que les 600 gènes non-redondants les plus forts
                ))
            ])

        # -----------------------------------------------------------------
        # Les Modèles Experts (entraînés sur les features ultra-propres)
        # -----------------------------------------------------------------
        pipe_ridge = Pipeline([
            ('selector', get_feature_selector()),
            ('ridge', RidgeCV(alphas=np.logspace(-1, 4, 50)))
        ])

        pipe_huber = Pipeline([
            ('selector', get_feature_selector()),
            ('huber', HuberRegressor(epsilon=1.35, alpha=5.0, max_iter=2000))
        ])

        # Pour la PLS, on ne passe pas par le SelectFromModel car elle gère 
        # elle-même la colinéarité. On lui donne plus de données brutes.
        pipe_pls = Pipeline([
            ('var', VarianceThreshold()),
            ('kbest', SelectKBest(score_func=f_regression, k=min(2500, n_features))),
            ('scaler', StandardScaler()),
            ('pls', PLSRegression(n_components=8))
        ])
        
        # Le LinearSVR est excellent pour ignorer le bruit des données médicales
        pipe_lsvr = Pipeline([
            ('selector', get_feature_selector()),
            ('lsvr', LinearSVR(C=0.5, epsilon=0.5, dual="auto", random_state=self.random_state, max_iter=5000))
        ])

        # -----------------------------------------------------------------
        # L'Ensemble Final
        # -----------------------------------------------------------------
        self.model_ = VotingRegressor([
            ('ridge', pipe_ridge),
            ('huber', pipe_huber),
            ('pls', pipe_pls),
            ('lsvr', pipe_lsvr)
        ], weights=[0.30, 0.25, 0.25, 0.20])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(X_np, y_np)

    def predict(self, X):
        X_np = self._prepare_features(X, fit=False)
        preds = np.ravel(self.model_.predict(X_np))
        return np.clip(preds, self.target_min_, self.target_max_)