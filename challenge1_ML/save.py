import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Model:
    def __init__(self):
        # Saved at fit-time to guarantee train/test column alignment.
        self.feature_columns_ = None
        # Best estimator selected via CV among all candidates.
        self.best_model_ = None
        self.best_name_ = None
        self.best_cv_rmse_ = None
        self.cv_scores_ = {}
        self.selected_models_ = []
        self.selected_weights_ = []
        self.target_min_ = None
        self.target_max_ = None
        self.random_state = 42

    def _prepare_features(self, X, fit=False):
        # Work on a copy to avoid mutating caller dataframes.
        X_df = X.copy()

        # Encode categorical gender to numeric values expected by sklearn estimators.
        if "gender" in X_df.columns:
            X_df["gender"] = X_df["gender"].map({"f": 0.0, "m": 1.0})
            X_df["gender"] = X_df["gender"].fillna(0.0)

        # Ensure homogeneous numeric dtype for all downstream operations.
        X_df = X_df.apply(lambda col: col.astype(float))

        if fit:
            # Persist training column order for deterministic test-time reindexing.
            self.feature_columns_ = list(X_df.columns)
        else:
            # Protect against column order mismatch or missing columns at inference.
            X_df = X_df.reindex(columns=self.feature_columns_, fill_value=0.0)

        return X_df.to_numpy(dtype=np.float64)

    def _candidate_models(self, n_samples):
        # Fast search space: strong regularized linear models only.
        models = {}

        # Ridge with feature filtering is often very effective in p >> n.
        # Keep only a tiny set of strong candidates for fast runtime.
        for k in [1500, 3000]:
            for alpha in [0.3]:
                name = "kbest{}_ridge_alpha_{}".format(k, alpha)
                models[name] = Pipeline(
                    steps=[
                        ("kbest", SelectKBest(score_func=f_regression, k=k)),
                        ("scaler", StandardScaler()),
                        ("ridge", Ridge(alpha=alpha, random_state=self.random_state)),
                    ]
                )

        for alpha in [1.0]:
            name = "ridge_alpha_{}".format(alpha)
            models[name] = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha, random_state=self.random_state)),
                ]
            )

        return models

    def _cv_rmse(self, model, X, y, splitter):
        # Manual CV loop to expose exactly how RMSE is computed fold-by-fold.
        fold_rmses = []
        for train_idx, valid_idx in splitter.split(X):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            # Clone to keep each fold independent (no parameter leakage).
            m = clone(model)
            m.fit(X_train, y_train)
            preds = m.predict(X_valid)
            rmse = np.sqrt(np.mean((preds - y_valid) ** 2))
            fold_rmses.append(rmse)

        return float(np.mean(fold_rmses))

    def fit(self, X, y):
        # Prepare features and flatten target to shape (n_samples,).
        X_np = self._prepare_features(X, fit=True)
        y_np = np.ravel(y).astype(float)
        self.target_min_ = float(np.min(y_np))
        self.target_max_ = float(np.max(y_np))

        # 3-fold is much faster than 5-fold and usually still reliable for ranking models.
        splitter = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        candidates = self._candidate_models(n_samples=X_np.shape[0])

        best_name = None
        best_model = None
        best_rmse = np.inf
        scored_models = []

        for name, model in candidates.items():
            rmse = self._cv_rmse(model, X_np, y_np, splitter)
            self.cv_scores_[name] = rmse
            scored_models.append((name, rmse, model))
            if rmse < best_rmse:
                best_rmse = rmse
                best_name = name
                best_model = clone(model)

        # Keep metadata for introspection and retrain winner on all training data.
        self.best_model_ = best_model
        self.best_name_ = best_name
        self.best_cv_rmse_ = best_rmse
        self.best_model_.fit(X_np, y_np)

        # Build a simple weighted blend of top-2 CV models for better generalization.
        scored_models.sort(key=lambda x: x[1])
        top_models = scored_models[:2]
        self.selected_models_ = []
        self.selected_weights_ = []

        # Higher weight for lower RMSE via inverse-error weighting.
        inv_scores = np.array([1.0 / max(t[1], 1e-8) for t in top_models], dtype=float)
        weights = inv_scores / np.sum(inv_scores)

        for i, (name, rmse, model) in enumerate(top_models):
            fitted = clone(model)
            fitted.fit(X_np, y_np)
            self.selected_models_.append((name, fitted, rmse))
            self.selected_weights_.append(float(weights[i]))

    def predict(self, X):
        # Reuse exact same feature prep logic used during training.
        X_np = self._prepare_features(X, fit=False)
        if self.selected_models_:
            blend = np.zeros(X_np.shape[0], dtype=float)
            for w, (_, model, _) in zip(self.selected_weights_, self.selected_models_):
                blend += w * np.ravel(model.predict(X_np))
            # Keep predictions in a realistic age range observed in training labels.
            return np.clip(blend, self.target_min_, self.target_max_)

        preds = np.ravel(self.best_model_.predict(X_np))
        return np.clip(preds, self.target_min_, self.target_max_)
