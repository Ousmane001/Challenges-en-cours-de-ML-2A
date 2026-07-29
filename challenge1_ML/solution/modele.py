import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import loguniform, randint
import warnings
import pandas as pd  # Import local pour flexibilité



class Model:
    def __init__(self, max_search_iter=30, use_ensemble=True, verbose=0):
        """
        Parameters
        ----------
        max_search_iter : int
            Budget computationnel par famille de modèles (défaut: 30).
            Réduire à 10 pour debug, augmenter à 100 pour recherche fine.
        use_ensemble : bool
            Si True, active l'ensemble pondéré des top modèles.
        verbose : int
            0=silencieux, 1=infos essentielles, 2=détail recherche.
        """
        self.feature_columns_ = None
        self.best_model_ = None
        self.best_name_ = None
        self.best_cv_rmse_ = None
        self.cv_scores_ = {}
        self.selected_models_ = []
        self.selected_weights_ = []
        self.target_min_ = None
        self.target_max_ = None
        self.random_state = 42
        self.max_search_iter = max_search_iter
        self.use_ensemble = use_ensemble
        self.verbose = verbose
        
        # Stockage des résultats détaillés
        self.search_results_ = {}
        self.best_params_ = {}

    def _log(self, msg, level=1):
        """Logging conditionnel."""
        if self.verbose >= level:
            print(msg)

    def _prepare_features(self, X, fit=False):
        """Préparation robuste des features (inchangée, déjà solide)."""
        X_df = X.copy()

        if "gender" in X_df.columns:
            X_df["gender"] = X_df["gender"].map({"f": 0.0, "m": 1.0})
            X_df["gender"] = X_df["gender"].fillna(0.0)

        # Conversion numérique robuste
        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0.0)

        if fit:
            self.feature_columns_ = list(X_df.columns)
        else:
            X_df = X_df.reindex(columns=self.feature_columns_, fill_value=0.0)

        return X_df.to_numpy(dtype=np.float64)

    def _get_param_distributions(self, n_features):
        """Distributions intelligentes pour RandomizedSearch."""
        # k adaptatif : entre 10% et 100% des features, mais max 5000
        k_min = max(50, int(n_features * 0.1))
        k_max = min(n_features, 5000)
        
        return {
            'kbest_ridge': {
                'kbest__k': randint(k_min, k_max),
                'ridge__alpha': loguniform(1e-3, 1e3)
            },
            'ridge': {
                'ridge__alpha': loguniform(1e-3, 1e3)
            },
            'elasticnet': {
                'elasticnet__alpha': loguniform(1e-4, 1e2),
                'elasticnet__l1_ratio': loguniform(0.01, 0.99)
            },
            'lasso': {
                'lasso__alpha': loguniform(1e-4, 1e1)
            }
        }

    def _create_pipeline(self, model_type):
        """Factory de pipelines."""
        scaler = StandardScaler()
        
        if model_type == 'kbest_ridge':
            return Pipeline([
                ('kbest', SelectKBest(score_func=f_regression)),
                ('scaler', scaler),
                ('ridge', Ridge(random_state=self.random_state))
            ])
        elif model_type == 'ridge':
            return Pipeline([
                ('scaler', scaler),
                ('ridge', Ridge(random_state=self.random_state))
            ])
        elif model_type == 'elasticnet':
            return Pipeline([
                ('scaler', scaler),
                ('elasticnet', ElasticNet(
                    random_state=self.random_state, 
                    max_iter=5000,
                    tol=1e-3
                ))
            ])
        elif model_type == 'lasso':
            return Pipeline([
                ('scaler', scaler),
                ('lasso', Lasso(
                    random_state=self.random_state, 
                    max_iter=5000,
                    tol=1e-3
                ))
            ])

    def _rmse_scorer(self):
        """Scorer RMSE négatif (convention sklearn)."""
        return make_scorer(
            lambda y, p: -np.sqrt(mean_squared_error(y, p)),
            greater_is_better=True
        )

    def _adapt_cv_splits(self, n_samples):
        """Adapte le nombre de splits selon la taille du dataset."""
        if n_samples < 100:
            return 3
        elif n_samples < 1000:
            return 5
        else:
            return 3  # Pour grands datasets, 3 suffisent et c'est plus rapide

    def _search_model(self, name, X, y, cv_splitter, n_features):
        """Exécute RandomizedSearchCV avec budget fixe."""
        pipeline = self._create_pipeline(name)
        param_dist = self._get_param_distributions(n_features)[name]
        
        # Ajustement du nombre d'itérations selon la complexité du modèle
        n_iter = self.max_search_iter
        if name == 'kbest_ridge':
            n_iter = int(n_iter * 1.5)  # Plus de paramètres à optimiser
        
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=self._rmse_scorer(),
            cv=cv_splitter,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=0,
            return_train_score=False
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Silence convergence warnings
            search.fit(X, y)
        
        rmse = -search.best_score_
        self.search_results_[name] = {
            'best_params': search.best_params_,
            'best_rmse': rmse,
            'cv_results': search.cv_results_
        }
        
        self._log(f"  {name:15s}: RMSE={rmse:.4f} | {search.best_params_}", level=2)
        
        return name, rmse, search.best_estimator_

    def fit(self, X, y):
        """Entraînement optimisé avec budget contrôlé."""
        
        # Préparation des données
        X_np = self._prepare_features(X, fit=True)
        y_np = np.ravel(y).astype(float)
        n_samples, n_features = X_np.shape
        
        self.target_min_ = float(np.min(y_np))
        self.target_max_ = float(np.max(y_np))
        
        # Configuration CV adaptative
        n_splits = self._adapt_cv_splits(n_samples)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        self._log(f"Recherche sur {n_samples} échantillons, {n_features} features "
                 f"(CV={n_splits}-fold, budget={self.max_search_iter} itérations/modèle)", level=1)
        
        # Recherche parallèle sur toutes les familles de modèles
        candidates = ['ridge', 'kbest_ridge', 'elasticnet', 'lasso']
        results = []
        
        for candidate in candidates:
            try:
                result = self._search_model(candidate, X_np, y_np, splitter, n_features)
                results.append(result)
            except Exception as e:
                self._log(f"  {candidate:15s}: ÉCHEC ({str(e)})", level=1)
        
        if not results:
            raise RuntimeError("Aucun modèle n'a convergé. Vérifiez vos données.")
        
        # Sélection du meilleur modèle
        results.sort(key=lambda x: x[1])
        best_name, best_rmse, best_estimator = results[0]
        
        self.best_name_ = best_name
        self.best_cv_rmse_ = best_rmse
        self.best_params_ = self.search_results_[best_name]['best_params']
        
        self._log(f"\nMeilleur: {best_name} (RMSE={best_rmse:.4f})", level=1)
        
        # Réentraînement final sur toutes les données
        self.best_model_ = clone(best_estimator)
        self.best_model_.fit(X_np, y_np)
        
        # Construction conditionnelle de l'ensemble
        if self.use_ensemble and len(results) >= 2:
            self._build_smart_ensemble(results, X_np, y_np)
        else:
            self.selected_models_ = []
            self._log("Mode single-model (ensemble désactivé ou insuffisant de modèles)", level=2)

    def _build_smart_ensemble(self, results, X, y):
        """Construit un ensemble seulement si bénéfique."""
        top_3 = results[:min(3, len(results))]
        
        # Vérifie si l'ensemble vaut le coup (diversité des erreurs)
        rmse_values = [r[1] for r in top_3]
        rmse_std = np.std(rmse_values)
        
        # Si les modèles sont trop proches (< 1% d'écart), pas besoin d'ensemble
        if len(top_3) >= 2 and (rmse_values[1] - rmse_values[0]) / rmse_values[0] < 0.01:
            self._log("Top modèles trop similaires, ensemble désactivé", level=2)
            return
        
        self._log(f"\nConstruction ensemble (top {len(top_3)}):", level=1)
        
        # Pondération inverse de l'erreur avec régularisation pour stabilité
        inv_errors = 1.0 / (np.array(rmse_values) + 1e-6)
        weights = inv_errors / np.sum(inv_errors)
        
        self.selected_models_ = []
        self.selected_weights_ = []
        
        for i, (name, rmse, estimator) in enumerate(top_3):
            # Réentraînement sur full data
            model = clone(estimator)
            model.fit(X, y)
            self.selected_models_.append((name, model, rmse))
            self.selected_weights_.append(float(weights[i]))
            self._log(f"  {name}: poids={weights[i]:.3f}", level=2)

    def predict(self, X):
        """Prédiction avec gestion fallback."""
        X_np = self._prepare_features(X, fit=False)
        
        # Mode ensemble
        if self.selected_models_:
            predictions = np.zeros((X_np.shape[0], len(self.selected_models_)))
            for i, (_, model, _) in enumerate(self.selected_models_):
                predictions[:, i] = np.ravel(model.predict(X_np))
            
            # Moyenne pondérée
            blend = np.average(predictions, axis=1, weights=self.selected_weights_)
            return np.clip(blend, self.target_min_, self.target_max_)
        
        # Mode single best
        preds = np.ravel(self.best_model_.predict(X_np))
        return np.clip(preds, self.target_min_, self.target_max_)

    def get_feature_importance(self):
        """Extrait l'importance des features si disponible."""
        if self.best_model_ is None:
            return None
            
        importance = {}
        
        # Pour SelectKBest
        if hasattr(self.best_model_, 'named_steps') and 'kbest' in self.best_model_.named_steps:
            kbest = self.best_model_.named_steps['kbest']
            mask = kbest.get_support()
            scores = kbest.scores_
            importance['selected_features'] = [f for f, m in zip(self.feature_columns_, mask) if m]
            importance['f_scores'] = scores[mask] if scores is not None else None
        
        # Pour coefficients linéaires
        for step in ['ridge', 'elasticnet', 'lasso']:
            if hasattr(self.best_model_, 'named_steps') and step in self.best_model_.named_steps:
                coefs = self.best_model_.named_steps[step].coef_
                importance['coefficients'] = dict(zip(self.feature_columns_, coefs))
                importance['coef_abs_mean'] = np.mean(np.abs(coefs))
                break
                
        return importance

    def get_summary(self):
        """Résumé des performances pour comparaison."""
        summary = []
        for name, res in self.search_results_.items():
            summary.append({
                'model': name,
                'rmse_cv': res['best_rmse'],
                'best_params': res['best_params']
            })
        return sorted(summary, key=lambda x: x['rmse_cv'])


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    import pandas as pd
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    # Génération de données test réalistes (1000 features, 500 samples)
    X, y = make_regression(n_samples=500, n_features=1000, n_informative=50, 
                          noise=10, random_state=42)
    X = pd.DataFrame(X, columns=[f'f_{i}' for i in range(X.shape[1])])
    X['gender'] = np.random.choice(['f', 'm'], size=len(X))  # Test feature catégorielle
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Version rapide (debug)
    print("=== MODE RAPIDE (debug) ===")
    model_fast = Model(max_search_iter=10, use_ensemble=False, verbose=1)
    model_fast.fit(X_train, y_train)
    preds_fast = model_fast.predict(X_test)
    rmse_fast = np.sqrt(mean_squared_error(y_test, preds_fast))
    print(f"RMSE Test: {rmse_fast:.4f}")
    
    # Version standard (production)
    print("\n=== MODE PRODUCTION ===")
    model_prod = Model(max_search_iter=30, use_ensemble=True, verbose=1)
    model_prod.fit(X_train, y_train)
    preds_prod = model_prod.predict(X_test)
    rmse_prod = np.sqrt(mean_squared_error(y_test, preds_prod))
    print(f"RMSE Test: {rmse_prod:.4f}")
    
    # Affichage du résumé
    print("\nClassement CV:")
    for item in model_prod.get_summary():
        print(f"  {item['model']:15s}: {item['rmse_cv']:.4f}")