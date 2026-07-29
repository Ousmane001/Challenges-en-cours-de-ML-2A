"""
BCI Motor Imagery - Version compétition (objectif > 0.95)
==========================================================
Techniques utilisées :
  - Filter Bank CSP custom (scipy.linalg.eigh) sur sous-bandes Mu/Beta
  - Riemann TangentSpace + Euclidean Alignment (EA)
  - Multi-fenêtres temporelles (sliding crops)
  - Stacking OOF avec meta-learner LogReg
  - Pseudo-labeling itératif sur le test set
  - Sélection de stratégie par sujet (single / ensemble / stacking)
  - fs = 512 Hz, sosfiltfilt
"""

import numpy as np
import pandas as pd
import os
import zipfile
from scipy.signal import butter, sosfiltfilt
from scipy.linalg import eigh, sqrtm, inv

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin, clone

from pyriemann.estimation import Covariances, Shrinkage
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIG
# =========================
DATA_PATH = "data"
SUBJECTS = ['A', 'B', 'C', 'D', 'E', 'F']
OUTPUT_DIR = "predictions"
ZIP_NAME = "BCI_predictions.zip"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 512
SEED = 42
N_FOLDS = 5

# Bandes
WIDE_BAND    = (4, 40)
CLASSIC_BAND = (8, 30)
SUB_BANDS = [
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
    (24, 30),
    (30, 38),
]

# Multi-fenêtres : si tes données sont à 512 Hz et durent ~4s = 2048 samples
# adapte ces fenêtres à la longueur réelle de tes essais
USE_WINDOWS = True
# Définies en proportion (start_ratio, end_ratio) de la longueur du signal
WINDOW_RATIOS = [
    (0.10, 0.65),
    (0.15, 0.70),
    (0.20, 0.80),
    (0.25, 0.85),
    (0.30, 0.95),
]

# Pseudo-labeling
USE_PSEUDO_LABEL = True
PSEUDO_CONF_THRESHOLD = 0.85   # seuil de confiance pour ajouter au train
PSEUDO_ITERATIONS = 2

# =========================
# PREPROCESS
# =========================
def bandpass_sos(X, low, high, fs=FS, order=4):
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, X, axis=-1)

def normalize(X):
    return (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)

def crop_ratio(X, start_ratio, end_ratio):
    T = X.shape[-1]
    s = int(start_ratio * T)
    e = int(end_ratio * T)
    return X[:, :, s:e]

# =========================
# EUCLIDEAN ALIGNMENT (clé pour générer ~0.05 de gain)
# =========================
def euclidean_alignment(X):
    """
    Recentre les covariances par sujet : X_aligned = R^{-1/2} @ X
    où R est la moyenne des covariances par essai.
    Très efficace pour homogénéiser la distribution des covariances.
    """
    covs = np.array([trial @ trial.T / np.trace(trial @ trial.T) for trial in X])
    R = covs.mean(axis=0)
    R_inv_sqrt = np.real(sqrtm(inv(R)))
    X_aligned = np.einsum('ij,kjl->kil', R_inv_sqrt, X)
    return X_aligned, R_inv_sqrt

def apply_alignment(X, R_inv_sqrt):
    return np.einsum('ij,kjl->kil', R_inv_sqrt, X)

# =========================
# CSP CUSTOM
# =========================
class CSPCustom(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=6, reg=0.01):
        self.n_components = n_components
        self.reg = reg

    def _cov(self, X):
        cov = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
        for i, t in enumerate(X):
            c = t @ t.T
            cov[i] = c / (np.trace(c) + 1e-12)
        m = cov.mean(axis=0)
        # Régularisation par diagonal loading
        m = m + self.reg * np.trace(m) / m.shape[0] * np.eye(m.shape[0])
        return m

    def fit(self, X, y):
        classes = np.unique(y)
        filters = []
        if len(classes) == 2:
            C1 = self._cov(X[y == classes[0]])
            C2 = self._cov(X[y == classes[1]])
            evals, evecs = eigh(C1, C1 + C2)
            order = np.argsort(evals)[::-1]
            evecs = evecs[:, order]
            n_pick = self.n_components // 2
            W = np.concatenate([evecs[:, :n_pick], evecs[:, -n_pick:]], axis=1)
            filters.append(W)
        else:
            n_pick = max(1, self.n_components // (2 * len(classes)))
            for c in classes:
                C1 = self._cov(X[y == c])
                C2 = self._cov(X[y != c])
                evals, evecs = eigh(C1, C1 + C2)
                order = np.argsort(evals)[::-1]
                evecs = evecs[:, order]
                W = np.concatenate([evecs[:, :n_pick], evecs[:, -n_pick:]], axis=1)
                filters.append(W)
        self.filters_ = np.concatenate(filters, axis=1)
        return self

    def transform(self, X):
        Xp = np.einsum('ij,kjl->kil', self.filters_.T, X)
        var = np.var(Xp, axis=-1)
        var = var / (var.sum(axis=-1, keepdims=True) + 1e-12)
        return np.log(var + 1e-12)

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X, y).transform(X)

# =========================
# FEATURES
# =========================
def logvar(X):
    var = np.var(X, axis=-1)
    var = var / (var.sum(axis=-1, keepdims=True) + 1e-12)
    return np.log(var + 1e-12)

def fbcsp_features(X_train, y_train, X_test, bands, n_components=6, reg=0.01, fs=FS):
    Ftr, Fte = [], []
    for (low, high) in bands:
        Xtr_b = bandpass_sos(X_train, low, high, fs=fs)
        Xte_b = bandpass_sos(X_test, low, high, fs=fs)
        csp = CSPCustom(n_components=n_components, reg=reg)
        Ftr.append(csp.fit_transform(Xtr_b, y_train))
        Fte.append(csp.transform(Xte_b))
    return np.concatenate(Ftr, axis=1), np.concatenate(Fte, axis=1)

def riemann_features(X_train, X_test, shrinkage=0.05, metric='riemann'):
    cov = Covariances(estimator='oas')
    Xc_tr = cov.fit_transform(X_train)
    Xc_te = cov.transform(X_test)
    if shrinkage > 0:
        sh = Shrinkage(shrinkage=shrinkage)
        Xc_tr = sh.fit_transform(Xc_tr)
        Xc_te = sh.transform(Xc_te)
    ts = TangentSpace(metric=metric)
    Ftr = ts.fit_transform(Xc_tr)
    Fte = ts.transform(Xc_te)
    return Ftr, Fte

# =========================
# OOF PROBA (stacking)
# =========================
def oof_proba(pipe, X, y, X_test, cv, n_classes, classes_ref):
    oof = np.zeros((len(X), n_classes))
    test_proba = np.zeros((len(X_test), n_classes))
    for tr_idx, va_idx in cv.split(X, y):
        m = clone(pipe)
        m.fit(X[tr_idx], y[tr_idx])
        cls = m.classes_ if hasattr(m, 'classes_') else m.named_steps['clf'].classes_
        p_va = m.predict_proba(X[va_idx])
        p_te = m.predict_proba(X_test)
        if not np.array_equal(cls, classes_ref):
            idx = [list(cls).index(c) for c in classes_ref]
            p_va = p_va[:, idx]
            p_te = p_te[:, idx]
        oof[va_idx] = p_va
        test_proba += p_te / cv.n_splits
    return oof, test_proba

# =========================
# CONSTRUCTION DES CANDIDATS
# =========================
def build_candidates(X_train, y_train, X_test, prefix=""):
    """Renvoie {nom: (F_train, F_test, classifier)} pour un crop donné"""
    cands = {}

    Xtr_w = bandpass_sos(X_train, *WIDE_BAND)
    Xte_w = bandpass_sos(X_test, *WIDE_BAND)
    Xtr_c = bandpass_sos(X_train, *CLASSIC_BAND)
    Xte_c = bandpass_sos(X_test, *CLASSIC_BAND)

    # --- log-variance ---
    cands[f'{prefix}logvar_4-40_LDA'] = (
        logvar(Xtr_w), logvar(Xte_w),
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    )
    cands[f'{prefix}logvar_8-30_LDA'] = (
        logvar(Xtr_c), logvar(Xte_c),
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    )

    # --- CSP simples ---
    csp_c = CSPCustom(n_components=6, reg=0.01)
    cands[f'{prefix}CSP_8-30_LDA'] = (
        csp_c.fit_transform(Xtr_c, y_train), csp_c.transform(Xte_c),
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    )
    csp_w = CSPCustom(n_components=6, reg=0.01)
    cands[f'{prefix}CSP_4-40_LDA'] = (
        csp_w.fit_transform(Xtr_w, y_train), csp_w.transform(Xte_w),
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    )

    # --- FBCSP variantes ---
    Ftr_fb, Fte_fb = fbcsp_features(X_train, y_train, X_test, SUB_BANDS, n_components=6)
    cands[f'{prefix}FBCSP_LDA'] = (
        Ftr_fb, Fte_fb,
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    )
    cands[f'{prefix}FBCSP_LR'] = (
        Ftr_fb, Fte_fb,
        LogisticRegression(max_iter=5000, C=0.1, penalty='l2', solver='lbfgs')
    )
    cands[f'{prefix}FBCSP_SVM'] = (
        Ftr_fb, Fte_fb,
        SVC(C=1.0, kernel='rbf', probability=True, gamma='scale')
    )

    Ftr_fb4, Fte_fb4 = fbcsp_features(X_train, y_train, X_test, SUB_BANDS, n_components=4)
    cands[f'{prefix}FBCSP_n4_LDA'] = (
        Ftr_fb4, Fte_fb4,
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    )

    # --- Riemann TS (sans EA) ---
    Ftr_r1, Fte_r1 = riemann_features(Xtr_c, Xte_c)
    cands[f'{prefix}Riemann_8-30_LR'] = (
        Ftr_r1, Fte_r1,
        LogisticRegression(max_iter=5000, C=1.0, penalty='l2', solver='lbfgs')
    )
    cands[f'{prefix}Riemann_8-30_LDA'] = (
        Ftr_r1, Fte_r1,
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    )

    # --- Riemann TS AVEC Euclidean Alignment (gain majeur) ---
    # On aligne train ET test ensemble (transductif) pour homogénéiser
    X_all = np.concatenate([X_train, X_test], axis=0)
    Xtr_c_all = bandpass_sos(X_all, *CLASSIC_BAND)
    Xtr_c_aligned, _ = euclidean_alignment(Xtr_c_all)
    Xtr_ea = Xtr_c_aligned[:len(X_train)]
    Xte_ea = Xtr_c_aligned[len(X_train):]
    Ftr_ea, Fte_ea = riemann_features(Xtr_ea, Xte_ea)
    cands[f'{prefix}Riemann_EA_LR'] = (
        Ftr_ea, Fte_ea,
        LogisticRegression(max_iter=5000, C=1.0, penalty='l2', solver='lbfgs')
    )
    cands[f'{prefix}Riemann_EA_LDA'] = (
        Ftr_ea, Fte_ea,
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    )
    cands[f'{prefix}Riemann_EA_SVM'] = (
        Ftr_ea, Fte_ea,
        SVC(C=1.0, kernel='rbf', probability=True, gamma='scale')
    )

    # --- Riemann TS sur signal brut + EA (large bande) ---
    Xall_w = bandpass_sos(np.concatenate([X_train, X_test], axis=0), *WIDE_BAND)
    Xall_w_aligned, _ = euclidean_alignment(Xall_w)
    Xtr_eaw = Xall_w_aligned[:len(X_train)]
    Xte_eaw = Xall_w_aligned[len(X_train):]
    Ftr_eaw, Fte_eaw = riemann_features(Xtr_eaw, Xte_eaw)
    cands[f'{prefix}Riemann_EA_4-40_LR'] = (
        Ftr_eaw, Fte_eaw,
        LogisticRegression(max_iter=5000, C=1.0, penalty='l2', solver='lbfgs')
    )

    return cands

# =========================
# PIPELINE COMPLET PAR SUJET
# =========================
def process_subject(X_train_raw, y_train, X_test_raw, classes_ref):
    n_classes = len(classes_ref)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # --- Construction multi-fenêtres ---
    all_candidates = {}
    if USE_WINDOWS:
        for w_idx, (sr, er) in enumerate(WINDOW_RATIOS):
            X_tr_w = normalize(crop_ratio(X_train_raw, sr, er))
            X_te_w = normalize(crop_ratio(X_test_raw, sr, er))
            cands = build_candidates(X_tr_w, y_train, X_te_w, prefix=f"w{w_idx}_")
            all_candidates.update(cands)
    else:
        X_tr = normalize(X_train_raw)
        X_te = normalize(X_test_raw)
        all_candidates = build_candidates(X_tr, y_train, X_te)

    # --- Évaluation CV ---
    cv_scores = {}
    pipes = {}
    for name, (Ftr, Fte, clf) in all_candidates.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        scores = cross_val_score(pipe, Ftr, y_train, cv=cv,
                                  scoring='accuracy', n_jobs=-1)
        cv_scores[name] = scores.mean()
        pipes[name] = pipe

    # Affichage condensé : top 10
    sorted_names = sorted(cv_scores, key=cv_scores.get, reverse=True)
    print(f"   --- Top 10 modèles (sur {len(cv_scores)}) ---")
    for nm in sorted_names[:10]:
        print(f"   [{nm:35s}] CV: {cv_scores[nm]:.4f}")

    # --- STRATÉGIE 1 : meilleur modèle seul ---
    best_name = sorted_names[0]
    best_score_single = cv_scores[best_name]

    # --- STRATÉGIE 2 : ensemble pondéré top-K ---
    K_VALUES = [3, 5, 7, 10]
    best_ensemble_score = -1
    best_ensemble_test = None

    # Compute OOF for top candidates (les K=10 meilleurs uniquement, par souci de temps)
    top_for_ens = sorted_names[:max(K_VALUES)]
    oof_dict, test_dict = {}, {}
    for nm in top_for_ens:
        Ftr, Fte, _ = all_candidates[nm]
        oof, test_p = oof_proba(pipes[nm], Ftr, y_train, Fte, cv, n_classes, classes_ref)
        oof_dict[nm] = oof
        test_dict[nm] = test_p

    y_idx = np.array([list(classes_ref).index(yy) for yy in y_train])
    for K in K_VALUES:
        top_k = sorted_names[:K]
        weights = np.array([cv_scores[nm] for nm in top_k])
        weights = weights ** 4   # accentue les meilleurs
        weights = weights / weights.sum()
        oof_ens = sum(w * oof_dict[nm] for w, nm in zip(weights, top_k))
        test_ens = sum(w * test_dict[nm] for w, nm in zip(weights, top_k))
        ens_score = (np.argmax(oof_ens, axis=1) == y_idx).mean()
        print(f"   [Ensemble top-{K:2d}                       ] CV: {ens_score:.4f}")
        if ens_score > best_ensemble_score:
            best_ensemble_score = ens_score
            best_ensemble_test = test_ens
            best_ensemble_K = K

    # --- STRATÉGIE 3 : stacking avec meta-learner ---
    # Concaténation des probas OOF du top-10
    meta_train = np.concatenate([oof_dict[nm] for nm in top_for_ens], axis=1)
    meta_test  = np.concatenate([test_dict[nm] for nm in top_for_ens], axis=1)

    meta_clf = LogisticRegression(max_iter=5000, C=0.5, penalty='l2', solver='lbfgs')
    cv_meta = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED+1)
    meta_oof, meta_test_p = oof_proba(meta_clf, meta_train, y_train, meta_test,
                                       cv_meta, n_classes, classes_ref)
    stacking_score = (np.argmax(meta_oof, axis=1) == y_idx).mean()
    print(f"   [Stacking meta-LR                       ] CV: {stacking_score:.4f}")

    # --- DÉCISION : quelle stratégie ? ---
    candidates_strategies = [
        ('Single', best_score_single, None),
        (f'Ensemble top-{best_ensemble_K}', best_ensemble_score, best_ensemble_test),
        ('Stacking', stacking_score, meta_test_p),
    ]
    candidates_strategies.sort(key=lambda x: x[1], reverse=True)
    chosen_strategy, chosen_score, chosen_test = candidates_strategies[0]

    if chosen_strategy == 'Single':
        Ftr, Fte, _ = all_candidates[best_name]
        pipes[best_name].fit(Ftr, y_train)
        y_pred = pipes[best_name].predict(Fte)
        # On a aussi besoin des probas pour le pseudo-labeling
        proba_test = pipes[best_name].predict_proba(Fte)
        cls = pipes[best_name].classes_
        if not np.array_equal(cls, classes_ref):
            idx = [list(cls).index(c) for c in classes_ref]
            proba_test = proba_test[:, idx]
        chosen_strategy_full = f"Single ({best_name})"
    else:
        proba_test = chosen_test
        y_pred = classes_ref[np.argmax(proba_test, axis=1)]
        chosen_strategy_full = chosen_strategy

    return y_pred, proba_test, chosen_strategy_full, chosen_score, all_candidates, pipes, cv_scores


# =========================
# MAIN
# =========================
all_results = {}

for subject in SUBJECTS:
    print(f"\n{'='*70}")
    print(f"🚀 Subject {subject}")
    print(f"{'='*70}")

    X_train = np.load(f"{DATA_PATH}/subject_{subject}_X_train.npy")
    y_train = np.load(f"{DATA_PATH}/subject_{subject}_y_train.npy")
    X_test  = np.load(f"{DATA_PATH}/subject_{subject}_X_test.npy")

    classes_ref = np.sort(np.unique(y_train))
    print(f"   Train: {X_train.shape} | Test: {X_test.shape} | Classes: {classes_ref}")

    # === Passe 1 : prédiction de base ===
    print("\n   [Passe 1 : modèle initial]")
    y_pred, proba_test, strat, score, all_cands, pipes, cv_scores = process_subject(
        X_train, y_train, X_test, classes_ref
    )
    print(f"   ⇒ Stratégie initiale : {strat} (CV={score:.4f})")

    # === Passe 2+ : pseudo-labeling itératif ===
    if USE_PSEUDO_LABEL:
        for it in range(PSEUDO_ITERATIONS):
            print(f"\n   [Passe {it+2} : pseudo-labeling]")
            confidences = proba_test.max(axis=1)
            high_conf_mask = confidences > PSEUDO_CONF_THRESHOLD
            n_added = high_conf_mask.sum()
            print(f"   Échantillons test confiants (>{PSEUDO_CONF_THRESHOLD}): "
                  f"{n_added}/{len(X_test)}")

            if n_added < 10:
                print("   Pas assez d'échantillons confiants, on s'arrête.")
                break

            # Augmente le train avec les pseudo-labels
            X_train_aug = np.concatenate([X_train, X_test[high_conf_mask]], axis=0)
            y_pseudo = classes_ref[np.argmax(proba_test[high_conf_mask], axis=1)]
            y_train_aug = np.concatenate([y_train, y_pseudo], axis=0)

            print(f"   Train augmenté : {X_train_aug.shape}")

            y_pred_new, proba_test_new, strat_new, score_new, _, _, _ = process_subject(
                X_train_aug, y_train_aug, X_test, classes_ref
            )
            print(f"   ⇒ Stratégie : {strat_new} (CV={score_new:.4f})")

            # On accepte le pseudo-labeling seulement si ça améliore
            if score_new > score - 0.005:  # tolérance
                y_pred = y_pred_new
                proba_test = proba_test_new
                strat = strat_new + f" + PL_it{it+1}"
                score = score_new
                print(f"   ✅ Pseudo-labeling accepté")
            else:
                print(f"   ❌ Pseudo-labeling rejeté (régression)")
                break

    all_results[subject] = {
        'strategy': strat,
        'cv_score': score,
    }

    pd.DataFrame({'y_pred': y_pred}).to_csv(
        f"{OUTPUT_DIR}/subject_{subject}_y_pred.csv",
        index=False
    )
    print(f"\n   ✅ Saved predictions for subject {subject}")

# =========================
# RÉSUMÉ
# =========================
print("\n" + "="*70)
print("📊 RÉSUMÉ FINAL PAR SUJET")
print("="*70)
mean_cv = []
for s, info in all_results.items():
    print(f"   {s}: {info['strategy']:50s}  CV={info['cv_score']:.4f}")
    mean_cv.append(info['cv_score'])
print(f"\n   Moyenne CV : {np.mean(mean_cv):.4f}")

# =========================
# ZIP
# =========================
with zipfile.ZipFile(ZIP_NAME, 'w') as zipf:
    for subject in SUBJECTS:
        zipf.write(
            f"{OUTPUT_DIR}/subject_{subject}_y_pred.csv",
            arcname=f"subject_{subject}_y_pred.csv"
        )

print(f"\n🏆 ZIP READY: {ZIP_NAME}")