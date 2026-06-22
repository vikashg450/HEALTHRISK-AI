"""
HealthRisk AI — Stacking Ensemble Module (Days 8–9)
Combines XGBoost, LightGBM, ClinicalBERT embeddings,
GNN patient embeddings, and Survival hazard scores
into a stacking meta-learner.

Target: Ensemble AUROC > 0.80, AUPRC > 0.40, Brier Score < 0.15
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, classification_report
)
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional, Tuple
import xgboost as xgb
import lightgbm as lgb
import joblib
import os


# ──────────────────────────────────────────────────────
# BASE MODELS
# ──────────────────────────────────────────────────────

class XGBoostModel:
    """XGBoost classifier for tabular clinical/financial features."""

    def __init__(self, seed: int = 42, **kwargs):
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_lambda": 1.0,
            "reg_alpha": 0.1,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": seed,
            "n_jobs": -1,
        }
        params.update(kwargs)
        self.model = xgb.XGBClassifier(**params)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None):
        fit_params = {"verbose": 50}
        if eval_set:
            fit_params["eval_set"] = eval_set
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.Series:
        importance = self.model.feature_importances_
        idx = np.argsort(importance)[::-1]
        names = feature_names or [f"f{i}" for i in range(len(importance))]
        return pd.Series(importance[idx], index=np.array(names)[idx])


class LightGBMModel:
    """LightGBM classifier — fast and effective on large datasets."""

    def __init__(self, seed: int = 42, **kwargs):
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "reg_alpha": 0.1,
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        params.update(kwargs)
        self.model = lgb.LGBMClassifier(**params)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None):
        callbacks = [lgb.log_evaluation(period=100)]
        self.model.fit(X, y, callbacks=callbacks)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


# ──────────────────────────────────────────────────────
# TIME-AWARE CROSS-VALIDATION
# ──────────────────────────────────────────────────────

def time_aware_cv_predictions(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    gap: int = 0,  # samples gap between train and val
) -> np.ndarray:
    """
    Generate out-of-fold predictions using time-aware cross-validation.
    Respects temporal ordering — future data never leaks into training.
    """
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    oof_predictions = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"  Fold {fold + 1}/{n_splits}: train={len(train_idx):,}, val={len(val_idx):,}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        oof_predictions[val_idx] = model.predict_proba(X_val)

        fold_auroc = roc_auc_score(y_val, oof_predictions[val_idx])
        print(f"           Fold AUROC: {fold_auroc:.4f}")

    total_auroc = roc_auc_score(y, oof_predictions)
    print(f"  OOF AUROC: {total_auroc:.4f}")
    return oof_predictions


# ──────────────────────────────────────────────────────
# STACKING META-LEARNER
# ──────────────────────────────────────────────────────

class HealthRiskEnsemble:
    """
    Stacking ensemble combining:
      - XGBoost tabular predictions
      - LightGBM tabular predictions
      - ClinicalBERT embeddings (PCA-reduced)
      - GNN patient embeddings
      - Survival model hazard scores

    Meta-learner: Logistic Regression (well-calibrated, interpretable)
    """

    def __init__(self, n_splits: int = 5, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed
        self.xgb_model = XGBoostModel(seed=seed)
        self.lgb_model = LightGBMModel(seed=seed)
        self.meta_learner = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        self.scaler = StandardScaler()
        self.pca_bert = None
        self.is_fitted = False

    def _reduce_embeddings(
        self, embeddings: np.ndarray, n_components: int = 32, fit: bool = True
    ) -> np.ndarray:
        """Reduce high-dim embeddings (e.g., 768-dim BERT) via PCA."""
        from sklearn.decomposition import PCA
        if fit:
            self.pca_bert = PCA(n_components=min(n_components, embeddings.shape[1]),
                                random_state=self.seed)
            return self.pca_bert.fit_transform(embeddings)
        else:
            return self.pca_bert.transform(embeddings) if self.pca_bert else embeddings[:, :n_components]

    def fit(
        self,
        X_tabular: np.ndarray,
        y: np.ndarray,
        bert_embeddings: Optional[np.ndarray] = None,
        gnn_embeddings: Optional[np.ndarray] = None,
        survival_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Fit the stacking ensemble using out-of-fold predictions.
        """
        print("=" * 55)
        print("TRAINING STACKING ENSEMBLE")
        print("=" * 55)
        n = len(X_tabular)

        # Step 1: OOF predictions from base models
        print("\n[XGBoost] Out-of-fold predictions...")
        xgb_oof = time_aware_cv_predictions(self.xgb_model, X_tabular, y, self.n_splits)

        print("\n[LightGBM] Out-of-fold predictions...")
        lgb_oof = time_aware_cv_predictions(self.lgb_model, X_tabular, y, self.n_splits)

        # Step 2: Build meta-features
        meta_features = [
            xgb_oof.reshape(-1, 1),
            lgb_oof.reshape(-1, 1),
        ]

        if bert_embeddings is not None:
            print(f"\n[BERT] Reducing {bert_embeddings.shape[1]}-dim → 32-dim...")
            bert_reduced = self._reduce_embeddings(bert_embeddings, n_components=32, fit=True)
            meta_features.append(bert_reduced)

        if gnn_embeddings is not None:
            print(f"\n[GNN] Using {gnn_embeddings.shape[1]}-dim patient embeddings...")
            meta_features.append(gnn_embeddings)

        if survival_scores is not None:
            print("\n[Survival] Adding hazard scores...")
            meta_features.append(survival_scores.reshape(-1, 1))

        X_meta = np.hstack(meta_features)
        X_meta_scaled = self.scaler.fit_transform(X_meta)

        # Step 3: Fit meta-learner
        print(f"\n[Meta-learner] Fitting LogReg on {X_meta_scaled.shape[1]} meta-features...")
        self.meta_learner.fit(X_meta_scaled, y)

        # Step 4: Final evaluation
        meta_probs = self.meta_learner.predict_proba(X_meta_scaled)[:, 1]
        metrics = self._evaluate(y, meta_probs, label="[Ensemble] Training")

        # Refit base models on full data
        print("\n[Final] Refitting base models on full training data...")
        self.xgb_model.fit(X_tabular, y)
        self.lgb_model.fit(X_tabular, y)

        self.is_fitted = True
        return metrics

    def predict_proba(
        self,
        X_tabular: np.ndarray,
        bert_embeddings: Optional[np.ndarray] = None,
        gnn_embeddings: Optional[np.ndarray] = None,
        survival_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return ensemble predicted probabilities."""
        meta_features = [
            self.xgb_model.predict_proba(X_tabular).reshape(-1, 1),
            self.lgb_model.predict_proba(X_tabular).reshape(-1, 1),
        ]
        if bert_embeddings is not None:
            meta_features.append(self._reduce_embeddings(bert_embeddings, fit=False))
        if gnn_embeddings is not None:
            meta_features.append(gnn_embeddings)
        if survival_scores is not None:
            meta_features.append(survival_scores.reshape(-1, 1))

        X_meta = np.hstack(meta_features)
        X_meta_scaled = self.scaler.transform(X_meta)
        return self.meta_learner.predict_proba(X_meta_scaled)[:, 1]

    def _evaluate(self, y_true: np.ndarray, y_prob: np.ndarray, label: str = "") -> Dict[str, float]:
        """Compute all evaluation metrics."""
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {"auroc": auroc, "auprc": auprc, "brier_score": brier}
        print(f"\n{label} Metrics:")
        print(f"  AUROC:       {auroc:.4f}  (target > 0.80)")
        print(f"  AUPRC:       {auprc:.4f}  (target > 0.40)")
        print(f"  Brier Score: {brier:.4f}  (target < 0.15)")
        print(classification_report(y_true, y_pred, target_names=["Low Risk", "High Risk"]))
        return metrics

    def evaluate(
        self,
        X_tabular: np.ndarray,
        y: np.ndarray,
        bert_embeddings=None,
        gnn_embeddings=None,
        survival_scores=None,
    ) -> Dict[str, float]:
        probs = self.predict_proba(X_tabular, bert_embeddings, gnn_embeddings, survival_scores)
        return self._evaluate(y, probs, label="[Ensemble] Test")

    def save(self, output_dir: str = "models/ensemble"):
        """Persist trained ensemble."""
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self, os.path.join(output_dir, "healthrisk_ensemble.pkl"))
        print(f"Ensemble saved to {output_dir}/healthrisk_ensemble.pkl")

    @classmethod
    def load(cls, path: str) -> "HealthRiskEnsemble":
        return joblib.load(path)


if __name__ == "__main__":
    print("Testing HealthRisk Ensemble with synthetic data...")
    np.random.seed(42)
    n = 2000

    X = np.random.randn(n, 20)
    y = np.random.binomial(1, 0.15, n)
    bert_emb = np.random.randn(n, 768)
    gnn_emb = np.random.randn(n, 32)
    surv_scores = np.random.randn(n)

    ensemble = HealthRiskEnsemble(n_splits=3, seed=42)
    train_metrics = ensemble.fit(X[:1600], y[:1600], bert_emb[:1600], gnn_emb[:1600], surv_scores[:1600])
    test_metrics = ensemble.evaluate(X[1600:], y[1600:], bert_emb[1600:], gnn_emb[1600:], surv_scores[1600:])
    print(f"\nTrain: {train_metrics}")
    print(f"Test:  {test_metrics}")
