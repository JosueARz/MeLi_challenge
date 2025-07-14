"""
ModelPredictor v4 – GridSearch + Persistencia de modelo
-------------------------------------------------------
• Train/test split + CV (average precision)
• Selección entre LogReg y HistGB + GridSearchCV
• Umbral óptimo (máx F1)
• Métodos save_model() y load_model()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


# --------------------------------------------------------------------
# helper global (pickle-safe)
def _log1p_selected(X: pd.DataFrame, cols: Tuple[str, ...]) -> pd.DataFrame:
    """Aplica log1p solo a las columnas listadas que existan en X."""
    X = X.copy()
    for c in cols:
        if c in X:
            X[c] = np.log1p(X[c])
    return X
# --------------------------------------------------------------------


@dataclass
class ModelPredictor:
    df: pd.DataFrame
    feature_cols: List[str]
    target_col: str = "sold_flag"
    test_size: float = 0.2
    random_state: int = 42
    model_dir: str | Path = "models"
    model_name: str = "best_pipeline.pkl"
    _log_cols: Tuple[str, ...] = ("base_price", "price", "price_per_unit")

    # internos (populados en .train)
    X_train: pd.DataFrame = field(init=False, repr=False)
    X_test: pd.DataFrame = field(init=False, repr=False)
    y_train: pd.Series = field(init=False, repr=False)
    y_test: pd.Series = field(init=False, repr=False)
    cv_results_: Dict[str, float] = field(init=False, default_factory=dict)
    best_model_name_: str | None = field(init=False, default=None)
    best_pipeline_: Pipeline | None = field(init=False, default=None)
    best_threshold_: float | None = field(init=False, default=None)

    # ---------- helpers ----------
    def _preproc(self) -> Pipeline:
        log_tf = FunctionTransformer(
            partial(_log1p_selected, cols=self._log_cols),
            validate=False,
        )
        return Pipeline([("log", log_tf), ("scale", StandardScaler())])

    def _pipelines(self) -> Dict[str, Pipeline]:
        pre = self._preproc()

        logreg = Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="saga",
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        hist = Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.01,
                        class_weight={0: 1, 1: 5},
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        return {"LogReg": logreg, "HistGB": hist}

    # ---------- entrenamiento ----------
    def train(self, cv_splits: int = 3,
              scoring: str = "average_precision") -> None:
        X = self.df[self.feature_cols]
        y = self.df[self.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y,
            random_state=self.random_state
        )

        cv = StratifiedKFold(
            n_splits=cv_splits, shuffle=True, random_state=self.random_state
        )

        # 1) evaluación base
        for name, pipe in self._pipelines().items():
            score = cross_val_score(
                pipe, self.X_train, self.y_train,
                cv=cv, scoring=scoring, n_jobs=-1
            ).mean()
            self.cv_results_[name] = score
            if (self.best_model_name_ is None
                    or score > self.cv_results_[self.best_model_name_]):
                self.best_model_name_, self.best_pipeline_ = name, pipe

        # 2) búsqueda de hiper-parámetros en el ganador
        self._gridsearch_best(cv_splits, scoring)

    def _gridsearch_best(self, cv_splits: int, scoring: str) -> None:
        if self.best_model_name_ == "HistGB":
            param_grid = {
                "clf__learning_rate": [0.05, 0.1, 0.01, 0.08],
                "clf__max_depth": [None, 3, 4, 5],
                "clf__l2_regularization": [0.0, 0.5],
                "clf__max_leaf_nodes": [31, 63],
                "clf__class_weight":
                    [{0: 1, 1: w} for w in (2, 3, 4, 5, 7, 10)],
            }
        else:  # LogReg
            param_grid = {
                "clf__C": [0.05, 0.1, 0.8, 1.0, 10.0],
                "clf__penalty": ["l2", "l1", "elasticnet"],
                "clf__l1_ratio": [None, 0.5],
            }

        grid = GridSearchCV(
            self.best_pipeline_,
            param_grid,
            cv=StratifiedKFold(
                n_splits=cv_splits, shuffle=True,
                random_state=self.random_state
            ),
            scoring=scoring,
            refit=True,
            n_jobs=-1,
        )
        grid.fit(self.X_train, self.y_train)

        self.best_pipeline_ = grid.best_estimator_
        self.cv_results_[f"{self.best_model_name_}_GS"] = grid.best_score_
        self.best_model_name_ += "_GS"

    # ---------- persistencia ----------
    def save_model(self, filename: str | None = None) -> None:
        """
        Guarda (pipeline, meta) en /models/.
        meta = columnas usadas, fecha de entrenamiento y umbral óptimo.
        """
        from datetime import datetime

        path = Path(self.model_dir)
        path.mkdir(parents=True, exist_ok=True)
        file = path / (filename or self.model_name)

        meta = {
            "feature_cols": self.feature_cols,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "model_name": self.best_model_name_,
            "threshold_opt": self.best_threshold_,
        }
        joblib.dump((self.best_pipeline_, meta), file)
        print(f"\n Modelo + metadata guardados en: {file}")

    @classmethod
    def load_model(cls, model_path: str | Path):
        """Devuelve (pipeline, meta)"""
        return joblib.load(model_path)

    # ---------- reporting ----------
    def summary(self) -> None:
        print("=== Validación cruzada (PR-AUC promedio) ===")
        for k, v in self.cv_results_.items():
            mark = "  <-- ganador" if k == self.best_model_name_ else ""
            print(f"{k:12s}: {v:.3f}{mark}")

    def evaluate_test(self, threshold: float = 0.5) -> float:
        proba = self.best_pipeline_.predict_proba(self.X_test)[:, 1]
        pred  = (proba >= threshold).astype(int)
        print(f"\n=== Classification report (thr={threshold:.2f}) ===")
        print(classification_report(self.y_test, pred, digits=3))
        pr_auc  = average_precision_score(self.y_test, proba)
        roc_auc = roc_auc_score(self.y_test, proba)
        print(f"PR-AUC test: {pr_auc:.3f} · ROC-AUC test: {roc_auc:.3f}")
        return pr_auc

    # ---------- umbral óptimo ----------
    def find_best_threshold(self) -> float:
        proba = self.best_pipeline_.predict_proba(self.X_test)[:, 1]
        prec, rec, thr = precision_recall_curve(self.y_test, proba)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        idx = np.argmax(f1)
        self.best_threshold_ = thr[idx]
        print(f"\nMejor F1 = {f1[idx]:.3f} con umbral {self.best_threshold_:.3f}")
        return self.best_threshold_

    # ---------- inferencia ----------
    def predict(self, df_new: pd.DataFrame,
                threshold: float | None = None) -> pd.Series:
        thr = threshold if threshold is not None else (
            self.best_threshold_ or 0.5
        )
        proba = self.best_pipeline_.predict_proba(
            df_new[self.feature_cols]
        )[:, 1]
        return (proba >= thr).astype(int)
