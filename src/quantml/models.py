"""Model factory. Preprocessing lives inside each scikit-learn ``Pipeline`` so scaling statistics
are always fitted on the training rows passed to ``fit`` and never on validation/test data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quantml.config import ModelConfig

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest (experimental)",
}


def base_params(cfg: ModelConfig) -> dict[str, Any]:
    """Model hyper-parameters from config, without the tuning grids."""
    params = dict(getattr(cfg, cfg.name))
    params.pop("C_grid", None)
    params.pop("max_depth_grid", None)
    return params


def candidate_params(cfg: ModelConfig) -> list[dict[str, Any]]:
    """Hyper-parameter candidates evaluated on the validation set."""
    params = base_params(cfg)
    if cfg.name == "logistic_regression":
        grid = cfg.logistic_regression.get("C_grid") or [params["C"]]
        return [{**params, "C": float(c)} for c in grid]
    grid = cfg.random_forest.get("max_depth_grid") or [params["max_depth"]]
    return [{**params, "max_depth": None if d is None else int(d)} for d in grid]


def build_model(name: str, params: dict[str, Any], seed: int = 42) -> Pipeline:
    if name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=float(params.get("C", 1.0)),
                                         max_iter=int(params.get("max_iter", 2000)),
                                         class_weight=params.get("class_weight"),
                                         random_state=seed)),
        ])
    if name == "random_forest":
        return Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=int(params.get("n_estimators", 300)),
                max_depth=params.get("max_depth"),
                min_samples_leaf=int(params.get("min_samples_leaf", 50)),
                max_features=params.get("max_features", "sqrt"),
                class_weight=params.get("class_weight"),
                random_state=seed, n_jobs=-1)),
        ])
    raise ValueError(f"Unknown model '{name}'. Choose one of {list(MODEL_LABELS)}.")


def fit_model(name: str, params: dict[str, Any], X: pd.DataFrame, y: pd.Series,
              seed: int = 42) -> Pipeline:
    """Fit a fresh pipeline; refuses single-class training data with an actionable message."""
    classes = np.unique(np.asarray(y))
    if len(classes) < 2:
        raise ValueError("Training labels contain only one class; the model cannot learn. Use a "
                         "longer training window or a different target threshold.")
    model = build_model(name, params, seed)
    model.fit(X, np.asarray(y).astype(int))
    return model


def predict_up_probability(model: Pipeline, X: pd.DataFrame) -> pd.Series:
    proba = model.predict_proba(X)
    col = list(model.classes_).index(1)
    return pd.Series(proba[:, col], index=X.index, name="prob_up")


def coefficient_table(model: Pipeline, feature_names: list[str]) -> pd.DataFrame | None:
    """Standardised logistic coefficients or forest impurity importances (descriptive only)."""
    est = model.named_steps["model"]
    if hasattr(est, "coef_"):
        return pd.DataFrame({"feature": feature_names, "value": est.coef_[0],
                             "kind": "standardised coefficient"}).sort_values(
            "value", key=np.abs, ascending=False)
    if hasattr(est, "feature_importances_"):
        return pd.DataFrame({"feature": feature_names, "value": est.feature_importances_,
                             "kind": "impurity importance"}).sort_values("value", ascending=False)
    return None
