"""
Sklearn Pipelines: matched preprocessing per model family.
Linear models: imputation + one-hot + scaling on numeric features.
Tree / boosting: imputation + one-hot; no scaling (scale-invariant).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from xgboost import XGBClassifier, XGBRegressor


def _column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    num_cols, cat_cols = _column_types(X)
    num_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(num_steps)

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def classification_models(
    random_state: int,
) -> Dict[str, Tuple[Pipeline, Dict[str, Any]]]:
    """
    Returns model_key -> (unfitted Pipeline template without preprocessor, param_grid for RandomizedSearchCV).
    Preprocessor is attached per-dataset in training code.
    """

    def pipe_lr(pre: ColumnTransformer) -> Pipeline:
        return Pipeline(
            [
                ("prep", pre),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    def pipe_dt(pre: ColumnTransformer) -> Pipeline:
        return Pipeline(
            [
                ("prep", pre),
                ("clf", DecisionTreeClassifier(random_state=random_state)),
            ]
        )

    def pipe_rf(pre: ColumnTransformer) -> Pipeline:
        return Pipeline(
            [
                ("prep", pre),
                (
                    "clf",
                    RandomForestClassifier(
                        random_state=random_state,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def pipe_xgb(pre: ColumnTransformer) -> Pipeline:
        return Pipeline(
            [
                ("prep", pre),
                (
                    "clf",
                    XGBClassifier(
                        random_state=random_state,
                        n_jobs=-1,
                        eval_metric="logloss",
                        tree_method="hist",
                    ),
                ),
            ]
        )

    def pipe_hgb(pre: ColumnTransformer) -> Pipeline:
        return Pipeline(
            [
                ("prep", pre),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        random_state=random_state,
                    ),
                ),
            ]
        )

    # Param grids use step__param for Pipeline
    grids: Dict[str, Tuple[Any, Dict[str, Any]]] = {
        "logistic_regression": (
            lambda pre: pipe_lr(pre),
            {
                "clf__C": np.logspace(-2, 2, 16),
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs"],
            },
        ),
        "decision_tree": (
            lambda pre: pipe_dt(pre),
            {
                "clf__max_depth": [2, 3, 4, 6, 8, 12, 16, None],
                "clf__min_samples_leaf": [1, 2, 5, 10, 20],
                "clf__min_samples_split": [2, 5, 10],
            },
        ),
        "random_forest": (
            lambda pre: pipe_rf(pre),
            {
                "clf__n_estimators": [100, 200, 400],
                "clf__max_depth": [None, 8, 16, 24],
                "clf__min_samples_leaf": [1, 2, 4],
            },
        ),
        "hist_gradient_boosting": (
            lambda pre: pipe_hgb(pre),
            {
                "clf__max_iter": [100, 200, 400],
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "clf__max_depth": [None, 3, 5, 7],
                "clf__min_samples_leaf": [10, 20, 40],
                "clf__l2_regularization": [0.0, 0.1, 1.0],
            },
        ),
        "xgboost": (
            lambda pre: pipe_xgb(pre),
            {
                "clf__n_estimators": [100, 200, 400],
                "clf__max_depth": [3, 4, 6, 8],
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "clf__subsample": [0.7, 0.9, 1.0],
                "clf__colsample_bytree": [0.7, 0.9, 1.0],
            },
        ),
    }
    return grids  # type: ignore


def regression_models(random_state: int) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    def pipe_ridge(pre: ColumnTransformer) -> Pipeline:
        return Pipeline(
            [
                ("prep", pre),
                ("reg", Ridge()),
            ]
        )

    def pipe_dt(pre: ColumnTransformer) -> Pipeline:
        return Pipeline(
            [
                ("prep", pre),
                ("reg", DecisionTreeRegressor(random_state=random_state)),
            ]
        )

    def pipe_rf(pre: ColumnTransformer) -> Pipeline:
        return Pipeline(
            [
                ("prep", pre),
                ("reg", RandomForestRegressor(random_state=random_state, n_jobs=-1)),
            ]
        )

    def pipe_xgb(pre: ColumnTransformer) -> Pipeline:
        return Pipeline(
            [
                ("prep", pre),
                ("reg", XGBRegressor(random_state=random_state, n_jobs=-1, tree_method="hist")),
            ]
        )

    def pipe_hgb(pre: ColumnTransformer) -> Pipeline:
        return Pipeline(
            [
                ("prep", pre),
                ("reg", HistGradientBoostingRegressor(random_state=random_state)),
            ]
        )

    grids: Dict[str, Tuple[Any, Dict[str, Any]]] = {
        "ridge": (
            lambda pre: pipe_ridge(pre),
            {"reg__alpha": np.logspace(-1, 3, 20)},
        ),
        "decision_tree": (
            lambda pre: pipe_dt(pre),
            {
                "reg__max_depth": [2, 4, 6, 8, 12, 16, None],
                "reg__min_samples_leaf": [1, 2, 5, 10, 20],
            },
        ),
        "random_forest": (
            lambda pre: pipe_rf(pre),
            {
                "reg__n_estimators": [100, 200, 400],
                "reg__max_depth": [None, 8, 16, 24],
                "reg__min_samples_leaf": [1, 2, 4],
            },
        ),
        "hist_gradient_boosting": (
            lambda pre: pipe_hgb(pre),
            {
                "reg__max_iter": [100, 200, 400],
                "reg__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "reg__max_depth": [None, 3, 5, 7],
                "reg__min_samples_leaf": [10, 20, 40],
                "reg__l2_regularization": [0.0, 0.1, 1.0],
            },
        ),
        "xgboost": (
            lambda pre: pipe_xgb(pre),
            {
                "reg__n_estimators": [100, 200, 400],
                "reg__max_depth": [3, 4, 6, 8],
                "reg__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "reg__subsample": [0.7, 0.9, 1.0],
                "reg__colsample_bytree": [0.7, 0.9, 1.0],
            },
        ),
    }
    return grids


def scale_numeric_for_model(model_key: str) -> bool:
    return model_key in ("logistic_regression", "ridge")
