"""
Train and evaluate models with matched RandomizedSearchCV budget per model.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

F1_AVERAGE = "weighted"  # valid for binary and multiclass
from sklearn.model_selection import RandomizedSearchCV

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # pragma: no cover

    def root_mean_squared_error(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


import os

import config
from src.datasets import DatasetBundle
from src.pipelines import classification_models, regression_models, make_preprocessor, scale_numeric_for_model


def _baselines_classification(bundle: DatasetBundle) -> Dict[str, Any]:
    dc = DummyClassifier(strategy="most_frequent")
    dc.fit(bundle.X_train, bundle.y_train)
    pred = dc.predict(bundle.X_test)
    return {
        "baseline_majority_accuracy": float(accuracy_score(bundle.y_test, pred)),
        "baseline_majority_f1": float(f1_score(bundle.y_test, pred, average=F1_AVERAGE, zero_division=0)),
    }


def _baselines_regression(bundle: DatasetBundle) -> Dict[str, Any]:
    dr = DummyRegressor(strategy="mean")
    dr.fit(bundle.X_train, bundle.y_train)
    pred = dr.predict(bundle.X_test)
    return {
        "baseline_mean_rmse": float(root_mean_squared_error(bundle.y_test, pred)),
    }


def train_classification(bundle: DatasetBundle) -> List[Dict[str, Any]]:
    models = classification_models(config.RANDOM_STATE)
    rows: List[Dict[str, Any]] = []
    for name, (factory, param_grid) in models.items():
        pre = make_preprocessor(bundle.X_train, scale_numeric=scale_numeric_for_model(name))
        pipe = factory(pre)
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grid,
            n_iter=config.N_ITER_SEARCH,
            scoring="f1_weighted",
            cv=config.INNER_CV_SPLITS,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        t0 = time.perf_counter()
        search.fit(bundle.X_train, bundle.y_train)
        fit_s = time.perf_counter() - t0

        best = search.best_estimator_
        tr_pred = best.predict(bundle.X_train)
        te_pred = best.predict(bundle.X_test)
        tr_acc = accuracy_score(bundle.y_train, tr_pred)
        te_acc = accuracy_score(bundle.y_test, te_pred)
        tr_f1 = f1_score(bundle.y_train, tr_pred, average=F1_AVERAGE, zero_division=0)
        te_f1 = f1_score(bundle.y_test, te_pred, average=F1_AVERAGE, zero_division=0)

        # Inference timing (batch)
        t_inf0 = time.perf_counter()
        _ = best.predict(bundle.X_test)
        inf_ms = (time.perf_counter() - t_inf0) * 1000 / max(len(bundle.X_test), 1)

        rows.append(
            {
                "dataset": bundle.key,
                "task": "classification",
                "model": name,
                "best_cv_f1": float(search.best_score_),
                "train_accuracy": float(tr_acc),
                "test_accuracy": float(te_acc),
                "train_f1": float(tr_f1),
                "test_f1": float(te_f1),
                "gap_f1": float(tr_f1 - te_f1),
                "fit_time_s": float(fit_s),
                "infer_ms_per_row_batch": float(inf_ms),
                "best_params": search.best_params_,
            }
        )
    return rows


def train_regression(bundle: DatasetBundle) -> List[Dict[str, Any]]:
    models = regression_models(config.RANDOM_STATE)
    rows: List[Dict[str, Any]] = []
    for name, (factory, param_grid) in models.items():
        pre = make_preprocessor(bundle.X_train, scale_numeric=scale_numeric_for_model(name))
        pipe = factory(pre)
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grid,
            n_iter=config.N_ITER_SEARCH,
            scoring="neg_root_mean_squared_error",
            cv=config.INNER_CV_SPLITS,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        t0 = time.perf_counter()
        search.fit(bundle.X_train, bundle.y_train)
        fit_s = time.perf_counter() - t0

        best = search.best_estimator_
        tr_pred = best.predict(bundle.X_train)
        te_pred = best.predict(bundle.X_test)
        tr_rmse = root_mean_squared_error(bundle.y_train, tr_pred)
        te_rmse = root_mean_squared_error(bundle.y_test, te_pred)

        t_inf0 = time.perf_counter()
        _ = best.predict(bundle.X_test)
        inf_ms = (time.perf_counter() - t_inf0) * 1000 / max(len(bundle.X_test), 1)

        rows.append(
            {
                "dataset": bundle.key,
                "task": "regression",
                "model": name,
                "best_cv_rmse": float(-search.best_score_),
                "train_rmse": float(tr_rmse),
                "test_rmse": float(te_rmse),
                "gap_rmse": float(tr_rmse - te_rmse),
                "fit_time_s": float(fit_s),
                "infer_ms_per_row_batch": float(inf_ms),
                "best_params": search.best_params_,
            }
        )
    return rows


def run_bundle(bundle: DatasetBundle) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "dataset_meta": {
            "key": bundle.key,
            "name": bundle.name,
            "task": bundle.task,
            "source": bundle.source,
            "license_note": bundle.license_note,
            "target_description": bundle.target_description,
            "leakage_notes": bundle.leakage_notes,
            "n_samples": bundle.n_samples,
            "n_features": bundle.n_features,
            "n_classes": getattr(bundle, "n_classes", None),
            "class_imbalance_ratio": bundle.class_imbalance_ratio,
        },
        "baselines": {},
        "results": [],
    }
    if bundle.task == "classification":
        out["baselines"] = _baselines_classification(bundle)
        out["results"] = train_classification(bundle)
    else:
        out["baselines"] = _baselines_regression(bundle)
        out["results"] = train_regression(bundle)
    return out


def save_all(results: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    serializable = []
    for r in results:
        block = json.loads(json.dumps(r, default=str))
        serializable.append(block)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
