"""
Real-world tabular datasets loaded from sklearn (bundled, no network required).

For OpenML/Kaggle mirrors, see docs/DATASETS.md — same protocol applies once CSVs are wired in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_wine,
)
from sklearn.model_selection import train_test_split

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import config


TaskType = Literal["classification", "regression"]


@dataclass
class DatasetBundle:
    key: str
    name: str
    task: TaskType
    source: str
    license_note: str
    target_description: str
    leakage_notes: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list
    n_samples: int
    n_features: int
    class_imbalance_ratio: Optional[float]
    n_classes: int


def _imbalance_ratio(y: np.ndarray) -> float:
    counts = np.bincount(y.astype(int))
    return float(counts.max() / max(counts.min(), 1))


def load_breast_cancer_bundle() -> DatasetBundle:
    """
    Wisconsin Breast Cancer (binary classification). UCI via sklearn bundle.
    """
    raw = load_breast_cancer(as_frame=True)
    X = raw.data
    y = raw.target.values
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    return DatasetBundle(
        key="breast_cancer",
        name="Breast Cancer Wisconsin",
        task="classification",
        source="sklearn.datasets.load_breast_cancer (UCI)",
        license_note="BSD-style (scikit-learn dataset)",
        target_description="Binary: malignant (1) vs benign (0)",
        leakage_notes="Classic medical tabular benchmark; random split (not patient-level if duplicates exist—limitation for clinical claims).",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        n_samples=len(y),
        n_features=X.shape[1],
        class_imbalance_ratio=_imbalance_ratio(y_train),
        n_classes=int(len(np.unique(y))),
    )


def load_wine_bundle() -> DatasetBundle:
    """
    Wine recognition (multiclass). UCI via sklearn bundle.
    """
    raw = load_wine(as_frame=True)
    X = raw.data
    y = raw.target.values
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    return DatasetBundle(
        key="wine",
        name="Wine Recognition",
        task="classification",
        source="sklearn.datasets.load_wine (UCI)",
        license_note="BSD-style (scikit-learn dataset)",
        target_description="Multiclass: cultivar (3 classes)",
        leakage_notes="Small n; high variance in test metrics; good for comparing overfitting, weak for broad claims.",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        n_samples=len(y),
        n_features=X.shape[1],
        class_imbalance_ratio=_imbalance_ratio(y_train),
        n_classes=int(len(np.unique(y))),
    )


def load_california_housing() -> DatasetBundle:
    """
    California Housing regression. Pace & Barry; sklearn bundle.
    """
    data = fetch_california_housing(as_frame=True)
    X = data.frame.drop(columns=["MedHouseVal"])
    y = data.frame["MedHouseVal"].values
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )
    return DatasetBundle(
        key="california_housing",
        name="California Housing",
        task="regression",
        source="sklearn / Pace & Barry (1997)",
        license_note="See sklearn dataset description",
        target_description="Median house value (sklearn scale)",
        leakage_notes="Random split ignores geography; spatial CV would be stricter.",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        n_samples=len(y),
        n_features=X.shape[1],
        class_imbalance_ratio=None,
        n_classes=0,
    )


def iter_datasets():
    yield load_breast_cancer_bundle()
    yield load_wine_bundle()
    yield load_california_housing()


def get_dataset(key: str) -> DatasetBundle:
    for d in iter_datasets():
        if d.key == key:
            return d
    raise KeyError(key)
