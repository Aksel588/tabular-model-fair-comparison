"""
Tabular benchmarks: sklearn bundles (offline) plus OpenML German Credit (cached on first download).

For more OpenML/Kaggle CSVs, see docs/DATASETS.md — same protocol applies once loaders are wired in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_wine,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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


def load_digits_bundle() -> DatasetBundle:
    """
    Optical recognition of handwritten digits (10-class classification).
    8×8 pixels as 64 numeric features — common sklearn tabular-style benchmark.
    """
    raw = load_digits(as_frame=True)
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
        key="digits",
        name="Digits (UCI / sklearn)",
        task="classification",
        source="sklearn.datasets.load_digits (UCI)",
        license_note="BSD-style (scikit-learn dataset)",
        target_description="Multiclass: digit 0–9",
        leakage_notes="Image pixels as features; i.i.d. split ignores any group structure.",
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


def load_german_credit_bundle() -> DatasetBundle:
    """
    German Credit (OpenML id 31): binary classification with mixed numeric/categorical columns.
    First run downloads into ``config.DATA_CACHE``; offline runs use the cache.
    """
    os.makedirs(config.DATA_CACHE, exist_ok=True)
    raw = fetch_openml(
        data_id=31,
        as_frame=True,
        parser="auto",
        data_home=config.DATA_CACHE,
    )
    X = raw.data
    y = LabelEncoder().fit_transform(raw.target)
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    return DatasetBundle(
        key="german_credit",
        name="German Credit (Statlog)",
        task="classification",
        source="OpenML data_id=31 (credit-g / Statlog)",
        license_note="See OpenML dataset page; research use — check redistribution terms",
        target_description="Binary: credit risk (encoded labels)",
        leakage_notes="Historical credit data; sensitive attributes; random split not temporal; not for production fairness claims without audit.",
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


def load_diabetes_bundle() -> DatasetBundle:
    """
    Diabetes progression (regression). Small n; sklearn bundle.
    """
    raw = load_diabetes(as_frame=True)
    X = raw.data
    y = raw.target.values
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )
    return DatasetBundle(
        key="diabetes",
        name="Diabetes (regression)",
        task="regression",
        source="sklearn.datasets.load_diabetes",
        license_note="BSD-style (scikit-learn dataset)",
        target_description="Disease progression quantitative response (1 year)",
        leakage_notes="Small sample; high metric variance; baseline ridge vs trees is still informative.",
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


DATASET_KEYS: tuple[str, ...] = (
    "breast_cancer",
    "wine",
    "digits",
    "german_credit",
    "california_housing",
    "diabetes",
)

_DATASET_LOADERS = {
    "breast_cancer": load_breast_cancer_bundle,
    "wine": load_wine_bundle,
    "digits": load_digits_bundle,
    "german_credit": load_german_credit_bundle,
    "california_housing": load_california_housing,
    "diabetes": load_diabetes_bundle,
}


def list_dataset_keys() -> list[str]:
    return list(DATASET_KEYS)


def iter_datasets():
    for key in DATASET_KEYS:
        yield _DATASET_LOADERS[key]()


def get_dataset(key: str) -> DatasetBundle:
    if key not in _DATASET_LOADERS:
        raise KeyError(key)
    return _DATASET_LOADERS[key]()
