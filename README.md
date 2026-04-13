# Simple vs Complex Tabular Models — Reproducible Benchmark

**A controlled experiment:** compare *simple* models (logistic regression / ridge, shallow decision trees) with *complex* ones (random forest, XGBoost) on public tabular data using **matched preprocessing rules**, **matched hyperparameter search budget**, and a **single held-out test set**—no peeking at test labels during tuning.

---

## Author

Created by **[Aksel Aghajanyan](https://github.com/Aksel588)** · [github.com/Aksel588](https://github.com/Aksel588)

---

## Why this exists

People often default to gradient boosting on tabular tasks. That can be right—but not always. This repository turns the question into something **measurable**: under explicit fairness constraints, how close do simple models get to ensembles, where do gaps open up, and what do train–test gaps suggest about overfitting?

This is **reproducible benchmarking / experimental methodology**, not a claim of novel theory or state-of-the-art on every dataset.

---

## What you get

| Output | Description |
| --- | --- |
| [`results/all_results.json`](results/all_results.json) | Per-dataset metrics, CV scores, train/test gaps, fit times, best hyperparameters |
| [`results/ANALYSIS.md`](results/ANALYSIS.md) | Tables: competitiveness vs best model, overfitting proxy |
| [`results/complexity_tradeoff.png`](results/complexity_tradeoff.png) | Ordinal model complexity vs test metric |

Documentation:

| Doc | Content |
| --- | --- |
| [`docs/PROTOCOL.md`](docs/PROTOCOL.md) | Locked splits, CV, tuning budget, “competitive” margins |
| [`docs/DATASETS.md`](docs/DATASETS.md) | Dataset sources, licenses, leakage / limitation notes |
| [`docs/RESEARCH_WRITEUP.md`](docs/RESEARCH_WRITEUP.md) | Paper-style narrative (intro → conclusion) |
| [`docs/THREAD.md`](docs/THREAD.md) | Outline for a short social thread |
| [`docs/MEDIUM_OUTLINE.md`](docs/MEDIUM_OUTLINE.md) | Outline for a blog post |

---

## Requirements

- **Python** 3.10+ recommended  
- See [`requirements.txt`](requirements.txt): `numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`

---

## Quick start

```bash
git clone <your-repo-url>
cd <repo-folder>

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python run_experiment.py
```

Artifacts are written under [`results/`](results/).

---

## Protocol (summary)

Constants live in [`config.py`](config.py); do not change them to chase test-set performance.

1. **Train / test:** 80% train, 20% test (stratified for classification). Test used **once** for reporting.
2. **Tuning:** `RandomizedSearchCV` with **`N_ITER_SEARCH` trials per model** (same budget for every model), **`INNER_CV_SPLITS`-fold** inner CV on **training data only**.
3. **Competitive:** classification — within **`COMPETITIVE_MARGIN_F1`** of best test F1; regression — within **`COMPETITIVE_MARGIN_RMSE_REL`** of best test RMSE.
4. **Baselines:** majority class (classification) / mean predictor (regression).
5. **Optional:** set `N_REPEATED_SPLITS` in `config.py` to repeat outer splits for mean±std over seeds (more compute).

**Fair comparison rules**

- One **sklearn `Pipeline`** per model: imputation → one-hot for categoricals → scaling **only** for linear models; trees skip numeric scaling (scale-invariant).
- Same search method and iteration budget for each estimator family.

---

## Datasets (bundled via sklearn)

| Key | Task | Notes |
| --- | --- | --- |
| `breast_cancer` | Binary classification | Classic numeric benchmark |
| `wine` | Multiclass (3 classes) | **Small n** — perfect test scores are not deployment evidence |
| `california_housing` | Regression | Random split ignores geography; spatial CV would be stricter |

Swap loaders in [`src/datasets.py`](src/datasets.py) for OpenML/Kaggle CSVs; keep the training code unchanged.

---

## Models compared

| Kind | Models |
| --- | --- |
| Simple | `LogisticRegression`, `DecisionTreeClassifier` / `Ridge`, `DecisionTreeRegressor` |
| Complex | `RandomForestClassifier` / `RandomForestRegressor`, `XGBClassifier` / `XGBRegressor` |

---

## Metrics

- **Classification:** accuracy, **F1 (weighted)** — works for binary and multiclass.  
- **Regression:** **RMSE** (root mean squared error).

---

## Project layout

| Path | Role |
| --- | --- |
| [`config.py`](config.py) | Protocol constants |
| [`run_experiment.py`](run_experiment.py) | Entry point |
| [`src/datasets.py`](src/datasets.py) | Data loading + metadata |
| [`src/pipelines.py`](src/pipelines.py) | Pipelines + hyperparameter grids |
| [`src/train.py`](src/train.py) | Search, fit, metrics, baselines |
| [`src/analyze.py`](src/analyze.py) | Analysis tables + figure |

---

## Sharing this work

- **GitHub:** pin Python version in this README; commit `results/` if you want visitors to see numbers without rerunning (or document that they must run `run_experiment.py`).
- **Kaggle:** attach a zip as a **Dataset** or paste the pipeline into a **Notebook**; `pip install` deps in the first cell; adjust paths if using `/kaggle/input/`.
- **Portfolio:** one-liner — *“Reproducible tabular benchmark: fair tuning budget, inner CV, simple vs ensemble comparison (sklearn + XGBoost).”*

---

## Limitations (read before citing)

- Three sklearn/UCI-style sets are **not** the full “real world.”
- **Wine** is tiny; treat extreme metrics as a cautionary tale about sample size.
- **California housing:** random split underestimates spatial correlation; claims about geographic deployment need stronger validation design.

---

## License

This project is licensed under the [MIT License](LICENSE). Third-party libraries (scikit-learn, XGBoost, etc.) and bundled datasets follow their respective licenses—see their documentation.

---

## Acknowledgments

Built with [scikit-learn](https://scikit-learn.org/) and [XGBoost](https://xgboost.readthedocs.io/). Datasets are provided via sklearn’s dataset loaders; see each dataset’s sklearn/UCI documentation for terms.
# tabular-model-fair-comparison
