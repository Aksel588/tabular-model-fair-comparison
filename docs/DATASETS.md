# Datasets

Most loaders use **bundled** `sklearn.datasets` (no network). `german_credit` uses **OpenML** (id 31, credit-g): the first run downloads into the repo’s `data_cache/` directory (`DATA_CACHE` in [`config.py`](../config.py)); later runs read from that cache offline. The same `DatasetBundle` pattern works for CSVs from [Kaggle](https://www.kaggle.com/datasets), [OpenML](https://www.openml.org/), or [UCI](https://archive.ics.uci.edu/) — add a loader in [`src/datasets.py`](../src/datasets.py).

Each run writes a block into [`results/all_results.json`](../results/all_results.json) with `dataset_meta` (`source`, `license_note`, `target_description`, `leakage_notes`, `n_samples`, `n_features`, `n_classes`, and `class_imbalance_ratio` for classification) alongside model metrics.

| Key | Task | Rows (approx) | Features | Notes |
| --- | --- | --- | --- | --- |
| `breast_cancer` | Binary classification | 569 | 30 numeric | Strong linear signal; good for comparing calibration of ensembles vs logistic. |
| `wine` | Multiclass (3) | 178 | 13 numeric | **Small n**: test metrics can hit 1.0 by chance; use for overfitting discussion, not broad generalization claims. |
| `digits` | Multiclass (10) | 1,797 | 64 numeric | Digits as 8×8 pixels; common sklearn “tabular” benchmark. |
| `german_credit` | Binary classification | 1,000 | 20 mixed | Statlog German Credit; numeric + categorical; sensitive attributes — document fairness limits. |
| `california_housing` | Regression | 20,640 | 8 numeric | Random split ignores geography; spatial CV would be stricter for deployment claims. |
| `diabetes` | Regression | 442 | 10 numeric | Small n; high variance in test RMSE. |

**Licenses**: see sklearn dataset `DESCR` / OpenML page for terms.

**Leakage**: document any column that is a proxy for the label; here we use standard sklearn/OpenML bundles as-is. For external CSVs, add a short audit table (column → risk).
