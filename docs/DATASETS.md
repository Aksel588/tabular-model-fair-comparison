# Datasets

All three load **without network** from `sklearn.datasets` (bundled UCI-style benchmarks). The same code path can be pointed at CSVs from [Kaggle](https://www.kaggle.com/datasets), [OpenML](https://www.openml.org/), or [UCI](https://archive.ics.uci.edu/) by replacing loaders in [`src/datasets.py`](../src/datasets.py).

| Key | Task | Rows (approx) | Features | Notes |
| --- | --- | --- | --- | --- |
| `breast_cancer` | Binary classification | 569 | 30 numeric | Strong linear signal; good for comparing calibration of ensembles vs logistic. |
| `wine` | Multiclass (3) | 178 | 13 numeric | **Small n**: test metrics can hit 1.0 by chance; use for overfitting discussion, not broad generalization claims. |
| `california_housing` | Regression | 20,640 | 8 numeric | Random split ignores geography; spatial CV would be stricter for deployment claims. |

**Licenses**: see sklearn dataset `DESCR` / bundled terms (BSD-style for these bundles).

**Leakage**: document any column that is a proxy for the label; here we use standard sklearn bundles as-is. For external CSVs, add a short audit table (column → risk).
