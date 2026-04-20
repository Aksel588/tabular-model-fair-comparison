# Analysis: simple vs complex models

> **Generated file** — Overwritten by `run_experiment.py` (via `write_analysis_report`). Content matches the datasets included in that run.

Protocol: competitive F1 margin ≤ 0.02; competitive RMSE relative margin ≤ 5%.

## breast_cancer (classification)

- Baseline majority F1: 0.4890
- Best test F1: 0.9564
- Largest train–test F1 gap (model): **random_forest** (max gap 0.0526)
- Competitive simple models: logistic_regression, decision_tree
- Competitive complex models: random_forest, hist_gradient_boosting, xgboost

| model | test F1 | within margin | train–test gap | fit (s) |
| --- | --- | --- | --- | --- |
| logistic_regression | 0.9564 | True | 0.0304 | 1.29 |
| hist_gradient_boosting | 0.9558 | True | 0.0442 | 1.83 |
| xgboost | 0.9558 | True | 0.0442 | 0.88 |
| random_forest | 0.9474 | True | 0.0526 | 3.07 |
| decision_tree | 0.9387 | True | 0.0480 | 0.16 |

## Interpretation notes

- **When simple models work well**: near-linear decision boundaries, strong numeric features, 
or limited sample size where deep trees overfit (watch train–test gap).
- **When complex models pull ahead**: non-additive interactions, heterogeneous subpopulations, 
or smooth nonlinear structure (housing) where ensembles reduce variance.
- **Overfitting**: large positive F1 gap or RMSE gap indicates memorization; compare against CV score.
- **Tradeoff**: higher complexity rank vs. marginal test gain and `fit_time_s` / inference ms.
