# Analysis: simple vs complex models

Protocol: competitive F1 margin ≤ 0.02; competitive RMSE relative margin ≤ 5%.

## breast_cancer (classification)

- Baseline majority F1: 0.4890
- Best test F1: 0.9564
- Largest train–test F1 gap (model): **random_forest** (max gap 0.0526)
- Competitive simple models: logistic_regression, decision_tree
- Competitive complex models: random_forest, xgboost

| model | test F1 | within margin | train–test gap | fit (s) |
| --- | --- | --- | --- | --- |
| logistic_regression | 0.9564 | True | 0.0304 | 1.92 |
| xgboost | 0.9558 | True | 0.0442 | 0.87 |
| random_forest | 0.9474 | True | 0.0526 | 3.71 |
| decision_tree | 0.9387 | True | 0.0480 | 0.21 |

## wine (classification)

- Baseline majority F1: 0.2178
- Best test F1: 1.0000
- Largest train–test F1 gap (model): **random_forest** (max gap 0.0000)
- Competitive simple models: logistic_regression, decision_tree
- Competitive complex models: random_forest, xgboost

| model | test F1 | within margin | train–test gap | fit (s) |
| --- | --- | --- | --- | --- |
| logistic_regression | 1.0000 | True | -0.0141 | 0.12 |
| decision_tree | 1.0000 | True | -0.0141 | 0.08 |
| random_forest | 1.0000 | True | 0.0000 | 2.34 |
| xgboost | 1.0000 | True | 0.0000 | 0.76 |

## california_housing (regression)

- Baseline mean RMSE: 1.1449
- Best test RMSE: 0.4390
- Largest train–test RMSE gap (model): **ridge** (max gap -0.0259)
- Competitive simple models: (none)
- Competitive complex models: xgboost

| model | test RMSE | within margin | train–test gap | fit (s) |
| --- | --- | --- | --- | --- |
| xgboost | 0.4390 | True | -0.2622 | 4.92 |
| random_forest | 0.5026 | False | -0.3181 | 77.74 |
| decision_tree | 0.5978 | False | -0.0923 | 0.66 |
| ridge | 0.7456 | False | -0.0259 | 0.29 |

## Interpretation notes

- **When simple models work well**: near-linear decision boundaries, strong numeric features, 
or limited sample size where deep trees overfit (watch train–test gap).
- **When complex models pull ahead**: non-additive interactions, heterogeneous subpopulations, 
or smooth nonlinear structure (housing) where ensembles reduce variance.
- **Overfitting**: large positive F1 gap or RMSE gap indicates memorization; compare against CV score.
- **Tradeoff**: higher complexity rank vs. marginal test gain and `fit_time_s` / inference ms.
