# Locked experimental protocol

These choices are fixed in [`config.py`](../config.py) **before** looking at test metrics.

| Item | Setting |
| --- | --- |
| Train / test | 80/20 holdout; stratified for classification |
| Inner CV | `INNER_CV_SPLITS` folds on training data only |
| Inner CV objective | Classification: maximize `f1_weighted`. Regression: maximize `neg_root_mean_squared_error` (same RMSE units as test reporting after sign flip) |
| Tuning budget | `N_ITER_SEARCH` RandomizedSearchCV trials **per model** |
| Competitiveness | Classification: within `COMPETITIVE_MARGIN_F1` of best test F1. Regression: within `COMPETITIVE_MARGIN_RMSE_REL` of best test RMSE |
| Baselines | Majority class (classification), mean predictor (regression) |
| Class imbalance (classification) | `LogisticRegression`: `class_weight="balanced"`. `RandomForestClassifier`: `class_weight="balanced_subsample"`. Other estimators: defaults in `src/pipelines.py` |
| Seeds | `RANDOM_STATE` for splits and search |

**Optional stability**: set `N_REPEATED_SPLITS` > 0 to repeat outer splits (more compute, stronger claims).
