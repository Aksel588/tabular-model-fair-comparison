# Are simple models competitive with complex models on real-world tabular data?

## Introduction

Classical machine learning practice often assumes that tree ensembles and gradient boosting dominate tabular problems once hyperparameters are tuned. That pattern is real in many competitions, but it is not universal: **linear models can match or approach ensemble performance** when the signal is roughly additive, features are well behaved, or data are scarce relative to model capacity. This study asks a narrower, testable question: on fixed public benchmarks with **matched preprocessing budgets** and **matched hyperparameter search effort**, how often do “simple” and “complex” models land in the same performance band on a held-out test set?

**Limitations stated up front**: three sklearn/UCI-style datasets are not the full universe of “real world”; the Wine set is small enough that perfect test scores are uninformative about deployment. The value is methodological: reproducible protocol and explicit tradeoffs.

## Hypothesis

**H1**: After equal tuning budget on training data only, at least one simple model (logistic regression or shallow decision tree) will fall within the pre-defined margin of the best test metric on each dataset.

**H2**: Complex models will show larger **train–test gaps** more often when capacity is unnecessary (overfitting proxy), and longer **fit times** for similar test scores.

These hypotheses are **falsifiable** by the JSON metrics in `results/all_results.json`.

## Data

See [`DATASETS.md`](DATASETS.md). We use two classification tasks (binary + small multiclass) and one regression task, all loadable offline for reproducibility.

## Methodology

- **Split**: 80% train / 20% test (`TEST_SIZE`), stratified for classification.  
- **Tuning**: `RandomizedSearchCV` with `N_ITER_SEARCH` trials per model, `INNER_CV_SPLITS`-fold inner CV, scoring `f1_weighted` (classification) or RMSE (regression).  
- **Metrics**: Accuracy + **F1 (weighted)** for classification; **RMSE** for regression.  
- **Baselines**: majority class / mean predictor.  
- **Complexity proxy**: ordinal rank (logistic/ridge < decision tree < random forest < XGBoost) plus wall-clock fit time.

## Results

Consult the generated [`results/ANALYSIS.md`](../results/ANALYSIS.md) after running `python run_experiment.py`. Expect:

- **Breast cancer**: strong linear separability → logistic competitive with ensembles.  
- **Wine**: trivially high test scores → emphasize **variance** and small-sample caveats, not winner declarations.  
- **California housing**: nonlinear structure → boosting often leads RMSE; ridge may lag beyond the competitiveness margin.

## Analysis

1. **When simple models work**: approximately linear decision boundaries, informative numeric features, or insufficient data for high-variance fits.  
2. **When complex models win**: interaction-rich targets, smooth nonlinear response surfaces (housing), or variance reduction from ensembling—**if** the train–test gap stays controlled.  
3. **Overfitting**: compare `gap_f1` / `gap_rmse` (train vs test on refitted model); large positive F1 gap suggests memorization.  
4. **Performance vs complexity**: use `fit_time_s` and `infer_ms_per_row_batch` alongside test metrics; a 1% metric gain may not justify 10× training cost.

## Conclusion

“Simple vs complex” is not a single answer: it is a **conditional** statement about dataset geometry, sample size, and tuning fairness. This repo makes the condition explicit through a locked protocol; extending it to larger OpenML or Kaggle CSVs is a direct swap in `src/datasets.py` without changing evaluation logic.
