# Medium article outline

## Title options

- “Simple Models Aren’t ‘Dead’: A Fair Tabular Benchmark”  
- “When XGBoost Beats Logistic Regression—and When It Doesn’t”

## Subtitle

A beginner-friendly, research-style experiment with fixed splits, matched tuning budgets, and honest limitations.

## Sections

1. **Hook** — The tension between default ensemble usage and interpretable baselines.  
2. **What we actually measured** — Operational definitions of “simple”, “complex”, and “competitive”.  
3. **Protocol** — Train/test split, inner CV, tuning objectives (`f1_weighted` / RMSE on folds), metrics (F1 weighted, RMSE), baselines.  
4. **Datasets** — Six benchmarks (sklearn bundles + optional OpenML German credit); why Wine’s and Diabetes’s small *n* makes headline metrics noisy.  
5. **Results** — Embed tables from `ANALYSIS.md` and the complexity figure.  
6. **What to conclude (and not)** — Conditions favoring linear models vs boosting; overfitting read from gaps.  
7. **How to extend** — OpenML IDs, Kaggle CSVs, spatial CV for housing, repeated seeds.  
8. **Reproducibility** — Link to GitHub, `requirements.txt`, runtime notes.

## Callouts (sidebars)

- “Fairness” = same search iterations per model, not the same hyperparameter grid size.  
- “Real world” = messy; public benchmarks are a starting point, not proof of production behavior.
