# X (Twitter) thread outline

**Post 1**  
We’re told to “just use XGBoost” on tabular data. Sometimes that’s right—but *when* is a linear model enough? I ran a fixed experiment: same train/test split, same number of hyperparameter trials per model, same preprocessing rules. Thread on what happened 👇

**Post 2**  
Setup: Logistic / Ridge + shallow Decision Tree vs Random Forest + XGBoost. Metrics: F1 (weighted) + accuracy for classification, RMSE for regression. Inner CV for tuning; test set used once. No peeking.

**Post 3**  
“Competitive” = within a small margin of the best test score (see `config.py`). That’s stricter than eyeballing leaderboards—it forces you to say *how close* is close enough.

**Post 4**  
Dataset caveat: I used reproducible sklearn/UCI bundles (see `docs/DATASETS.md`). Real Kaggle-scale data can differ; the point is the *protocol*, not one leaderboard rank.

**Post 5**  
Findings to look for in `results/ANALYSIS.md`: (1) Did logistic land in the winner band? (2) Who had the biggest train–test gap? (3) Was the RMSE win worth the fit time?

**Post 6**  
Takeaway: Complexity buys flexibility, not free performance. If your simple model is within the margin, you may prefer interpretability, speed, and simpler ops—*if* your error profile allows it.

**Post 7**  
Repo: run `python run_experiment.py`, read `results/all_results.json` + figures. Criticism welcome—especially better splits (e.g. spatial CV for housing) and more datasets.
