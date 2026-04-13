"""
Research protocol: simple vs complex models (locked before experiments).
Do not tune these constants based on test-set results.
"""

import os

# Reproducibility
RANDOM_STATE = 42

# Train / test split (holdout; test used once for final reporting)
TEST_SIZE = 0.2

# Inner CV for hyperparameter search (training data only)
INNER_CV_SPLITS = 3
N_ITER_SEARCH = 50  # RandomizedSearchCV trials per model (matched budget)

# "Competitive" definition: simple model within this margin of best validated score
# Classification: absolute F1 margin (e.g. 0.02 = 2 points on F1 scale)
COMPETITIVE_MARGIN_F1 = 0.02
# Regression: relative RMSE margin vs best RMSE on validation (e.g. 0.05 = 5% worse)
COMPETITIVE_MARGIN_RMSE_REL = 0.05

# Optional: repeated outer splits for stability (same protocol; more compute)
N_REPEATED_SPLITS = 0  # set to 3–5 to report mean±std over seeds

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_CACHE = os.path.join(ROOT, "data_cache")
