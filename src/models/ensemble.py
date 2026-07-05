"""Ensemble model wrapper - see note below for where this logic actually lives."""


def build_ensemble(models):
    """Build an ensemble from fitted base models."""
    raise NotImplementedError(
        "The weighted-ensemble and stacking-ensemble construction actually "
        "used by this project lives in src/train/train.py (weighted: "
        "alpha*GPR + (1-alpha)*SVR with alpha tuned by "
        "src.optuna_tuning.optimize_ensemble_weight; stacking: a RidgeCV "
        "meta-learner over out-of-fold base-model predictions). This stub "
        "was never wired up to that - not touched as part of the API/"
        "drift-monitoring upgrade. See UPGRADE_NOTES.md."
    )
