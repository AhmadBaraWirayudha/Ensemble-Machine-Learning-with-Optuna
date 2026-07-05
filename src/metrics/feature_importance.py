"""
Feature importance via permutation importance.

The original prototype computed permutation importance *inside* the
polynomial-expanded feature space and then tried to map auto-generated
names like "x3 x7" back to the original Vc/Fz/ap variables by string
matching - fragile, and not reimplemented as part of this upgrade (see
UPGRADE_NOTES.md).

This is simpler and doesn't have that problem: sklearn's
permutation_importance takes a fitted *pipeline* (poly expansion + scaling
+ estimator, all of it) and permutes columns of whatever X you hand it.
Since we hand it our 12 named engineered features (Vc, Fz, ap, Vc2, ...),
scikit-learn permutes at that level and the pipeline re-expands internally
for scoring - so the result is already indexed by clean, meaningful
feature names, with no name-mapping step required at all.
"""

import pandas as pd
from sklearn.inspection import permutation_importance

BASE_VARIABLES = ("Vc", "Fz", "ap")


def compute_permutation_importance(model, X, y, feature_names, scoring="r2", n_repeats=30, random_state=151101):
    """Permutation importance for one fitted pipeline, as a Series indexed
    by feature name, sorted descending."""

    result = permutation_importance(
        model, X, y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return pd.Series(result.importances_mean, index=feature_names, name="importance").sort_values(ascending=False)


def aggregate_by_base_variable(importance: pd.Series, base_vars=BASE_VARIABLES) -> pd.Series:
    """
    Roll engineered-feature importance up to the raw machining parameter
    it derives from, e.g. Vc, Vc2, Vc_Fz, and Vc_ap all count toward Vc.

    An interaction term counts toward *every* variable it involves rather
    than being split between them - Vc_Fz's importance is genuinely joint
    information about both Vc and Fz, not half-and-half. That means these
    group sums can add up to more than the un-grouped total; treat this as
    a "what matters" ranking, not a precise decomposition.
    """

    return pd.Series(
        {var: importance[[f for f in importance.index if var in f]].sum() for var in base_vars},
        name="importance",
    ).sort_values(ascending=False)


def compute_all_feature_importance(models: dict, X, y, feature_names, n_repeats=30, random_state=151101):
    """
    Permutation importance for an arbitrary set of fitted models (e.g.
    {"SVR": svr_model, "GPR": gpr_model, "RandomForest": rf_model, ...}).
    Returns (detailed, by_variable) - both pd.DataFrames with one column
    per model, normalized so each model's column sums to 1 (values
    clipped at 0 first: a *negative* importance_mean means permuting that
    feature randomly, on average, improved the score, which isn't
    meaningful to report as "importance").
    """

    columns = {
        label: compute_permutation_importance(
            model, X, y, feature_names, n_repeats=n_repeats, random_state=random_state,
        )
        for label, model in models.items()
    }

    detailed = pd.DataFrame(columns).clip(lower=0)
    detailed = detailed.div(detailed.sum(axis=0).replace(0, 1), axis=1)

    by_variable = pd.DataFrame({name: aggregate_by_base_variable(col) for name, col in detailed.items()})

    return detailed, by_variable
