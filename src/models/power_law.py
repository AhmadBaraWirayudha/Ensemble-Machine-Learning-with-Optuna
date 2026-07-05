"""
Classical power-law surface roughness model.

Empirical machining models of the form

    Ra = C * Vc^a * Fz^b * ap^c

(fit via ordinary least squares on log-transformed variables) are the
textbook Taguchi / Response Surface Methodology approach to exactly this
problem - surface roughness as a function of cutting speed, feed, and
depth of cut - and are standard in the machining literature. Notably,
nothing in this project (original prototype or otherwise) ever tried it;
the modeling here went straight to SVR/GPR. It's worth having as a
baseline: it's simple, has only 4 parameters (so essentially no
overfitting risk on 119 samples), is directly interpretable (the fitted
exponents a/b/c say how sensitive Ra is to each parameter), and gives the
stacking ensemble a qualitatively different, much lower-variance signal
to combine with the flexible SVR/GPR models.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


class PowerLawRegressor(BaseEstimator, RegressorMixin):
    """
    Fits Ra = C * Vc^a * Fz^b * ap^c by ordinary least squares on
    log(Ra) = log(C) + a*log(Vc) + b*log(Fz) + c*log(ap).

    Expects X's first 3 columns to be [Vc, Fz, ap] in that order (true of
    MODEL_FEATURE_COLUMNS - see src/preprocessing/features.py) - any
    further engineered columns are ignored, since this model only uses
    the 3 raw machining parameters, exactly like the classical formula it
    represents.
    """

    def fit(self, X, y):
        X = np.asarray(X)[:, :3]
        y = np.asarray(y)

        if (X <= 0).any() or (y <= 0).any():
            raise ValueError(
                "PowerLawRegressor requires strictly positive Vc, Fz, ap, "
                "and Ra (it fits in log-space) - got a non-positive value."
            )

        log_X = np.log(X)
        log_y = np.log(y)

        self._ols = LinearRegression().fit(log_X, log_y)
        self.log_C_ = self._ols.intercept_
        self.exponents_ = dict(zip(["Vc", "Fz", "ap"], self._ols.coef_))
        return self

    def predict(self, X):
        X = np.asarray(X)[:, :3]
        log_X = np.log(np.clip(X, 1e-12, None))
        return np.exp(self._ols.predict(log_X))

    def formula_string(self):
        C = np.exp(self.log_C_)
        a, b, c = (self.exponents_[k] for k in ["Vc", "Fz", "ap"])
        return f"Ra = {C:.4f} * Vc^{a:.4f} * Fz^{b:.4f} * ap^{c:.4f}"
