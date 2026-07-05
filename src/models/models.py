from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    RobustScaler,
    PolynomialFeatures,
    PowerTransformer
)

from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    RationalQuadratic,
    WhiteKernel,
    ConstantKernel as C
)

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor

from src.config import POLY_DEGREE


def build_svr_pipeline(
    C_value=1.5,
    epsilon=0.2,
    gamma="scale",
    poly_degree=POLY_DEGREE
):
    """
    Build SVR pipeline.
    """

    pipeline = Pipeline([
        (
            "poly",
            PolynomialFeatures(
                degree=poly_degree,
                include_bias=False
            )
        ),
        ("scale", RobustScaler()),
        (
            "power",
            PowerTransformer(
                method="yeo-johnson",
                standardize=True
            )
        ),
        (
            "svr",
            SVR(
                kernel="rbf",
                C=C_value,
                epsilon=epsilon,
                gamma=gamma
            )
        )
    ])

    return TransformedTargetRegressor(
        regressor=pipeline,
        transformer=PowerTransformer()
    )


def build_gpr_pipeline(
    amplitude=1.0,
    length_scale=1.0,
    alpha=1.0,
    noise=1e-4
):
    """
    Build Gaussian Process Regression pipeline.
    """

    kernel = (
        C(amplitude)
        * (
            RBF(length_scale=length_scale)
            * RationalQuadratic(alpha=alpha)
        )
        + WhiteKernel(noise_level=noise)
    )

    pipeline = Pipeline([
        (
            "poly",
            PolynomialFeatures(
                degree=POLY_DEGREE,
                include_bias=False
            )
        ),
        ("scale", RobustScaler()),
        (
            "power",
            PowerTransformer(
                method="yeo-johnson"
            )
        ),
        (
            "gpr",
            GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True
            )
        )
    ])

    return pipeline


def build_gbm_pipeline(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=1.0,
    min_samples_leaf=1,
    random_state=151101
):
    """
    Build a Gradient Boosting pipeline - like build_rf_pipeline, on the 12
    engineered features directly with no PolynomialFeatures step, for the
    same reason (boosted trees capture nonlinearity/interactions via
    splits, not via an explicit polynomial expansion of the inputs).
    """

    return Pipeline([
        (
            "gbm",
            GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )
        )
    ])
