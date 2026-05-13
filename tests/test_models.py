from src.models import (
    build_svr_pipeline,
    build_gpr_pipeline
)


def test_svr_pipeline():

    model = build_svr_pipeline()

    assert model is not None


def test_gpr_pipeline():

    model = build_gpr_pipeline()

    assert model is not None
