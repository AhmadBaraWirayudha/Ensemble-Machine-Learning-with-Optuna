from src.data_loader import load_dataset

from src.features import (
    add_engineered_features
)


def test_feature_engineering():

    df = load_dataset()

    df = add_engineered_features(df)

    assert "Vc2" in df.columns

    assert "Vc_Fz" in df.columns

    assert "log_Vc" in df.columns
