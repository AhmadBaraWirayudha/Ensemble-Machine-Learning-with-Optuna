from src.data_loader import load_dataset


def test_dataset_loads():

    df = load_dataset()

    assert df is not None

    assert len(df) > 0

    assert "Ra" in df.columns
