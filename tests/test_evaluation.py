from src.evaluation import (
    calculate_metrics
)


def test_metrics():

    y_true = [1, 2, 3, 4]

    y_pred = [1.1, 2.1, 2.9, 4.0]

    results = calculate_metrics(
        y_true,
        y_pred
    )

    assert "RMSE" in results

    assert "R2" in results
