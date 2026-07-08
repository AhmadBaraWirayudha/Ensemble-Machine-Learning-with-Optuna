import pandas as pd
import pytest

import monitoring.storage as storage_module
from monitoring.storage import (
    log_prediction,
    read_prediction_log,
    log_measurement,
    read_measurements,
    compute_accuracy_report,
    log_retrain_event,
    read_retrain_log,
)


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_module, "DB_PATH", tmp_path / "test.db")
    return tmp_path / "test.db"


FAKE_PREDICTION = {
    "svr_prediction": 0.7, "gpr_prediction": 0.75, "rf_prediction": 0.6, "gbm_prediction": 0.55,
    "power_law_prediction": 0.8, "weighted_ensemble_prediction": 0.72, "stacking_ensemble_prediction": 0.65,
    "recommended_model": "GradientBoosting", "recommended_prediction": 0.55,
}


def test_read_prediction_log_empty_by_default(isolated_db):
    df = read_prediction_log()
    assert len(df) == 0
    assert "Vc" in df.columns


def test_log_and_read_prediction_round_trip(isolated_db):
    log_prediction(10.0, 0.1, 1.0, FAKE_PREDICTION, job_id="J1")
    log_prediction(12.5, 0.1, 1.0, FAKE_PREDICTION)

    df = read_prediction_log()
    assert len(df) == 2
    assert df.iloc[0]["job_id"] == "J1"
    assert pd.isna(df.iloc[1]["job_id"])  # NULL job_id comes back as NaN, not None, via pandas
    assert df.iloc[0]["recommended_model"] == "GradientBoosting"


def test_db_file_is_actually_created_on_disk(isolated_db):
    log_prediction(10.0, 0.1, 1.0, FAKE_PREDICTION)
    assert isolated_db.exists()


def test_measurements_round_trip(isolated_db):
    log_measurement(ra_measured=0.6, job_id="J1", device="TIME3233", raw_payload="Ra=0.60um")
    df = read_measurements()
    assert len(df) == 1
    assert df.iloc[0]["Ra_measured"] == 0.6
    assert df.iloc[0]["device"] == "TIME3233"


def test_accuracy_report_insufficient_data_when_nothing_logged(isolated_db):
    report = compute_accuracy_report()
    assert report["status"] == "insufficient_data"


def test_accuracy_report_insufficient_data_when_no_shared_job_id(isolated_db):
    log_prediction(10.0, 0.1, 1.0, FAKE_PREDICTION, job_id="A")
    log_measurement(ra_measured=0.6, job_id="B")

    report = compute_accuracy_report()
    assert report["status"] == "insufficient_data"
    assert report["n_predictions_with_job_id"] == 1
    assert report["n_measurements_with_job_id"] == 1


def test_accuracy_report_computes_correct_residuals(isolated_db):
    # recommended_prediction is 0.55 (see FAKE_PREDICTION); measure 0.65 -> residual +0.10
    log_prediction(10.0, 0.1, 1.0, FAKE_PREDICTION, job_id="J1")
    log_measurement(ra_measured=0.65, job_id="J1")

    report = compute_accuracy_report()
    assert report["status"] == "ok"
    assert report["n_matched_pairs"] == 1
    assert abs(report["bias_measured_minus_predicted"] - 0.10) < 1e-9
    assert abs(report["mae_actual_vs_predicted"] - 0.10) < 1e-9


def test_accuracy_report_uses_latest_measurement_on_remeasure(isolated_db):
    log_prediction(10.0, 0.1, 1.0, FAKE_PREDICTION, job_id="J1")
    log_measurement(ra_measured=0.90, job_id="J1")  # a bad first reading
    log_measurement(ra_measured=0.55, job_id="J1")  # re-measured, matches prediction exactly

    report = compute_accuracy_report()
    assert report["status"] == "ok"
    assert abs(report["bias_measured_minus_predicted"] - 0.0) < 1e-9


def test_retrain_events_round_trip(isolated_db):
    log_retrain_event({"action": "promoted", "reason": "improved", "old_rmse": 0.3, "new_rmse": 0.24, "pct_change": -20.0})
    events = read_retrain_log()
    assert len(events) == 1
    assert events[0]["action"] == "promoted"
    assert events[0]["new_rmse"] == 0.24


def test_read_retrain_log_empty_by_default(isolated_db):
    assert read_retrain_log() == []
