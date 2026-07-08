import pytest
from fastapi.testclient import TestClient

import monitoring.storage as storage_module
from app.main import app


@pytest.fixture(autouse=True)
def isolated_prediction_log(tmp_path, monkeypatch):
    """Redirect the production database to a temp file so running this
    test suite doesn't pollute the real logs/production.db (which the
    drift monitor and accuracy report treat as real production traffic)."""
    monkeypatch.setattr(storage_module, "DB_PATH", tmp_path / "test_production.db")
    yield


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["recommended_model"] in (
        "SVR", "GPR", "RandomForest", "GradientBoosting", "PowerLaw",
        "Weighted_Ensemble", "Stacking_Ensemble",
    )


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert "metrics" in body
    assert "feature_columns" in body


def test_model_feature_importance(client):
    r = client.get("/model/feature-importance")
    assert r.status_code == 200
    body = r.json()
    assert "detailed" in body
    assert "by_variable" in body
    assert set(body["by_variable"].keys()) == {"Vc", "Fz", "ap"}


def test_predict_valid_input(client):
    r = client.post("/predict", json={"Vc": 10.0, "Fz": 0.1, "ap": 1.0})
    assert r.status_code == 200
    body = r.json()
    assert body["range_check"]["within_training_envelope"] is True
    assert isinstance(body["recommended_prediction"], float)


def test_predict_out_of_range_still_succeeds_with_warning(client):
    r = client.post("/predict", json={"Vc": 500.0, "Fz": 0.1, "ap": 1.0})
    assert r.status_code == 200
    body = r.json()
    assert body["range_check"]["within_training_envelope"] is False
    assert "Vc" in body["range_check"]["out_of_range_features"]


@pytest.mark.parametrize("bad_payload", [
    {"Vc": -1.0, "Fz": 0.1, "ap": 1.0},   # negative
    {"Vc": 0.0, "Fz": 0.1, "ap": 1.0},    # zero (must be > 0)
    {"Vc": 10.0, "Fz": 0.1},              # missing ap
    {"Vc": "fast", "Fz": 0.1, "ap": 1.0}, # wrong type
])
def test_predict_rejects_invalid_input(client, bad_payload):
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_predict_batch(client):
    r = client.post("/predict/batch", json={"items": [
        {"Vc": 10.0, "Fz": 0.1, "ap": 1.0},
        {"Vc": 12.5, "Fz": 0.125, "ap": 1.25},
    ]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_predict_batch_rejects_empty_list(client):
    r = client.post("/predict/batch", json={"items": []})
    assert r.status_code == 422


def test_drift_report_insufficient_data(client):
    # isolated_prediction_log guarantees a fresh, empty log for this test
    r = client.get("/drift/report")
    assert r.status_code == 200
    assert r.json()["status"] == "insufficient_data"


def test_drift_report_runs_once_enough_traffic_logged(client):
    for _ in range(35):
        client.post("/predict", json={"Vc": 10.0, "Fz": 0.1, "ap": 1.0})

    r = client.get("/drift/report")
    assert r.status_code == 200
    body = r.json()
    assert "overall_verdict" in body
    assert body["n_current_samples"] == 35


def test_predict_returns_and_echoes_job_id(client):
    r = client.post("/predict", json={"Vc": 10.0, "Fz": 0.1, "ap": 1.0, "job_id": "PART-1"})
    assert r.status_code == 200
    assert r.json()["job_id"] == "PART-1"


def test_predict_auto_generates_job_id_when_omitted(client):
    r = client.post("/predict", json={"Vc": 10.0, "Fz": 0.1, "ap": 1.0})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert job_id  # non-empty
    # a second call without job_id gets a *different* auto-generated one
    r2 = client.post("/predict", json={"Vc": 10.0, "Fz": 0.1, "ap": 1.0})
    assert r2.json()["job_id"] != job_id


def test_submit_measurement(client):
    r = client.post("/measurements", json={"Ra_measured": 0.75, "job_id": "PART-2", "device": "TIME3233"})
    assert r.status_code == 200
    assert r.json() == {"status": "recorded", "job_id": "PART-2"}


def test_submit_measurement_rejects_non_positive_ra(client):
    r = client.post("/measurements", json={"Ra_measured": -1.0, "job_id": "PART-3"})
    assert r.status_code == 422


def test_accuracy_report_insufficient_data_by_default(client):
    r = client.get("/accuracy/report")
    assert r.status_code == 200
    assert r.json()["status"] == "insufficient_data"


def test_accuracy_report_matches_predict_and_measurement_by_job_id(client):
    pred = client.post("/predict", json={"Vc": 10.0, "Fz": 0.1, "ap": 1.0, "job_id": "PART-4"}).json()
    client.post("/measurements", json={"Ra_measured": pred["recommended_prediction"] + 0.05, "job_id": "PART-4"})

    r = client.get("/accuracy/report")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_matched_pairs"] == 1
    assert abs(body["bias_measured_minus_predicted"] - 0.05) < 1e-6


def test_retrain_history_empty_by_default(client):
    # isolated_prediction_log (autouse) already points at a fresh, empty
    # temp database for this test
    r = client.get("/retrain/history")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 0
    assert body["events"] == []
