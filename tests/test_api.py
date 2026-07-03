import pytest
from fastapi.testclient import TestClient

import monitoring.request_log as request_log_module
from app.main import app


@pytest.fixture(autouse=True)
def isolated_prediction_log(tmp_path, monkeypatch):
    """Redirect the prediction log to a temp file so running this test
    suite doesn't pollute the real logs/prediction_log.jsonl (which the
    drift monitor treats as real production traffic)."""
    monkeypatch.setattr(request_log_module, "PREDICTION_LOG_PATH", tmp_path / "test_prediction_log.jsonl")
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
    assert body["recommended_model"] in ("Weighted_Ensemble", "Stacking_Ensemble")


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert "metrics" in body
    assert "feature_columns" in body


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
