import pytest
from fastapi.testclient import TestClient

import monitoring.request_log as request_log_module
from app.main import app
from app.auth import API_KEY_ENV_VAR


@pytest.fixture(autouse=True)
def isolated_prediction_log(tmp_path, monkeypatch):
    """Same isolation as test_api.py - avoid polluting the real prediction log."""
    monkeypatch.setattr(request_log_module, "PREDICTION_LOG_PATH", tmp_path / "test_prediction_log.jsonl")
    yield


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_open_mode_by_default(client, monkeypatch):
    """No CNC_API_KEY set anywhere in this test suite's environment -
    every endpoint should work with no header, exactly as before auth
    existed. This is the most important test here: it's what guarantees
    every other test in this project (written before auth existed)
    doesn't need to change."""
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["auth_enabled"] is False

    r = client.post("/predict", json={"Vc": 10.0, "Fz": 0.1, "ap": 1.0})
    assert r.status_code == 200


def test_health_never_requires_a_key(client, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "secret123")
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["auth_enabled"] is True


def test_protected_endpoint_rejects_missing_key(client, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "secret123")
    r = client.post("/predict", json={"Vc": 10.0, "Fz": 0.1, "ap": 1.0})
    assert r.status_code == 401


def test_protected_endpoint_rejects_wrong_key(client, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "secret123")
    r = client.post(
        "/predict",
        json={"Vc": 10.0, "Fz": 0.1, "ap": 1.0},
        headers={"X-API-Key": "wrong-key"},
    )
    assert r.status_code == 401


def test_protected_endpoint_accepts_correct_key(client, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "secret123")
    r = client.post(
        "/predict",
        json={"Vc": 10.0, "Fz": 0.1, "ap": 1.0},
        headers={"X-API-Key": "secret123"},
    )
    assert r.status_code == 200


@pytest.mark.parametrize("method,path,body", [
    ("get", "/model/info", None),
    ("get", "/model/feature-importance", None),
    ("post", "/predict", {"Vc": 10.0, "Fz": 0.1, "ap": 1.0}),
    ("post", "/predict/batch", {"items": [{"Vc": 10.0, "Fz": 0.1, "ap": 1.0}]}),
    ("get", "/drift/report", None),
    ("get", "/retrain/history", None),
])
def test_all_data_endpoints_are_protected(client, monkeypatch, method, path, body):
    monkeypatch.setenv(API_KEY_ENV_VAR, "secret123")
    call = getattr(client, method)
    r = call(path, json=body) if body is not None else call(path)
    assert r.status_code == 401, f"{method.upper()} {path} should require a key but returned {r.status_code}"
