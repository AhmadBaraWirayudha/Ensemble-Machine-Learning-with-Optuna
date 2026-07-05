import json
import sys

import pytest

import monitoring.retrain_trigger as rt


@pytest.fixture()
def isolated_model_dir(tmp_path, monkeypatch):
    """Redirect every path retrain_trigger touches into tmp_path, so tests
    never read/write the real trained model."""

    saved_model_dir = tmp_path / "saved_models"
    archive_dir = saved_model_dir / "archive"
    saved_model_dir.mkdir(parents=True)

    monkeypatch.setattr(rt, "SAVED_MODEL_DIR", saved_model_dir)
    monkeypatch.setattr(rt, "MODEL_ARCHIVE_DIR", archive_dir)
    monkeypatch.setattr(rt, "MODEL_BUNDLE_PATH", saved_model_dir / "model_bundle.joblib")
    monkeypatch.setattr(rt, "RETRAIN_LOG_PATH", tmp_path / "retrain_log.jsonl")

    return saved_model_dir


def _write_fake_bundle_files(saved_model_dir, tag):
    """Write placeholder files standing in for a model bundle - content
    doesn't matter for backup/restore tests, only that the files exist and
    are distinguishable."""
    (saved_model_dir / "model_bundle.joblib").write_text(f"bundle-{tag}")
    (saved_model_dir / "model_metadata.json").write_text(json.dumps({"tag": tag}))
    (saved_model_dir / "training_baseline.csv").write_text(f"Vc,Fz,ap,Ra\n{tag}\n")
    # deliberately omit feature_importance.json for one case - backup/restore
    # should tolerate a missing optional sidecar file


def test_backup_returns_none_when_no_existing_model(isolated_model_dir):
    assert rt.backup_current_model() is None


def test_backup_and_restore_round_trip(isolated_model_dir):
    _write_fake_bundle_files(isolated_model_dir, "v1")

    archive_path = rt.backup_current_model()
    assert archive_path is not None
    assert (archive_path / "model_bundle.joblib").read_text() == "bundle-v1"
    assert not (archive_path / "feature_importance.json").exists()

    # simulate a retrain overwriting the live files with a new version
    _write_fake_bundle_files(isolated_model_dir, "v2")
    assert (isolated_model_dir / "model_bundle.joblib").read_text() == "bundle-v2"

    # roll back
    rt.restore_from_backup(archive_path)
    assert (isolated_model_dir / "model_bundle.joblib").read_text() == "bundle-v1"


def test_get_recommended_rmse():
    metadata = {"recommended_model": "Stacking_Ensemble", "metrics": {"Stacking_Ensemble": {"RMSE": 0.31}, "SVR": {"RMSE": 0.40}}}
    assert rt.get_recommended_rmse(metadata) == 0.31


def test_log_retrain_event_appends_jsonl(isolated_model_dir, tmp_path):
    log_path = tmp_path / "retrain_log.jsonl"
    rt.log_retrain_event({"action": "skipped", "reason": "no_drift"}, log_path=log_path)
    rt.log_retrain_event({"action": "promoted"}, log_path=log_path)

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["action"] == "skipped"
    assert json.loads(lines[1])["action"] == "promoted"
    assert "timestamp" in json.loads(lines[0])


@pytest.fixture()
def cli_environment(isolated_model_dir, tmp_path, monkeypatch):
    """Full environment for exercising rt.main(): isolated paths, a fake
    'old' model already in place, drift check and training both mocked out
    so tests run in milliseconds instead of minutes."""

    _write_fake_bundle_files(isolated_model_dir, "old")

    monkeypatch.setattr(rt, "load_metadata", lambda: {
        "recommended_model": "Stacking_Ensemble",
        "metrics": {"Stacking_Ensemble": {"RMSE": 0.30}},
    })
    monkeypatch.setattr(rt, "load_baseline_data", lambda: "fake-baseline-df")
    monkeypatch.setattr(rt, "read_prediction_log", lambda: list(range(35)))  # len() >= min_samples

    return tmp_path


def _run_cli(monkeypatch, args, drift_verdict="DRIFT_DETECTED", new_rmse=0.30):
    monkeypatch.setattr(sys, "argv", ["retrain_trigger.py"] + args)
    monkeypatch.setattr(rt, "analyze_drift", lambda baseline, current: {"overall_verdict": drift_verdict})
    monkeypatch.setattr(
        "src.train.main",
        lambda **kwargs: {"metrics": {"Stacking_Ensemble": {"RMSE": new_rmse}}, "recommended_model": "Stacking_Ensemble"},
    )
    return rt.main()


def test_skips_when_no_drift(cli_environment, monkeypatch):
    exit_code = _run_cli(monkeypatch, ["--quiet"], drift_verdict="STABLE")
    assert exit_code == 0

    log_lines = (cli_environment / "retrain_log.jsonl").read_text().strip().split("\n")
    assert json.loads(log_lines[-1])["action"] == "skipped"
    # model file should be untouched (still says "old")
    assert (cli_environment / "saved_models" / "model_bundle.joblib").read_text() == "bundle-old"


def test_promotes_improved_model(cli_environment, monkeypatch, isolated_model_dir):
    # new RMSE (0.20) is better than old (0.30)
    exit_code = _run_cli(monkeypatch, ["--quiet"], drift_verdict="DRIFT_DETECTED", new_rmse=0.20)
    assert exit_code == 0

    log_lines = (cli_environment / "retrain_log.jsonl").read_text().strip().split("\n")
    last_event = json.loads(log_lines[-1])
    assert last_event["action"] == "promoted"
    assert last_event["new_rmse"] == 0.20

    # a backup of the old model should exist
    archive_dirs = list((isolated_model_dir / "archive").iterdir())
    assert len(archive_dirs) == 1
    assert (archive_dirs[0] / "model_bundle.joblib").read_text() == "bundle-old"


def test_rolls_back_regressed_model(cli_environment, monkeypatch, isolated_model_dir):
    # new RMSE (0.50) is 66% worse than old (0.30) - well past the 15% default threshold
    exit_code = _run_cli(monkeypatch, ["--quiet"], drift_verdict="DRIFT_DETECTED", new_rmse=0.50)
    assert exit_code == 1

    log_lines = (cli_environment / "retrain_log.jsonl").read_text().strip().split("\n")
    last_event = json.loads(log_lines[-1])
    assert last_event["action"] == "rolled_back"

    # NOTE: src.train.main is mocked, so it never actually wrote new files -
    # this confirms restore_from_backup ran without erroring, not that it
    # overwrote a "new" file with the old one (there's nothing else to
    # verify that against here since the mock doesn't touch disk).
    assert (isolated_model_dir / "model_bundle.joblib").read_text() == "bundle-old"


def test_dry_run_does_not_retrain_or_modify_files(cli_environment, monkeypatch, isolated_model_dir):
    exit_code = _run_cli(monkeypatch, ["--dry-run", "--quiet"], drift_verdict="DRIFT_DETECTED")
    assert exit_code == 0

    log_lines = (cli_environment / "retrain_log.jsonl").read_text().strip().split("\n")
    assert json.loads(log_lines[-1])["action"] == "dry_run"
    assert (isolated_model_dir / "model_bundle.joblib").read_text() == "bundle-old"
    archive_dir = isolated_model_dir / "archive"
    assert not archive_dir.exists() or list(archive_dir.iterdir()) == []
