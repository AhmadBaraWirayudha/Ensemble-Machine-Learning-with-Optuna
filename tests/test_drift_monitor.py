import numpy as np
import pandas as pd

from monitoring.drift_monitor import (
    compute_psi,
    classify_psi,
    analyze_feature_drift,
    analyze_drift,
    simulate_drifted_batch,
    PSI_WARN_THRESHOLD,
    PSI_DRIFT_THRESHOLD,
)
from src.models.persistence import load_baseline_data


def test_compute_psi_identical_distributions_is_near_zero():
    rng = np.random.default_rng(0)
    baseline = rng.normal(10, 2, 500)
    current = baseline.copy()
    rng.shuffle(current)

    psi = compute_psi(baseline, current)
    assert psi < 0.01


def test_compute_psi_shifted_distribution_is_large():
    rng = np.random.default_rng(0)
    baseline = rng.normal(10, 2, 500)
    current = rng.normal(20, 2, 500)  # 5 std devs away

    psi = compute_psi(baseline, current)
    assert psi > PSI_DRIFT_THRESHOLD


def test_compute_psi_handles_narrow_discrete_baseline():
    # mirrors this project's actual data: a handful of discrete levels
    baseline = np.array([7.5, 10, 12.5, 15, 17.5] * 24)
    current = np.array([7.5, 10, 12.5, 15, 17.5] * 8)

    psi = compute_psi(baseline, current)
    assert np.isfinite(psi)
    assert psi >= 0


def test_classify_psi_thresholds():
    assert classify_psi(0.0) == "stable"
    assert classify_psi(PSI_WARN_THRESHOLD) == "moderate_shift"
    assert classify_psi(PSI_DRIFT_THRESHOLD) == "significant_shift"


def test_analyze_feature_drift_structure():
    rng = np.random.default_rng(0)
    baseline = pd.Series(rng.normal(10, 2, 200))
    current = pd.Series(rng.normal(10, 2, 200))

    result = analyze_feature_drift(baseline, current)
    for key in ["psi", "psi_verdict", "ks_statistic", "ks_pvalue", "drifted", "baseline_range"]:
        assert key in result


def test_analyze_drift_stable_case():
    rng = np.random.default_rng(1)
    baseline_df = pd.DataFrame({"Vc": rng.normal(10, 2, 300), "Fz": rng.normal(0.1, 0.02, 300)})
    current_df = pd.DataFrame({"Vc": rng.normal(10, 2, 100), "Fz": rng.normal(0.1, 0.02, 100)})

    analysis = analyze_drift(baseline_df, current_df, features=["Vc", "Fz"])
    assert analysis["overall_verdict"] in ("STABLE", "WARNING")  # not the severe tier
    assert analysis["n_baseline_samples"] == 300
    assert analysis["n_current_samples"] == 100


def test_analyze_drift_detects_real_shift():
    rng = np.random.default_rng(1)
    baseline_df = pd.DataFrame({"Vc": rng.normal(10, 1, 300)})
    current_df = pd.DataFrame({"Vc": rng.normal(25, 1, 100)})  # 15 std devs away - unmistakable

    analysis = analyze_drift(baseline_df, current_df, features=["Vc"])
    assert analysis["overall_verdict"] == "DRIFT_DETECTED"
    assert analysis["features"]["Vc"]["drifted"] is True


def test_simulate_drifted_batch_shape_and_ranges():
    baseline = load_baseline_data()
    drifted = simulate_drifted_batch(baseline, n_samples=25)

    assert len(drifted) == 25
    assert set(["Vc", "Fz", "ap"]).issubset(drifted.columns)
    assert (drifted["Vc"] > 0).all()
    assert (drifted["Fz"] > 0).all()
    assert (drifted["ap"] > 0).all()
    # the whole point of this function is that it shifts Vc up
    assert drifted["Vc"].mean() > baseline["Vc"].mean()
