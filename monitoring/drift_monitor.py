#!/usr/bin/env python3
"""
Data drift monitor for the CNC surface-roughness model.

The training data is a narrow Design-of-Experiments grid (Vc, Fz, and ap
each only take ~4-5 discrete values across 119 rows). Production readings
won't land on that grid - and as tool wear progresses, operators
compensate by changing speeds/feeds, so the real operating envelope
gradually walks away from what the model was trained on. This script
flags that before it silently degrades prediction quality.

For each raw machining parameter (Vc, Fz, ap) it compares a "current"
batch of readings against the training baseline using:

  - PSI (Population Stability Index): the standard drift metric. Below
    0.1 is considered stable, 0.1-0.25 a moderate shift worth watching,
    0.25+ a significant shift.
  - Kolmogorov-Smirnov two-sample test: are the two samples plausibly
    drawn from the same distribution (p < 0.05 says no).
  - Mean shift in baseline standard deviations, and % of samples falling
    entirely outside the min/max range the model was trained on.

Usage
-----
  # Compare against everything the API has logged so far
  python -m monitoring.drift_monitor --from-log

  # Compare against a CSV of new production readings (needs Vc, Fz, ap columns)
  python -m monitoring.drift_monitor --input production_batch.csv

  # No production data yet? Generate a synthetic "tool wear" batch to see
  # the monitor actually catch something.
  python -m monitoring.drift_monitor --simulate-drift

Exit codes: 0 = stable, 1 = warning, 2 = drift detected. That makes this
usable directly as a scheduled/CI job: a non-zero exit can gate or
trigger a retraining pipeline.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DRIFT_REPORT_DIR
from src.models.persistence import load_baseline_data
from monitoring.request_log import read_prediction_log

DEFAULT_FEATURES = ["Vc", "Fz", "ap"]

PSI_WARN_THRESHOLD = 0.1
PSI_DRIFT_THRESHOLD = 0.25
KS_ALPHA = 0.05

# This dataset's raw features are a narrow DOE grid (Vc/Fz/ap each take
# only ~4-5 discrete values). Empirically, PSI on a current-batch smaller
# than ~30 samples is unreliable here - a batch of 11 samples resampled
# from the *exact same* grid (zero real drift) produced a PSI of 1.4 on
# Vc, purely from a few bins randomly landing at 0 count. The KS test
# stayed correctly non-significant (p=0.57) on the same data. 30 isn't a
# magic number that eliminates the effect, just a threshold below which
# it gets dramatic rather than occasional - see LOW_SAMPLE_WARN below.
MIN_SAMPLES = 30

# Below this size, PSI can still occasionally throw a borderline WARNING
# on genuinely non-drifted data (observed: PSI=0.13 on Fz at n=40, just
# over the 0.1 line). Surface that as a caveat on the report rather than
# silently presenting PSI as equally trustworthy at every sample size.
LOW_SAMPLE_CAVEAT_THRESHOLD = 50


def compute_psi(baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index between two 1-D samples, binned using the
    baseline's quantiles. Falls back to the baseline's unique values as
    bin edges when it has too few distinct values for quantile binning to
    produce more than one or two bins - relevant here since this dataset's
    raw features only take a handful of discrete values.
    """

    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)

    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(baseline, quantiles))

    if len(edges) < 3:
        uniques = np.unique(baseline)
        if len(uniques) <= 1:
            return 0.0
        midpoints = uniques[:-1] + np.diff(uniques) / 2
        edges = np.concatenate([[-np.inf], midpoints, [np.inf]])
    else:
        edges = edges.copy()
        edges[0] = -np.inf
        edges[-1] = np.inf

    baseline_counts, _ = np.histogram(baseline, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    baseline_pct = baseline_counts / max(len(baseline), 1)
    current_pct = current_counts / max(len(current), 1)

    epsilon = 1e-4
    baseline_pct = np.where(baseline_pct == 0, epsilon, baseline_pct)
    current_pct = np.where(current_pct == 0, epsilon, current_pct)

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)


def classify_psi(psi: float) -> str:
    if psi >= PSI_DRIFT_THRESHOLD:
        return "significant_shift"
    if psi >= PSI_WARN_THRESHOLD:
        return "moderate_shift"
    return "stable"


def analyze_feature_drift(baseline: pd.Series, current: pd.Series) -> dict:
    baseline_vals = baseline.dropna().to_numpy(dtype=float)
    current_vals = current.dropna().to_numpy(dtype=float)

    psi = compute_psi(baseline_vals, current_vals)
    ks_stat, ks_pvalue = ks_2samp(baseline_vals, current_vals)

    baseline_mean, baseline_std = baseline_vals.mean(), baseline_vals.std()
    current_mean = current_vals.mean()
    mean_shift_in_std = (
        (current_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0.0
    )

    lo, hi = baseline_vals.min(), baseline_vals.max()
    pct_out_of_range = float(np.mean((current_vals < lo) | (current_vals > hi)) * 100)

    # A feature only counts toward the severe DRIFT_DETECTED verdict if the
    # KS test agrees the distributions actually differ, AND there's a
    # corroborating signal (moderate-or-higher PSI, or a real chunk of
    # samples outside the trained range). KS alone as the gate matters:
    # on this dataset's narrow discrete grid, PSI alone spiked to 1.4 on a
    # batch that was pure resampling with zero real drift (KS p=0.57 on
    # that same batch correctly said "not significant"). Requiring KS
    # agreement is what keeps that false positive from reaching the
    # "schedule retraining" verdict. WARNING-level psi_verdict below stays
    # PSI-only on purpose - it's meant to be a cheap, sensitive, low-stakes
    # early signal, unlike DRIFT_DETECTED which implies real action.
    is_drifted = bool(
        ks_pvalue < KS_ALPHA and (psi >= PSI_WARN_THRESHOLD or pct_out_of_range >= 15)
    )

    return {
        "psi": round(psi, 4),
        "psi_verdict": classify_psi(psi),
        "ks_statistic": round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_pvalue), 4),
        "ks_significant": bool(ks_pvalue < KS_ALPHA),
        "baseline_mean": round(float(baseline_mean), 4),
        "current_mean": round(float(current_mean), 4),
        "mean_shift_in_std": round(float(mean_shift_in_std), 4),
        "baseline_range": [round(float(lo), 4), round(float(hi), 4)],
        "pct_samples_out_of_range": round(pct_out_of_range, 2),
        "drifted": bool(is_drifted),
    }


def analyze_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame, features=None) -> dict:
    features = features or DEFAULT_FEATURES

    per_feature = {}
    for feature in features:
        if feature not in baseline_df.columns or feature not in current_df.columns:
            continue
        per_feature[feature] = analyze_feature_drift(baseline_df[feature], current_df[feature])

    any_drift = any(f["drifted"] for f in per_feature.values())
    any_warning = any(f["psi_verdict"] != "stable" for f in per_feature.values())

    if any_drift:
        overall = "DRIFT_DETECTED"
    elif any_warning:
        overall = "WARNING"
    else:
        overall = "STABLE"

    low_sample_caveat = len(current_df) < LOW_SAMPLE_CAVEAT_THRESHOLD

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_baseline_samples": int(len(baseline_df)),
        "n_current_samples": int(len(current_df)),
        "overall_verdict": overall,
        "low_sample_size_caveat": low_sample_caveat,
        "low_sample_size_note": (
            f"Current batch is {len(current_df)} samples, below the "
            f"{LOW_SAMPLE_CAVEAT_THRESHOLD}-sample comfort zone for PSI on "
            "this dataset's narrow discrete-value grid. The psi_verdict "
            "field below (used for the mild WARNING tier) is PSI-only and "
            "can read 'moderate_shift' from sampling noise alone at low n. "
            "The 'drifted' field (used for the severe DRIFT_DETECTED tier) "
            "requires the KS test to also agree, which is more robust at "
            "low sample sizes - trust that one first."
            if low_sample_caveat
            else None
        ),
        "features": per_feature,
    }


def simulate_drifted_batch(baseline_df: pd.DataFrame, n_samples: int = 40, random_state: int = 151101) -> pd.DataFrame:
    """
    Build a synthetic "current" batch that mimics tool wear: cutting speed
    creeps up as an operator compensates, and readings spread continuously
    around the original DOE grid points instead of landing exactly on
    them (as they would from a real running machine, unlike the tight
    experimental grid the model was trained on).
    """

    rng = np.random.default_rng(random_state)
    sample = baseline_df.sample(n=n_samples, replace=True, random_state=random_state).reset_index(drop=True)

    drifted = pd.DataFrame({
        "Vc": sample["Vc"] * 1.35 + rng.normal(0, sample["Vc"].std() * 0.4, n_samples),
        "Fz": sample["Fz"] + rng.normal(0, sample["Fz"].std() * 0.5, n_samples),
        "ap": sample["ap"] * 1.15 + rng.normal(0, sample["ap"].std() * 0.5, n_samples),
    })

    drifted["Vc"] = drifted["Vc"].clip(lower=0.1)
    drifted["Fz"] = drifted["Fz"].clip(lower=0.01)
    drifted["ap"] = drifted["ap"].clip(lower=0.1)

    return drifted


def format_report(analysis: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("DATA DRIFT REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated:        {analysis['generated_at']}")
    lines.append(f"Baseline samples: {analysis['n_baseline_samples']}")
    lines.append(f"Current samples:  {analysis['n_current_samples']}")
    lines.append(f"Overall verdict:  {analysis['overall_verdict']}")
    if analysis.get("low_sample_size_caveat"):
        lines.append(f"NOTE: {analysis['low_sample_size_note']}")
    lines.append("-" * 60)

    for feature, result in analysis["features"].items():
        flag = "DRIFT" if result["drifted"] else result["psi_verdict"].upper()
        lines.append(f"\n{feature}  [{flag}]")
        lines.append(f"  PSI                 : {result['psi']}  ({result['psi_verdict']})")
        lines.append(f"  KS test             : stat={result['ks_statistic']}, p={result['ks_pvalue']}"
                      f"  {'(significant)' if result['ks_significant'] else '(not significant)'}")
        lines.append(f"  mean shift          : {result['mean_shift_in_std']} baseline std-devs"
                      f"  ({result['baseline_mean']} -> {result['current_mean']})")
        lines.append(f"  trained range       : {result['baseline_range']}")
        lines.append(f"  % outside that range: {result['pct_samples_out_of_range']}%")

    lines.append("\n" + "=" * 60)
    if analysis["overall_verdict"] == "DRIFT_DETECTED":
        lines.append("ACTION: distribution has shifted meaningfully - schedule retraining.")
    elif analysis["overall_verdict"] == "WARNING":
        lines.append("ACTION: early signs of shift - keep watching, not yet urgent.")
    else:
        lines.append("ACTION: none - incoming data still matches the training distribution.")
    lines.append("=" * 60)

    return "\n".join(lines)


def save_report(analysis: dict) -> Path:
    DRIFT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = DRIFT_REPORT_DIR / f"drift_report_{ts}.json"
    with open(path, "w") as f:
        json.dump(analysis, f, indent=2)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=str, help="CSV of new production readings (needs Vc, Fz, ap columns)")
    source.add_argument("--from-log", action="store_true", help="Use the API's accumulated prediction log (default)")
    source.add_argument("--simulate-drift", action="store_true", help="Generate a synthetic drifted batch to demo the monitor")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES, help=f"Minimum current-batch samples required to report (default: {MIN_SAMPLES})")
    parser.add_argument("--n-simulated", type=int, default=40, help="Sample count for --simulate-drift (default: 40)")
    parser.add_argument("--quiet", action="store_true", help="Only print the JSON report, not the formatted text version")
    return parser.parse_args()


def main():
    args = parse_args()

    baseline_df = load_baseline_data()

    if args.input:
        current_df = pd.read_csv(args.input)
        source_desc = f"CSV: {args.input}"
    elif args.simulate_drift:
        current_df = simulate_drifted_batch(baseline_df, n_samples=args.n_simulated)
        source_desc = f"simulated tool-wear batch (n={args.n_simulated})"
    else:
        current_df = read_prediction_log()
        source_desc = "API prediction log"

    if len(current_df) < args.min_samples:
        print(
            f"Only {len(current_df)} samples available from {source_desc} "
            f"(need at least {args.min_samples}). Not enough traffic yet to "
            "report on - this is expected for a freshly-deployed service, "
            "not an error."
        )
        return 0

    analysis = analyze_drift(baseline_df, current_df)
    report_path = save_report(analysis)

    if not args.quiet:
        print(format_report(analysis))
        print(f"\nSource: {source_desc}")
        print(f"Report saved: {report_path}")
    else:
        print(json.dumps(analysis, indent=2))

    if analysis["overall_verdict"] == "DRIFT_DETECTED":
        return 2
    if analysis["overall_verdict"] == "WARNING":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
