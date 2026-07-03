#!/usr/bin/env python3
"""
Train the surface-roughness ensemble and persist it for the API to serve.

Examples
--------
  # Fast default: ~25/25/40 Optuna trials, poly_degree capped at 3.
  python scripts/train_model.py

  # Faithful to the original prototype's search space (60/60/80 trials,
  # poly_degree up to 4). Meaningfully slower - see the note in
  # src/train/train.py about degree=4's cost on this dataset.
  python scripts/train_model.py --preset full

  # Fine-grained overrides on top of a preset:
  python scripts/train_model.py --preset quick --svr-trials 40
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import main as run_training
from src.train.train import PRESETS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=list(PRESETS), default="quick", help="Trial-count/search-space preset (default: quick)")
    parser.add_argument("--svr-trials", type=int, default=None, help="Override SVR Optuna trial count")
    parser.add_argument("--gpr-trials", type=int, default=None, help="Override GPR Optuna trial count")
    parser.add_argument("--ensemble-trials", type=int, default=None, help="Override ensemble-weight Optuna trial count")
    parser.add_argument("--max-poly-degree", type=int, default=None, choices=[2, 3, 4], help="Override max PolynomialFeatures degree searched for SVR")
    parser.add_argument("--n-splits", type=int, default=None, help="Override CV fold count (default: 5)")
    parser.add_argument("--random-state", type=int, default=None, help="Override random seed (default: 151101)")
    parser.add_argument("--no-pruning", action="store_true", help="Disable Optuna median pruning of unpromising trials")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating report plots")
    parser.add_argument("--no-save", action="store_true", help="Don't persist the model bundle (dry run / metrics only)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logging")
    return parser.parse_args()


def main():
    args = parse_args()

    result = run_training(
        preset=args.preset,
        svr_trials=args.svr_trials,
        gpr_trials=args.gpr_trials,
        ensemble_trials=args.ensemble_trials,
        max_poly_degree=args.max_poly_degree,
        n_splits=args.n_splits,
        random_state=args.random_state,
        use_pruning=not args.no_pruning,
        save=not args.no_save,
        make_plots=not args.no_plots,
        verbose=not args.quiet,
    )

    print("\n=== Summary ===")
    print(f"Recommended model : {result['recommended_model']}")
    print(f"Weighted RMSE     : {result['metrics']['Weighted_Ensemble']['RMSE']:.4f}")
    print(f"Weighted R2       : {result['metrics']['Weighted_Ensemble']['R2']:.4f}")
    print(f"Stacking RMSE     : {result['metrics']['Stacking_Ensemble']['RMSE']:.4f}")
    print(f"Stacking R2       : {result['metrics']['Stacking_Ensemble']['R2']:.4f}")
    if "bundle_path" in result:
        print(f"Model bundle      : {result['bundle_path']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
