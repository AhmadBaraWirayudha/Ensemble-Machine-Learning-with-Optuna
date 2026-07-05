"""
Tests for the CV-setup helpers in src/tuning/optuna_tuning.py (this file
previously just asserted True as a placeholder).
"""

import numpy as np

from src.data_loader import load_dataset
from src.optuna_tuning import create_stratified_bins, create_replicate_groups


def test_create_stratified_bins_returns_one_bin_per_row():
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    bins = create_stratified_bins(y, n_bins=5)
    assert len(bins) == len(y)


def test_create_replicate_groups_matches_known_dataset_structure():
    # This dataset is a complete 5x5x4 factorial DOE (100 unique Vc/Fz/ap
    # combinations) where 19 of those 100 were measured twice - see
    # UPGRADE_NOTES.md. This test pins that down so a future data update
    # that changes the replicate structure doesn't silently go unnoticed.
    df = load_dataset()
    groups = create_replicate_groups(df)

    assert len(groups) == len(df)

    n_unique_groups = len(np.unique(groups))
    assert n_unique_groups == 100

    group_sizes = np.bincount(groups)
    n_replicated_groups = int((group_sizes == 2).sum())
    assert n_replicated_groups == 19


def test_create_replicate_groups_gives_identical_rows_the_same_group():
    df = load_dataset()
    groups = create_replicate_groups(df)

    # any two rows with identical (Vc, Fz, ap) must share a group id, and
    # any two rows that differ in at least one of them must not
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            same_params = (
                df.iloc[i]["Vc"] == df.iloc[j]["Vc"]
                and df.iloc[i]["Fz"] == df.iloc[j]["Fz"]
                and df.iloc[i]["ap"] == df.iloc[j]["ap"]
            )
            assert (groups[i] == groups[j]) == same_params
