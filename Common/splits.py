"""Reproducible speaker-level outer train/test splitting.

The split is stratified within language so that every sufficiently sized
language contributes speakers to both partitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_speaker_split(stats, test_size=0.20, random_state=20260822):
    """Deterministic approximately-80/20 speaker holdout within every language."""
    rng = np.random.default_rng(random_state)
    train_parts, test_parts = [], []
    for lang, g in stats.groupby("language_id", sort=True):
        g = g.reset_index(drop=True)
        order = rng.permutation(len(g))
        n_test = max(1, int(round(test_size * len(g))))
        if n_test >= len(g):
            n_test = len(g) - 1
        test_parts.append(g.iloc[order[:n_test]])
        train_parts.append(g.iloc[order[n_test:]])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return train, test
