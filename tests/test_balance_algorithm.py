from __future__ import annotations

import numpy as np

from valuelens.core.balance import (
    distribution_from_hist,
    levels_to_wgb,
    optimize_balance_params,
)


def test_levels_to_wgb_supports_common_levels() -> None:
    for levels in (2, 3, 5, 8):
        values = [100.0 / levels] * levels
        wgb = levels_to_wgb(values)
        assert len(wgb) == 3
        assert abs(sum(wgb) - 100.0) < 1e-6


def test_distribution_from_hist_returns_normalized_ratio() -> None:
    hist = np.zeros(256, dtype=np.float64)
    hist[32:220] = 1.0
    dist = distribution_from_hist(hist, lower=8, upper=240, levels=5, exp_value=0.1, mode=0)
    assert dist.shape == (3,)
    assert np.isclose(float(dist.sum()), 1.0)
    assert np.all(dist >= 0.0)


def test_auto_balance_optimization_improves_loss() -> None:
    gray = np.tile(np.arange(256, dtype=np.uint8), (128, 1))
    target = (70.0, 20.0, 10.0)
    levels = 5
    current = (0, 255, 0.0, 0)

    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    target_ratios = np.array(target, dtype=np.float64) / sum(target)
    baseline_dist = distribution_from_hist(hist, current[0], current[1], levels, current[2], current[3])
    baseline_loss = float(np.sum((baseline_dist - target_ratios) ** 2))

    best = optimize_balance_params(
        gray,
        target=target,
        current_min=current[0],
        current_max=current[1],
        current_exp=current[2],
        levels_count=levels,
        hysteresis=0.0,
        current_mode=current[3],
        search_all_modes=True,
    )
    best_dist = distribution_from_hist(hist, best[0], best[1], levels, best[2], best[3])
    best_loss = float(np.sum((best_dist - target_ratios) ** 2))

    assert best[1] > best[0]
    assert best_loss <= baseline_loss + 1e-12
