from __future__ import annotations

import numpy as np

from valuelens.core.balance import calc_level_distribution, levels_to_wgb, optimize_balance_params
from valuelens.core.quantize import quantize_gray_with_indices


def test_manual_then_auto_balance_flow_runs() -> None:
    rng = np.random.default_rng(100)
    frame = rng.integers(0, 256, size=(200, 300, 3), dtype=np.uint8)

    # Manual path: explicit min/max/exp with selected curve mode.
    _, indices, _, _ = quantize_gray_with_indices(
        frame,
        levels=5,
        min_value=20,
        max_value=220,
        exp_value=0.25,
        curve_mode=1,
    )
    assert indices.shape == frame.shape[:2]

    gray = frame[:, :, 0]
    raw_dist = calc_level_distribution(gray, levels=5)
    target_wgb = levels_to_wgb(raw_dist)

    # Auto path: optimize from reset params.
    best = optimize_balance_params(
        gray=gray,
        target=target_wgb,
        current_min=0,
        current_max=255,
        current_exp=0.0,
        levels_count=5,
        hysteresis=0.0,
        current_mode=0,
        search_all_modes=True,
    )

    assert isinstance(best, tuple) and len(best) == 4
    assert 0 <= best[0] < best[1] <= 255
