from __future__ import annotations

import numpy as np

from valuelens.core.quantize import get_quantization_lut, quantize_gray_with_indices


def _sample_frame(seed: int = 7, shape: tuple[int, int] = (240, 320)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(shape[0], shape[1], 3), dtype=np.uint8)


def test_quantize_is_deterministic_for_same_input() -> None:
    frame = _sample_frame()
    out_a, idx_a, edges_a, counts_a = quantize_gray_with_indices(
        frame,
        levels=5,
        min_value=12,
        max_value=230,
        exp_value=0.15,
        blur_radius=0,
        dither_strength=0,
        edge_strength=0,
    )
    out_b, idx_b, edges_b, counts_b = quantize_gray_with_indices(
        frame,
        levels=5,
        min_value=12,
        max_value=230,
        exp_value=0.15,
        blur_radius=0,
        dither_strength=0,
        edge_strength=0,
    )
    assert np.array_equal(out_a, out_b)
    assert np.array_equal(idx_a, idx_b)
    assert edges_a is None and edges_b is None
    assert np.array_equal(counts_a, counts_b)


def test_quantize_counts_align_with_levels() -> None:
    frame = _sample_frame(seed=9, shape=(180, 260))
    _, _, _, counts = quantize_gray_with_indices(frame, levels=8, min_value=0, max_value=255, exp_value=0.0)
    assert counts.shape == (8,)
    assert int(counts.sum()) > 0


def test_manual_lut_is_monotonic() -> None:
    lut = get_quantization_lut(levels=8, min_val=10, max_val=240, exp_val=0.2, curve_mode=0)
    assert lut.shape == (256,)
    assert np.all(np.diff(lut.astype(np.int32)) >= 0)
