from __future__ import annotations

import numpy as np
import cv2

import pytest

from valuelens.core.quantize import (
    apply_ordered_dither,
    get_color_lut_bgr,
    get_quantization_lut,
    native_color_map_rgb,
    quantize_gray_with_indices,
)


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


def test_profile_collection_does_not_change_output() -> None:
    frame = _sample_frame(seed=11)
    baseline = quantize_gray_with_indices(frame, levels=5, min_value=8, max_value=240, exp_value=0.1)

    profile: dict[str, float] = {}
    profiled = quantize_gray_with_indices(
        frame,
        levels=5,
        min_value=8,
        max_value=240,
        exp_value=0.1,
        profile=profile,
    )

    assert np.array_equal(baseline[0], profiled[0])
    assert np.array_equal(baseline[1], profiled[1])
    assert np.array_equal(baseline[3], profiled[3])
    assert profile["total_ms"] >= 0.0
    assert "filters_ms" in profile


def test_color_lut_cache_is_stable() -> None:
    palette = [(0, 0, 0), (120, 80, 40), (255, 255, 255)]
    lut_a = get_color_lut_bgr(3, 0, 255, 0.0, palette=palette, edge_mix=25, edge_active=True)
    lut_b = get_color_lut_bgr(3, 0, 255, 0.0, palette=palette, edge_mix=25, edge_active=True)
    assert lut_a is lut_b
    assert np.array_equal(lut_a, lut_b)


def test_morph_mask_positions_are_black() -> None:
    frame = np.zeros((80, 96, 3), dtype=np.uint8)
    frame[:, 48:] = 255
    out, _, _, _ = quantize_gray_with_indices(
        frame,
        levels=5,
        min_value=0,
        max_value=255,
        exp_value=0.0,
        morph_enabled=True,
        morph_strength=1,
        morph_threshold=1,
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel) > 1
    assert np.any(mask)
    assert np.all(out[mask] == 0)


def test_native_color_map_matches_python_when_available() -> None:
    indices = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
    lut_bgr = get_color_lut_bgr(3, 0, 255, 0.0)
    native_rgb = native_color_map_rgb(indices, lut_bgr)
    if native_rgb is None:
        pytest.skip("native color_map_rgb is not available")

    python_rgb = cv2.cvtColor(lut_bgr[indices], cv2.COLOR_BGR2RGB)
    assert np.array_equal(native_rgb, python_rgb)


def test_dither_strength_scales_ordered_dither() -> None:
    gray = np.full((16, 16), 128, dtype=np.uint8)
    off = apply_ordered_dither(gray, 0)
    half = apply_ordered_dither(gray, 50)
    full = apply_ordered_dither(gray, 100)

    assert np.array_equal(off, gray)
    assert not np.array_equal(half, gray)
    assert not np.array_equal(full, half)

    half_delta = np.mean(np.abs(half.astype(np.int16) - gray.astype(np.int16)))
    full_delta = np.mean(np.abs(full.astype(np.int16) - gray.astype(np.int16)))
    assert full_delta > half_delta > 0


def test_filter_parameter_values_are_wired() -> None:
    frame = _sample_frame(seed=21, shape=(96, 128))
    weak_dither = quantize_gray_with_indices(frame, levels=5, min_value=8, max_value=240, dither_strength=10)
    strong_dither = quantize_gray_with_indices(frame, levels=5, min_value=8, max_value=240, dither_strength=100)
    assert not np.array_equal(weak_dither[1], strong_dither[1])

    no_edge_mix = quantize_gray_with_indices(frame, levels=5, edge_strength=60, edge_mix=0)
    full_edge_mix = quantize_gray_with_indices(frame, levels=5, edge_strength=60, edge_mix=100)
    assert not np.array_equal(no_edge_mix[0], full_edge_mix[0])
