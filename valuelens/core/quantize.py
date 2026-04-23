from __future__ import annotations

import numpy as np
import cv2


def quantize_gray_with_indices(
    bgr: np.ndarray,
    levels: int,
    min_value: int,
    max_value: int,
    exp_value: float = 0.0,
    fixed_output_levels: bool = True,
    blur_radius: int = 0,
    dither_strength: int = 0,
    dither_first: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    levels = max(2, int(levels))
    min_value = max(0, min(255, int(min_value)))
    max_value = max(min_value + 1, min(255, int(max_value)))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype(np.float32)

    if dither_first:
        if dither_strength > 0:
            noise = np.random.uniform(-dither_strength, dither_strength, gray_float.shape).astype(np.float32)
            gray_float += noise
        if blur_radius > 0:
            ksize = blur_radius * 2 + 1
            gray_float = cv2.GaussianBlur(gray_float, (ksize, ksize), 0)
    else:
        if blur_radius > 0:
            ksize = blur_radius * 2 + 1
            gray_float = cv2.GaussianBlur(gray_float, (ksize, ksize), 0)
        if dither_strength > 0:
            noise = np.random.uniform(-dither_strength, dither_strength, gray_float.shape).astype(np.float32)
            gray_float += noise

    clipped = np.clip(gray_float, min_value, max_value)

    span = float(max_value - min_value)
    normalized = (clipped - min_value) / span
    gamma = float(np.exp(float(exp_value)))
    normalized = np.power(normalized, gamma)
    indices = np.floor(normalized * levels)
    indices = np.clip(indices, 0, levels - 1).astype(np.int32)
    if fixed_output_levels:
        # Fixed anchors in global grayscale space.
        quantized = (indices / (levels - 1)) * 255.0
    else:
        # Range-mapped anchors: min lifts black point, max lowers white point.
        quantized = (indices / (levels - 1)) * span + min_value
    out = np.clip(quantized, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR), indices


def quantize_gray(
    bgr: np.ndarray,
    levels: int,
    min_value: int,
    max_value: int,
    exp_value: float = 0.0,
    fixed_output_levels: bool = True,
    blur_radius: int = 0,
    dither_strength: int = 0,
    dither_first: bool = False,
) -> np.ndarray:
    out, _ = quantize_gray_with_indices(
        bgr,
        levels,
        min_value,
        max_value,
        exp_value,
        fixed_output_levels=fixed_output_levels,
        blur_radius=blur_radius,
        dither_strength=dither_strength,
        dither_first=dither_first,
    )
    return out

