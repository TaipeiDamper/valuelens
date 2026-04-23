from __future__ import annotations

import numpy as np
import cv2


_dither_cache = {}

def get_tiled_bayer(h: int, w: int) -> np.ndarray:
    key = (h, w)
    if key in _dither_cache:
        return _dither_cache[key]
    bayer = np.array([
        [ 0,  8,  2, 10],
        [12,  4, 14,  6],
        [ 3, 11,  1,  9],
        [15,  7, 13,  5]
    ], dtype=np.float32) / 16.0 - 0.5
    tiled_bayer = np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w]
    _dither_cache[key] = tiled_bayer
    return tiled_bayer

def quantize_gray_with_indices(
    bgr: np.ndarray,
    levels: int,
    min_value: int,
    max_value: int,
    exp_value: float = 0.0,
    display_min: int | None = None,
    display_max: int | None = None,
    display_exp: float | None = None,
    blur_radius: int = 0,
    dither_strength: int = 0,
    dither_first: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    levels = max(2, int(levels))
    min_value = max(0, min(255, int(min_value)))
    max_value = max(min_value + 1, min(255, int(max_value)))

    # Step 1: Grayscale (uint8 is much faster for Bilateral)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def apply_bilateral(img_u8):
        if blur_radius <= 0: return img_u8
        # Allow slightly larger diameter for stronger effect, but cap at 15 for performance
        d = min(15, max(5, blur_radius // 2))
        # Multiply sigma by 4 (from 2.0 to 8.0) for a 4x stronger color bleed/smoothing
        sig = blur_radius * 8.0
        return cv2.bilateralFilter(img_u8, d, sig, sig)

    def apply_ordered_dither(img_float):
        if dither_strength <= 0: return img_float
        h, w = img_float.shape[:2]
        tiled_bayer = get_tiled_bayer(h, w)
        step_size = 255.0 / (levels - 1)
        # Multiply dither amplitude by 4 (from 0.5 to 2.0)
        img_float += tiled_bayer * step_size * (dither_strength / 100.0) * 2.0
        return img_float

    # Pipeline
    if dither_first:
        gray_float = gray.astype(np.float32)
        gray_float = apply_ordered_dither(gray_float)
        # If dither first, we have to do bilateral on float32 (slower but mathematically correct here)
        if blur_radius > 0:
            d = min(9, blur_radius)
            sig = blur_radius * 2.0
            gray_float = cv2.bilateralFilter(gray_float, d, sig, sig)
    else:
        # Fast path: Bilateral on uint8, then convert to float for dither
        gray = apply_bilateral(gray)
        gray_float = gray.astype(np.float32)
        gray_float = apply_ordered_dither(gray_float)

    # --- Stage 1: Logic Mapping ---
    np.clip(gray_float, min_value, max_value, out=gray_float) # In-place clip
    span = float(max_value - min_value)
    if span == 0: span = 1.0 # Prevent division by zero
    
    # In-place normalization
    gray_float -= min_value
    gray_float /= span
    
    if exp_value != 0.0:
        gamma_logic = float(np.power(2.0, float(exp_value)))
        cv2.pow(gray_float, gamma_logic, gray_float) # cv2.pow is often faster than np.power
    
    gray_float *= levels
    indices = np.floor(gray_float)
    indices = np.clip(indices, 0, levels - 1).astype(np.int32)
    
    # --- Stage 2: Display Mapping (One-pass) ---
    norm_display = indices.astype(np.float32) / (levels - 1)
    
    if display_exp is not None and display_exp != 0:
        gamma_display = float(np.power(2.0, float(display_exp)))
        norm_display = np.power(norm_display, gamma_display)
        
    if display_min is not None and display_max is not None:
        d_min = float(display_min)
        d_max = float(display_max)
        out_float = norm_display * (d_max - d_min) + d_min
    else:
        out_float = norm_display * 255.0
        
    out = np.clip(out_float, 0, 255).astype(np.uint8)
    return out, indices


def quantize_gray(
    bgr: np.ndarray,
    levels: int,
    min_value: int,
    max_value: int,
    exp_value: float = 0.0,
    display_min: int | None = None,
    display_max: int | None = None,
    display_exp: float | None = None,
    blur_radius: int = 0,
    dither_strength: int = 0,
    dither_first: bool = False,
) -> np.ndarray:
    out_gray, _ = quantize_gray_with_indices(
        bgr, levels, min_value, max_value, exp_value,
        display_min=display_min,
        display_max=display_max,
        display_exp=display_exp,
        blur_radius=blur_radius,
        dither_strength=dither_strength,
        dither_first=dither_first,
    )
    return cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR)

