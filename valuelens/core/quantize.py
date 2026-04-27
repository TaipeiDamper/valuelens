from __future__ import annotations

import cv2
import numpy as np

# Cache for performance
_BAYER_CACHE: dict[tuple[int, int], np.ndarray] = {}
_GAMMA_LUT_CACHE: dict[float, np.ndarray] = {}

def get_bayer_tiled(h: int, w: int) -> np.ndarray:
    key = (h, w)
    if key in _BAYER_CACHE:
        return _BAYER_CACHE[key]
    
    bayer = np.array([
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ], dtype=np.float32)
    bayer = (bayer + 0.5) * (255.0 / 16.0) - 128.0
    
    tiled = np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w]
    _BAYER_CACHE[key] = tiled
    return tiled

def get_gamma_lut(exp_value: float) -> np.ndarray:
    if exp_value == 0:
        return np.arange(256, dtype=np.uint8)
    
    if exp_value in _GAMMA_LUT_CACHE:
        return _GAMMA_LUT_CACHE[exp_value]
    
    gamma = float(np.power(2.0, float(exp_value)))
    lut = np.power(np.arange(256) / 255.0, gamma) * 255.0
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    _GAMMA_LUT_CACHE[exp_value] = lut
    return lut

def apply_bilateral(gray: np.ndarray, radius: int = 5) -> np.ndarray:
    if radius <= 0:
        return gray
    d = radius * 2 + 1
    # Optimization: if radius is large, process on a smaller scale
    if radius > 15:
        h, w = gray.shape
        small = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
        filtered = cv2.bilateralFilter(small, d // 2 + 1, 75, 75)
        return cv2.resize(filtered, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return cv2.bilateralFilter(gray, d, 75, 75)

def apply_ordered_dither(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    bayer_tiled = get_bayer_tiled(h, w)
    # Using float32 for stable calculation, then back to uint8
    res = gray.astype(np.float32) + (bayer_tiled * 0.5)
    return np.clip(res, 0, 255).astype(np.uint8)

def quantize_gray_with_indices(
    bgr: np.ndarray,
    levels: int,
    min_value: int = 0,
    max_value: int = 255,
    exp_value: float = 0.0,
    display_min: int | None = None,
    display_max: int | None = None,
    display_exp: float | None = None,
    blur_radius: int = 0,
    dither_strength: int = 0,
    edge_strength: int = 0,
    process_order: list[str] = ("blur", "dither", "edge", "morph"),
    morph_enabled: bool = False,
    morph_strength: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    levels = max(2, int(levels))
    
    # --- Stage 1: Core Quantization ---
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    working_gray = gray
    
    indices = None
    edges = None
    morph_mask = None

    def apply_quantization(img_gray):
        # Use LUT for Gamma
        lut = get_gamma_lut(exp_value)
        img_lut = cv2.LUT(img_gray, lut)
        
        # Fast normalization and floor
        denom = max(1, max_value - min_value)
        f = img_lut.astype(np.float32)
        norm = (f - min_value) * (levels / denom)
        # Fix: ensure no NaN/Inf and clip before casting to int
        norm = np.nan_to_num(norm, nan=0.0, posinf=float(levels-1), neginf=0.0)
        idx = np.clip(norm, 0, levels - 1).astype(np.int32)
        return idx

    # Execute Modular Pipeline
    for step in process_order:
        if step == "blur" and blur_radius > 0:
            working_gray = apply_bilateral(working_gray, blur_radius)
        elif step == "dither" and dither_strength > 0:
            working_gray = apply_ordered_dither(working_gray)
            indices = apply_quantization(working_gray)
        elif step == "edge" and edge_strength > 0:
            if indices is not None:
                # Use cached levels logic
                edge_input = (indices * (255 // (levels - 1))).astype(np.uint8)
            else:
                edge_input = working_gray
            t1 = max(1, int(255 - edge_strength * 2.54))
            t2 = t1 * 2
            edges = cv2.Canny(edge_input, t1, t2)
        elif step == "morph" and morph_enabled and morph_strength > 0:
            k_size = morph_strength * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            # Optimization: Morph can be expensive on high-res, but we'll stick to full res for quality unless asked
            morph_grad = cv2.morphologyEx(working_gray, cv2.MORPH_GRADIENT, kernel)
            morph_mask = morph_grad > 30

    if indices is None:
        indices = apply_quantization(working_gray)

    # --- Stage 2: Display Mapping ---
    d_min = display_min if display_min is not None else 0
    d_max = display_max if display_max is not None else 255
    d_exp = display_exp if display_exp is not None else 0
    
    # Pre-calc display map to avoid slow power/float ops per frame if simple
    if d_exp == 0 and d_min == 0 and d_max == 255:
        out = (indices * (255 // (levels - 1))).astype(np.uint8)
    else:
        norm = indices.astype(np.float32) / (levels - 1)
        if d_exp != 0:
            gamma_display = float(np.power(2.0, float(d_exp)))
            norm = np.power(norm, gamma_display)
        out = (norm * (d_max - d_min) + d_min).astype(np.uint8)
    
    if morph_mask is not None:
        out[morph_mask] = 0
        
    return out, indices, edges

def quantize_gray(
    bgr: np.ndarray,
    levels: int,
    min_value: int = 0,
    max_value: int = 255,
    exp_value: float = 0.0,
    display_min: int | None = None,
    display_max: int | None = None,
    display_exp: float | None = None,
    blur_radius: int = 0,
    dither_strength: int = 0,
    edge_strength: int = 0,
    process_order: list[str] = ("blur", "dither", "edge", "morph"),
    morph_enabled: bool = False,
    morph_strength: int = 1,
) -> np.ndarray:
    out_gray, _, _ = quantize_gray_with_indices(
        bgr, levels, min_value, max_value, exp_value,
        display_min=display_min,
        display_max=display_max,
        display_exp=display_exp,
        blur_radius=blur_radius,
        dither_strength=dither_strength,
        edge_strength=edge_strength,
        process_order=process_order,
        morph_enabled=morph_enabled,
        morph_strength=morph_strength,
    )
    return cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR)
