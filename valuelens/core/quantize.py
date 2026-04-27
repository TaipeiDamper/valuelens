from __future__ import annotations

import cv2
import numpy as np

try:
    import valuelens_native as _native_quant
except Exception:
    _native_quant = None

# Cache for the LUT to avoid re-calculating if parameters haven't changed
_CURRENT_QUANT_LUT: tuple[tuple, np.ndarray] | None = None

def get_quantization_lut(levels: int, min_val: int, max_val: int, exp_val: float) -> np.ndarray:
    global _CURRENT_QUANT_LUT
    params = (levels, min_val, max_val, exp_val)
    if _CURRENT_QUANT_LUT is not None and _CURRENT_QUANT_LUT[0] == params:
        return _CURRENT_QUANT_LUT[1]
    
    inputs = np.arange(256, dtype=np.float32)
    denom = max(1, max_val - min_val)
    norm = (inputs - min_val) / denom
    norm = np.clip(norm, 0.0, 1.0)
    
    if exp_val != 0:
        gamma = float(np.power(2.0, float(exp_val)))
        norm = np.power(norm, gamma)
        
    idx = np.floor(norm * levels)
    idx = np.clip(idx, 0, levels - 1).astype(np.uint8)
    
    _CURRENT_QUANT_LUT = (params, idx)
    return idx

# Bayer Cache
_BAYER_CACHE: dict[tuple[int, int], np.ndarray] = {}

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

def apply_bilateral(gray: np.ndarray, radius: int = 5) -> np.ndarray:
    if radius <= 0:
        return gray
    d = radius * 2 + 1
    if radius > 15:
        h, w = gray.shape
        small = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
        filtered = cv2.bilateralFilter(small, d // 2 + 1, 75, 75)
        return cv2.resize(filtered, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.bilateralFilter(gray, d, 75, 75)

def apply_ordered_dither(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    bayer_tiled = get_bayer_tiled(h, w)
    res = gray.astype(np.float32) + (bayer_tiled * 0.5)
    return np.clip(res, 0, 255).astype(np.uint8)


def has_native_acceleration() -> bool:
    return _native_quant is not None


def native_distribution_from_indices(indices: np.ndarray, levels: int) -> np.ndarray | None:
    """Return level distribution (%) using native module when available."""
    if _native_quant is None:
        return None
    try:
        out = _native_quant.distribution_from_indices(indices, max(2, int(levels)))
        return np.asarray(out, dtype=np.float64)
    except Exception:
        return None

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

    # Native fast path covers baseline quantization only (no additional effects).
    can_use_native = (
        _native_quant is not None
        and blur_radius <= 0
        and dither_strength <= 0
        and edge_strength <= 0
        and (not morph_enabled or morph_strength <= 0)
        and bgr.dtype == np.uint8
        and bgr.ndim == 3
        and bgr.shape[2] >= 3
    )
    if can_use_native:
        try:
            d_min = display_min if display_min is not None else 0
            d_max = display_max if display_max is not None else 255
            d_exp = display_exp if display_exp is not None else 0
            out_gray, indices = _native_quant.quantize_fast(
                np.ascontiguousarray(bgr),
                levels,
                int(min_value),
                int(max_value),
                float(exp_value),
                int(d_min),
                int(d_max),
                float(d_exp),
            )
            return (
                np.asarray(out_gray, dtype=np.uint8),
                np.asarray(indices, dtype=np.int32),
                None,
            )
        except Exception:
            # Any native failure should gracefully fallback to Python/OpenCV path.
            pass
    
    # --- Stage 1: Core Quantization ---
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # We maintain two versions: 
    # - working_gray: used for final color/indices (can be dithered)
    # - structure_gray: used for edge/morph detection (always clean)
    working_gray = gray
    structure_gray = gray
    
    indices = None
    edges = None
    morph_mask = None

    def apply_quantization(img_gray):
        lut = get_quantization_lut(levels, min_value, max_value, exp_value)
        idx = cv2.LUT(img_gray, lut)
        return idx.astype(np.int32)

    # Execute Modular Pipeline
    for step in process_order:
        if step == "blur" and blur_radius > 0:
            # Blur applies to both
            working_gray = apply_bilateral(working_gray, blur_radius)
            structure_gray = working_gray
        elif step == "dither" and dither_strength > 0:
            # Dither ONLY applies to working_gray
            working_gray = apply_ordered_dither(working_gray)
            indices = apply_quantization(working_gray)
        elif step == "edge" and edge_strength > 0:
            # Edges should use structure_gray or clean indices to avoid dither noise
            if indices is not None:
                # If dither was first, indices has noise. We use structure_gray instead or re-calc clean indices
                clean_indices = apply_quantization(structure_gray)
                edge_input = (clean_indices * (255 // (levels - 1))).astype(np.uint8)
            else:
                edge_input = structure_gray
            t1 = max(1, int(255 - edge_strength * 2.54))
            t2 = t1 * 2
            edges = cv2.Canny(edge_input, t1, t2)
        elif step == "morph" and morph_enabled and morph_strength > 0:
            # Morph ALWAYS uses structure_gray to avoid dither noise
            k_size = morph_strength * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            morph_grad = cv2.morphologyEx(structure_gray, cv2.MORPH_GRADIENT, kernel)
            # Use a slightly higher threshold for morph to be cleaner
            morph_mask = morph_grad > 35

    if indices is None:
        indices = apply_quantization(working_gray)

    # --- Stage 2: Display Mapping ---
    d_min = display_min if display_min is not None else 0
    d_max = display_max if display_max is not None else 255
    d_exp = display_exp if display_exp is not None else 0
    
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
