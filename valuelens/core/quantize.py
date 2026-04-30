from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

try:
    import valuelens_native as _native_quant
except Exception:
    _native_quant = None

# Cache for the LUT to avoid re-calculating if parameters haven't changed
_CURRENT_QUANT_LUT: tuple[tuple, np.ndarray] | None = None
_CURRENT_COLOR_LUT: tuple[tuple, np.ndarray] | None = None


def _palette_key(palette: list[tuple[int, int, int]] | None, levels: int) -> tuple | None:
    if not palette or len(palette) != levels:
        return None
    return tuple((int(r), int(g), int(b)) for r, g, b in palette)


def get_color_lut_bgr(
    levels: int,
    display_min: int,
    display_max: int,
    display_exp: float,
    palette: list[tuple[int, int, int]] | None = None,
    edge_mix: int = 0,
    edge_active: bool = False,
) -> np.ndarray:
    """Return a 256-entry BGR color LUT for quantized indices."""
    global _CURRENT_COLOR_LUT
    levels = max(2, int(levels))
    mix = float(edge_mix) / 100.0 if edge_active else 0.0
    params = (
        levels,
        int(display_min),
        int(display_max),
        float(display_exp),
        _palette_key(palette, levels),
        float(mix),
    )
    if _CURRENT_COLOR_LUT is not None and _CURRENT_COLOR_LUT[0] == params:
        return _CURRENT_COLOR_LUT[1]

    lut_inputs = np.arange(levels, dtype=np.float32)
    norm_lut = lut_inputs / (levels - 1)
    if display_exp != 0:
        gamma_display = float(np.power(2.0, float(display_exp)))
        norm_lut = np.power(norm_lut, gamma_display)
    out_lut = (norm_lut * (display_max - display_min) + display_min).astype(np.uint8)

    full_lut_bgr = np.zeros((256, 3), dtype=np.uint8)
    if palette and len(palette) == levels:
        for i, color in enumerate(palette):
            r, g, b = color
            full_lut_bgr[i] = [
                int(b * (1.0 - mix) + 255.0 * mix),
                int(g * (1.0 - mix) + 255.0 * mix),
                int(r * (1.0 - mix) + 255.0 * mix),
            ]
    else:
        for i in range(levels):
            val = out_lut[i]
            mixed_val = int(val * (1.0 - mix) + 255.0 * mix)
            full_lut_bgr[i] = [mixed_val, mixed_val, mixed_val]

    _CURRENT_COLOR_LUT = (params, full_lut_bgr)
    return full_lut_bgr

def get_quantization_lut(levels: int, min_val: int, max_val: int, exp_val: float, curve_mode: int = 0) -> np.ndarray:
    global _CURRENT_QUANT_LUT
    params = (levels, min_val, max_val, exp_val, curve_mode)
    if _CURRENT_QUANT_LUT is not None and _CURRENT_QUANT_LUT[0] == params:
        return _CURRENT_QUANT_LUT[1]
    
    x_raw = np.arange(256, dtype=np.float32)
    denom = max(1, max_val - min_val)
    x = np.clip((x_raw - min_val) / denom, 0.0, 1.0)
    
    gamma = 2.0 ** float(exp_val)
    
    if curve_mode == 0: # Gamma
        y = np.power(x, gamma)
    elif curve_mode == 1: # Sigmoid (S-Curve)
        eps = 1e-6
        x_g = np.power(x, gamma)
        inv_x_g = np.power(1.0 - x + eps, gamma)
        y = x_g / (x_g + inv_x_g)
    elif curve_mode == 2: # Log
        k = (float(exp_val) + 2.0) * 10.0
        if k < 0.1: k = 0.1
        y = np.log(1.0 + k * x) / np.log(1.0 + k)
    else:
        y = x
        
    idx = np.floor(y * levels)
    idx = np.clip(idx, 0, levels - 1).astype(np.uint8)
    
    _CURRENT_QUANT_LUT = (params, idx)
    return idx

# Bayer Cache
_BAYER_CACHE: dict[tuple[int, int], np.ndarray] = {}
_DITHER_OFFSET_CACHE: tuple[tuple[int, int, int], np.ndarray] | None = None

def get_bayer_tiled(h: int, w: int) -> np.ndarray:
    key = (h, w)
    if key in _BAYER_CACHE:
        return _BAYER_CACHE[key]
    
    bayer = np.array([
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ], dtype=np.int16)
    
    # 將浮點計算 (bayer + 0.5) * (255/32) - 64 預先轉為 Int16
    bayer_int = np.round((bayer + 0.5) * (255.0 / 32.0) - 64.0).astype(np.int16)
    
    tiled = np.tile(bayer_int, (h // 4 + 1, w // 4 + 1))[:h, :w]
    _BAYER_CACHE[key] = tiled
    return tiled


def get_dither_offset(h: int, w: int, strength: int) -> np.ndarray:
    global _DITHER_OFFSET_CACHE
    strength = max(0, min(100, int(strength)))
    key = (h, w, strength)
    if _DITHER_OFFSET_CACHE is not None and _DITHER_OFFSET_CACHE[0] == key:
        return _DITHER_OFFSET_CACHE[1]

    bayer_tiled = get_bayer_tiled(h, w)
    if strength >= 100:
        offset = bayer_tiled
    else:
        offset = np.rint(bayer_tiled * (strength / 100.0)).astype(np.int16)
    _DITHER_OFFSET_CACHE = (key, offset)
    return offset

def apply_bilateral(gray: np.ndarray, radius: int = 5) -> np.ndarray:
    if radius <= 0:
        return gray
    
    # --- 原本的高級 Bilateral Filter (保邊濾鏡)，目前先註解掉以追求極速 ---
    # d = radius * 2 + 1
    # h, w = gray.shape
    # target_w = 400
    # if w > target_w:
    #     scale = target_w / w
    #     target_h = max(1, int(h * scale))
    #     d_small = max(3, int(d * scale))
    #     small = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    #     filtered = cv2.bilateralFilter(small, d_small, 75, 75)
    #     return cv2.resize(filtered, (w, h), interpolation=cv2.INTER_LINEAR)
    # return cv2.bilateralFilter(gray, d, 75, 75)

    # --- 高速版：Gaussian Blur (搭配 Edge Mix 功能通常已經足夠) ---
    k = (radius * 2) | 1
    return cv2.GaussianBlur(gray, (k, k), 0)

def apply_ordered_dither(gray: np.ndarray, strength: int = 100) -> np.ndarray:
    strength = max(0, min(100, int(strength)))
    if strength <= 0:
        return gray

    h, w = gray.shape
    dither_offset = get_dither_offset(h, w, strength)

    # 採用快速的 Int16 向量運算取代耗能的 Float32
    res = gray.astype(np.int16) + dither_offset
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


def native_color_map_rgb(indices: np.ndarray, lut_bgr: np.ndarray) -> np.ndarray | None:
    """Return RGB output from quantized indices and BGR LUT using native module."""
    if _native_quant is None or not hasattr(_native_quant, "color_map_rgb"):
        return None
    try:
        return np.asarray(_native_quant.color_map_rgb(indices, lut_bgr), dtype=np.uint8)
    except Exception:
        return None

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
    morph_threshold: int = 35,
) -> np.ndarray:
    out_gray, _, _, _ = quantize_gray_with_indices(
        bgr,
        levels,
        min_value=min_value,
        max_value=max_value,
        exp_value=exp_value,
        display_min=display_min,
        display_max=display_max,
        display_exp=display_exp,
        blur_radius=blur_radius,
        dither_strength=dither_strength,
        edge_strength=edge_strength,
        process_order=process_order,
        morph_enabled=morph_enabled,
        morph_strength=morph_strength,
        morph_threshold=morph_threshold,
    )
    return out_gray


# =====================================================================
# 【V2 模組化管線架構實驗區】
# =====================================================================

class FilterContext:
    """
    影像處理管線的 Context 封裝箱（狀態管理者）。
    負責存放原始影像與各階段衍生資料，確保資料流向明確。
    """
    def __init__(self, bgr: np.ndarray, levels: int, min_val: int, max_val: int, exp_val: float, curve_mode: int = 0, buffer_manager: Any = None):
        self.levels = max(2, int(levels))
        self.min_val = int(min_val)
        self.max_val = int(max_val)
        self.exp_val = float(exp_val)
        self.curve_mode = int(curve_mode)
        self.buffer_manager = buffer_manager
        
        # 【超高清輸入源】：強制使用副本，徹底隔離後續濾鏡的污染
        if len(bgr.shape) == 3 and bgr.shape[2] == 3:
            self.original_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            self.original_gray = bgr.copy()
        
        # 【Ａ軌】：主色調畫布（必須是另一個獨立複本）
        self.working_gray = self.original_gray.copy()
        
        # 【成果輸出區】
        self.indices = None      # 量化後的色彩索引陣列
        self.edges = None        # 連線（Canny）偵測結果
        self.morph_mask = None   # 勾邊遮罩結果


class BaseFilter:
    def apply(self, ctx: FilterContext, **kwargs) -> None:
        raise NotImplementedError

class BlurFilter(BaseFilter):
    def apply(self, ctx: FilterContext, **kwargs) -> None:
        radius = kwargs.get("blur_radius", 0)
        if radius > 0:
            ctx.working_gray = apply_bilateral(ctx.working_gray, radius)

class DitherFilter(BaseFilter):
    def apply(self, ctx: FilterContext, **kwargs) -> None:
        strength = kwargs.get("dither_strength", 0)
        if strength > 0:
            ctx.working_gray = apply_ordered_dither(ctx.working_gray, strength)

class EdgeFilter(BaseFilter):
    def apply(self, ctx: FilterContext, **kwargs) -> None:
        strength = kwargs.get("edge_strength", 0)
        if strength > 0:
            # 【核心解耦】：永遠使用超高清原始影像，不吃污染
            t1 = max(1, int(255 - strength * 2.54))
            t2 = t1 * 2
            ctx.edges = cv2.Canny(ctx.original_gray, t1, t2)

class MorphFilter(BaseFilter):
    def apply(self, ctx: FilterContext, **kwargs) -> None:
        enabled = kwargs.get("morph_enabled", False)
        strength = kwargs.get("morph_strength", 1)
        threshold = kwargs.get("morph_threshold", 35)
        if enabled and strength > 0:
            k_size = strength * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            morph_grad = cv2.morphologyEx(ctx.original_gray, cv2.MORPH_GRADIENT, kernel)
            ctx.morph_mask = morph_grad > threshold


FILTERS: dict[str, BaseFilter] = {
    "blur": BlurFilter(),
    "dither": DitherFilter(),
    "edge": EdgeFilter(),
    "morph": MorphFilter(),
}


def quantize_gray_with_indices(
    bgr: np.ndarray,
    levels: int,
    min_value: int = 0,
    max_value: int = 255,
    exp_value: float = 0.0,
    curve_mode: int = 0,
    display_min: int | None = None,
    display_max: int | None = None,
    display_exp: float | None = None,
    blur_radius: int = 0,
    dither_strength: int = 0,
    edge_strength: int = 0,
    process_order: list[str] = ("blur", "dither", "edge", "morph"),
    morph_enabled: bool = False,
    morph_strength: int = 1,
    morph_threshold: int = 35,
    buffer_manager: Any = None,
    palette: list[tuple[int, int, int]] | None = None,
    edge_mix: int = 0,
    edge_color: tuple[int, int, int] | None = None,
    profile: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """模組化影像濾鏡與量化入口點，回傳 (量化彩色影像, 索引矩陣, 邊緣矩陣, 真實比例計數)。"""
    if profile is not None:
        profile.clear()
    t_total = time.perf_counter() if profile is not None else 0.0
    t_last = t_total

    def _mark(name: str) -> None:
        nonlocal t_last
        if profile is None:
            return
        now = time.perf_counter()
        profile[name] = (now - t_last) * 1000.0
        t_last = now

    levels = max(2, int(levels))
    ctx = FilterContext(bgr, levels, min_value, max_value, exp_value, curve_mode, buffer_manager=buffer_manager)
    _mark("prepare_gray_ms")
    
    # --- 關鍵：在濾鏡之前獲取「真實比例」 (採用抽樣優化) ---
    lut = get_quantization_lut(ctx.levels, ctx.min_val, ctx.max_val, ctx.exp_val, ctx.curve_mode)
    _mark("logic_lut_ms")
    
    # 對統計資料進行降採樣，解析度越高省下的時間越多
    h_orig, w_orig = ctx.original_gray.shape
    if w_orig > 400:
        scale_stats = 400 / w_orig
        h_stats = max(1, int(h_orig * scale_stats))
        stats_gray = cv2.resize(ctx.original_gray, (400, h_stats), interpolation=cv2.INTER_NEAREST)
    else:
        stats_gray = ctx.original_gray
        
    true_indices = cv2.LUT(stats_gray, lut)
    true_counts = np.bincount(true_indices.ravel(), minlength=levels)
    _mark("stats_ms")
    
    kwargs = {
        "blur_radius": blur_radius,
        "dither_strength": dither_strength,
        "edge_strength": edge_strength,
        "morph_enabled": morph_enabled,
        "morph_strength": morph_strength,
        "morph_threshold": morph_threshold,
    }
    
    for step in process_order:
        if step in FILTERS:
            FILTERS[step].apply(ctx, **kwargs)
    _mark("filters_ms")
            
    # 最終量化索引 (吃過濾鏡後的 working_gray)
    ctx.indices = cv2.LUT(ctx.working_gray, lut).astype(np.uint8)
    _mark("final_index_ms")
    
    d_min = display_min if display_min is not None else 0
    d_max = display_max if display_max is not None else 255
    d_exp = display_exp if display_exp is not None else 0
    full_lut_bgr = get_color_lut_bgr(
        ctx.levels,
        int(d_min),
        int(d_max),
        float(d_exp),
        palette=palette,
        edge_mix=edge_mix,
        edge_active=ctx.edges is not None,
    )
    _mark("display_lut_ms")

    native_rgb = None
    if edge_color is None and ctx.morph_mask is None:
        native_rgb = native_color_map_rgb(ctx.indices, full_lut_bgr)

    if native_rgb is not None:
        out_rgb = native_rgb
        _mark("color_map_ms")
        _mark("edge_morph_apply_ms")
        _mark("rgb_convert_ms")
    else:
        # 直接進行彩色查表 (NumPy 方式)
        out_bgr = full_lut_bgr[ctx.indices]
        _mark("color_map_ms")

        # 處理邊緣顏色
        if ctx.edges is not None and edge_color:
            ec_bgr = edge_color[::-1] # RGB -> BGR
            out_bgr[ctx.edges > 0] = ec_bgr

        # 處理形態學遮罩
        if ctx.morph_mask is not None:
            mask = np.ascontiguousarray(ctx.morph_mask)
            np.copyto(out_bgr, 0, where=mask[:, :, None])
        _mark("edge_morph_apply_ms")

        # 在背景執行緒完成 RGB 轉換，徹底解放主執行緒
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        _mark("rgb_convert_ms")
    if profile is not None:
        profile["total_ms"] = (time.perf_counter() - t_total) * 1000.0
    
    return out_rgb, ctx.indices, ctx.edges, true_counts
