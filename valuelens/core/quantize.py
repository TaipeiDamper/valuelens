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
    ], dtype=np.int16)
    
    # 將浮點計算 (bayer + 0.5) * (255/32) - 64 預先轉為 Int16
    bayer_int = np.round((bayer + 0.5) * (255.0 / 32.0) - 64.0).astype(np.int16)
    
    tiled = np.tile(bayer_int, (h // 4 + 1, w // 4 + 1))[:h, :w]
    _BAYER_CACHE[key] = tiled
    return tiled

def apply_bilateral(gray: np.ndarray, radius: int = 5) -> np.ndarray:
    if radius <= 0:
        return gray
    d = radius * 2 + 1
    h, w = gray.shape
    
    # 鎖定最大運算寬度為 400 像素，徹底擺脫解析度造成的效能懲罰
    target_w = 400
    if w > target_w:
        scale = target_w / w
        target_h = max(1, int(h * scale))
        
        # 為了讓模糊視覺體驗一致，濾波半徑 d 也需要隨比例縮放
        d_small = max(3, int(d * scale))
        
        small = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        filtered = cv2.bilateralFilter(small, d_small, 75, 75)
        return cv2.resize(filtered, (w, h), interpolation=cv2.INTER_LINEAR)
        
    return cv2.bilateralFilter(gray, d, 75, 75)

def apply_ordered_dither(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    bayer_tiled = get_bayer_tiled(h, w)
    
    # 採用快速的 Int16 向量運算取代耗能的 Float32
    res = gray.astype(np.int16) + bayer_tiled
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
    def __init__(self, bgr: np.ndarray, levels: int, min_val: int, max_val: int, exp_val: float):
        self.levels = max(2, int(levels))
        self.min_val = int(min_val)
        self.max_val = int(max_val)
        self.exp_val = float(exp_val)
        
        # 【超高清輸入源】：永遠維持 100% 乾淨無污染
        self.original_bgr = bgr
        if len(bgr.shape) == 3 and bgr.shape[2] == 3:
            self.original_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            self.original_gray = bgr.copy()
        
        # 【Ａ軌】：主色調畫布（允許被平滑、抖動演算法修改）
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
            ctx.working_gray = apply_ordered_dither(ctx.working_gray)

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
    morph_threshold: int = 35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """模組化影像濾鏡與量化入口點。"""
    levels = max(2, int(levels))
    ctx = FilterContext(bgr, levels, min_value, max_value, exp_value)
    
    # 註冊可供調配的積木庫
    FILTERS = {
        "blur": BlurFilter(),
        "dither": DitherFilter(),
        "edge": EdgeFilter(),
        "morph": MorphFilter(),
    }
    
    kwargs = {
        "blur_radius": blur_radius,
        "dither_strength": dither_strength,
        "edge_strength": edge_strength,
        "morph_enabled": morph_enabled,
        "morph_strength": morph_strength,
        "morph_threshold": morph_threshold,
    }
    
    # 【積木隨意搬風】：依照 process_order 動態呼叫
    import time
    for step in process_order:
        if step in FILTERS:
            t0 = time.perf_counter()
            FILTERS[step].apply(ctx, **kwargs)
            t1 = time.perf_counter()
            print(f"  [Filter Step] {step}: {(t1-t0)*1000:.2f}ms", flush=True)
            
    # 最終量化查表
    lut = get_quantization_lut(ctx.levels, ctx.min_val, ctx.max_val, ctx.exp_val)
    ctx.indices = cv2.LUT(ctx.working_gray, lut).astype(np.int32)
    
    # --- 顯示輸出映射 (Display Mapping) 查表優化 ---
    d_min = display_min if display_min is not None else 0
    d_max = display_max if display_max is not None else 255
    d_exp = display_exp if display_exp is not None else 0
    
    # 僅針對少數離散階調值進行浮點與 Gamma 運算
    lut_inputs = np.arange(levels, dtype=np.float32)
    norm_lut = lut_inputs / (levels - 1)
    if d_exp != 0:
        gamma_display = float(np.power(2.0, float(d_exp)))
        norm_lut = np.power(norm_lut, gamma_display)
    out_lut = (norm_lut * (d_max - d_min) + d_min).astype(np.uint8)
    
    # 填入適用於 cv2.LUT 的 256 長度映射表
    full_display_lut = np.zeros(256, dtype=np.uint8)
    full_display_lut[:levels] = out_lut
    
    # 透過 OpenCV 的 C++ 底層引擎瞬間完成千萬像素查表
    out = cv2.LUT(ctx.indices.astype(np.uint8), full_display_lut)
    
    if ctx.morph_mask is not None:
        out[ctx.morph_mask] = 0
        
    return out, ctx.indices, ctx.edges
