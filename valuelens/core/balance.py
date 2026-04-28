from __future__ import annotations

import itertools
import numpy as np

def calc_level_distribution(gray: np.ndarray | None, levels: int) -> list[float]:
    """計算給定灰階圖在特定階調數下的分佈比例 (0.0~100.0)"""
    level_count = max(2, int(levels))
    if gray is None or gray.size == 0:
        return [0.0] * level_count
    scaled = np.floor(gray.astype(np.float32) * level_count / 256.0)
    indices = np.clip(scaled, 0, level_count - 1).astype(np.int32)
    counts = np.bincount(indices.reshape(-1), minlength=level_count).astype(np.float64)
    total = float(counts.sum())
    if total <= 0:
        return [0.0] * level_count
    return ((counts / total) * 100.0).tolist()

def calc_indices_distribution(indices: np.ndarray | None, levels: int) -> list[float]:
    """直接從量化完的 index map 計算分佈比例"""
    level_count = max(2, int(levels))
    if indices is None or indices.size == 0:
        return [0.0] * level_count
    
    from valuelens.core.quantize import native_distribution_from_indices
    native_dist = native_distribution_from_indices(indices, level_count)
    if native_dist is not None and native_dist.size == level_count:
        return native_dist.tolist()
        
    clamped = np.clip(indices.astype(np.int32), 0, level_count - 1)
    counts = np.bincount(clamped.reshape(-1), minlength=level_count).astype(np.float64)
    total = float(counts.sum())
    if total <= 0:
        return [0.0] * level_count
    return ((counts / total) * 100.0).tolist()

def levels_to_wgb(values: list[float]) -> tuple[float, float, float]:
    """將 N 階層分佈折疊為 (white%, gray%, black%)。"""
    v = list(reversed(values))
    n = len(v)

    def _s(*idx):
        return float(sum(v[i] for i in idx if 0 <= i < n))

    def _best_dist(wgb, target_base):
        total = sum(wgb) or 1.0
        norm = [val / total for val in wgb]
        t_total = sum(target_base)
        tn = [val / t_total for val in target_base]
        best = float("inf")
        for p in itertools.permutations(tn):
            d = sum((norm[i] - p[i])**2 for i in range(3))
            if d < best:
                best = d
        return best

    if n == 2:
        return (_s(0), 0.0, _s(1))
    if n == 3:
        return (_s(0), _s(1), _s(2))
    if n == 5:
        target = (0.7, 0.2, 0.1)
        a = (_s(0, 1), _s(2, 3), _s(4))
        b = (_s(0),    _s(1, 2), _s(3, 4))
        return a if _best_dist(a, target) <= _best_dist(b, target) else b
    if n == 8:
        target = (0.7, 0.2, 0.1)
        a = (_s(0, 1),    _s(2, 3, 4), _s(5, 6, 7))
        b = (_s(0, 1, 2), _s(3, 4, 5), _s(6, 7))
        return a if _best_dist(a, target) <= _best_dist(b, target) else b

    edges = np.linspace(0, n, 4, dtype=np.float64)
    e1, e2 = int(np.floor(edges[1])), int(np.floor(edges[2]))
    return (_s(*range(0, e1)), _s(*range(e1, e2)), _s(*range(e2, n)))

def distribution_from_hist(
    hist: np.ndarray, lower: int, upper: int, levels: int, exp_value: float
) -> np.ndarray:
    """從灰階直方圖預測量化後的結果比例"""
    levels = max(2, int(levels))
    lower = max(0, min(254, int(lower)))
    upper = max(lower + 1, min(255, int(upper)))
    values = np.arange(256, dtype=np.float64)
    clipped = np.clip(values, lower, upper)
    normalized = (clipped - float(lower)) / float(upper - lower)
    gamma = float(np.power(2.0, float(exp_value)))
    mapped = np.power(normalized, gamma)
    indices = np.floor(mapped * levels)
    indices = np.clip(indices, 0, levels - 1).astype(np.int32)
    counts = np.bincount(indices, weights=hist, minlength=levels).astype(np.float64)
    total = float(counts.sum())
    if total <= 0:
        return np.zeros(3, dtype=np.float64)
    
    if levels == 2:
        return np.array([counts[0]/total, 0.0, counts[1]/total], dtype=np.float64)

    level_edges = np.linspace(0, levels, 4, dtype=np.float64)
    edge1 = int(np.floor(level_edges[1]))
    edge2 = int(np.floor(level_edges[2]))
    low = float(np.sum(counts[:edge1]) / total)
    mid = float(np.sum(counts[edge1:edge2]) / total)
    high = float(np.sum(counts[edge2:]) / total)
    return np.array([low, mid, high], dtype=np.float64)

def optimize_balance_params(
    gray: np.ndarray,
    target: tuple[float, float, float], # (White, Gray, Black)
    current_min: int,
    current_max: int,
    current_exp: float,
    levels_count: int,
    hysteresis: float = 0.0
) -> tuple[int, int, float]:
    """全自動分析最佳平衡參數的核心算法 - 多解析度階層搜尋 (Multi-resolution Search)"""
    
    # 記錄呼叫次數，用於定期「強制校正與抖動探索」
    optimize_balance_params._call_count = getattr(optimize_balance_params, '_call_count', 0) + 1
    force_recalibrate = (optimize_balance_params._call_count % 15 == 0)
    
    hist = np.bincount(gray.reshape(-1).astype(np.uint8), minlength=256).astype(np.float64)
    total = float(hist.sum())
    if total <= 0: return current_min, current_max, current_exp

    levels = max(2, int(levels_count))
    t_white, t_gray, t_black = target
    t_total = t_white + t_gray + t_black
    if t_total <= 0: return current_min, current_max, current_exp
    
    target_ratios = np.array([t_black, t_gray, t_white]) / t_total
    
    # 評估目前參數的 Loss (防震盪)
    current_dist = distribution_from_hist(hist, current_min, current_max, levels, current_exp)
    diff_curr = current_dist - target_ratios
    current_loss = float(np.dot(diff_curr, diff_curr))
    
    # 定期校正時不使用主場優勢 (hysteresis=0)，確保隨時都有機會逃離局部最佳解
    eff_hysteresis = 0.0 if force_recalibrate else hysteresis
    best_loss = current_loss - eff_hysteresis
    best_params = (current_min, current_max, current_exp)

    # --- 階段一：粗算 (Coarse Search) ---
    # 鋪設覆蓋全域的稀疏網格
    lo_coarse = np.linspace(0, 240, 12, dtype=int)
    hi_coarse = np.linspace(10, 255, 12, dtype=int)
    ex_coarse = np.linspace(-2.0, 2.0, 12)
    
    coarse_best_params = best_params
    coarse_best_loss = best_loss
    
    for lo in lo_coarse:
        if lo > 240: continue
        for hi in hi_coarse:
            if hi <= lo + 4: continue
            if hi > 255: continue
            for ex in ex_coarse:
                r_now = distribution_from_hist(hist, lo, hi, levels, float(ex))
                diff = r_now - target_ratios
                l = float(np.dot(diff, diff))
                if l < coarse_best_loss:
                    coarse_best_loss = l
                    coarse_best_params = (lo, hi, float(ex))

    # --- 階段二：精算 (Fine Search) ---
    # 圍繞粗算結果鋪設精細的網格
    c_lo, c_hi, c_ex = coarse_best_params
    
    # 定期校正時，故意給參數微小的隨機抖動 (Nudge)，打破對稱性以探索邊緣極限
    if force_recalibrate:
        c_lo = max(0, min(240, c_lo + int(np.random.randint(-6, 7))))
        c_hi = max(c_lo + 5, min(255, c_hi + int(np.random.randint(-6, 7))))
        c_ex = max(-2.0, min(2.0, c_ex + float(np.random.uniform(-0.25, 0.25))))

    lo_fine = np.linspace(max(0, c_lo - 24), min(240, c_lo + 24), 10, dtype=int)
    hi_fine = np.linspace(max(c_lo + 5, c_hi - 24), min(255, c_hi + 24), 10, dtype=int)
    ex_fine = np.linspace(max(-2.0, c_ex - 0.4), min(2.0, c_ex + 0.4), 10)

    for lo in lo_fine:
        if lo > 240: continue
        for hi in hi_fine:
            if hi <= lo + 4: continue
            if hi > 255: continue
            for ex in ex_fine:
                r_now = distribution_from_hist(hist, lo, hi, levels, float(ex))
                diff = r_now - target_ratios
                l = float(np.dot(diff, diff))
                if l < best_loss:
                    best_loss = l
                    best_params = (lo, hi, float(ex))
                    
    return best_params
