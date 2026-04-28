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
    levels_count: int
) -> tuple[int, int, float]:
    """全自動分析最佳平衡參數的核心算法"""
    hist = np.bincount(gray.reshape(-1).astype(np.uint8), minlength=256).astype(np.float64)
    total = float(hist.sum())
    if total <= 0: return current_min, current_max, current_exp

    cdf = np.cumsum(hist) / total
    
    t_white, t_gray, t_black = target
    t_total = t_white + t_gray + t_black
    if t_total <= 0: return current_min, current_max, current_exp
    
    pct_black = t_black / t_total
    pct_not_white = (t_black + t_gray) / t_total if t_gray > 0 else (1.0 - t_white / t_total)
    
    def find_val(p):
        return int(np.searchsorted(cdf, max(0.0, min(1.0, p))))

    guess_min = find_val(pct_black)
    guess_max = find_val(pct_not_white)
    
    p_samples = np.linspace(-0.12, 0.12, 13) 
    
    lo_candidates = sorted(list(set([find_val(pct_black + s) for s in p_samples] + [guess_min])))
    hi_candidates = sorted(list(set([find_val(pct_not_white + s) for s in p_samples] + [guess_max])))
    exp_range = np.linspace(-1.5, 1.5, 25)

    best_loss = float('inf')
    best_params = (current_min, current_max, current_exp)
    levels = max(2, int(levels_count))
    target_ratios = np.array([t_black, t_gray, t_white]) / t_total
    
    for lo in lo_candidates:
        if lo > 240: continue
        for hi in hi_candidates:
            if hi <= lo + 4: continue
            if hi > 255: continue
            for ex in exp_range:
                r_now = distribution_from_hist(hist, lo, hi, levels, float(ex))
                diff = r_now - target_ratios
                l = float(np.dot(diff, diff))
                if l < best_loss:
                    best_loss = l
                    best_params = (lo, hi, float(ex))
                    
    return best_params
