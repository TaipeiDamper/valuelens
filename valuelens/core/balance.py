import itertools
import numpy as np

# 預分配常用陣列以提升速度
_VALUES_256 = np.arange(256, dtype=np.float32)
_DEFAULT_WGB_TARGET = np.array([0.7, 0.2, 0.1], dtype=np.float64)


def _best_partition_distance(wgb: np.ndarray) -> float:
    total = float(wgb.sum()) or 1.0
    norm = wgb / total
    best = float("inf")
    for p in itertools.permutations(_DEFAULT_WGB_TARGET):
        d = float(np.sum((norm - np.asarray(p)) ** 2))
        if d < best:
            best = d
    return best


def _fold_level_values_to_wgb(values: np.ndarray) -> np.ndarray:
    """Fold low-to-high level values into W:G:B order."""
    v = values.astype(np.float64, copy=False)[::-1]
    n = int(v.size)

    if n == 0:
        return np.zeros(3, dtype=np.float64)
    if n == 2:
        return np.array([v[0], 0.0, v[1]], dtype=np.float64)
    if n == 3:
        return np.array([v[0], v[1], v[2]], dtype=np.float64)
    if n == 5:
        a = np.array([v[0] + v[1], v[2] + v[3], v[4]], dtype=np.float64)
        b = np.array([v[0], v[1] + v[2], v[3] + v[4]], dtype=np.float64)
        return a if _best_partition_distance(a) <= _best_partition_distance(b) else b
    if n == 8:
        a = np.array([v[0] + v[1], v[2] + v[3] + v[4], v[5] + v[6] + v[7]], dtype=np.float64)
        b = np.array([v[0] + v[1] + v[2], v[3] + v[4] + v[5], v[6] + v[7]], dtype=np.float64)
        return a if _best_partition_distance(a) <= _best_partition_distance(b) else b

    edges = np.linspace(0, n, 4)
    e1, e2 = int(edges[1]), int(edges[2])
    return np.array([v[:e1].sum(), v[e1:e2].sum(), v[e2:].sum()], dtype=np.float64)

def calc_level_distribution(gray: np.ndarray | None, levels: int) -> list[float]:
    """計算給定灰階圖在特定階調數下的分佈比例 (0.0~100.0)"""
    level_count = max(2, int(levels))
    if gray is None or gray.size == 0:
        return [0.0] * level_count
    scaled = np.floor(gray.astype(np.float32) * level_count / 256.0)
    indices = np.clip(scaled, 0, level_count - 1).astype(np.int32)
    counts = np.bincount(indices.ravel(), minlength=level_count).astype(np.float64)
    total = float(counts.sum())
    if total <= 0:
        return [0.0] * level_count
    return ((counts / total) * 100.0).tolist()

def calc_indices_distribution(indices: np.ndarray | None, levels: int) -> list[float]:
    """直接從量化完的 index map 計算分佈比例"""
    level_count = max(2, int(levels))
    if indices is None or indices.size == 0:
        return [0.0] * level_count
    counts = np.bincount(indices.ravel(), minlength=level_count).astype(np.float64)
    total = float(counts.sum())
    if total <= 0:
        return [0.0] * level_count
    return ((counts / total) * 100.0).tolist()

def levels_to_wgb(values: list[float]) -> tuple[float, float, float]:
    """將 N 階層分佈折疊為 (white%, gray%, black%)。"""
    folded = _fold_level_values_to_wgb(np.asarray(values, dtype=np.float64))
    return float(folded[0]), float(folded[1]), float(folded[2])

def distribution_from_hist(
    hist: np.ndarray, lower: int, upper: int, levels: int, exp_value: float, mode: int = 0
) -> np.ndarray:
    """從灰階直方圖預測量化後的結果比例 (多模式優化版)"""
    levels = max(2, int(levels))
    lower = max(0, min(240, int(lower)))
    upper = max(lower + 1, min(255, int(upper)))
    
    inv_range = 1.0 / (upper - lower)
    x = np.clip((_VALUES_256 - lower) * inv_range, 0, 1)
    
    # 根據模式決定映射曲線
    gamma = 2.0 ** float(exp_value)
    
    if mode == 0: # Gamma 模式
        mapped = x ** gamma
    elif mode == 1: # Sigmoid (S-Curve) 模式
        # 使用 Hill Equation: y = x^g / (x^g + (1-x)^g)
        eps = 1e-6
        x_g = np.power(x, gamma)
        inv_x_g = np.power(1.0 - x + eps, gamma)
        mapped = x_g / (x_g + inv_x_g)
    elif mode == 2: # Log (對數) 模式
        # y = ln(1 + kx) / ln(1 + k)
        k = (exp_value + 2.0) * 10.0 # 簡單映射 k 值
        if k < 0.1: k = 0.1
        mapped = np.log(1.0 + k * x) / np.log(1.0 + k)
    else:
        mapped = x
        
    indices = (mapped * levels).astype(np.int32)
    indices[indices >= levels] = levels - 1
    
    counts = np.bincount(indices, weights=hist, minlength=levels).astype(np.float64)
    total = counts.sum()
    if total <= 0: return np.zeros(3)
    return _fold_level_values_to_wgb(counts) / total

def optimize_balance_params(
    gray: np.ndarray,
    target: tuple[float, float, float],
    current_min: int,
    current_max: int,
    current_exp: float,
    levels_count: int,
    hysteresis: float = 0.0,
    current_mode: int = 0,
    search_all_modes: bool = False
) -> tuple[int, int, float, int]:
    """全自動分析最佳平衡參數與曲線模式 (多模式極速版)"""
    
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0: return current_min, current_max, current_exp, current_mode

    levels = max(2, int(levels_count))
    t_white, t_gray, t_black = target
    t_total = t_white + t_gray + t_black
    if t_total <= 0: return current_min, current_max, current_exp, current_mode
    target_ratios = np.array([t_white, t_gray, t_black]) / t_total
    
    # 評估目前參數 Loss
    current_dist = distribution_from_hist(hist, current_min, current_max, levels, current_exp, current_mode)
    best_loss = np.sum((current_dist - target_ratios)**2) - hysteresis
    best_params = (current_min, current_max, current_exp, current_mode)

    def _search(lo_list, hi_list, ex_list, mode_list, b_l, b_p):
        for m in mode_list:
            for lo in lo_list:
                for hi in hi_list:
                    if hi <= lo + 2: continue
                    inv_r = 1.0 / (hi - lo)
                    x_base = np.clip((_VALUES_256 - lo) * inv_r, 0, 1)
                    
                    for ex in ex_list:
                        gamma = 2.0 ** ex
                        # 核心映射計算
                        if m == 0: mapped = x_base ** gamma
                        elif m == 1:
                            x_g = x_base ** gamma
                            mapped = x_g / (x_g + (1.0 - x_base + 1e-6)**gamma)
                        else:
                            k = (ex + 2.0) * 10.0
                            mapped = np.log(1.0 + k * x_base) / np.log(1.0 + k)
                            
                        idx = (mapped * levels).astype(np.int32)
                        idx[idx >= levels] = levels - 1
                        cnts = np.bincount(idx, weights=hist, minlength=levels)
                        dist = _fold_level_values_to_wgb(cnts) / total
                        loss = np.sum((dist - target_ratios)**2)
                        if loss < b_l:
                            b_l = loss
                            b_p = (int(lo), int(hi), float(ex), int(m))
        return b_l, b_p

    # 決定要搜尋的模式
    modes_to_search = [0, 1, 2] if search_all_modes else [current_mode]

    # 階段一：粗算
    lo_c = np.linspace(0, 240, 10, dtype=int)
    hi_c = np.linspace(20, 255, 10, dtype=int)
    ex_c = np.linspace(-1.5, 1.5, 8)
    best_loss, best_params = _search(lo_c, hi_c, ex_c, modes_to_search, best_loss, best_params)

    # 階段二：精算
    c_lo, c_hi, c_ex, c_m = best_params
    lo_f = np.linspace(max(0, c_lo - 20), min(240, c_lo + 20), 8, dtype=int)
    hi_f = np.linspace(max(c_lo + 5, c_hi - 20), min(255, c_hi + 20), 8, dtype=int)
    ex_f = np.linspace(max(-2.0, c_ex - 0.3), min(2.0, c_ex + 0.3), 8)
    best_loss, best_params = _search(lo_f, hi_f, ex_f, [c_m], best_loss, best_params)

    return best_params
