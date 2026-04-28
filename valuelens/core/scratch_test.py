import time
import numpy as np
from balance import distribution_from_hist

def optimize_balance_params_slow(hist, target_ratios, lo_candidates, hi_candidates, exp_range, levels):
    best_loss = float('inf')
    best_params = (0, 255, 0.0)
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

def optimize_balance_params_fast(hist, target_ratios, lo_candidates, hi_candidates, exp_range, levels):
    lo_grid, hi_grid, ex_grid = np.meshgrid(lo_candidates, hi_candidates, exp_range, indexing='ij')
    valid = (lo_grid <= 240) & (hi_grid > lo_grid + 4) & (hi_grid <= 255)
    lo_valid = lo_grid[valid].astype(np.float32)
    hi_valid = hi_grid[valid].astype(np.float32)
    ex_valid = ex_grid[valid].astype(np.float32)
    
    if len(lo_valid) == 0:
        return (0, 255, 0.0)
        
    values = np.arange(256, dtype=np.float32)
    clipped = np.clip(values[None, :], lo_valid[:, None], hi_valid[:, None])
    normalized = (clipped - lo_valid[:, None]) / (hi_valid[:, None] - lo_valid[:, None])
    gamma = np.power(2.0, ex_valid)
    mapped = np.power(normalized, gamma[:, None])
    indices = np.floor(mapped * levels).astype(np.int32)
    indices = np.clip(indices, 0, levels - 1)
    
    counts = np.zeros((len(lo_valid), levels), dtype=np.float32)
    for lvl in range(levels):
        counts[:, lvl] = np.sum((indices == lvl) * hist[None, :], axis=1)
        
    totals = np.sum(counts, axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    r_now = counts / totals
    
    diff = r_now - target_ratios[None, :]
    losses = np.sum(diff * diff, axis=1)
    
    best_idx = np.argmin(losses)
    return (int(lo_valid[best_idx]), int(hi_valid[best_idx]), float(ex_valid[best_idx]))

if __name__ == '__main__':
    np.random.seed(42)
    hist = np.random.rand(256) * 100
    target_ratios = np.array([0.1, 0.2, 0.7])
    lo_candidates = np.linspace(0, 100, 31).astype(int)
    hi_candidates = np.linspace(150, 255, 31).astype(int)
    exp_range = np.linspace(-2.0, 2.0, 35)
    levels = 3
    
    t0 = time.time()
    res1 = optimize_balance_params_slow(hist, target_ratios, lo_candidates, hi_candidates, exp_range, levels)
    t1 = time.time()
    res2 = optimize_balance_params_fast(hist, target_ratios, lo_candidates, hi_candidates, exp_range, levels)
    t2 = time.time()
    
    print(f"Slow: {t1-t0:.4f}s, Res: {res1}")
    print(f"Fast: {t2-t1:.4f}s, Res: {res2}")
