from __future__ import annotations

import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from valuelens.core.balance import distribution_from_hist, optimize_balance_params


def bench_case(shape: tuple[int, int], search_all_modes: bool, loops: int) -> tuple[float, tuple[int, int, float, int], float]:
    rng = np.random.default_rng(2026)
    gray = rng.integers(0, 256, size=shape, dtype=np.uint8)
    target = (70.0, 20.0, 10.0)
    times: list[float] = []
    best = (0, 255, 0.0, 0)

    for _ in range(loops):
        start = time.perf_counter()
        best = optimize_balance_params(
            gray,
            target=target,
            current_min=0,
            current_max=255,
            current_exp=0.0,
            levels_count=5,
            hysteresis=0.0,
            current_mode=0,
            search_all_modes=search_all_modes,
        )
        times.append((time.perf_counter() - start) * 1000.0)

    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    dist = distribution_from_hist(hist, best[0], best[1], 5, best[2], best[3])
    target_ratios = np.array(target, dtype=np.float64) / sum(target)
    loss = float(np.sum((dist - target_ratios) ** 2))
    return mean(times), best, loss


def main() -> int:
    for shape, loops in (((720, 1280), 12), ((1080, 1920), 8), ((2160, 3840), 3)):
        print(f"\n{shape[1]}x{shape[0]} loops={loops}")
        for search_all in (False, True):
            ms, best, loss = bench_case(shape, search_all, loops)
            mode = "all_modes" if search_all else "current_mode"
            print(f"  {mode:<12} avg_ms={ms:.3f} best={best} loss={loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
