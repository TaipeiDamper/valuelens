from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from valuelens.core.quantize import has_native_acceleration, quantize_gray_with_indices


def bench_case(shape: tuple[int, int], loops: int = 40) -> float:
    rng = np.random.default_rng(2026)
    frame = rng.integers(0, 256, size=(shape[0], shape[1], 3), dtype=np.uint8)

    start = time.perf_counter()
    for _ in range(loops):
        quantize_gray_with_indices(
            frame,
            levels=5,
            min_value=8,
            max_value=240,
            exp_value=0.1,
            display_min=0,
            display_max=255,
            display_exp=0.0,
            blur_radius=0,
            dither_strength=0,
            edge_strength=0,
            morph_enabled=False,
            morph_strength=1,
        )
    elapsed = time.perf_counter() - start
    return elapsed / loops * 1000.0


def main() -> int:
    print(f"HAS_NATIVE={has_native_acceleration()}")
    for shape in ((720, 1280), (1080, 1920)):
        ms = bench_case(shape)
        print(f"{shape[1]}x{shape[0]} avg_ms={ms:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

