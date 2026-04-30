from __future__ import annotations

import sys
import time
from pathlib import Path
from statistics import mean

import cv2
import mss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from valuelens.core.scene_detector import RandomSceneDetector


def bench_capture(shape: tuple[int, int], loops: int) -> dict[str, float]:
    h, w = shape
    monitor = {"left": 100, "top": 100, "width": w, "height": h}
    detector = RandomSceneDetector(threshold=10.0, sample_count=1024)

    buckets: dict[str, list[float]] = {
        "grab_ms": [],
        "buffer_ms": [],
        "bgr_ms": [],
        "gray_ms": [],
        "scene_ms": [],
        "total_ms": [],
    }

    with mss.mss() as sct:
        for _ in range(loops):
            t0 = time.perf_counter()
            shot = sct.grab(monitor)
            t1 = time.perf_counter()

            bgra = np.frombuffer(shot.bgra, dtype=np.uint8).reshape((shot.height, shot.width, 4))
            t2 = time.perf_counter()

            # Materialize contiguous BGR (same flow as live source).
            bgr = np.ascontiguousarray(bgra[:, :, :3])
            t3 = time.perf_counter()

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            t4 = time.perf_counter()

            detector.detect_change(gray)
            t5 = time.perf_counter()

            buckets["grab_ms"].append((t1 - t0) * 1000.0)
            buckets["buffer_ms"].append((t2 - t1) * 1000.0)
            buckets["bgr_ms"].append((t3 - t2) * 1000.0)
            buckets["gray_ms"].append((t4 - t3) * 1000.0)
            buckets["scene_ms"].append((t5 - t4) * 1000.0)
            buckets["total_ms"].append((t5 - t0) * 1000.0)

    return {name: mean(values) for name, values in buckets.items()}


def main() -> int:
    for shape, loops in (((720, 1280), 40), ((1080, 1920), 24), ((2160, 3840), 8)):
        stats = bench_capture(shape, loops)
        print(f"\n{shape[1]}x{shape[0]} loops={loops}")
        print(
            "  "
            f"grab={stats['grab_ms']:.2f}ms | "
            f"buffer={stats['buffer_ms']:.2f}ms | "
            f"bgr={stats['bgr_ms']:.2f}ms | "
            f"gray={stats['gray_ms']:.2f}ms | "
            f"scene={stats['scene_ms']:.2f}ms | "
            f"total={stats['total_ms']:.2f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
