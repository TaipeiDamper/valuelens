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


def _empty_buckets() -> dict[str, list[float]]:
    return {
        "grab_ms": [],
        "buffer_ms": [],
        "bgr_ms": [],
        "gray_ms": [],
        "scene_ms": [],
        "total_ms": [],
    }


def _average_buckets(buckets: dict[str, list[float]]) -> dict[str, float]:
    return {name: mean(values) if values else 0.0 for name, values in buckets.items()}


def bench_capture_mss(shape: tuple[int, int], loops: int) -> dict[str, float]:
    h, w = shape
    monitor = {"left": 100, "top": 100, "width": w, "height": h}
    detector = RandomSceneDetector(threshold=10.0, sample_count=1024)
    buckets = _empty_buckets()

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

    return _average_buckets(buckets)


def _bench_dx_style_backend(module_name: str, shape: tuple[int, int], loops: int) -> dict[str, float] | str:
    try:
        module = __import__(module_name)
    except Exception as exc:
        return f"SKIP: {module_name} not importable ({exc})"

    if not hasattr(module, "create"):
        return f"SKIP: {module_name}.create is unavailable"

    h, w = shape
    region = (100, 100, 100 + w, 100 + h)
    detector = RandomSceneDetector(threshold=10.0, sample_count=1024)
    buckets = _empty_buckets()

    try:
        camera = module.create(output_color="BGR")
    except Exception as exc:
        return f"SKIP: {module_name}.create failed ({exc})"

    try:
        for _ in range(loops):
            t0 = time.perf_counter()
            frame = camera.grab(region=region)
            t1 = time.perf_counter()
            if frame is None:
                continue

            bgr = np.ascontiguousarray(frame[:, :, :3])
            t2 = time.perf_counter()
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            t3 = time.perf_counter()
            detector.detect_change(gray)
            t4 = time.perf_counter()

            buckets["grab_ms"].append((t1 - t0) * 1000.0)
            buckets["buffer_ms"].append(0.0)
            buckets["bgr_ms"].append((t2 - t1) * 1000.0)
            buckets["gray_ms"].append((t3 - t2) * 1000.0)
            buckets["scene_ms"].append((t4 - t3) * 1000.0)
            buckets["total_ms"].append((t4 - t0) * 1000.0)
    except Exception as exc:
        return f"SKIP: {module_name} benchmark failed ({exc})"
    finally:
        stop = getattr(camera, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:
                pass

    if not buckets["total_ms"]:
        return f"SKIP: {module_name} returned no frames"
    return _average_buckets(buckets)


def _print_stats(label: str, stats: dict[str, float] | str) -> None:
    if isinstance(stats, str):
        print(f"  {label:<9} {stats}")
        return
    print(
        f"  {label:<9} "
        f"grab={stats['grab_ms']:.2f}ms | "
        f"buffer={stats['buffer_ms']:.2f}ms | "
        f"bgr={stats['bgr_ms']:.2f}ms | "
        f"gray={stats['gray_ms']:.2f}ms | "
        f"scene={stats['scene_ms']:.2f}ms | "
        f"total={stats['total_ms']:.2f}ms"
    )


def main() -> int:
    for shape, loops in (((720, 1280), 40), ((1080, 1920), 24), ((2160, 3840), 8)):
        print(f"\n{shape[1]}x{shape[0]} loops={loops}")
        _print_stats("mss", bench_capture_mss(shape, loops))
        _print_stats("dxcam", _bench_dx_style_backend("dxcam", shape, loops))
        _print_stats("bettercam", _bench_dx_style_backend("bettercam", shape, loops))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
