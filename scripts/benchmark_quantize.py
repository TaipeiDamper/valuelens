from __future__ import annotations

import time
import sys
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from valuelens.core.quantize import has_native_acceleration, quantize_gray_with_indices


SCENARIOS = [
    ("base_5lv", {"levels": 5}),
    ("base_8lv", {"levels": 8}),
    ("blur", {"levels": 5, "blur_radius": 3}),
    ("dither", {"levels": 5, "dither_strength": 100}),
    ("edge", {"levels": 5, "edge_strength": 60, "edge_mix": 100}),
    ("morph", {"levels": 5, "morph_enabled": True, "morph_strength": 1}),
]


def bench_case(shape: tuple[int, int], params: dict, loops: int = 20) -> tuple[float, dict[str, float]]:
    rng = np.random.default_rng(2026)
    frame = rng.integers(0, 256, size=(shape[0], shape[1], 3), dtype=np.uint8)
    profile_totals: dict[str, list[float]] = {}

    start = time.perf_counter()
    for _ in range(loops):
        profile: dict[str, float] = {}
        quantize_gray_with_indices(
            frame,
            levels=params.get("levels", 5),
            min_value=8,
            max_value=240,
            exp_value=0.1,
            display_min=0,
            display_max=255,
            display_exp=0.0,
            blur_radius=params.get("blur_radius", 0),
            dither_strength=params.get("dither_strength", 0),
            edge_strength=params.get("edge_strength", 0),
            morph_enabled=params.get("morph_enabled", False),
            morph_strength=params.get("morph_strength", 1),
            edge_mix=params.get("edge_mix", 0),
            profile=profile,
        )
        for key, value in profile.items():
            profile_totals.setdefault(key, []).append(value)
    elapsed = time.perf_counter() - start
    return elapsed / loops * 1000.0, {key: mean(values) for key, values in profile_totals.items()}


def _print_profile(profile: dict[str, float]) -> None:
    ordered = [
        "prepare_gray_ms",
        "stats_ms",
        "filters_ms",
        "final_index_ms",
        "color_map_ms",
        "edge_morph_apply_ms",
        "rgb_convert_ms",
        "total_ms",
    ]
    parts = [f"{key.replace('_ms', '')}={profile.get(key, 0.0):.2f}" for key in ordered]
    print("    " + " | ".join(parts))


def main() -> int:
    print(f"HAS_NATIVE={has_native_acceleration()}")
    for shape, loops in (((720, 1280), 20), ((1080, 1920), 12), ((2160, 3840), 4)):
        print(f"\n{shape[1]}x{shape[0]} loops={loops}")
        for name, params in SCENARIOS:
            ms, profile = bench_case(shape, params, loops=loops)
            print(f"  {name:<9} avg_ms={ms:.3f}")
            _print_profile(profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

