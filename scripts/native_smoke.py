from __future__ import annotations

import numpy as np

from valuelens.core.quantize import has_native_acceleration, quantize_gray_with_indices


def main() -> int:
    rng = np.random.default_rng(1234)
    frame = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)

    out_a, idx_a, _ = quantize_gray_with_indices(
        frame,
        5,
        min_value=16,
        max_value=230,
        exp_value=0.2,
        display_min=0,
        display_max=255,
        display_exp=0.0,
        blur_radius=0,
        dither_strength=0,
        edge_strength=0,
        morph_enabled=False,
        morph_strength=1,
    )
    out_b, idx_b, _ = quantize_gray_with_indices(
        frame,
        5,
        min_value=16,
        max_value=230,
        exp_value=0.2,
        display_min=0,
        display_max=255,
        display_exp=0.0,
        blur_radius=0,
        dither_strength=0,
        edge_strength=0,
        morph_enabled=False,
        morph_strength=1,
    )

    same_out = np.array_equal(out_a, out_b)
    same_idx = np.array_equal(idx_a, idx_b)
    print(f"HAS_NATIVE={has_native_acceleration()}")
    print(f"out_gray_deterministic={same_out}")
    print(f"indices_deterministic={same_idx}")
    return 0 if same_out and same_idx else 1


if __name__ == "__main__":
    raise SystemExit(main())

