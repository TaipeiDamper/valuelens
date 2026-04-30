from __future__ import annotations

import json
import os
import numpy as np
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AppSettings:
    levels: int = 3
    min_value: int = 0
    max_value: int = 255
    exp_value: float = 0.0
    display_min_value: int = 0
    display_max_value: int = 255
    display_exp_value: float = 0.0
    compare_mode: bool = False
    compare_bw: bool = False
    hotkey: str = "ctrl+alt+g"
    enabled: bool = True
    blur_enabled: bool = False
    blur_radius: int = 0
    dither_enabled: bool = False
    dither_strength: int = 0
    edge_enabled: bool = False
    edge_strength: int = 50
    edge_mix: int = 100
    edge_color: tuple[int, int, int] = (0, 0, 0)
    morph_enabled: bool = False
    morph_strength: int = 1
    morph_threshold: int = 35
    refresh_ms: int = 16
    scene_threshold: float = 10.0
    sync_timeout_s: float = 1.0
    sample_count: int = 256
    x: int = 200
    y: int = 200
    width: int = 640
    height: int = 360
    startup_preset: dict | None = None
    presets: list[dict | None] = field(default_factory=lambda: [None] * 40)
    process_order: list[str] = ("blur", "dither", "edge", "morph")
    custom_palette: list[tuple[int, int, int]] = field(default_factory=list)
    last_state: dict | None = None
    last_color_state: dict | None = None


class SettingsEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class SettingsManager:
    def __init__(self) -> None:
        appdata = Path(os.getenv("APPDATA", Path.home()))
        self._path = appdata / "ValueLens" / "settings.json"

    def load(self) -> AppSettings:
        if not self._path.exists():
            return AppSettings()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return AppSettings()

        # Ensure presets list is at least 40 entries long
        if "presets" in raw and isinstance(raw["presets"], list):
            if len(raw["presets"]) < 40:
                raw["presets"].extend([None] * (40 - len(raw["presets"])))

        defaults = asdict(AppSettings())
        # Migration Revert
        if "clean_strength" in raw:
            raw["blur_radius"] = raw.pop("clean_strength") // 2
            raw["blur_enabled"] = raw.pop("clean_enabled", True)
        if "bilateral_radius" in raw:
            raw["blur_radius"] = raw.pop("bilateral_radius")
            raw["blur_enabled"] = raw.pop("bilateral_enabled", True)
            
        # Filter only valid fields to prevent TypeError
        valid_raw = {k: v for k, v in raw.items() if k in defaults}
        merged = {**defaults, **valid_raw}
        return AppSettings(**merged)

    def save(self, settings: AppSettings) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(asdict(settings), ensure_ascii=False, indent=2, cls=SettingsEncoder), encoding="utf-8"
        )
