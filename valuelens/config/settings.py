from __future__ import annotations

import json
import os
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
    refresh_ms: int = 100
    x: int = 200
    y: int = 200
    width: int = 640
    height: int = 360
    startup_preset: dict | None = None
    presets: list[dict | None] = field(default_factory=lambda: [None] * 20)
    process_order: list[str] = ("blur", "dither", "edge", "morph")


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
            json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8"
        )


