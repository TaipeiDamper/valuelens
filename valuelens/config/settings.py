from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
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
    dither_first: bool = False
    refresh_ms: int = 100
    x: int = 200
    y: int = 200
    width: int = 640
    height: int = 360


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
        merged = {**defaults, **raw}
        return AppSettings(**merged)

    def save(self, settings: AppSettings) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8"
        )


