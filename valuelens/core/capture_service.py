from __future__ import annotations

import ctypes
import sys
from typing import Optional, Tuple

import mss
import numpy as np


_WDA_NONE = 0x00
_WDA_EXCLUDEFROMCAPTURE = 0x11


class CaptureService:
    def __init__(self) -> None:
        self._sct = mss.mss()
        self._user32 = ctypes.windll.user32 if sys.platform.startswith("win") else None

    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        exclude_hwnd: Optional[int] = None,
    ) -> np.ndarray:
        monitor = {
            "left": int(x),
            "top": int(y),
            "width": max(1, int(width)),
            "height": max(1, int(height)),
        }

        excluded = self._apply_affinity(exclude_hwnd, _WDA_EXCLUDEFROMCAPTURE)
        try:
            shot = self._sct.grab(monitor)
        finally:
            if excluded:
                self._apply_affinity(exclude_hwnd, _WDA_NONE)

        img = np.array(shot, dtype=np.uint8)
        return img[:, :, :3]

    def _apply_affinity(self, hwnd: Optional[int], affinity: int) -> bool:
        if not self._user32 or hwnd is None:
            return False
        try:
            return bool(self._user32.SetWindowDisplayAffinity(int(hwnd), affinity))
        except Exception:
            return False

    @staticmethod
    def to_tuple(rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        return int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])

