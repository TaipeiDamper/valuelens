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

    def bind_to_current_thread(self) -> None:
        """mss 在 Windows 上是 Thread-local 的，若由背景執行緒接管，必須呼叫此方法重新綁定。"""
        # 不要呼叫 self._sct.close()，否則會因為跨執行緒存取 thread-local variables 報錯
        self._sct = mss.mss()

    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        monitor = {
            "left": int(x),
            "top": int(y),
            "width": max(1, int(width)),
            "height": max(1, int(height)),
        }

        shot = self._sct.grab(monitor)
        # 使用更快的 frombuffer 方式，避免 np.array 的二次拷貝
        img = np.frombuffer(shot.bgra, dtype=np.uint8).reshape((shot.height, shot.width, 4))
        return img[:, :, :3]

    def set_affinity(self, hwnd: int | None, enabled: bool) -> bool:
        """設定視窗隱身屬性。enabled=True 則擷取時不可見。"""
        affinity = _WDA_EXCLUDEFROMCAPTURE if enabled else _WDA_NONE
        return self._apply_affinity(hwnd, affinity)

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

