from __future__ import annotations

import ctypes
import os
import sys
import time
from typing import Optional, Tuple

import cv2
import mss
import numpy as np


_WDA_NONE = 0x00
_WDA_EXCLUDEFROMCAPTURE = 0x11


class CaptureService:
    def __init__(self) -> None:
        self._requested_backend = os.getenv("VALUELENS_CAPTURE_BACKEND", "auto").lower()
        self._sct = mss.mss()
        self._bettercam = None
        self._bettercam_region = None
        self._last_bettercam_frame = None
        self._active_backend = "mss"
        self._backend_initialized = False
        self._user32 = ctypes.windll.user32 if sys.platform.startswith("win") else None

    def bind_to_current_thread(self) -> None:
        """Capture backends keep thread-local/native resources, so bind them inside the worker thread."""
        # 不要呼叫 self._sct.close()，否則會因為跨執行緒存取 thread-local variables 報錯
        self._sct = mss.mss()
        self._bettercam = None
        self._bettercam_region = None
        self._last_bettercam_frame = None
        self._active_backend = "mss"
        self._backend_initialized = False
        self._init_preferred_backend()

    @property
    def active_backend(self) -> str:
        return self._active_backend

    def _init_preferred_backend(self) -> None:
        self._backend_initialized = True
        if self._requested_backend not in {"auto", "bettercam"}:
            return
        if not sys.platform.startswith("win"):
            return
        try:
            import bettercam

            self._bettercam = bettercam.create(output_color="BGR")
            self._active_backend = "bettercam"
            print("[Capture] backend=bettercam")
        except Exception as exc:
            if self._requested_backend == "bettercam":
                print(f"[Capture] bettercam unavailable, fallback=mss ({exc})")
            self._bettercam = None
            self._active_backend = "mss"

    def _fallback_to_mss(self, exc: Exception) -> None:
        if self._active_backend == "bettercam":
            print(f"[Capture] bettercam failed, fallback=mss ({exc})")
        self._bettercam = None
        self._bettercam_region = None
        self._last_bettercam_frame = None
        self._active_backend = "mss"
        self._backend_initialized = True

    def close(self) -> None:
        stop = getattr(self._bettercam, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:
                pass
        self._bettercam = None
        self._bettercam_region = None
        self._last_bettercam_frame = None
        try:
            self._sct.close()
        except Exception:
            pass

    def _grab_bgra(
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
        return np.frombuffer(shot.bgra, dtype=np.uint8).reshape((shot.height, shot.width, 4))

    def _grab_bgr(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        if not self._backend_initialized:
            self._init_preferred_backend()

        if self._active_backend == "bettercam" and self._bettercam is not None:
            region = (
                int(x),
                int(y),
                int(x) + max(1, int(width)),
                int(y) + max(1, int(height)),
            )
            try:
                if self._bettercam_region != region:
                    self._bettercam_region = region
                    self._last_bettercam_frame = None

                frame = self._bettercam.grab(region=region)
                if frame is not None:
                    self._last_bettercam_frame = np.ascontiguousarray(frame[:, :, :3])
                    return self._last_bettercam_frame
                if self._last_bettercam_frame is not None:
                    return self._last_bettercam_frame
                time.sleep(0.002)
                frame = self._bettercam.grab(region=region)
                if frame is not None:
                    self._last_bettercam_frame = np.ascontiguousarray(frame[:, :, :3])
                    return self._last_bettercam_frame
                raise RuntimeError("no frame returned")
            except Exception as exc:
                self._fallback_to_mss(exc)

        bgra = self._grab_bgra(x, y, width, height)
        return np.ascontiguousarray(bgra[:, :, :3])

    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Capture BGR frame as contiguous array for downstream OpenCV usage."""
        return self._grab_bgr(x, y, width, height)

    def capture_region_bgra(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Capture BGRA frame (view on MSS buffer) to avoid extra conversion copies."""
        if self._active_backend == "bettercam":
            bgr = self._grab_bgr(x, y, width, height)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        return self._grab_bgra(x, y, width, height)

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

