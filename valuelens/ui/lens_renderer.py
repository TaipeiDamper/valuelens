from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtGui import QImage, QPixmap

from valuelens.core.capture_service import CaptureService
from valuelens.core.quantize import quantize_gray


class LensRenderer:
    def __init__(self, capture: CaptureService) -> None:
        self.capture = capture
        self.frame = QPixmap()
        self.raw_frame = QPixmap()
        self._last_capture_rect: tuple[int, int, int, int] | None = None
        self._last_signature: np.ndarray | None = None
        self._stable_frame_count = 0
        self._is_refreshing = False

    @property
    def is_refreshing(self) -> bool:
        return self._is_refreshing

    @property
    def stable_frame_count(self) -> int:
        return self._stable_frame_count

    def reset(self) -> None:
        self.frame = QPixmap()
        self.raw_frame = QPixmap()
        self._last_capture_rect = None
        self._last_signature = None
        self._stable_frame_count = 0

    def refresh(
        self,
        *,
        enabled: bool,
        compare_mode: bool,
        lens_rect: QRect,
        dpr: float,
        physical_window_rect: tuple[int, int, int, int] | None,
        panel_height: int,
        window_x: int,
        window_y: int,
        hwnd: int | None,
        levels: int,
        min_value: int,
        max_value: int,
        exp_value: float,
    ) -> tuple[bool, bool]:
        """Returns (updated, capture_rect_changed)."""
        if self._is_refreshing:
            return False, False

        self._is_refreshing = True
        try:
            if not enabled:
                changed = not self.frame.isNull() or not self.raw_frame.isNull()
                if changed:
                    self.reset()
                return changed, False

            if lens_rect.width() <= 0 or lens_rect.height() <= 0:
                return False, False

            if physical_window_rect is not None:
                win_x, win_y, win_pw, win_ph = physical_window_rect
                panel_h_phys = int(round(panel_height * dpr))
                cap_x = win_x
                cap_y = win_y + panel_h_phys
                cap_w = win_pw
                cap_h = max(1, win_ph - panel_h_phys)
            else:
                cap_x = int(round((window_x + lens_rect.x()) * dpr))
                cap_y = int(round((window_y + lens_rect.y()) * dpr))
                cap_w = max(1, int(round(lens_rect.width() * dpr)))
                cap_h = max(1, int(round(lens_rect.height() * dpr)))

            capture_rect = (cap_x, cap_y, cap_w, cap_h)
            capture_rect_changed = capture_rect != self._last_capture_rect
            self._last_capture_rect = capture_rect

            frame = self.capture.capture_region(
                cap_x, cap_y, cap_w, cap_h, exclude_hwnd=hwnd
            )

            signature = self._frame_signature(frame)
            if (
                self._last_signature is not None
                and signature.shape == self._last_signature.shape
                and np.array_equal(signature, self._last_signature)
                and not self.frame.isNull()
            ):
                self._stable_frame_count += 1
                return False, capture_rect_changed

            self._stable_frame_count = 0
            self._last_signature = signature

            quantized = quantize_gray(frame, levels, min_value, max_value, exp_value)
            h, w, _ = quantized.shape
            rgb = cv2.cvtColor(quantized, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
            qimg.setDevicePixelRatio(dpr)
            self.frame = QPixmap.fromImage(qimg)

            if compare_mode:
                raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_qimg = QImage(
                    raw_rgb.data, w, h, raw_rgb.strides[0], QImage.Format.Format_RGB888
                ).copy()
                raw_qimg.setDevicePixelRatio(dpr)
                self.raw_frame = QPixmap.fromImage(raw_qimg)
            else:
                self.raw_frame = QPixmap()

            return True, capture_rect_changed
        finally:
            self._is_refreshing = False

    @staticmethod
    def _frame_signature(frame: np.ndarray) -> np.ndarray:
        if frame.size == 0:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return frame[::8, ::8].copy()


