from __future__ import annotations

import ctypes
import itertools
import sys
import time
from ctypes import wintypes

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QMainWindow, QToolButton

from valuelens.config.settings import AppSettings, SettingsManager
from valuelens.core.capture_service import CaptureService
from valuelens.core.hotkey_service import HotkeyService
from valuelens.core.quantize import quantize_gray, quantize_gray_with_indices
from valuelens.modes.image_mode import ImageModeDialog
from valuelens.ui.control_panel import ControlPanel


_RESIZE_MARGIN = 12
_MIN_WIDTH = 320
_MIN_HEIGHT = 160

_EDGE_LEFT = 1
_EDGE_RIGHT = 2
_EDGE_TOP = 4
_EDGE_BOTTOM = 8

_RANGE_PRESETS: tuple[tuple[int, int], ...] = (
    (0, 255),
    (0, 248),
    (0, 224),
    (0, 208),
    (0, 192),
    (8, 255),
    (16, 240),
    (16, 224),
    (24, 255),
    (24, 224),
    (32, 224),
    (32, 208),
    (40, 208),
    (48, 240),
    (56, 224),
    (64, 255),
    (72, 240),
    (80, 224),
    (88, 248),
    (96, 240),
    (104, 248),
    (112, 255),
)
_EXP_PRESETS: tuple[float, ...] = (-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
_EXP_COARSE_GRID: tuple[float, ...] = (
    -2.0,
    -1.5,
    -1.0,
    -0.75,
    -0.5,
    -0.25,
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
)


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class OverlayWindow(QMainWindow):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings
        self.settings_manager = SettingsManager()
        self.capture = CaptureService()
        self.hotkeys = HotkeyService()
        self.image_mode = ImageModeDialog(
            levels=settings.levels,
            min_value=settings.min_value,
            max_value=settings.max_value,
            exp_value=settings.exp_value,
            blur_enabled=settings.blur_enabled,
            blur_radius=settings.blur_radius,
            dither_enabled=settings.dither_enabled,
            dither_strength=settings.dither_strength,
            dither_first=getattr(settings, 'dither_first', False),
            parent=self,
        )
        self.image_mode.set_import_callback(self._get_current_raw_frame)

        self.setGeometry(settings.x, settings.y, settings.width, settings.height)
        self.setMinimumSize(_MIN_WIDTH, _MIN_HEIGHT)
        self.setWindowTitle("ValueLens")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Window
        )
        self.setMouseTracking(True)
        central = self.centralWidget()
        if central is not None:
            central.setMouseTracking(True)

        self._drag_pos: QPoint | None = None
        self._resize_edges = 0
        self._resize_start_geom: QRect | None = None
        self._resize_start_global: QPoint | None = None
        self._frame = QPixmap()
        self._raw_frame = QPixmap()
        self._panel_height = 136
        self._compare_gap = 6
        self._last_toggle_ts = 0.0
        self._is_dragging = False
        self._is_resizing = False
        self._is_refreshing = False
        self._active_refresh_ms = max(80, int(self.settings.refresh_ms))
        self._idle_refresh_ms = 2000
        self._motion_boost_until = 0.0
        self._last_capture_rect: tuple[int, int, int, int] | None = None
        self._last_frame_signature: bytes | None = None
        self._stable_frame_count = 0
        self._compare_mode = bool(settings.compare_mode)
        self._processed_distribution_pct = [0.0] * max(2, int(settings.levels))
        self._raw_distribution_pct = [0.0] * max(2, int(settings.levels))
        self._last_gray_frame: np.ndarray | None = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)
        self.timer.start(self._active_refresh_ms)

        self._coalesce_timer = QTimer(self)
        self._coalesce_timer.setSingleShot(True)
        self._coalesce_timer.timeout.connect(self.refresh_frame)

        self.panel = ControlPanel(settings=self.settings, parent=self)
        self.panel.settings_changed.connect(self.on_settings_changed)
        self.panel.display_settings_changed.connect(self.on_display_settings_changed)
        self.panel.effect_settings_changed.connect(self.on_effect_settings_changed)
        self.panel.collapse_toggled.connect(self.on_collapse_toggled)
        self.panel.compare_mode_changed.connect(self.on_compare_mode_changed)
        self.panel.hotkey_changed.connect(self.on_hotkey_changed)
        self.panel.auto_balance_raw_requested.connect(self.on_auto_balance_raw_requested)
        self.panel.auto_balance_target_requested.connect(self.on_auto_balance_target_requested)
        self.panel.image_mode_requested.connect(self.open_image_mode)
        self.panel.quit_requested.connect(self.force_quit)
        self.panel.minimize_requested.connect(self.showMinimized)
        self.panel.drag_started.connect(self._start_drag_from_panel)
        self.panel.show()
        self.panel.raise_()
        self._layout_panel()

        self.btn_compare_bw = QToolButton(self)
        self.btn_compare_bw.setText("對照")
        self.btn_compare_bw.setCheckable(True)
        self.btn_compare_bw.setChecked(self.settings.compare_bw)
        self.btn_compare_bw.setToolTip("將對比圖轉換為純灰階顯示")
        self.btn_compare_bw.setStyleSheet("background: rgba(0,0,0,150); color: white; border-radius: 3px; padding: 2px 6px;")
        self.btn_compare_bw.toggled.connect(self.on_compare_bw_changed)
        self.btn_compare_bw.hide()

        self.hotkeys.register("toggle", self.settings.hotkey, self.toggle_enabled)
        self.hotkeys.register("quit", "ctrl+alt+shift+q", self.force_quit)
        self.request_refresh()

    def _get_current_raw_frame(self) -> np.ndarray | None:
        if hasattr(self, '_last_raw_bgr_frame'):
            return self._last_raw_bgr_frame
        return None

    def open_image_mode(self) -> None:
        self.image_mode.set_quantize_settings(
            self.settings.levels,
            self.settings.min_value,
            self.settings.max_value,
            self.settings.exp_value,
            self.settings.blur_enabled,
            self.settings.blur_radius,
            self.settings.dither_enabled,
            self.settings.dither_strength,
            getattr(self.settings, 'dither_first', False),
        )
        self.image_mode.show()
        self.image_mode.raise_()

    def on_settings_changed(
        self, levels: int, min_value: int, max_value: int, exp_value: float
    ) -> None:
        self.settings.levels = levels
        self.settings.min_value = min_value
        self.settings.max_value = max_value
        self.settings.exp_value = exp_value
        self._processed_distribution_pct = [0.0] * max(2, int(levels))
        self._raw_distribution_pct = [0.0] * max(2, int(levels))
        self.image_mode.set_quantize_settings(
            levels, min_value, max_value, exp_value,
            self.settings.blur_enabled, self.settings.blur_radius,
            self.settings.dither_enabled, self.settings.dither_strength,
            getattr(self.settings, 'dither_first', False)
        )
        self._last_frame_signature = None
        self._boost_motion(2.0)
        self.request_refresh()

    def on_display_settings_changed(self, min_value: int, max_value: int, exp_value: float) -> None:
        self.settings.display_min_value = min_value
        self.settings.display_max_value = max_value
        self.settings.display_exp_value = exp_value
        self._last_frame_signature = None
        self._boost_motion(1.0)
        self.request_refresh()

    def on_effect_settings_changed(self, blur_enabled: bool, blur_radius: int, dither_enabled: bool, dither_strength: int, dither_first: bool) -> None:
        self.settings.blur_enabled = blur_enabled
        self.settings.blur_radius = blur_radius
        self.settings.dither_enabled = dither_enabled
        self.settings.dither_strength = dither_strength
        self.settings.dither_first = dither_first
        self.image_mode.set_quantize_settings(
            self.settings.levels, self.settings.min_value, self.settings.max_value, self.settings.exp_value,
            blur_enabled, blur_radius, dither_enabled, dither_strength, dither_first
        )
        self._last_frame_signature = None
        self._boost_motion(1.0)
        self.request_refresh()

    def on_collapse_toggled(self, collapsed: bool) -> None:
        self._panel_height = 36 if collapsed else 136
        self._layout_panel()
        self.request_refresh(10)

    def on_compare_mode_changed(self, enabled: bool) -> None:
        self._compare_mode = enabled
        self.settings.compare_mode = enabled
        self._last_frame_signature = None
        self._raw_frame = QPixmap()
        self._layout_overlay_buttons()
        self._boost_motion(1.0)
        self.request_refresh()

    def on_compare_bw_changed(self, bw_enabled: bool) -> None:
        self.settings.compare_bw = bw_enabled
        self._last_frame_signature = None
        self._raw_frame = QPixmap()
        self._boost_motion(1.0)
        self.request_refresh()

    def on_hotkey_changed(self, hotkey: str) -> None:
        self.settings.hotkey = hotkey
        self.hotkeys.register("toggle", hotkey, self.toggle_enabled)

    def on_auto_balance_target_requested(self, ratios: tuple[float, float, float]) -> None:
        if self._last_gray_frame is None or self._last_gray_frame.size == 0:
            return
        # UI preset order is white:gray:black, optimizer expects black:gray:white.
        target = (ratios[2], ratios[1], ratios[0])
        lower, upper, exp_value = self._optimize_balance_params(
            self._last_gray_frame,
            target,
            self.settings.min_value,
            self.settings.max_value,
            self.settings.exp_value,
        )
        self.panel.exp_slider.setValue(int(round(-exp_value * 100.0)))
        self.panel.range_slider.set_values(lower, upper)
        self._boost_motion(1.0)
        self.request_refresh()

    def on_auto_balance_raw_requested(self) -> None:
        raw_wgb = self._levels_to_wgb(self._raw_distribution_pct)
        target_wgb = self.panel.nearest_balance_preset(raw_wgb)
        self.panel.set_balance_preset(target_wgb, mark_best=True)
        self.on_auto_balance_target_requested(target_wgb)

    def toggle_enabled(self) -> None:
        now = time.monotonic()
        if now - self._last_toggle_ts < 0.25:
            return
        self._last_toggle_ts = now
        self.settings.enabled = not self.settings.enabled
        self.panel.enabled_check.setChecked(self.settings.enabled)
        self._last_frame_signature = None
        self.request_refresh()

    def force_quit(self) -> None:
        self.close()

    def _layout_panel(self) -> None:
        self.panel.setGeometry(0, 0, self.width(), self._panel_height)
        if not self.panel.isVisible():
            self.panel.show()
        self.panel.raise_()

    def _layout_overlay_buttons(self) -> None:
        if self._compare_mode and hasattr(self, 'btn_compare_bw'):
            c = self._compare_rect()
            if not c.isNull():
                self.btn_compare_bw.setGeometry(c.right() - 60, c.top() + 6, 54, 24)
                self.btn_compare_bw.show()
                self.btn_compare_bw.raise_()
        elif hasattr(self, 'btn_compare_bw'):
            self.btn_compare_bw.hide()

    def _lens_rect(self) -> QRect:
        lens_h = max(40, self.height() - self._panel_height)
        if self._compare_mode:
            half = max(80, max(0, self.width() - self._compare_gap) // 2)
            return QRect(0, self._panel_height, half, lens_h)
        return QRect(0, self._panel_height, self.width(), lens_h)

    def _compare_rect(self) -> QRect:
        if not self._compare_mode:
            return QRect()
        lens = self._lens_rect()
        return QRect(
            lens.right() + 1 + self._compare_gap,
            lens.y(),
            lens.width(),
            lens.height(),
        )

    def request_refresh(self, delay_ms: int = 33) -> None:
        if not self._coalesce_timer.isActive():
            self._coalesce_timer.start(delay_ms)

    def _boost_motion(self, seconds: float) -> None:
        self._motion_boost_until = max(self._motion_boost_until, time.monotonic() + seconds)
        self._stable_frame_count = 0
        if self.timer.interval() != self._active_refresh_ms:
            self.timer.setInterval(self._active_refresh_ms)

    def _on_timer_tick(self) -> None:
        now = time.monotonic()
        in_active_window = now < self._motion_boost_until
        target_ms = self._active_refresh_ms if in_active_window else self._idle_refresh_ms
        if self.timer.interval() != target_ms:
            self.timer.setInterval(target_ms)
        self.request_refresh(0 if in_active_window else 20)

    def _hwnd(self) -> int | None:
        try:
            return int(self.winId())
        except Exception:
            return None

    def _physical_window_rect(self) -> tuple[int, int, int, int] | None:
        if not sys.platform.startswith("win"):
            return None
        hwnd = self._hwnd()
        if hwnd is None:
            return None
        try:
            rect = _RECT()
            if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return None
            return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
        except Exception:
            return None

    def refresh_frame(self) -> None:
        if self._is_refreshing or self._is_dragging or self._is_resizing:
            return
        self._is_refreshing = True
        try:
            lens = self._lens_rect()
            if lens.width() <= 0 or lens.height() <= 0:
                return

            dpr = self.devicePixelRatioF()
            phys = self._physical_window_rect()
            if phys is not None:
                win_x, win_y, win_pw, win_ph = phys
                panel_h_phys = int(round(self._panel_height * dpr))
                cap_x = win_x
                cap_y = win_y + panel_h_phys
                cap_w = win_pw
                cap_h = max(1, win_ph - panel_h_phys)
            else:
                cap_x = int(round((self.x() + lens.x()) * dpr))
                cap_y = int(round((self.y() + lens.y()) * dpr))
                cap_w = max(1, int(round(lens.width() * dpr)))
                cap_h = max(1, int(round(lens.height() * dpr)))

            capture_rect = (cap_x, cap_y, cap_w, cap_h)
            if capture_rect != self._last_capture_rect:
                self._boost_motion(1.0)
                self._last_capture_rect = capture_rect

            frame = self.capture.capture_region(
                cap_x, cap_y, cap_w, cap_h, exclude_hwnd=self._hwnd()
            )
            self._last_raw_bgr_frame = frame.copy()
            self._last_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            signature = self._frame_signature(frame)
            if signature == self._last_frame_signature and not self._frame.isNull():
                self._stable_frame_count += 1
                if self._stable_frame_count >= 3:
                    self._motion_boost_until = 0.0
                return
            self._stable_frame_count = 0
            self._last_frame_signature = signature

            eff_blur = self.settings.blur_radius if self.settings.blur_enabled else 0
            eff_dither = self.settings.dither_strength if self.settings.dither_enabled else 0

            logic_quantized, logic_indices = quantize_gray_with_indices(
                frame,
                self.settings.levels,
                self.settings.min_value,
                self.settings.max_value,
                self.settings.exp_value,
                fixed_output_levels=True,
                blur_radius=eff_blur,
                dither_strength=eff_dither,
                dither_first=getattr(self.settings, 'dither_first', False),
            )
            # Display stage is applied on top of logic output.
            display_quantized = quantize_gray(
                logic_quantized,
                self.settings.levels,
                self.settings.display_min_value,
                self.settings.display_max_value,
                self.settings.display_exp_value,
                fixed_output_levels=False,
            )
            h, w, _ = display_quantized.shape
            rgb = cv2.cvtColor(display_quantized, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
            qimg.setDevicePixelRatio(dpr)
            self._frame = QPixmap.fromImage(qimg)
            self._update_distributions(self._last_gray_frame, logic_indices)
            if self._compare_mode:
                if self.settings.compare_bw:
                    raw_rgb = cv2.cvtColor(self._last_gray_frame, cv2.COLOR_GRAY2RGB)
                else:
                    raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_qimg = QImage(
                    raw_rgb.data, w, h, raw_rgb.strides[0], QImage.Format.Format_RGB888
                ).copy()
                raw_qimg.setDevicePixelRatio(dpr)
                self._raw_frame = QPixmap.fromImage(raw_qimg)
                self.update(self.rect())
            else:
                self._raw_frame = QPixmap()
                self.update(lens)
        finally:
            self._is_refreshing = False

    @staticmethod
    def _frame_signature(frame: np.ndarray) -> bytes:
        if frame.size == 0:
            return b""
        sampled = frame[::8, ::8]
        return sampled.tobytes()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        lens = self._lens_rect()
        compare = self._compare_rect()
        painter.setClipRect(self.rect())
        if self._frame.isNull():
            painter.fillRect(lens, Qt.GlobalColor.black)
        else:
            painter.drawPixmap(lens.topLeft(), self._frame)
        if self._compare_mode and not compare.isNull():
            if self._raw_frame.isNull():
                painter.fillRect(compare, Qt.GlobalColor.black)
            else:
                painter.drawPixmap(compare.topLeft(), self._raw_frame)
            painter.setPen(Qt.GlobalColor.darkGray)
            divider_x = lens.right() + (self._compare_gap // 2) + 1
            painter.drawLine(divider_x, lens.y(), divider_x, lens.bottom())
        painter.setPen(Qt.GlobalColor.white)
        painter.drawRect(lens.adjusted(0, 0, -1, -1))
        self._draw_distribution_overlay(painter, lens, self._processed_distribution_pct, "處理")
        if self._compare_mode and not compare.isNull():
            painter.drawRect(compare.adjusted(0, 0, -1, -1))
            self._draw_distribution_overlay(painter, compare, self._raw_distribution_pct, "原始")

    def _calc_balance_distribution(self, gray: np.ndarray | None) -> tuple[float, float, float]:
        if gray is None or gray.size == 0:
            return (0.0, 0.0, 0.0)
        low = float(np.mean(gray < self.settings.min_value) * 100.0)
        mid = float(
            np.mean((gray >= self.settings.min_value) & (gray <= self.settings.max_value)) * 100.0
        )
        high = max(0.0, 100.0 - low - mid)
        return (low, mid, high)

    @staticmethod
    def _calc_level_distribution(gray: np.ndarray | None, levels: int) -> list[float]:
        level_count = max(2, int(levels))
        if gray is None or gray.size == 0:
            return [0.0] * level_count
        scaled = np.floor(gray.astype(np.float32) * level_count / 256.0)
        indices = np.clip(scaled, 0, level_count - 1).astype(np.int32)
        counts = np.bincount(indices.reshape(-1), minlength=level_count).astype(np.float64)
        total = float(counts.sum())
        if total <= 0:
            return [0.0] * level_count
        return ((counts / total) * 100.0).tolist()

    def _update_distributions(
        self, raw_gray: np.ndarray | None, processed_indices: np.ndarray | None
    ) -> None:
        level_count = max(2, int(self.settings.levels))
        self._raw_distribution_pct = self._calc_level_distribution(raw_gray, level_count)
        self._processed_distribution_pct = self._calc_indices_distribution(
            processed_indices, level_count
        )

    @staticmethod
    def _calc_indices_distribution(indices: np.ndarray | None, levels: int) -> list[float]:
        level_count = max(2, int(levels))
        if indices is None or indices.size == 0:
            return [0.0] * level_count
        clamped = np.clip(indices.astype(np.int32), 0, level_count - 1)
        counts = np.bincount(clamped.reshape(-1), minlength=level_count).astype(np.float64)
        total = float(counts.sum())
        if total <= 0:
            return [0.0] * level_count
        return ((counts / total) * 100.0).tolist()

    @staticmethod
    def _levels_to_wgb(values: list[float]) -> tuple[float, float, float]:
        n = max(2, len(values))
        edges = np.linspace(0, n, 4, dtype=np.float64)
        e1 = int(np.floor(edges[1]))
        e2 = int(np.floor(edges[2]))
        black = float(sum(values[:e1]))
        gray = float(sum(values[e1:e2]))
        white = float(sum(values[e2:]))
        return (white, gray, black)

    @staticmethod
    def _closest_target_ratios(
        base_ratios: tuple[float, float, float], current: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        candidates = list(itertools.permutations(base_ratios))
        return min(
            candidates,
            key=lambda cand: sum((float(cand[i]) - float(current[i])) ** 2 for i in range(3)),
        )

    def _optimize_balance_params(
        self,
        gray: np.ndarray,
        target: tuple[float, float, float],
        current_min: int,
        current_max: int,
        current_exp: float,
    ) -> tuple[int, int, float]:
        hist = np.bincount(gray.reshape(-1).astype(np.uint8), minlength=256).astype(np.float64)
        total = float(hist.sum())
        if total <= 0:
            return current_min, current_max, current_exp

        target_total = float(sum(target))
        if target_total <= 0:
            return current_min, current_max, current_exp
        target_ratios = np.array([v / target_total for v in target], dtype=np.float64)
        levels = max(2, int(self.settings.levels))

        def project(params: np.ndarray) -> np.ndarray:
            lo = float(np.clip(params[0], 0.0, 254.0))
            hi = float(np.clip(params[1], lo + 1.0, 255.0))
            exp = float(np.clip(params[2], -2.0, 2.0))
            return np.array([lo, hi, exp], dtype=np.float64)

        def loss(params: np.ndarray) -> float:
            p = project(params)
            ratios_now = self._distribution_from_hist(
                hist,
                int(round(p[0])),
                int(round(p[1])),
                levels,
                float(p[2]),
            )
            diff = ratios_now - target_ratios
            return float(np.dot(diff, diff))

        best_loss = float("inf")
        best = np.array([current_min, current_max, current_exp], dtype=np.float64)
        for lo, hi in _RANGE_PRESETS:
            for exp in _EXP_PRESETS:
                cand = project(np.array([lo, hi, exp], dtype=np.float64))
                cand_loss = loss(cand)
                if cand_loss < best_loss:
                    best_loss = cand_loss
                    best = cand

        # Broaden search with coarse grid so balance has visible effect.
        for lo in range(0, 255, 16):
            for hi in range(lo + 8, 256, 8):
                for exp in _EXP_COARSE_GRID:
                    cand = project(np.array([lo, hi, exp], dtype=np.float64))
                    cand_loss = loss(cand)
                    if cand_loss < best_loss:
                        best_loss = cand_loss
                        best = cand

        x = best.copy()
        steps = np.array([2.0, 2.0, 0.06], dtype=np.float64)
        for _ in range(10):
            base = loss(x)
            grad = np.zeros(3, dtype=np.float64)
            hdiag = np.zeros(3, dtype=np.float64)
            for i in range(3):
                d = np.zeros(3, dtype=np.float64)
                d[i] = steps[i]
                fp = loss(project(x + d))
                fm = loss(project(x - d))
                grad[i] = (fp - fm) / (2.0 * steps[i])
                hdiag[i] = (fp - 2.0 * base + fm) / (steps[i] * steps[i])

            denom = np.abs(hdiag) + 1e-4
            step_vec = grad / denom
            improved = False
            for alpha in (1.0, 0.5, 0.25, 0.1):
                cand = project(x - alpha * step_vec)
                cand_loss = loss(cand)
                if cand_loss + 1e-10 < base:
                    x = cand
                    improved = True
                    break
            if not improved:
                break
            if np.linalg.norm(step_vec) < 0.25:
                break

        out = project(x)
        return int(round(out[0])), int(round(out[1])), float(out[2])

    @staticmethod
    def _distribution_from_hist(
        hist: np.ndarray, lower: int, upper: int, levels: int, exp_value: float
    ) -> np.ndarray:
        levels = max(2, int(levels))
        lower = max(0, min(254, int(lower)))
        upper = max(lower + 1, min(255, int(upper)))
        values = np.arange(256, dtype=np.float64)
        clipped = np.clip(values, lower, upper)
        normalized = (clipped - float(lower)) / float(upper - lower)
        gamma = float(np.exp(float(exp_value)))
        mapped = np.power(normalized, gamma)
        indices = np.floor(mapped * levels)
        indices = np.clip(indices, 0, levels - 1).astype(np.int32)
        counts = np.bincount(indices, weights=hist, minlength=levels).astype(np.float64)
        total = float(counts.sum())
        if total <= 0:
            return np.zeros(3, dtype=np.float64)
        level_edges = np.linspace(0, levels, 4, dtype=np.float64)
        edge1 = int(np.floor(level_edges[1]))
        edge2 = int(np.floor(level_edges[2]))
        low = float(np.sum(counts[:edge1]) / total)
        mid = float(np.sum(counts[edge1:edge2]) / total)
        high = float(np.sum(counts[edge2:]) / total)
        return np.array([low, mid, high], dtype=np.float64)

    def _calc_processed_three_distribution(
        self, gray: np.ndarray | None, lower: int, upper: int, exp_value: float
    ) -> tuple[float, float, float]:
        if gray is None or gray.size == 0:
            return (0.0, 0.0, 0.0)
        levels = max(2, int(self.settings.levels))
        hist = np.bincount(gray.reshape(-1).astype(np.uint8), minlength=256).astype(np.float64)
        ratios = self._distribution_from_hist(hist, lower, upper, levels, exp_value)
        if np.sum(ratios) <= 0:
            return (0.0, 0.0, 0.0)
        low = float(ratios[0] * 100.0)
        mid = float(ratios[1] * 100.0)
        high = float(ratios[2] * 100.0)
        return (low, mid, high)

    def _draw_distribution_overlay(
        self, painter: QPainter, rect: QRect, values: list[float], title: str
    ) -> None:
        _ = title  # keep signature stable for current call sites
        display_values = list(reversed(values))  # white -> black
        rows_count = max(1, len(display_values))
        row_h = 24
        block_height = rows_count * row_h + 2
        block_rect = QRect(rect.left() + 8, rect.top() + 8, 116, block_height)
        painter.fillRect(block_rect, Qt.GlobalColor.black)
        painter.setPen(Qt.GlobalColor.white)
        painter.drawRect(block_rect.adjusted(0, 0, -1, -1))

        for idx, pct in enumerate(display_values):
            top = block_rect.top() + 1 + idx * row_h
            row_rect = QRect(block_rect.left() + 1, top, block_rect.width() - 2, row_h - 1)
            tone = int(round(((rows_count - 1 - idx) / max(1, rows_count - 1)) * 255))
            fill_color = QColor(tone, tone, tone)
            text_color = Qt.GlobalColor.black if tone >= 128 else Qt.GlobalColor.white
            painter.fillRect(row_rect, fill_color)
            painter.setPen(Qt.GlobalColor.white)
            painter.drawRect(row_rect.adjusted(0, 0, -1, -1))
            painter.setPen(text_color)
            painter.drawText(row_rect, Qt.AlignmentFlag.AlignCenter, f"{pct:.1f}%")

    def _edges_at(self, pos: QPoint) -> int:
        edges = 0
        if pos.x() <= _RESIZE_MARGIN:
            edges |= _EDGE_LEFT
        if pos.x() >= self.width() - _RESIZE_MARGIN:
            edges |= _EDGE_RIGHT
        if pos.y() <= _RESIZE_MARGIN:
            edges |= _EDGE_TOP
        if pos.y() >= self.height() - _RESIZE_MARGIN:
            edges |= _EDGE_BOTTOM
        return edges

    def _cursor_for_edges(self, edges: int) -> Qt.CursorShape:
        if edges in (_EDGE_LEFT | _EDGE_TOP, _EDGE_RIGHT | _EDGE_BOTTOM):
            return Qt.CursorShape.SizeFDiagCursor
        if edges in (_EDGE_RIGHT | _EDGE_TOP, _EDGE_LEFT | _EDGE_BOTTOM):
            return Qt.CursorShape.SizeBDiagCursor
        if edges & (_EDGE_LEFT | _EDGE_RIGHT):
            return Qt.CursorShape.SizeHorCursor
        if edges & (_EDGE_TOP | _EDGE_BOTTOM):
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.ArrowCursor

    def _start_drag_from_panel(self, global_point: QPoint) -> None:
        self._is_dragging = True
        self._boost_motion(2.0)
        self._drag_pos = global_point - self.frameGeometry().topLeft()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            edges = self._edges_at(pos)
            if edges:
                self._is_resizing = True
                self._resize_edges = edges
                self._resize_start_global = event.globalPosition().toPoint()
                self._resize_start_geom = self.geometry()
                event.accept()
                return
            if self.panel.isVisible() and self.panel.geometry().contains(pos):
                super().mousePressEvent(event)
                return
            self._is_dragging = True
            self._boost_motion(2.0)
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        pos = event.position().toPoint()
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            self.setCursor(self._cursor_for_edges(self._edges_at(pos)))
            return

        if self._is_resizing and self._resize_start_geom is not None and self._resize_start_global is not None:
            delta = event.globalPosition().toPoint() - self._resize_start_global
            g = QRect(self._resize_start_geom)
            if self._resize_edges & _EDGE_LEFT:
                new_left = min(g.right() - _MIN_WIDTH + 1, g.left() + delta.x())
                g.setLeft(new_left)
            if self._resize_edges & _EDGE_RIGHT:
                g.setRight(max(g.left() + _MIN_WIDTH - 1, g.right() + delta.x()))
            if self._resize_edges & _EDGE_TOP:
                new_top = min(g.bottom() - _MIN_HEIGHT + 1, g.top() + delta.y())
                g.setTop(new_top)
            if self._resize_edges & _EDGE_BOTTOM:
                g.setBottom(max(g.top() + _MIN_HEIGHT - 1, g.bottom() + delta.y()))
            self.setGeometry(g)
            self._boost_motion(1.5)
            self.request_refresh(60)
            return

        if self._drag_pos is None:
            return
        self.move(event.globalPosition().toPoint() - self._drag_pos)
        self._boost_motion(2.0)
        self.request_refresh(120)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        self._drag_pos = None
        self._is_dragging = False
        self._is_resizing = False
        self._resize_edges = 0
        self._resize_start_geom = None
        self._resize_start_global = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._boost_motion(1.5)
        self.request_refresh(20)

    def closeEvent(self, event: QCloseEvent) -> None:
        geom = self.geometry()
        self.settings.x = geom.x()
        self.settings.y = geom.y()
        self.settings.width = geom.width()
        self.settings.height = geom.height()
        self.settings_manager.save(self.settings)
        self.hotkeys.shutdown()
        self.panel.close()
        self.image_mode.close()
        event.accept()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._layout_panel()
        self._layout_overlay_buttons()
        self._boost_motion(2.0)
        self.request_refresh(120)

    def moveEvent(self, event) -> None:  # type: ignore[override]
        super().moveEvent(event)
        self._last_frame_signature = None
        self._boost_motion(1.5)
        self.request_refresh(80)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Escape:
            self.force_quit()
            return
        super().keyPressEvent(event)


