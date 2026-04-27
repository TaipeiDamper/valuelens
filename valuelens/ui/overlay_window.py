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
from valuelens.core.qt_image import bgr_to_qimage, gray_to_qimage, qimage_to_bgr
from valuelens.core.quantize import native_distribution_from_indices, quantize_gray, quantize_gray_with_indices
from valuelens.modes.image_mode import ImageModeDialog
from valuelens.ui.control_panel import ControlPanel
from valuelens.ui.mirror_window import MirrorWindow


_RESIZE_MARGIN = 12
_MIN_WIDTH = 320
_MIN_HEIGHT = 160

_EDGE_LEFT = 1
_EDGE_RIGHT = 2
_EDGE_TOP = 4
_EDGE_BOTTOM = 8



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
        
        # 如果有啟動預設集，則在此處套用它
        if self.settings.startup_preset:
            self._apply_startup_preset(self.settings.startup_preset)

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
            process_order=settings.process_order,
            morph_enabled=settings.morph_enabled,
            morph_strength=settings.morph_strength,
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
        self._panel_height = 200
        self._compare_gap = 6
        self._last_toggle_ts = 0.0
        self._is_dragging = False
        self._is_resizing = False
        self._is_refreshing = False
        self._refresh_ms = 33             # 固定 30 FPS
        self._fps_count = 0                # FPS 計數
        self._fps_last_report = time.time() # 上次報告時間
        self._last_frame_signature = None
        self._stable_frame_count = 0
        self._compare_mode = bool(settings.compare_mode)
        self._processed_distribution_pct = [0.0] * max(2, int(settings.levels))
        self._raw_distribution_pct = [0.0] * max(2, int(settings.levels))
        self._last_gray_frame: np.ndarray | None = None
        self._auto_balance_pending = True   # 開啟時自動執行一次平衡
        self._auto_balance_use_current = True
        self._auto_continuous_enabled = False
        self._remembered_multi_ratio = None # 記憶 3, 5, 8 階
        self._remembered_two_ratio = None   # 記憶 2 階
        self._show_distribution = True      # 是否顯示灰階比例
        self._bypass_mode = False           # Bypass: 跳過所有處理
        self._is_static_mode = False        # 是否處於靜態分析模式 (Frozen 或 ImageMode)
        self._static_source_type = ""       # "frozen" 或 "image"
        self._source_image = None           # 靜態模式下的原始底圖 (BGR)
        self._full_image_gray = None        # 靜態模式下的整張灰階快取
        self._pan_offset_x = 0             # 圖片平移 X 偏移
        self._pan_offset_y = 0             # 圖片平移 Y 偏移
        self._zoom_factor = 1.0            # 圖片縮放倍率
        self._use_global_calc = False      # 是否使用全圖計算 (ImageMode)
        self._rect_compare_bw = QRect()    # 黑白對比按鈕區域 (手動繪製)
        self._rect_global_calc = QRect()   # 計算全圖按鈕區域 (手動繪製)
        self._pan_drag_start = None        # 平移拖曳起始點
        self._pan_start_offset = None      # 平移拖曳起始偏移
        self._global_calc_dirty = False    # 標記全圖計算是否需要重新執行一次
        self._last_auto_balance_ts = 0.0
        self._last_raw_bgr_frame = None
        self._mirror_window: MirrorWindow | None = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)
        self.timer.start(self._refresh_ms)

        self._coalesce_timer = QTimer(self)
        self._coalesce_timer.setSingleShot(True)
        self._coalesce_timer.timeout.connect(self.refresh_frame)

        self.panel = ControlPanel(settings=self.settings, parent=self)
        self.panel.settings_changed.connect(self.on_settings_changed)
        self.panel.display_settings_changed.connect(self.on_display_settings_changed)
        self.panel.effect_settings_changed.connect(self.on_effect_settings_changed)
        self.panel.edge_settings_changed.connect(self.on_edge_settings_changed)
        self.panel.morph_settings_changed.connect(self.on_morph_settings_changed)
        self.panel.order_changed.connect(self.on_order_changed)
        self.panel.collapse_toggled.connect(self.on_collapse_toggled)
        self.panel.compare_mode_changed.connect(self.on_compare_mode_changed)
        self.panel.hotkey_changed.connect(self.on_hotkey_changed)
        self.panel.auto_balance_raw_requested.connect(self.on_auto_balance_raw_requested)
        self.panel.auto_balance_target_requested.connect(self.on_auto_balance_target_requested)
        self.panel.auto_continuous_toggled.connect(self.on_auto_continuous_toggled)
        self.panel.edge_settings_changed.connect(self.on_edge_settings_changed)
        self.panel.import_requested.connect(self.toggle_freeze_mode)
        self.panel.screenshot_requested.connect(self.on_screenshot_requested)
        self.panel.image_mode_requested.connect(self.open_image_mode)
        self.panel.distribution_toggled.connect(self.on_distribution_toggled)
        self.panel.quit_requested.connect(self.force_quit)
        self.panel.minimize_requested.connect(self.showMinimized)
        self.panel.maximize_requested.connect(self.toggle_maximize)
        self.panel.drag_started.connect(self._start_drag_from_panel)
        self.panel.save_startup_requested.connect(self.on_save_startup_preset)
        self.panel.clear_startup_requested.connect(self.on_clear_startup_preset)
        self.panel.debug_screenshot_requested.connect(self.on_debug_screenshot_requested)
        self.panel.recording_window_toggled.connect(self.on_recording_window_toggled)
        self.panel.show()
        self.panel.raise_()
        self._layout_panel()

        self.hotkeys.register("toggle", self.settings.hotkey, self.toggle_enabled)
        self.hotkeys.register("image_mode", "ctrl+alt+i", self.open_image_mode)
        self.hotkeys.register("collapse", "ctrl+alt+p", lambda: self.panel.collapse_btn.toggle())
        self.hotkeys.register("quit", "ctrl+alt+q", self.force_quit)

        # 拖放支援
        self.setAcceptDrops(True)

        # 啟動時跑一次自動平衡
        self.panel.sync_from_settings(self.settings)
        QTimer.singleShot(500, self.on_auto_balance_raw_requested)

        self.show()
        self.request_refresh()

    def _get_current_raw_frame(self) -> np.ndarray | None:
        if hasattr(self, '_last_raw_bgr_frame'):
            return self._last_raw_bgr_frame
        return None

    def on_distribution_toggled(self, show: bool) -> None:
        """切換灰階比例顯示。"""
        self._show_distribution = show
        self._layout_overlay_buttons()
        self.update()

    def on_global_calc_toggled(self, enabled: bool) -> None:
        self._use_global_calc = enabled
        self._global_calc_dirty = True
        # 切換計算範圍時，自動重跑一次平衡
        self._auto_balance_pending = True
        self._auto_balance_use_current = True
        self.request_refresh(5)
        self.update()
    def on_bypass_toggled(self, bypass: bool) -> None:
        """切換 Bypass 模式：跳過所有處理，顯示原始影像。"""
        self._bypass_mode = bypass
        self._last_frame_signature = None
        self.refresh_frame()
        self.update()


    def on_recording_window_toggled(self, enabled: bool) -> None:
        """開啟或關閉錄製鏡像視窗。"""
        if enabled:
            if self._mirror_window is None:
                self._mirror_window = MirrorWindow()
            self._mirror_window.show()
            self._mirror_window.raise_()
        else:
            if self._mirror_window:
                self._mirror_window.hide()

    def on_screenshot_requested(self) -> None:
        """擷取鏡片區域（Lens Area）內容並存入剪貼簿。"""
        from PySide6.QtWidgets import QApplication
        cb = QApplication.clipboard()
        
        # 擷取可見的鏡片區域（包含對照圖）
        if self._compare_mode:
            lens = self._lens_rect()
            comp = self._compare_rect()
            full_rect = lens.united(comp)
            pix = self.grab(full_rect)
        else:
            pix = self.grab(self._lens_rect())
            
        cb.setPixmap(pix)

    def on_debug_screenshot_requested(self) -> None:
        """偵錯模式：從系統層級擷取全螢幕（強制包含 ValueLens 視窗本身）。"""
        from PySide6.QtWidgets import QApplication
        import time
        import ctypes
        
        # 1. 強制解除隱身 (確保 Windows DWM 將其納入擷取範圍)
        hwnd = self.effectiveWinId()
        if hwnd:
            # WDA_NONE = 0
            ctypes.windll.user32.SetWindowDisplayAffinity(int(hwnd), 0)
        
        # 2. 強制重繪並等待 DWM 生效
        self.update()
        QApplication.processEvents()
        time.sleep(0.3)  # 給 DWM 稍長時間反應

        # 3. 擷取螢幕 (利用 MSS 擷取主螢幕)
        sct = self.capture._sct
        monitor = sct.monitors[0]  # 抓取整個虛擬螢幕 (All monitors)
        shot = sct.grab(monitor)
        
        from PySide6.QtGui import QImage, QPixmap
        img = QImage(shot.raw, shot.width, shot.height, QImage.Format.Format_ARGB32)
        pix = QPixmap.fromImage(img)

        # 4. 存入剪貼簿
        cb = QApplication.clipboard()
        cb.setPixmap(pix)
        print("[Debug] Full screen (including window) captured to clipboard.")
        
        # 5. 下一輪 refresh 會自動根據 CaptureService 恢復隱身狀態

    def on_save_startup_preset(self) -> None:
        """將目前的所有參數儲存為啟動預設。"""
        from dataclasses import asdict
        # 排除非參數類的欄位 (如視窗位置 x, y)
        current = asdict(self.settings)
        excluded = {'x', 'y', 'width', 'height', 'startup_preset', 'enabled'}
        preset = {k: v for k, v in current.items() if k not in excluded}
        
        self.settings.startup_preset = preset
        self.settings_manager.save(self.settings)
        print("[Presets] Current configuration saved as Startup Preset.")

    def on_clear_startup_preset(self) -> None:
        """清除啟動預設，改回恢復上次關閉狀態。"""
        self.settings.startup_preset = None
        self.settings_manager.save(self.settings)
        print("[Presets] Startup Preset cleared.")

    def _apply_startup_preset(self, preset: dict) -> None:
        """將預設集的資料填充進目前的 settings 物件中。"""
        for k, v in preset.items():
            if hasattr(self.settings, k):
                setattr(self.settings, k, v)
        print("[Presets] Startup Preset applied.")

    def toggle_freeze_mode(self) -> None:
        """切換凍結模式：鎖定當前畫面或恢復即時擷取。同時處理 ImageMode 釋放。"""
        if self._is_static_mode:
            # 恢復即時模式
            self._is_static_mode = False
            self._static_source_type = ""
            self._source_image = None
            self._full_image_gray = None
            self._pan_offset_x = 0
            self._pan_offset_y = 0
            self._zoom_factor = 1.0
            self.panel.freeze_btn.setProperty("freeze_mode", "")
            self.panel.freeze_btn.style().unpolish(self.panel.freeze_btn)
            self.panel.freeze_btn.style().polish(self.panel.freeze_btn)
            self.panel.freeze_btn.setToolTip("鎖定當下畫面 (就地凍結)")
            print("[StaticMode] EXIT (Back to Live)")
            self._layout_overlay_buttons()
        else:
            # 擷取整個視窗背後的影像（包含工具列後面的區域）
            dpr = self.devicePixelRatioF()
            phys = self._physical_window_rect()
            if phys is not None:
                win_x, win_y, win_pw, win_ph = phys
                frame = self.capture.capture_region(
                    win_x, win_y, win_pw, win_ph,
                    exclude_hwnd=self._hwnd()
                )
            else:
                frame = self._last_raw_bgr_frame
                if frame is None:
                    rect = self._lens_rect()
                    frame = self.capture.capture_region(
                        rect.x(), rect.y(), rect.width(), rect.height(),
                        exclude_hwnd=self._hwnd()
                    )
                
            if frame is not None:
                self._is_static_mode = True
                self._static_source_type = "frozen"
                self._source_image = frame.copy()
                self._full_image_gray = cv2.cvtColor(self._source_image, cv2.COLOR_BGR2GRAY)
                
                # 進入靜態模式時重置縮放與偏移
                self._pan_offset_x = 0
                self._pan_offset_y = 0
                self._zoom_factor = 1.0
                
                self.panel.freeze_btn.setProperty("freeze_mode", "frozen")
                self.panel.freeze_btn.style().unpolish(self.panel.freeze_btn)
                self.panel.freeze_btn.style().polish(self.panel.freeze_btn)
                self.panel.freeze_btn.setToolTip("點擊解除鎖定 (目前已凍結)")
                print(f"[StaticMode] ENTERED via Freeze, shape={frame.shape}")
            else:
                print("[Freeze] FAILED - no frame available")
        
        self._last_frame_signature = None
        self._is_refreshing = False
        
        # 狀態轉換時自動執行一次平衡 (防止切換後畫面階調不正確)
        self._auto_balance_pending = True
        self._auto_balance_use_current = True
        
        self.refresh_frame()
        self.update()

    def import_image(self, bgr_image: np.ndarray) -> None:
        """匯入外部圖片，進入 StaticMode (Image 類型)。"""
        self._source_image = bgr_image.copy()
        self._full_image_gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        self._is_static_mode = True
        self._static_source_type = "image"
        
        self._pan_offset_x = 0
        self._pan_offset_y = 0
        self._zoom_factor = 1.0
        self.panel.freeze_btn.setProperty("freeze_mode", "image")
        self.panel.freeze_btn.style().unpolish(self.panel.freeze_btn)
        self.panel.freeze_btn.style().polish(self.panel.freeze_btn)
        self.panel.freeze_btn.setToolTip("點擊退出圖片模式")
        self._last_frame_signature = None
        self._is_refreshing = False
        print(f"[StaticMode] ENTERED via Import, shape={bgr_image.shape}")
        self._layout_overlay_buttons()
        # 匯入時自動執行一次平衡
        self._auto_balance_pending = True
        self._auto_balance_use_current = True
        self.refresh_frame()
        self.update()

    def _sync_image_mode_settings(self) -> None:
        self.image_mode.set_quantize_settings(
            self.settings.levels,
            self.settings.min_value,
            self.settings.max_value,
            self.settings.exp_value,
            self.settings.blur_enabled,
            self.settings.blur_radius,
            self.settings.dither_enabled,
            self.settings.dither_strength,
            self.settings.process_order,
            self.settings.morph_enabled,
            self.settings.morph_strength,
        )

    def open_image_mode(self) -> None:
        """開啟檔案選擇器匯入圖片。"""
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇圖片", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            return
        self.import_image(image)

    def dragEnterEvent(self, event) -> None:
        """接受圖片檔案拖放。"""
        if event.mimeData().hasUrls() or event.mimeData().hasImage():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        """處理拖放的圖片檔案。"""
        mime = event.mimeData()
        if mime.hasUrls():
            for url in mime.urls():
                path = url.toLocalFile()
                if path:
                    image = cv2.imread(path)
                    if image is not None:
                        self.import_image(image)
                        return
        if mime.hasImage():
            qimg = QImage(mime.imageData())
            if not qimg.isNull():
                self.import_image(qimage_to_bgr(qimg))

    def on_settings_changed(
        self, levels: int, min_value: int, max_value: int, exp_value: float
    ) -> None:
        if self.settings.levels != levels:
            old_lvl = self.settings.levels
            new_lvl = levels
            
            was_multi = old_lvl in (3, 5, 8)
            is_multi = new_lvl in (3, 5, 8)
            was_two = (old_lvl == 2)
            is_two = (new_lvl == 2)
            
            # --- 記憶目前的比例 ---
            if was_multi:
                self._remembered_multi_ratio = self.panel.balance_presets.currentData()
            elif was_two:
                self._remembered_two_ratio = self.panel.balance_presets.currentData()
                
            # --- 決定載回哪一組記憶 ---
            self._auto_balance_pending = True
            if is_multi and self._remembered_multi_ratio:
                # 恢復多階層記憶
                self.panel.set_balance_preset(self._remembered_multi_ratio)
                self._auto_balance_use_current = True
            elif is_two and self._remembered_two_ratio:
                # 恢復 2 階記憶
                self.panel.set_balance_preset(self._remembered_two_ratio)
                self._auto_balance_use_current = True
            elif was_multi and is_multi:
                # 在 3, 5, 8 之間切換，且沒有記憶（雖然通常會有）
                self._auto_balance_use_current = True
            else:
                # 其他情況（例如第一次進入），執行全域最佳
                self._auto_balance_use_current = False
            
        self.settings.levels = levels
        self.settings.min_value = min_value
        self.settings.max_value = max_value
        self.settings.exp_value = exp_value
        self._sync_image_mode_settings()
        self._processed_distribution_pct = [0.0] * max(2, int(levels))
        self._raw_distribution_pct = [0.0] * max(2, int(levels))
        self._last_frame_signature = None
        self.request_refresh(5)

    def on_display_settings_changed(self, min_value: int, max_value: int, exp_value: float) -> None:
        self.settings.display_min_value = min_value
        self.settings.display_max_value = max_value
        self.settings.display_exp_value = exp_value
        self._last_frame_signature = None
        self.request_refresh(16)

    def on_effect_settings_changed(
        self, blur_enabled: bool, blur_radius: int,
        dither_enabled: bool, dither_strength: int
    ) -> None:
        self.settings.blur_enabled = blur_enabled
        self.settings.blur_radius = blur_radius
        self.settings.dither_enabled = dither_enabled
        self.settings.dither_strength = dither_strength
        # Update image mode if active
        self._sync_image_mode_settings()
        self._last_frame_signature = None
        self.request_refresh(16)

    def on_order_changed(self, order: list[str]) -> None:
        self.settings.process_order = order
        self.image_mode.process_order = order
        self._last_frame_signature = None
        self.request_refresh(16)

    def on_morph_settings_changed(self, enabled: bool, strength: int) -> None:
        self.settings.morph_enabled = enabled
        self.settings.morph_strength = strength
        self._sync_image_mode_settings()
        self._last_frame_signature = None
        self.request_refresh(16)

    def on_collapse_toggled(self, collapsed: bool) -> None:
        self._panel_height = 36 if collapsed else 200
        self._layout_panel()
        
        # 面板收合會導致鏡片區域 (lens rect) 大小與物理座標改變
        # 我們必須清除快取，強制下一幀重新計算
        self._last_frame_signature = None
        
        self.request_refresh(10)

    def on_compare_mode_changed(self, enabled: bool) -> None:
        self._compare_mode = enabled
        self.settings.compare_mode = enabled
        self._last_frame_signature = None
        self._raw_frame = QPixmap()
        self._layout_overlay_buttons()
        self.request_refresh()

    def on_compare_bw_changed(self, bw_enabled: bool) -> None:
        self.settings.compare_bw = bw_enabled
        self._last_frame_signature = None
        self._raw_frame = QPixmap()
        self.request_refresh()

    def on_hotkey_changed(self, hotkey: str) -> None:
        self.settings.hotkey = hotkey
        self.hotkeys.register("toggle", hotkey, self.toggle_enabled)

    def _apply_balance_to_ui(self, lower: int, upper: int, exp_value: float) -> None:
        """統一將平衡參數套用到所有 UI 組件，不觸發循環信號。"""
        self.panel.exp_slider.blockSignals(True)
        self.panel.range_slider.blockSignals(True)
        
        self.panel.exp_slider.setValue(int(round(-exp_value * 100.0)))
        self.panel.range_slider.set_values(lower, upper)
        
        self.panel.exp_slider.blockSignals(False)
        self.panel.range_slider.blockSignals(False)
        
        # 同步更新本地與圖片模式設定
        self.settings.min_value = lower
        self.settings.max_value = upper
        self.settings.exp_value = exp_value
        self._sync_image_mode_settings()

    def on_auto_balance_target_requested(self, ratios: tuple[float, float, float]) -> None:
        # 決定計算來源：是當前可見的 viewport 還是整張圖片
        source_gray = self._last_gray_frame
        if self._is_static_mode and self._use_global_calc and self._full_image_gray is not None:
            source_gray = self._full_image_gray
            
        if source_gray is None or source_gray.size == 0:
            return
            
        lower, upper, exp_value = self._optimize_balance_params(
            source_gray,
            ratios,
            self.settings.min_value,
            self.settings.max_value,
            self.settings.exp_value,
        )
        
        # 參數平滑處理 (防止跳動)
        # 如果變動極小，則不更新；如果變動中等，則混合一部分舊值
        def smooth(new, old, alpha=0.3):
            if abs(new - old) < 0.5: return old
            return old * (1 - alpha) + new * alpha

        if self._auto_continuous_enabled:
            # 持續模式下使用平滑
            lower = int(round(smooth(lower, self.settings.min_value, 0.4)))
            upper = int(round(smooth(upper, self.settings.max_value, 0.4)))
            exp_value = smooth(exp_value, self.settings.exp_value, 0.3)
        
        changed = (
            abs(lower - self.settings.min_value) >= 1 or
            abs(upper - self.settings.max_value) >= 1 or
            abs(exp_value - self.settings.exp_value) > 0.02
        )
        
        if changed:
            self._apply_balance_to_ui(lower, upper, exp_value)
            self._last_frame_signature = None

    def on_auto_continuous_toggled(self, enabled: bool) -> None:
        """切換持續自動平衡。"""
        self._auto_continuous_enabled = enabled
        self._auto_balance_pending = False # 清除任何待處理的平衡請求
        if enabled:
            self._global_calc_dirty = True
            self._last_auto_balance_ts = 0.0 # 立即執行一次
            self.request_refresh()

    def on_edge_settings_changed(self, enabled: bool, strength: int, mix: int) -> None:
        """邊緣檢測設定變更。"""
        self.settings.edge_enabled = enabled
        self.settings.edge_strength = strength
        self.settings.edge_mix = mix
        self._last_frame_signature = None
        self.request_refresh(16)

    def on_auto_balance_raw_requested(self) -> None:
        # 點擊重新平衡時，關閉持續模式
        if self._auto_continuous_enabled:
            self._auto_continuous_enabled = False
            self.panel.auto_continuous_check.setChecked(False)

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
        # 黑白對照按鈕 (右側)
        show_bw = self._compare_mode and self._show_distribution
        if show_bw:
            c = self._compare_rect()
            if not c.isNull():
                self._rect_compare_bw = QRect(c.right() - 60, c.top() + 6, 54, 24)
            else:
                self._rect_compare_bw = QRect()
        else:
            self._rect_compare_bw = QRect()

        # 全圖計算按鈕 (左側)
        show_global = self._is_static_mode and self._show_distribution
        if show_global:
            l = self._lens_rect()
            if not l.isNull():
                self._rect_global_calc = QRect(l.left() + 130, l.top() + 6, 80, 24)
            else:
                self._rect_global_calc = QRect()
        else:
            self._rect_global_calc = QRect()

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

    def _on_timer_tick(self) -> None:
        # 固定頻率請求刷新
        self.request_refresh(0)
        
        # 確保懸浮按鈕位置正確
        if self._is_static_mode and self._stable_frame_count % 10 == 0:
            self._layout_overlay_buttons()

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
        """核心影像處理管線。"""
        if self._is_refreshing:
            return
        
        self._fps_count += 1
        now = time.time()
        if now - self._fps_last_report >= 1.0:
            fps = self._fps_count / (now - self._fps_last_report)
            self.panel.set_fps(fps)
            self._fps_count = 0
            self._fps_last_report = now

        self._stable_frame_count += 1
        self._is_refreshing = True
        try:
            # 1. 座標與參數準備
            rect = self._lens_rect()
            if rect.width() <= 0 or rect.height() <= 0: return
            dpr = self.devicePixelRatioF()
            
            # --- 核心邏輯：決定影像來源 ---
            if self._is_static_mode and self._source_image is not None:
                # 靜態分析模式 (Frozen 或 Import)：處理縮放、平移與 Padding
                src = self._source_image
                sh, sw = src.shape[:2]
                view_w_phys = max(1, int(round(rect.width() * dpr)))
                view_h_phys = max(1, int(round(rect.height() * dpr)))
                
                # 最小縮放限制 (確保長邊 1.5x)
                view_long = max(view_w_phys, view_h_phys)
                img_long = max(sw, sh)
                min_zoom = view_long / (1.5 * img_long)
                if self._zoom_factor < min_zoom:
                    self._zoom_factor = min_zoom

                vsw = sw * self._zoom_factor
                vsh = sh * self._zoom_factor
                
                # 限制平移偏移 (處理 Padding)
                if vsw > view_w_phys:
                    self._pan_offset_x = max(0, min(self._pan_offset_x, vsw - view_w_phys))
                else:
                    self._pan_offset_x = (vsw - view_w_phys) / 2
                    
                if vsh > view_h_phys:
                    self._pan_offset_y = max(0, min(self._pan_offset_y, vsh - view_h_phys))
                else:
                    self._pan_offset_y = (vsh - view_h_phys) / 2
                
                # 裁切並縮放有效像素
                sx1 = max(0, int(round(self._pan_offset_x / self._zoom_factor)))
                sy1 = max(0, int(round(self._pan_offset_y / self._zoom_factor)))
                sx2 = min(sw, int(round((self._pan_offset_x + view_w_phys) / self._zoom_factor)))
                sy2 = min(sh, int(round((self._pan_offset_y + view_h_phys) / self._zoom_factor)))
                
                crop = src[sy1:sy2, sx1:sx2]
                
                if crop.size > 0:
                    target_w = int(round((sx2 - sx1) * self._zoom_factor))
                    target_h = int(round((sy2 - sy1) * self._zoom_factor))
                    if target_w > 0 and target_h > 0:
                        resized_crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        frame = np.full((view_h_phys, view_w_phys, 3), 34, dtype=np.uint8)
                        dx = max(0, int(round(-self._pan_offset_x)))
                        dy = max(0, int(round(-self._pan_offset_y)))
                        fh, fw = resized_crop.shape[:2]
                        # 安全複製：確保不超出邊界且形狀匹配
                        jh = min(fh, view_h_phys - dy)
                        jw = min(fw, view_w_phys - dx)
                        if jh > 0 and jw > 0:
                            frame[dy:dy+jh, dx:dx+jw] = resized_crop[:jh, :jw]
                        
                        # 更新計算用的灰階影像 (排除 Padding)
                        self._last_gray_frame = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2GRAY)
                    else:
                        frame = np.full((view_h_phys, view_w_phys, 3), 34, dtype=np.uint8)
                        self._last_gray_frame = None
                else:
                    frame = np.full((view_h_phys, view_w_phys, 3), 34, dtype=np.uint8)
                    self._last_gray_frame = None
            else:
                # 即時模式 (Live)：從螢幕擷取
                phys = self._physical_window_rect()
                if phys is not None:
                    win_x, win_y, win_pw, win_ph = phys
                    panel_h_phys = int(round(self._panel_height * dpr))
                    cap_x, cap_y = win_x, win_y + panel_h_phys
                    cap_w, cap_h = win_pw, max(1, win_ph - panel_h_phys)
                else:
                    cap_x = int(round((self.x() + rect.x()) * dpr))
                    cap_y = int(round((self.y() + rect.y()) * dpr))
                    cap_w = max(1, int(round(rect.width() * dpr)))
                    cap_h = max(1, int(round(rect.height() * dpr)))
                
                hwnd = self._hwnd()
                frame = self.capture.capture_region(
                    cap_x, cap_y, cap_w, cap_h, exclude_hwnd=hwnd
                )
                if frame is not None:
                    self._last_raw_bgr_frame = frame.copy()
            
            if frame is None or frame.size == 0: 
                return

            self._last_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 特徵檢查
            signature = self._frame_signature(frame)
            if signature == self._last_frame_signature and not self._frame.isNull():
                return
            self._last_frame_signature = signature

            h, w = frame.shape[:2]

            # --- Bypass 模式：跳過量化，直接顯示原始影像 ---
            if self._bypass_mode:
                self._frame_array = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                qimg = bgr_to_qimage(frame)
                qimg.setDevicePixelRatio(dpr)
                self._frame = QPixmap.fromImage(qimg)
                self._processed_distribution_pct = [0.0] * max(2, int(self.settings.levels))
                self._raw_distribution_pct = [0.0] * max(2, int(self.settings.levels))
            else:
                # 正常量化處理
                eff_blur = self.settings.blur_radius if self.settings.blur_enabled else 0
                eff_dither = self.settings.dither_strength if self.settings.dither_enabled else 0
                eff_edge = self.settings.edge_strength if self.settings.edge_enabled else 0
                
                logic_quantized, logic_indices, edges = quantize_gray_with_indices(
                    frame,
                    self.settings.levels,
                    self.settings.min_value,
                    self.settings.max_value,
                    self.settings.exp_value,
                    display_min=self.settings.display_min_value,
                    display_max=self.settings.display_max_value,
                    display_exp=self.settings.display_exp_value,
                    blur_radius=eff_blur,
                    dither_strength=eff_dither,
                    edge_strength=eff_edge,
                    process_order=self.settings.process_order,
                    morph_enabled=self.settings.morph_enabled,
                    morph_strength=self.settings.morph_strength,
                )
                
                h, w = logic_quantized.shape[:2]
                
                # 處理邊緣與比例 (Edge Mix)
                if edges is not None:
                    mix = self.settings.edge_mix / 100.0
                    ec = self.settings.edge_color[::-1] # RGB -> BGR (預設為黑色)
                    
                    # 原始量化圖轉為 BGR 用於混合
                    base_bgr = cv2.cvtColor(logic_quantized, cv2.COLOR_GRAY2BGR)
                    
                    if mix >= 1.0:
                        # 純邊緣模式：背景全白，邊緣為黑色 (或設定色)
                        final_bgr = np.full_like(base_bgr, 255)
                        final_bgr[edges > 0] = ec
                    else:
                        # 混合模式：(1-mix) * 原圖 + mix * 白色背景，最後疊加邊緣
                        # 讓原圖隨著 mix 增加而淡出至白色
                        bg_part = (base_bgr.astype(np.float32) * (1.0 - mix) + 255.0 * mix).astype(np.uint8)
                        final_bgr = bg_part
                        final_bgr[edges > 0] = ec
                    
                    # 轉回 RGB 給 QImage
                    self._frame_array = np.ascontiguousarray(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))
                    qimg = bgr_to_qimage(final_bgr)
                else:
                    self._frame_array = logic_quantized
                    qimg = gray_to_qimage(logic_quantized)
                
                qimg.setDevicePixelRatio(dpr)
                self._frame = QPixmap.fromImage(qimg)
                self._update_distributions(self._last_gray_frame, logic_indices)
            
            # 3. 處理對照圖
            if self._compare_mode:
                if self.settings.compare_bw:
                    gray_cont = np.ascontiguousarray(self._last_gray_frame)
                    self._raw_frame_array = gray_cont
                    raw_qimg = gray_to_qimage(gray_cont)
                else:
                    self._raw_frame_array = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    raw_qimg = bgr_to_qimage(frame)
                raw_qimg.setDevicePixelRatio(dpr)
                self._raw_frame = QPixmap.fromImage(raw_qimg)
            else:
                self._raw_frame = QPixmap()
            
            self.update()

            # 自動平衡邏輯
            if self._auto_balance_pending:
                self._auto_balance_pending = False
                if self._auto_balance_use_current:
                    target = self.panel.balance_presets.currentData()
                    if target:
                        self.on_auto_balance_target_requested(target)
                    else:
                        self.on_auto_balance_raw_requested()
                else:
                    self.on_auto_balance_raw_requested()
            elif self._auto_continuous_enabled:
                # 如果開啟了「計算全圖」，且數據來源是靜態的 (StaticMode)，則只需計算一次
                if self._is_static_mode and self._use_global_calc:
                    if self._global_calc_dirty:
                        self._global_calc_dirty = False
                        current_target = self.panel.balance_presets.currentData()
                        if current_target:
                            self.on_auto_balance_target_requested(current_target)
                        else:
                            self.on_auto_balance_raw_requested()
                else:
                    # 情況 A: 即時模式 (Live)
                    # 情況 B: 靜態模式但關閉全圖計算 (Local) -> 隨縮放/平移持續追蹤
                    now = time.monotonic()
                    if now - self._last_auto_balance_ts > 0.15:
                        self._last_auto_balance_ts = now
                        current_target = self.panel.balance_presets.currentData()
                        if current_target:
                            self.on_auto_balance_target_requested(current_target)
        finally:
            self._is_refreshing = False
            
            # 如果錄製視窗開啟中，則將完整的視窗內容「照翻」過去
            if self._mirror_window and self._mirror_window.isVisible():
                # 使用 grab 擷取整個視窗（含工具列）
                pix = self.grab()
                self._mirror_window.update_frame(pix)

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
        if self._show_distribution:
            self._draw_distribution_overlay(painter, lens, self._processed_distribution_pct, "處理")
        if self._compare_mode and not compare.isNull():
            painter.drawRect(compare.adjusted(0, 0, -1, -1))
            if self._show_distribution:
                self._draw_distribution_overlay(painter, compare, self._raw_distribution_pct, "原始")
                
        # --- 繪製手動按鈕 ---
        if not self._rect_compare_bw.isNull():
            self._draw_manual_button(painter, self._rect_compare_bw, "黑白", self.settings.compare_bw)
        if not self._rect_global_calc.isNull():
            self._draw_manual_button(painter, self._rect_global_calc, "計算全圖", self._use_global_calc)

    def _draw_manual_button(self, painter: QPainter, rect: QRect, text: str, checked: bool) -> None:
        """手動繪製一個看起來像 QToolButton 的按鈕。"""
        bg_color = QColor(46, 123, 246) if checked else QColor(0, 0, 0, 150)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.GlobalColor.white)
        painter.setBrush(bg_color)
        painter.drawRoundedRect(rect, 4, 4)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


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
        native_dist = native_distribution_from_indices(indices, level_count)
        if native_dist is not None and native_dist.size == level_count:
            return native_dist.tolist()
        clamped = np.clip(indices.astype(np.int32), 0, level_count - 1)
        counts = np.bincount(clamped.reshape(-1), minlength=level_count).astype(np.float64)
        total = float(counts.sum())
        if total <= 0:
            return [0.0] * level_count
        return ((counts / total) * 100.0).tolist()


    @staticmethod
    def _levels_to_wgb(values: list[float]) -> tuple[float, float, float]:
        """將 N 階層分佈折疊為 (white%, gray%, black%)。

        重要：
        - 根據使用者規則，此處 index 0 必須是最白 (White)，index n-1 必須是最黑 (Black)。
        - 由於 quantize 輸出是 0=黑，因此我們在此先做 reversed。
        """
        # 反轉數據，使 v[0]=最白, v[n-1]=最黑
        v = list(reversed(values))
        n = len(v)

        def _s(*idx):
            """加總指定 index 的百分比。"""
            return float(sum(v[i] for i in idx if 0 <= i < n))

        def _best_dist(wgb, target_base):
            """計算 wgb 與目標比例（如 0.7, 0.2, 0.1）所有排列的最小距離。"""
            total = sum(wgb) or 1.0
            norm = [val / total for val in wgb]
            t_total = sum(target_base)
            tn = [val / t_total for val in target_base]
            best = float("inf")
            for p in itertools.permutations(tn):
                d = sum((norm[i] - p[i])**2 for i in range(3))
                if d < best:
                    best = d
            return best

        if n == 2:
            # 二分法：白=[0], 黑=[1]。基準比例 3:7 (0.3, 0.7)
            return (_s(0), 0.0, _s(1))

        if n == 3:
            # 三分法：白=[0], 灰=[1], 黑=[2]。基準比例 7:2:1
            return (_s(0), _s(1), _s(2))

        if n == 5:
            # 方案 A: [0,1]|[2,3]|[4] ; 方案 B: [0]|[1,2]|[3,4]
            target = (0.7, 0.2, 0.1)
            a = (_s(0, 1), _s(2, 3), _s(4))
            b = (_s(0),    _s(1, 2), _s(3, 4))
            return a if _best_dist(a, target) <= _best_dist(b, target) else b

        if n == 8:
            # 方案 A: [0,1]|[2,3,4]|[5,6,7] ; 方案 B: [0,1,2]|[3,4,5]|[6,7]
            target = (0.7, 0.2, 0.1)
            a = (_s(0, 1),    _s(2, 3, 4), _s(5, 6, 7))
            b = (_s(0, 1, 2), _s(3, 4, 5), _s(6, 7))
            return a if _best_dist(a, target) <= _best_dist(b, target) else b

        # 其他 n 值：均等三分
        edges = np.linspace(0, n, 4, dtype=np.float64)
        e1, e2 = int(np.floor(edges[1])), int(np.floor(edges[2]))
        return (_s(*range(0, e1)), _s(*range(e1, e2)), _s(*range(e2, n)))

    def _optimize_balance_params(
        self,
        gray: np.ndarray,
        target: tuple[float, float, float], # (White, Gray, Black)
        current_min: int,
        current_max: int,
        current_exp: float,
    ) -> tuple[int, int, float]:
        # 1. 取得直方圖
        hist = np.bincount(gray.reshape(-1).astype(np.uint8), minlength=256).astype(np.float64)
        total = float(hist.sum())
        if total <= 0: return current_min, current_max, current_exp

        # 2. 計算 CDF (累積分布)
        cdf = np.cumsum(hist) / total
        
        t_white, t_gray, t_black = target
        t_total = t_white + t_gray + t_black
        if t_total <= 0: return current_min, current_max, current_exp
        
        # 3. 百分位數預測 (直方圖 0=黑, 255=白)
        pct_black = t_black / t_total
        pct_not_white = (t_black + t_gray) / t_total if t_gray > 0 else (1.0 - t_white / t_total)
        
        def find_val(p):
            # p 必須在 [0, 1] 之間
            return int(np.searchsorted(cdf, max(0.0, min(1.0, p))))

        guess_min = find_val(pct_black)
        guess_max = find_val(pct_not_white)
        
        # 4. 增加採樣密度與範圍
        # 搜尋範圍為 +/- 12% 的人口比例，採樣 13 個點
        p_samples = np.linspace(-0.12, 0.12, 13) 
        
        lo_candidates = sorted(list(set([find_val(pct_black + s) for s in p_samples] + [guess_min])))
        hi_candidates = sorted(list(set([find_val(pct_not_white + s) for s in p_samples] + [guess_max])))
        # Exp 搜尋點位增加到 25 個點，覆蓋更細膩的曲線
        exp_range = np.linspace(-1.5, 1.5, 25)

        best_loss = float('inf')
        best_params = (current_min, current_max, current_exp)
        levels = max(2, int(self.settings.levels))
        target_ratios = np.array([t_black, t_gray, t_white]) / t_total
        
        for lo in lo_candidates:
            if lo > 240: continue
            for hi in hi_candidates:
                if hi <= lo + 4: continue
                if hi > 255: continue
                # 加速：如果這組 lo/hi 離預測太遠則跳過？(目前維持全搜尋以獲取最佳解)
                for ex in exp_range:
                    r_now = self._distribution_from_hist(hist, lo, hi, levels, float(ex))
                    diff = r_now - target_ratios
                    l = float(np.dot(diff, diff))
                    if l < best_loss:
                        best_loss = l
                        best_params = (lo, hi, float(ex))
                        
        return best_params

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
        gamma = float(np.power(2.0, float(exp_value)))
        mapped = np.power(normalized, gamma)
        indices = np.floor(mapped * levels)
        indices = np.clip(indices, 0, levels - 1).astype(np.int32)
        counts = np.bincount(indices, weights=hist, minlength=levels).astype(np.float64)
        total = float(counts.sum())
        if total <= 0:
            return np.zeros(3, dtype=np.float64)
        
        if levels == 2:
            # 特殊處理 n=2：[0]是黑, [1]是白，沒有灰
            return np.array([counts[0]/total, 0.0, counts[1]/total], dtype=np.float64)

        level_edges = np.linspace(0, levels, 4, dtype=np.float64)
        edge1 = int(np.floor(level_edges[1]))
        edge2 = int(np.floor(level_edges[2]))
        low = float(np.sum(counts[:edge1]) / total)
        mid = float(np.sum(counts[edge1:edge2]) / total)
        high = float(np.sum(counts[edge2:]) / total)
        return np.array([low, mid, high], dtype=np.float64)

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
        if self.isMaximized():
            # 記錄最大化之前的寬度比例，以便還原時鼠標能落在相對位置
            old_width = self.width()
            self.toggle_maximize()
            new_width = self.width()
            # 調整 drag_pos 讓滑鼠在還原後依然抓在標題列上的對應比例位置
            ratio = global_point.x() / old_width
            self.move(global_point.x() - int(new_width * ratio), global_point.y() - 15)
            
        self._is_dragging = True
        self._drag_pos = global_point - self.frameGeometry().topLeft()

    def toggle_maximize(self) -> None:
        """切換視窗最大化或還原狀態。"""
        if self.isMaximized():
            self.showNormal()
            self.panel.max_btn.setText("□")
        else:
            self.showMaximized()
            self.panel.max_btn.setText("❐")
        self._last_frame_signature = None
        self.request_refresh(50)

    def changeEvent(self, event) -> None:  # type: ignore[override]
        """監聽視窗狀態改變（如透過系統熱鍵或拖曳最大化），同步按鈕圖示。"""
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Type.WindowStateChange:
            if self.isMaximized():
                self.panel.max_btn.setText("❐")
            else:
                self.panel.max_btn.setText("□")
        super().changeEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        """雙擊工具列標題區域時切換最大化。"""
        pos = event.position().toPoint()
        if self.panel.geometry().contains(pos):
            child = self.childAt(pos)
            # 如果是點擊到按鈕以外的空白處、Layout、或是標籤，則觸發最大化
            if child in (None, self.panel, self.panel.top_row_widget, self.panel.extra_container) or isinstance(child, QLabel):
                self.toggle_maximize()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

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
            # 檢查是否點擊到控制面板
            target = self.childAt(pos)
            if target:
                curr = target
                while curr:
                    if curr == self.panel:
                        super().mousePressEvent(event)
                        return
                    curr = curr.parentWidget()
            
            # 檢查手動按鈕點擊 (優先於平移)
            if not self._rect_compare_bw.isNull() and self._rect_compare_bw.contains(pos):
                self.on_compare_bw_changed(not self.settings.compare_bw)
                self.update()
                event.accept()
                return
            if not self._rect_global_calc.isNull() and self._rect_global_calc.contains(pos):
                self.on_global_calc_toggled(not self._use_global_calc)
                self.update()
                event.accept()
                return

            # 靜態模式 (StaticMode) 下鏡片區域 → 平移圖片
            if self._is_static_mode and self._source_image is not None:
                lens = self._lens_rect()
                if lens.contains(pos) or (self._compare_mode and self._compare_rect().contains(pos)):
                    self._pan_drag_start = event.globalPosition().toPoint()
                    self._pan_start_offset = (self._pan_offset_x, self._pan_offset_y)
                    event.accept()
                    return
            self._is_dragging = True
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
            self.request_refresh(0)
            return

        # ImageMode 平移圖片
        if self._pan_drag_start is not None and self._pan_start_offset is not None:
            dpr = self.devicePixelRatioF()
            delta = self._pan_drag_start - event.globalPosition().toPoint()
            self._pan_offset_x = self._pan_start_offset[0] + delta.x() * dpr
            self._pan_offset_y = self._pan_start_offset[1] + delta.y() * dpr
            self._last_frame_signature = None
            self._is_refreshing = False
            self.refresh_frame()
            return

        if self._drag_pos is None:
            return
        self.move(event.globalPosition().toPoint() - self._drag_pos)
        self.request_refresh(0)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._is_dragging:
            # 模擬 Windows Snap: 如果釋放時滑鼠靠近螢幕頂端，則觸發最大化
            if event.globalPosition().y() < 10:
                if not self.isMaximized():
                    self.toggle_maximize()
        self._is_dragging = False
        super().mouseReleaseEvent(event)
        self._is_resizing = False
        self._resize_edges = 0
        self._resize_start_geom = None
        self._resize_start_global = None
        self._pan_drag_start = None
        self._pan_start_offset = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.request_refresh(5)

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
        self.request_refresh(20)

    def moveEvent(self, event) -> None:  # type: ignore[override]
        super().moveEvent(event)
        self._last_frame_signature = None
        self.request_refresh(20)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if not self._is_static_mode or self._source_image is None:
            super().wheelEvent(event)
            return

        source = self._source_image

        # 鏡片區域內的縮放
        pos = event.position().toPoint()
        lens = self._lens_rect()
        compare = self._compare_rect()
        
        in_lens = lens.contains(pos)
        in_compare = self._compare_mode and compare.contains(pos)
        
        if in_lens or in_compare:
            # 決定縮放中心 (相對於鏡片左上角)
            rel_pos = pos - (lens.topLeft() if in_lens else compare.topLeft())
            dpr = self.devicePixelRatioF()
            mx, my = rel_pos.x() * dpr, rel_pos.y() * dpr
            
            # 計算新的縮放倍率
            angle = event.angleDelta().y()
            zoom_step = 1.15 if angle > 0 else (1.0 / 1.15)
            old_zoom = self._zoom_factor
            
            # 計算最小縮放限制：視窗長邊 = 1.5 * 圖片長邊
            # 換言之：圖片長邊 = 視窗長邊 / 1.5 (確保容納全圖)
            sh, sw = source.shape[:2]
            view_w_phys = max(1, int(round(self.width() * dpr)))
            view_h_phys = max(1, int(round((self.height() - self._panel_height) * dpr)))
            view_long = max(view_w_phys, view_h_phys)
            img_long = max(sw, sh)
            
            min_zoom = view_long / (1.5 * img_long)
            max_zoom = 20.0
            
            new_zoom = max(min_zoom, min(max_zoom, old_zoom * zoom_step))
            
            if new_zoom != old_zoom:
                # 調整 pan_offset 讓縮放中心保持在滑鼠位置
                # 公式：new_offset = (old_offset + mouse_pos) * (new_zoom / old_zoom) - mouse_pos
                self._pan_offset_x = (self._pan_offset_x + mx) * (new_zoom / old_zoom) - mx
                self._pan_offset_y = (self._pan_offset_y + my) * (new_zoom / old_zoom) - my
                self._zoom_factor = new_zoom
                
                self._last_frame_signature = None
                self._is_refreshing = False
                self.refresh_frame()
            event.accept()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Escape:
            self.force_quit()
            return
        # Ctrl+V 貼上圖片
        if event.key() == Qt.Key.Key_V and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            from PySide6.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            qimg = clipboard.image()
            if not qimg.isNull():
                self.import_image(qimage_to_bgr(qimg))
            return
        super().keyPressEvent(event)





