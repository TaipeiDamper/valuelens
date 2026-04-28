from __future__ import annotations

import ctypes
import itertools
import sys
import time
from dataclasses import asdict
from ctypes import wintypes

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, QTimer, QThread, Signal
from PySide6.QtGui import QCloseEvent, QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QMainWindow, QToolButton

from valuelens.config.settings import AppSettings, SettingsManager
from valuelens.core.capture_service import CaptureService
from valuelens.core.hotkey_service import HotkeyService
from valuelens.core.qt_image import bgr_to_qimage, gray_to_qimage, qimage_to_bgr
from valuelens.core.quantize import native_distribution_from_indices, quantize_gray, quantize_gray_with_indices
from valuelens.core.engine import ImageProcessWorker
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
        from valuelens.core.store import AppStore
        self.store = AppStore()
        self.settings = self.store.settings
        self.settings_manager = self.store._manager
        
        # 如果有啟動預設集，則在此處套用它
        if self.settings.startup_preset:
            self._apply_startup_preset(self.settings.startup_preset)

        self.capture = CaptureService()
        self.hotkeys = HotkeyService()
        
        self.calc_worker = ImageProcessWorker(self)
        self.calc_worker.finished.connect(self._on_calc_finished)
        self.image_mode = ImageModeDialog(settings=self.settings, parent=self)
        self.image_mode.set_import_callback(self._get_current_raw_frame)

        from valuelens.ui.render_widget import RenderWidget
        self.canvas = RenderWidget(self)
        self.canvas.setGeometry(self.rect())
        self.canvas.lower()
        self.setGeometry(settings.x, settings.y, settings.width, settings.height)
        
        # 啟動安全性檢查：如果視窗完全在螢幕外，則自動居中
        from PySide6.QtGui import QGuiApplication
        visible = False
        window_rect = QRect(settings.x, settings.y, settings.width, settings.height)
        for screen in QGuiApplication.screens():
            if screen.geometry().intersects(window_rect):
                visible = True
                break
        if not visible:
            QTimer.singleShot(0, self.center_window)

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

        self._snap_timer = QTimer(self)
        self._snap_timer.setSingleShot(True)
        self._snap_timer.timeout.connect(self._on_snap_timeout)

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
        self.panel.save_preset_requested.connect(self.on_save_preset)
        self.panel.load_preset_requested.connect(self.on_load_preset)
        self.panel.clear_preset_requested.connect(self.on_clear_preset)
        self.panel.show()
        self.panel.raise_()
        self._layout_panel()

        self.hotkeys.register("toggle", self.settings.hotkey, self.toggle_enabled)
        self.hotkeys.register("image_mode", "ctrl+alt+i", self.open_image_mode)
        self.hotkeys.register("collapse", "ctrl+alt+p", lambda: self.panel.collapse_btn.toggle())
        self.hotkeys.register("center", "ctrl+alt+home", self.center_window)
        self.hotkeys.register("quit", "ctrl+alt+q", self.force_quit)

        # 拖放支援
        self.setAcceptDrops(True)

        # 啟動時跑一次自動平衡
        self.panel.sync_from_settings(self.settings)
        QTimer.singleShot(500, self._trigger_startup_auto_balance)

        self.show()
        self.request_refresh()

    def _trigger_startup_auto_balance(self) -> None:
        """啟動時執行的自動平衡。"""
        if getattr(self, '_last_gray_frame', None) is None or self._last_gray_frame.size == 0:
            QTimer.singleShot(500, self._trigger_startup_auto_balance)
            return
            
        target = self.panel.balance_presets.currentData()
        if target:
            self.on_auto_balance_target_requested(target)
        else:
            self.on_auto_balance_raw_requested()

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
            
        self.store.update(levels=levels, min_value=min_value, max_value=max_value, exp_value=exp_value)
        self._processed_distribution_pct = [0.0] * max(2, int(levels))
        self._raw_distribution_pct = [0.0] * max(2, int(levels))
        self._last_frame_signature = None
        self.request_refresh(5)

    def on_display_settings_changed(self, min_value: int, max_value: int, exp_value: float) -> None:
        self.store.update(display_min_value=min_value, display_max_value=max_value, display_exp_value=exp_value)
        self._last_frame_signature = None
        self.request_refresh(16)

    def on_effect_settings_changed(
        self, blur_enabled: bool, blur_radius: int,
        dither_enabled: bool, dither_strength: int
    ) -> None:
        self.store.update(blur_enabled=blur_enabled, blur_radius=blur_radius, dither_enabled=dither_enabled, dither_strength=dither_strength)
        self._last_frame_signature = None
        self.request_refresh(16)

    def on_order_changed(self, order: list[str]) -> None:
        self.store.update(process_order=order)
        self.image_mode.process_order = order
        self._last_frame_signature = None
        self.request_refresh(16)

    def on_morph_settings_changed(self, enabled: bool, strength: int, threshold: int = 35) -> None:
        self.store.update(morph_enabled=enabled, morph_strength=strength, morph_threshold=threshold)
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
        self.store.update(compare_mode=enabled)
        self._last_frame_signature = None
        self._raw_frame = QPixmap()
        self._layout_overlay_buttons()
        self.request_refresh()

    def on_compare_bw_changed(self, bw_enabled: bool) -> None:
        self.store.update(compare_bw=bw_enabled)
        self._last_frame_signature = None
        self._raw_frame = QPixmap()
        self.request_refresh()

    def on_hotkey_changed(self, hotkey: str) -> None:
        self.store.update(hotkey=hotkey)
        self.hotkeys.register("toggle", hotkey, self.toggle_enabled)

    def _apply_balance_to_ui(self, lower: int, upper: int, exp_value: float) -> None:
        """統一將平衡參數套用到所有 UI 組件，不觸發循環信號。"""
        self.panel.exp_slider.blockSignals(True)
        self.panel.range_slider.blockSignals(True)
        
        self.panel.exp_slider.setValue(int(round(-exp_value * 100.0)))
        self.panel.range_slider.set_values(lower, upper)
        
        self.panel.exp_slider.blockSignals(False)
        self.panel.range_slider.blockSignals(False)
        
        # 同步更新本地設定
        self.store.update(min_value=lower, max_value=upper, exp_value=exp_value)
    def on_auto_balance_target_requested(self, ratios: tuple[float, float, float]) -> None:
        # 決定計算來源：是當前可見的 viewport 還是整張圖片
        source_gray = self._last_gray_frame
        if self._is_static_mode and self._use_global_calc and self._full_image_gray is not None:
            source_gray = self._full_image_gray
            
        if source_gray is None or source_gray.size == 0:
            return
            
        from valuelens.core.balance import optimize_balance_params
        lower, upper, exp_value = optimize_balance_params(
            source_gray,
            ratios,
            self.settings.min_value,
            self.settings.max_value,
            self.settings.exp_value,
            self.settings.levels
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
            self.request_refresh(16)

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
        self.store.update(edge_enabled=enabled, edge_strength=strength, edge_mix=mix)
        self._last_frame_signature = None
        self.request_refresh(16)

    def on_auto_balance_raw_requested(self) -> None:
        # 點擊重新平衡時，關閉持續模式
        if self._auto_continuous_enabled:
            self._auto_continuous_enabled = False
            self.panel.auto_continuous_check.setChecked(False)

        from valuelens.core.balance import levels_to_wgb
        raw_wgb = levels_to_wgb(self._raw_distribution_pct)
        target_wgb = self.panel.nearest_balance_preset(raw_wgb)
        self.panel.set_balance_preset(target_wgb, mark_best=True)
        self.on_auto_balance_target_requested(target_wgb)

    def center_window(self) -> None:
        """將視窗移至目前螢幕的中央 (用於視窗跳出螢幕外時找回)。"""
        from PySide6.QtGui import QGuiApplication
        # 優先找滑鼠所在螢幕，若找不著則找視窗所在，再者主螢幕
        screen = QGuiApplication.screenAt(self.cursor().pos())
        if not screen:
            screen = self.screen()
        if not screen:
            screen = QGuiApplication.primaryScreen()
            
        if screen:
            geo = screen.availableGeometry()
            # 確保視窗不會大於螢幕
            w = min(self.width(), geo.width() - 40)
            h = min(self.height(), geo.height() - 40)
            x = geo.x() + (geo.width() - w) // 2
            y = geo.y() + (geo.height() - h) // 2
            self.setGeometry(x, y, w, h)
            print(f"[Window] Reset position to center: {x}, {y} on screen {screen.name()}")

    def toggle_enabled(self) -> None:
        now = time.monotonic()
        if now - self._last_toggle_ts < 0.25:
            return
        self._last_toggle_ts = now
        self.store.update(enabled=not self.settings.enabled)
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

    def closeEvent(self, event: QCloseEvent) -> None:
        self._coalesce_timer.stop()
        if hasattr(self, "calc_worker") and self.calc_worker.isRunning():
            self.calc_worker.quit()
            self.calc_worker.wait()
        event.accept()

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
            t_start = time.perf_counter()
            t_captured = t_start
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
                t_captured = time.perf_counter()
            
            if frame is None or frame.size == 0: 
                self._is_refreshing = False
                return

            self._last_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 特徵檢查
            signature = self._frame_signature(frame)
            if signature == self._last_frame_signature and not self._frame.isNull():
                self._is_refreshing = False
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
                self.update()
                self._is_refreshing = False
            else:
                # 正常量化處理：丟進 QThread 背景計算
                self.calc_worker.process_frame(frame.copy(), self.settings)
                
                # 暫存執行緒運算當下的關鍵變數給 finished 使用
                self._last_calc_frame = frame.copy()
                self._last_calc_dpr = dpr
                self._last_calc_t_start = t_start
                self._last_calc_t_captured = t_captured
                
        except Exception as e:
            print(f"[Error] refresh_frame crash: {e}")
            self._is_refreshing = False

    def _on_calc_finished(self, logic_quantized: np.ndarray, logic_indices: np.ndarray, edges: np.ndarray | None, t_computed: float) -> None:
        try:
            if not hasattr(self, '_last_calc_frame') or self._last_calc_frame is None:
                return
                
            frame = self._last_calc_frame
            dpr = self._last_calc_dpr
            t_start = self._last_calc_t_start
            t_captured = self._last_calc_t_captured
            
            # 釋放引用避免記憶體洩漏
            self._last_calc_frame = None
            
            h, w = logic_quantized.shape[:2]
            
            # 處理邊緣與比例 (Edge Mix)
            if edges is not None:
                mix = self.settings.edge_mix / 100.0
                ec = self.settings.edge_color[::-1] # RGB -> BGR
                
                base_bgr = cv2.cvtColor(logic_quantized, cv2.COLOR_GRAY2BGR)
                
                if mix >= 1.0:
                    final_bgr = np.full_like(base_bgr, 255)
                    final_bgr[edges > 0] = ec
                else:
                    bg_part = (base_bgr.astype(np.float32) * (1.0 - mix) + 255.0 * mix).astype(np.uint8)
                    final_bgr = bg_part
                    final_bgr[edges > 0] = ec
                
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
            if hasattr(self, 'canvas'):
                self.canvas.update()
            t_rendered = time.perf_counter()
            
            # --- Profiling 測速節流輸出 ---
            PROFILING = True
            if PROFILING:
                if not hasattr(self, '_last_profile_ts'):
                    self._last_profile_ts = 0.0
                now_ts = time.monotonic()
                if now_ts - self._last_profile_ts > 0.33:
                    self._last_profile_ts = now_ts
                    c_cap = (t_captured - t_start) * 1000
                    c_comp = t_computed * 1000
                    c_ui = (t_rendered - (t_captured + t_computed)) * 1000
                    c_total = (t_rendered - t_start) * 1000
                    
                    s = self.settings
                    lvl = s.levels
                    b_on = "B" if (s.blur_enabled and s.blur_radius > 0) else "-"
                    d_on = "D" if (s.dither_enabled and s.dither_strength > 0) else "-"
                    e_on = "E" if (s.edge_enabled and s.edge_strength > 0) else "-"
                    m_on = "M" if (s.morph_enabled and s.morph_strength > 0) else "-"
                    comp_on = "C" if self._compare_mode else "-"
                    
                    state_str = f"[Lv:{lvl} {b_on}{d_on}{e_on}{m_on}{comp_on}]"
                    print(f"[Profiling]{state_str} Cap:{c_cap:5.1f}ms | Calc:{c_comp:5.1f}ms | Render:{c_ui:5.1f}ms | Total:{c_total:5.1f}ms")

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
                if self._is_static_mode and self._use_global_calc:
                    if self._global_calc_dirty:
                        self._global_calc_dirty = False
                        current_target = self.panel.balance_presets.currentData()
                        if current_target:
                            self.on_auto_balance_target_requested(current_target)
                        else:
                            self.on_auto_balance_raw_requested()
                else:
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
                pix = self.grab()
                self._mirror_window.update_frame(pix)

    @staticmethod
    def _frame_signature(frame: np.ndarray) -> bytes:
        if frame.size == 0:
            return b""
        sampled = frame[::8, ::8]
        return sampled.tobytes()

    # paintEvent 與 _draw_manual_button 已經搬移至 RenderWidget (valuelens/ui/render_widget.py)


    def _update_distributions(
        self, raw_gray: np.ndarray | None, processed_indices: np.ndarray | None
    ) -> None:
        from valuelens.core.balance import calc_level_distribution, calc_indices_distribution
        level_count = max(2, int(self.settings.levels))
        self._raw_distribution_pct = calc_level_distribution(raw_gray, level_count)
        self._processed_distribution_pct = calc_indices_distribution(
            processed_indices, level_count
        )

    # _draw_distribution_overlay 已經搬移至 RenderWidget (valuelens/ui/render_widget.py)

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

    def _on_snap_timeout(self) -> None:
        """當拖曳至邊緣超過時限，觸發全螢幕。"""
        if self._is_dragging and not self.isMaximized():
            self.toggle_maximize()
            self._is_dragging = False
            self._snap_timer.stop()

    def on_save_preset(self, index: int) -> None:
        from PySide6.QtWidgets import QInputDialog
        slot = self.settings.presets[index]
        old_name = slot["name"] if slot else f"Slot {index + 1}"
        
        name, ok = QInputDialog.getText(self, "儲存預設集", f"請輸入 Slot {index+1} 的名稱:", text=old_name)
        if not ok: return
        
        settings_dict = asdict(self.settings)
        # 排除遞迴欄位與特定狀態
        for key in ["presets", "startup_preset", "x", "y", "width", "height"]:
            if key in settings_dict: del settings_dict[key]
            
        self.settings.presets[index] = {
            "name": name,
            "data": settings_dict
        }
        self.store._manager.save(self.settings)
        self.panel.update_presets_ui(self.settings.presets)

    def on_load_preset(self, index: int) -> None:
        slot = self.settings.presets[index]
        if not slot: return
        
        data = slot["data"]
        defaults = asdict(AppSettings())
        # 保留目前的 presets 與視窗幾何
        current_presets = self.settings.presets
        current_geom = (self.settings.x, self.settings.y, self.settings.width, self.settings.height)
        
        valid_data = {k: v for k, v in data.items() if k in defaults}
        merged = {**defaults, **valid_data}
        merged["presets"] = current_presets
        merged["x"], merged["y"], merged["width"], merged["height"] = current_geom
        
        for k, v in merged.items():
            setattr(self.settings, k, v)
            
        self.panel.sync_from_settings(self.settings)
        # 觸發 UI 同步
        self.on_settings_changed(self.settings.levels, self.settings.min_value, self.settings.max_value, self.settings.exp_value)
        self.on_display_settings_changed(self.settings.display_min_value, self.settings.display_max_value, self.settings.display_exp_value)
        self.on_effect_settings_changed(self.settings.blur_enabled, self.settings.blur_radius, self.settings.dither_enabled, self.settings.dither_strength)
        self.on_edge_settings_changed(self.settings.edge_enabled, self.settings.edge_strength, self.settings.edge_mix)
        self.on_morph_settings_changed(self.settings.morph_enabled, self.settings.morph_strength, self.settings.morph_threshold)
        self.on_order_changed(self.settings.process_order)

    def on_clear_preset(self, index: int) -> None:
        self.settings.presets[index] = None
        self.store._manager.save(self.settings)
        self.panel.update_presets_ui(self.settings.presets)

    def on_save_startup_preset(self) -> None:
        settings_dict = asdict(self.settings)
        for key in ["presets", "startup_preset", "x", "y", "width", "height"]:
            if key in settings_dict: del settings_dict[key]
        self.settings.startup_preset = settings_dict
        self.store._manager.save(self.settings)

    def on_clear_startup_preset(self) -> None:
        self.settings.startup_preset = None
        self.store._manager.save(self.settings)

    def on_debug_screenshot_requested(self) -> None:
        # 如果有實作擷取全螢幕的方法則呼叫，否則使用一般截圖
        self.on_screenshot_requested()

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
        # 允許滑鼠中鍵在任何地方拖動視窗 (解決 StaticMode 下左鍵被平移佔用的問題)
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_dragging = True
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return

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

            # 靜態模式 (StaticMode) 下鏡片區域 → 左鍵平移圖片
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
        # 檢查是否按下了左鍵或中鍵
        if not (event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.MiddleButton)):
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

        # ImageMode 平移圖片 (僅限左鍵)
        if (event.buttons() & Qt.MouseButton.LeftButton) and self._pan_drag_start is not None and self._pan_start_offset is not None:
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
        
        # 處理 Windows Snap 邏輯：檢查滑鼠是否撞到螢幕邊緣
        global_pos = event.globalPosition().toPoint()
        from PySide6.QtGui import QGuiApplication
        screen = QGuiApplication.screenAt(global_pos)
        if screen:
            geo = screen.availableGeometry()
            # 判斷是否撞到四周邊界 (容許 3px 誤差)
            margin = 3
            is_at_edge = (
                global_pos.x() <= geo.left() + margin or
                global_pos.x() >= geo.right() - margin or
                global_pos.y() <= geo.top() + margin or
                global_pos.y() >= geo.bottom() - margin
            )
            
            if is_at_edge:
                if not self._snap_timer.isActive():
                    self._snap_timer.start(500) # 0.5秒觸發
            else:
                self._snap_timer.stop()
        else:
            self._snap_timer.stop()

        self.move(global_pos - self._drag_pos)
        self.request_refresh(0)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        self._snap_timer.stop()
        if self._is_dragging:
            # 移除舊的即時 Top Snap，改用 MoveEvent 中的 Timer 觸發
            pass
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
        self.store.update(x=geom.x(), y=geom.y(), width=geom.width(), height=geom.height())
        self.hotkeys.shutdown()
        self.panel.close()
        self.image_mode.close()
        event.accept()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if hasattr(self, 'canvas'):
            self.canvas.setGeometry(self.rect())
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





