from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QAction, QColor, QMouseEvent, QPainter, QPaintEvent, QPainterPath
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QSlider,
    QToolButton,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QButtonGroup,
    QGridLayout,
    QSizePolicy,
    QFrame,
)

from valuelens.config.settings import AppSettings


_LEVEL_PRESETS = [2, 3, 5, 8]

# 核心比例定義
_RATIO_2 = (30.0, 0.0, 70.0)    # n=2 時使用 3:7 (無灰色)
_RATIO_3PLUS = (70.0, 20.0, 10.0) # n>=3 時使用 7:2:1

_PANEL_STYLE = """
QWidget#gl_panel { background: #1e1e22; color: #e0e0e0; border-top: 4px solid #3d89ff; }
QWidget#gl_top_row_bg { background: #252529; border-bottom: 1px solid #333; }
QLabel { color: #ccc; background: transparent; font-size: 10pt; }
QToolButton { 
    color: #eee; 
    background: #333; 
    border: 1px solid #444;
    padding: 3px 10px; 
    border-radius: 4px; 
    font-size: 9pt;
}
QToolButton:hover { background: #444; border: 1px solid #555; }
QToolButton:checked { background: #2e7bf6; border: 1px solid #1a5ed1; }
QToolButton[freeze_mode="frozen"] { background: #0078d7; color: white; border: 1px solid white; padding: 3px 10px; border-radius: 4px; font-size: 9pt; }
QToolButton[freeze_mode="image"] { background: #00a86b; color: white; border: 1px solid white; padding: 3px 10px; border-radius: 4px; font-size: 9pt; }
QToolButton#gl_winbtn { border: none; background: transparent; padding: 0 10px; font-size: 11pt; }

QToolButton#gl_winbtn:hover { background: rgba(255,255,255,20); }
QToolButton#gl_closebtn:hover { background: #e81123; color: white; }

QComboBox { 
    color: #eee; 
    background: #2d2d30; 
    border: 1px solid #444; 
    border-radius: 4px;
    padding: 2px 6px; 
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background-color: #2d2d30;
    color: #eee;
    selection-background-color: #2e7bf6;
    selection-color: white;
    border: 1px solid #444;
    outline: none;
}

QCheckBox { color: #ccc; background: transparent; spacing: 6px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #555;
                       background: #2d2d30; border-radius: 3px; }
QCheckBox::indicator:checked { background: #2e7bf6; border: 1px solid #1a5ed1;
                                image: none; }
QCheckBox::indicator:checked:hover { background: #3b8efb; }

QPushButton {
    color: #ccc;
    background: #2d2d30;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 8pt;
}
QPushButton:hover { background: #3d3d40; border: 1px solid #555; }
QPushButton:checked { background: #2e7bf6; color: white; border: 1px solid #1a5ed1; }
QPushButton[best="true"] { border: 1px solid #2e7bf6; color: #5fb4fa; font-weight: bold; }
QPushButton[best="true"]:checked { color: white; border: 1px solid #fff; }

QSlider::groove:horizontal {
    border: 1px solid #444;
    height: 6px;
    background: #222;
    margin: 2px 0;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #2e7bf6;
    border: 1px solid #444;
    height: 6px;
    margin: 2px 0;
    border-radius: 3px;
}
QSlider#blur_slider::sub-page:horizontal { background: #4CAF50; }
QSlider#dither_slider::sub-page:horizontal { background: #2196F3; }
QSlider#edge_slider::sub-page:horizontal { background: #FF9800; }
QSlider#edge_mix_slider::sub-page:horizontal { background: #FF9800; }
QSlider#morph_slider::sub-page:horizontal, QSlider#morph_thresh_slider::sub-page:horizontal { background: #E91E63; }

QSlider::handle:horizontal {
    background: #eee;
    border: 1px solid #aaa;
    width: 18px;
    height: 18px;
    margin: -7px 0;
    border-radius: 9px;
}
QSlider::handle:horizontal:hover {
    background: #fff;
    border: 1px solid #2e7bf6;
}
"""


class ControlPanel(QWidget):
    settings_changed = Signal(int, int, int, float)
    display_settings_changed = Signal(int, int, float)
    effect_settings_changed = Signal(bool, int, bool, int)
    collapse_toggled = Signal(bool)
    compare_mode_changed = Signal(bool)
    hotkey_changed = Signal(str)
    image_mode_requested = Signal()
    import_requested = Signal()
    screenshot_requested = Signal()
    distribution_toggled = Signal(bool)
    edge_settings_changed = Signal(bool, int, int)
    morph_settings_changed = Signal(bool, int, int)
    order_changed = Signal(list)
    quit_requested = Signal()
    minimize_requested = Signal()
    drag_started = Signal(object)
    auto_balance_raw_requested = Signal()
    auto_balance_target_requested = Signal(tuple)
    auto_continuous_toggled = Signal(bool)
    debug_screenshot_requested = Signal()
    maximize_requested = Signal()
    recording_window_toggled = Signal(bool)
    save_preset_requested = Signal(int)
    load_preset_requested = Signal(int)
    clear_preset_requested = Signal(int)
    save_startup_requested = Signal()
    clear_startup_requested = Signal()

    def __init__(self, settings: AppSettings, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("gl_panel")
        self.setFixedHeight(136) 
        self.setStyleSheet(_PANEL_STYLE)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self.settings = settings
        self._current_hotkey = settings.hotkey

        self.levels = QComboBox()
        for v in _LEVEL_PRESETS:
            self.levels.addItem(str(v), v)
        if settings.levels in _LEVEL_PRESETS:
            self.levels.setCurrentIndex(_LEVEL_PRESETS.index(settings.levels))
        else:
            self.levels.addItem(str(settings.levels), settings.levels)
            self.levels.setCurrentIndex(self.levels.count() - 1)
        self.levels.setFixedWidth(50)
        self.levels.setToolTip("階層數量 (Quantization Levels)")

        self.enabled_check = QCheckBox()
        self.enabled_check.setChecked(settings.compare_mode)
        self.enabled_check.setToolTip("開啟對照 (Toggle Reference View)")

        self.range_slider = DualHandleSlider(0, 255, settings.min_value, settings.max_value)
        self.range_slider.setToolTip("亮度上下限 (輸入範圍)")

        self.exp_slider = QSlider(Qt.Orientation.Horizontal)
        self.exp_slider.setMinimumWidth(60)
        self.exp_slider.setToolTip("偏差修正 (Bias/Gamma) - 左暗右亮")

        self.display_range_slider = DualHandleSlider(
            0, 255, settings.display_min_value, settings.display_max_value
        )
        self.display_range_slider.setToolTip("影像輸出範圍 (Output Range Mapping)")

        self.display_exp_slider = QSlider(Qt.Orientation.Horizontal)
        self.display_exp_slider.setRange(-200, 200)
        self.display_exp_slider.setValue(int(round(-settings.display_exp_value * 100)))
        self.display_exp_slider.setInvertedAppearance(False)
        self.display_exp_slider.setMinimumWidth(60)
        self.display_exp_slider.setToolTip("輸出偏差 (Output Bias)")

        self.more_btn = QToolButton()
        self.more_btn.setText("更多")
        self.more_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more_btn.setToolTip("選單")

        more_menu = QMenu(self.more_btn)
        
        # 預設集子選單
        self.preset_menu = QMenu("預設集 (Presets)", self)
        self.update_presets_ui(settings.presets)
        
        self.image_mode_action = QAction("圖片模式", self)
        self.hotkey_action = QAction(f"設定快捷鍵 ({settings.hotkey})", self)
        self.debug_screenshot_action = QAction("偵錯：擷取全螢幕 (含視窗)", self)
        self.quit_action = QAction("結束程式", self)
        
        more_menu.addMenu(self.preset_menu)
        more_menu.addSeparator()
        more_menu.addAction(self.image_mode_action)
        more_menu.addAction(self.hotkey_action)
        more_menu.addAction(self.debug_screenshot_action)
        more_menu.addSeparator()
        more_menu.addAction(self.quit_action)
        self.more_btn.setMenu(more_menu)

        self.min_btn = QToolButton()
        self.min_btn.setObjectName("gl_winbtn")
        self.min_btn.setText("－")
        self.min_btn.setToolTip("最小化")
        self.min_btn.setFixedWidth(34)

        self.max_btn = QToolButton()
        self.max_btn.setObjectName("gl_winbtn")
        self.max_btn.setText("□")  # 使用更簡單的方塊圖示
        self.max_btn.setToolTip("最大化/還原")
        self.max_btn.setFixedWidth(34)

        self.close_btn = QToolButton()
        self.close_btn.setObjectName("gl_closebtn")
        self.close_btn.setProperty("class", "gl_winbtn")
        self.close_btn.setText("×")
        self.close_btn.setToolTip("結束")
        self.close_btn.setFixedWidth(34)
        self.close_btn.setStyleSheet("QToolButton { border: none; padding: 0; font-weight: bold; font-size: 11pt; }")

        self.freeze_btn = QToolButton()
        self.freeze_btn.setText("🧊")
        self.freeze_btn.setToolTip("鎖定當下畫面 (就地凍結)")
        self.freeze_btn.clicked.connect(self.import_requested.emit)

        self.screenshot_btn = QToolButton()
        self.screenshot_btn.setText("📷")
        self.screenshot_btn.setToolTip("截圖至剪貼簿 (Copy to Clipboard)")
        self.screenshot_btn.clicked.connect(self.screenshot_requested.emit)

        self.image_mode_btn = QToolButton()
        self.image_mode_btn.setText("🖼️")
        self.image_mode_btn.setToolTip("圖片模式 (匯入外部圖片)")
        self.image_mode_btn.clicked.connect(self.image_mode_requested.emit)

        self.distribution_btn = QToolButton()
        self.distribution_btn.setText("📊")
        self.distribution_btn.setToolTip("顯示/隱藏灰階比例")
        self.distribution_btn.setCheckable(True)
        self.distribution_btn.setChecked(True)
        self.distribution_btn.toggled.connect(self.distribution_toggled.emit)
        
        self.record_btn = QToolButton()
        self.record_btn.setText("🎥")
        self.record_btn.setCheckable(True)
        self.record_btn.setToolTip("開啟/關閉 錄製專用鏡像視窗 (非透明)")
        self.record_btn.toggled.connect(self.recording_window_toggled.emit)

        self.balance_raw_btn = QToolButton()
        self.balance_raw_btn.setText("⚖️")
        self.balance_raw_btn.setToolTip("自動平衡：根據目前畫面分佈重新計算預設比例")

        self.auto_continuous_check = QCheckBox("Auto")
        self.auto_continuous_check.setToolTip("開啟後會不斷自動調整參數，鎖定目前選擇的比例")
        self.auto_continuous_check.setChecked(False)

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #888; font-family: monospace; font-weight: bold; margin-right: 10px;")

        self.balance_presets = QComboBox()
        self._update_balance_presets(settings.levels)
        self.balance_presets.setFixedWidth(85)
        self.balance_presets.setToolTip("平衡預設比例 (W:G:B)")

        self.logic_reset_btn = QToolButton()
        self.logic_reset_btn.setText("Reset")
        self.logic_reset_btn.setToolTip("重置第一階段 (Logic) 參數")
        self.display_reset_btn = QToolButton()
        self.display_reset_btn.setText("Reset")
        self.display_reset_btn.setToolTip("重置第二階段 (Display) 參數")

        self.clear_palette_btn = QToolButton()
        self.clear_palette_btn.setText("🎨清空染色")
        self.clear_palette_btn.setToolTip("清除目前色彩對應，還原為預設黑白灰")
        
        self.clear_all_btn = QToolButton()
        self.clear_all_btn.setText("🧹清空全部")
        self.clear_all_btn.setToolTip("還原所有參數到預設、最安全乾淨的狀態")

        # Sliders
        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setObjectName("blur_slider")
        self.blur_slider.setRange(0, 50)
        self.blur_slider.setValue(settings.blur_radius)
        self.blur_slider.setMinimumWidth(80)

        self.dither_slider = QSlider(Qt.Orientation.Horizontal)
        self.dither_slider.setObjectName("dither_slider")
        self.dither_slider.setRange(0, 100)
        self.dither_slider.setValue(settings.dither_strength)
        self.dither_slider.setMinimumWidth(80)

        self.edge_slider = QSlider(Qt.Orientation.Horizontal)
        self.edge_slider.setObjectName("edge_slider")
        self.edge_slider.setRange(0, 100)
        self.edge_slider.setValue(settings.edge_strength)
        self.edge_slider.setMinimumWidth(80)
        self.edge_slider.setToolTip("連線偵測強度 (Canny Threshold)")

        self.edge_mix_slider = QSlider(Qt.Orientation.Horizontal)
        self.edge_mix_slider.setObjectName("edge_mix_slider")
        self.edge_mix_slider.setRange(0, 100)
        self.edge_mix_slider.setValue(settings.edge_mix)
        self.edge_mix_slider.setMinimumWidth(80)
        self.edge_mix_slider.setToolTip("連線混合比例 (0:原圖, 100:純邊緣)")

        self.morph_slider = QSlider(Qt.Orientation.Horizontal)
        self.morph_slider.setObjectName("morph_slider")
        self.morph_slider.setRange(0, 5)
        self.morph_slider.setValue(min(5, settings.morph_strength))
        self.morph_slider.setMinimumWidth(80)
        self.morph_slider.setToolTip("勾邊粗細 (Morphological Gradient)")

        self.morph_thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.morph_thresh_slider.setObjectName("morph_thresh_slider")
        self.morph_thresh_slider.setRange(1, 255)
        self.morph_thresh_slider.setInvertedAppearance(True) # 反轉滑桿外觀：往右拉 = 門檻低 = 線條變多
        self.morph_thresh_slider.setValue(settings.morph_threshold)
        self.morph_thresh_slider.setMinimumWidth(80)
        self.morph_thresh_slider.setToolTip("勾邊靈敏度門檻 (降低防爆)")

        # Order Widget
        states = {
            "blur": settings.blur_enabled,
            "dither": settings.dither_enabled,
            "edge": settings.edge_enabled,
            "morph": settings.morph_enabled
        }
        self.order_widget = DraggableOrderWidget(settings.process_order, states)
        self.order_widget.order_changed.connect(self.order_changed.emit)
        self.order_widget.toggle_requested.connect(self._on_module_toggle)

        self.collapse_btn = QToolButton()
        self.collapse_btn.setText("▲")
        self.collapse_btn.setToolTip("收合/展開設定面板")
        self.collapse_btn.setCheckable(True)
        self.collapse_btn.setChecked(False)
        self.collapse_btn.toggled.connect(self._on_collapse_toggled)


        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(10, 8, 10, 4)
        top_layout.setSpacing(8)
        
        # 階層分組
        top_layout.addWidget(QLabel("階層"))
        top_layout.addWidget(self.levels)
        top_layout.addWidget(self.enabled_check)
        
        # 間隔線
        v_line1 = QFrame()
        v_line1.setFrameShape(QFrame.Shape.VLine)
        v_line1.setStyleSheet("color: #444;")
        top_layout.addWidget(v_line1)
        
        # 平衡工具組
        top_layout.addWidget(self.balance_presets)
        top_layout.addWidget(self.auto_continuous_check)
        top_layout.addWidget(self.balance_raw_btn)
        
        v_line2 = QFrame()
        v_line2.setFrameShape(QFrame.Shape.VLine)
        v_line2.setStyleSheet("color: #444;")
        top_layout.addWidget(v_line2)
        
        # 擷取與功能組
        top_layout.addWidget(self.freeze_btn)
        top_layout.addWidget(self.screenshot_btn)
        top_layout.addWidget(self.image_mode_btn)
        top_layout.addWidget(self.distribution_btn)
        top_layout.addWidget(self.record_btn)
        
        top_layout.addStretch(1)
        top_layout.addWidget(self.fps_label)
        top_layout.addWidget(self.collapse_btn)
        top_layout.addWidget(self.more_btn)
        top_layout.addSpacing(6)
        top_layout.addWidget(self.min_btn)
        top_layout.addWidget(self.max_btn)
        top_layout.addWidget(self.close_btn)

        self.top_row_widget = QWidget()
        self.top_row_widget.setObjectName("gl_top_row_bg")
        self.top_row_widget.setLayout(top_layout)

        row2 = QHBoxLayout()
        row2.setContentsMargins(10, 2, 10, 2)
        row2.setSpacing(8)
        row2.addWidget(QLabel("📥 輸入"))
        row2.addWidget(self.range_slider, 1)
        self.logic_exp_label = QLabel("偏差")
        self.logic_exp_label.setFixedWidth(100)
        row2.addWidget(self.logic_exp_label)
        row2.addWidget(self.exp_slider)

        # 標籤排序行 (改為自適應寬度)
        order_row = QHBoxLayout()
        order_row.setContentsMargins(8, 2, 8, 2)
        order_row.addWidget(self.order_widget)

        # 參數調整區域 (改用 GridLayout 以防炸掉)
        params_grid = QGridLayout()
        params_grid.setContentsMargins(8, 4, 8, 4)
        params_grid.setSpacing(10)
        params_grid.setHorizontalSpacing(30)
        
        params_grid.addWidget(QLabel("Blur"), 0, 0)
        params_grid.addWidget(self.blur_slider, 0, 1)
        params_grid.addWidget(QLabel("Dither"), 0, 2)
        params_grid.addWidget(self.dither_slider, 0, 3)
        params_grid.addWidget(QLabel("Morph"), 0, 4)
        params_grid.addWidget(self.morph_slider, 0, 5)
        
        params_grid.addWidget(QLabel("Edge Strength"), 1, 0)
        params_grid.addWidget(self.edge_slider, 1, 1)
        params_grid.addWidget(QLabel("Edge Mix"), 1, 2)
        params_grid.addWidget(self.edge_mix_slider, 1, 3)
        params_grid.addWidget(QLabel("Morph Thresh"), 1, 4)
        params_grid.addWidget(self.morph_thresh_slider, 1, 5)
        
        params_grid.setColumnStretch(1, 1)
        params_grid.setColumnStretch(3, 1)
        params_grid.setColumnStretch(5, 1)

        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(10, 2, 10, 6)
        bottom_row.setSpacing(8)
        bottom_row.addWidget(QLabel("📤 輸出"))
        bottom_row.addWidget(self.display_range_slider, 1)
        self.display_exp_label = QLabel("偏差")
        self.display_exp_label.setFixedWidth(100)
        bottom_row.addWidget(self.display_exp_label)
        bottom_row.addWidget(self.display_exp_slider)

        # 獨立的重置功能按鈕行
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(10, 4, 10, 6)
        btn_row.setSpacing(12)
        btn_row.addStretch(1)
        
        # 變更按鈕名稱以利對齊
        self.logic_reset_btn.setText("🔄重置輸入")
        self.display_reset_btn.setText("🔄重置輸出")
        
        btn_row.addWidget(self.logic_reset_btn)
        btn_row.addWidget(self.display_reset_btn)
        btn_row.addWidget(self.clear_palette_btn)
        btn_row.addWidget(self.clear_all_btn)
        btn_row.addStretch(1)

        self.extra_container = QWidget()
        extra_layout = QVBoxLayout(self.extra_container)
        extra_layout.setContentsMargins(0, 0, 0, 0)
        extra_layout.setSpacing(0)
        extra_layout.addLayout(row2)
        extra_layout.addLayout(order_row)
        extra_layout.addLayout(params_grid)
        extra_layout.addLayout(bottom_row)
        extra_layout.addLayout(btn_row)

        layout = QVBoxLayout(self)
        self.setFixedHeight(230) # 兩行參數後增加高度
        self.setMinimumWidth(500) # 防止視窗縮太小導致 UI 完全崩潰
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.top_row_widget)
        layout.addWidget(self.extra_container)

        self.levels.currentIndexChanged.connect(self._on_levels_changed)
        self.enabled_check.toggled.connect(self.compare_mode_changed.emit)
        self.range_slider.range_changed.connect(self._on_range_change)
        self.exp_slider.valueChanged.connect(self._emit_settings)
        self.display_range_slider.range_changed.connect(self._on_display_range_change)
        self.display_exp_slider.valueChanged.connect(self._emit_display_settings)
        self.blur_slider.valueChanged.connect(self._emit_effect_settings)
        self.dither_slider.valueChanged.connect(self._emit_effect_settings)
        self.edge_slider.valueChanged.connect(self._emit_edge_settings)
        self.edge_mix_slider.valueChanged.connect(self._emit_edge_settings)
        self.morph_slider.valueChanged.connect(self._emit_morph_settings)
        self.morph_thresh_slider.valueChanged.connect(self._emit_morph_settings)
        self.image_mode_action.triggered.connect(self.image_mode_requested.emit)
        self.hotkey_action.triggered.connect(self._prompt_hotkey)
        self.debug_screenshot_action.triggered.connect(self.debug_screenshot_requested.emit)
        self.quit_action.triggered.connect(self.quit_requested.emit)
        self.min_btn.clicked.connect(self.minimize_requested.emit)
        self.max_btn.clicked.connect(self.maximize_requested.emit)
        self.close_btn.clicked.connect(self.quit_requested.emit)
        self.balance_presets.currentIndexChanged.connect(self._request_target_auto_balance)
        self.balance_raw_btn.clicked.connect(self._request_raw_auto_balance)
        self.auto_continuous_check.toggled.connect(self.auto_continuous_toggled.emit)
        self.logic_reset_btn.clicked.connect(self._reset_logic_settings)
        self.display_reset_btn.clicked.connect(self._reset_display_settings)
        self.clear_palette_btn.clicked.connect(self._clear_palette)
        self.clear_all_btn.clicked.connect(self._clear_all_settings)
        self.save_startup_action.triggered.connect(self.save_startup_requested.emit)
        self.clear_startup_action.triggered.connect(self.clear_startup_requested.emit)

    def _clear_palette(self) -> None:
        """清除自訂調色盤，回復灰階模式。"""
        self.settings.custom_palette = []
        self._emit_settings()

    def _clear_all_settings(self) -> None:
        """完全清除所有濾鏡與重置參數。"""
        s = self.settings
        s.blur_enabled = False
        s.dither_enabled = False
        s.edge_enabled = False
        s.morph_enabled = False
        s.levels = 3
        s.min_value = 0
        s.max_value = 255
        s.exp_value = 0.0
        s.display_min_value = 0
        s.display_max_value = 255
        s.display_exp_value = 0.0
        s.blur_radius = 0
        s.dither_strength = 0
        s.edge_strength = 50
        s.edge_mix = 100
        s.morph_strength = 1
        s.morph_threshold = 35
        s.custom_palette = []
        
        self.sync_from_settings(s)
        self._emit_settings()

    def _on_collapse_toggled(self, checked: bool) -> None:
        if checked:
            self.extra_container.hide()
            self.setFixedHeight(36)
            self.collapse_btn.setText("▼")
        else:
            self.extra_container.show()
            self.setFixedHeight(230)
            self.collapse_btn.setText("▲")
        self.collapse_toggled.emit(checked)

    def _on_levels_changed(self, index: int) -> None:
        lvl = _LEVEL_PRESETS[index]
        self._update_balance_presets(lvl)
        self._emit_settings()

    def _current_levels(self) -> int:
        data = self.levels.currentData()
        return int(data) if data is not None else _LEVEL_PRESETS[0]

    def _on_range_change(self, min_value: int, max_value: int) -> None:
        self._emit_settings()

    def _emit_settings(self, *_) -> None:
        val = -self.exp_slider.value() / 100.0
        self.logic_exp_label.setText(f"偏差 (2^{val:+.2f})")
        
        levels = self._current_levels()
        l_val = self.range_slider.lower_value
        u_val = self.range_slider.upper_value
        self.range_slider.set_levels(levels, l_val, u_val, val)
        
        self.settings_changed.emit(
            levels,
            l_val,
            u_val,
            val,
        )

    def _on_display_range_change(self, min_value: int, max_value: int) -> None:
        self._emit_display_settings()

    def _emit_display_settings(self, *_) -> None:
        val = -self.display_exp_slider.value() / 100.0
        self.display_exp_label.setText(f"偏差 (2^{val:+.2f})")
        
        # Display slider segment colors
        # For output range, we just use a simple gradient or standard behavior
        # But to be consistent, we could also use segments.
        # However, output mapping is usually just linear (or with its own exp).
        # We'll use the same levels but with display_exp.
        l_val = self.display_range_slider.lower_value
        u_val = self.display_range_slider.upper_value
        self.display_range_slider.set_levels(self._current_levels(), l_val, u_val, val)
        
        self.display_settings_changed.emit(
            l_val,
            u_val,
            val,
        )

    def set_fps(self, fps: float) -> None:
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def _on_module_toggle(self, key: str, state: bool) -> None:
        """當 DraggableOrderWidget 的標籤被點擊時，切換對應功能的開關。"""
        if key == "blur" or key == "dither":
            self._emit_effect_settings()
        elif key == "edge":
            self._emit_edge_settings()
        elif key == "morph":
            self._emit_morph_settings()

    def _emit_effect_settings(self, *_) -> None:
        self.effect_settings_changed.emit(
            self.order_widget._states.get("blur", True),
            self.blur_slider.value(),
            self.order_widget._states.get("dither", True),
            self.dither_slider.value(),
        )

    def _emit_edge_settings(self, *_) -> None:
        self.edge_settings_changed.emit(
            self.order_widget._states.get("edge", True),
            self.edge_slider.value(),
            self.edge_mix_slider.value(),
        )

    def _emit_morph_settings(self, *_) -> None:
        self.morph_settings_changed.emit(
            self.order_widget._states.get("morph", True),
            self.morph_slider.value(),
            self.morph_thresh_slider.value(),
        )

    def _prompt_hotkey(self) -> None:
        text, ok = QInputDialog.getText(
            self,
            "設定快捷鍵",
            "請輸入全域快捷鍵組合 (如 ctrl+alt+g):",
            text=self._current_hotkey,
        )
        if not ok:
            return
        value = text.strip().lower()
        if not value:
            return
        self._current_hotkey = value
        self.hotkey_action.setText(f"設定快捷鍵 ({value})")
        self.hotkey_changed.emit(value)

    def _request_target_auto_balance(self) -> None:
        data = self.balance_presets.currentData()
        if data:
            self.auto_balance_target_requested.emit(data)

    def _request_raw_auto_balance(self) -> None:
        self.auto_balance_raw_requested.emit()

    def update_presets_ui(self, presets: list[dict | None]) -> None:
        """動態建立 20 組預設集選單。"""
        self.preset_menu.clear()
        
        # 啟動預設集 (原有功能)
        self.save_startup_action = QAction("儲存目前設定為「啟動預設」", self)
        self.save_startup_action.triggered.connect(self.save_startup_requested.emit)
        self.clear_startup_action = QAction("清除並恢復「上次關閉狀態」", self)
        self.clear_startup_action.triggered.connect(self.clear_startup_requested.emit)
        self.preset_menu.addAction(self.save_startup_action)
        self.preset_menu.addAction(self.clear_startup_action)
        # 讀取自動預設集
        if getattr(self.settings, 'last_state', None) is not None:
            load_last_act = QAction("讀取：上次最後的設定 (自動)", self)
            load_last_act.triggered.connect(lambda: self.load_preset_requested.emit(-1))
            self.preset_menu.addAction(load_last_act)
        else:
            act1 = QAction("上次最後的設定 (無資料)", self)
            act1.setEnabled(False)
            self.preset_menu.addAction(act1)

        if getattr(self.settings, 'last_color_state', None) is not None:
            load_color_act = QAction("讀取：上次染色的設定 (自動)", self)
            load_color_act.triggered.connect(lambda: self.load_preset_requested.emit(-2))
            self.preset_menu.addAction(load_color_act)
        else:
            act2 = QAction("上次染色的設定 (無資料)", self)
            act2.setEnabled(False)
            self.preset_menu.addAction(act2)

        self.preset_menu.addSeparator()

        # 20 組欄位
        for i in range(20):
            preset = presets[i]
            is_empty = preset is None
            raw_name = preset["name"] if not is_empty else f"Slot {i+1}"
            # 限制選單名稱長度
            display_name = (raw_name[:15] + "...") if len(raw_name) > 15 else raw_name
            
            slot_menu = QMenu(f"{i+1:02d}. {display_name}", self)
            if is_empty:
                slot_menu.setStyleSheet("color: #777;")
            
            load_act = QAction("讀取 (Load)", self)
            load_act.setEnabled(not is_empty)
            load_act.triggered.connect(lambda checked=False, idx=i: self.load_preset_requested.emit(idx))
            
            save_act = QAction("儲存/更名 (Save & Rename)", self)
            save_act.triggered.connect(lambda checked=False, idx=i: self.save_preset_requested.emit(idx))
            
            clear_act = QAction("清除 (Clear)", self)
            clear_act.setEnabled(not is_empty)
            clear_act.triggered.connect(lambda checked=False, idx=i: self.clear_preset_requested.emit(idx))
            
            slot_menu.addAction(load_act)
            slot_menu.addAction(save_act)
            slot_menu.addAction(clear_act)
            
            self.preset_menu.addMenu(slot_menu)

    def _update_balance_presets(self, n: int) -> None:
        """根據目前階層動態更新平衡預設選項。"""
        import itertools
        self.balance_presets.blockSignals(True)
        current_data = self.balance_presets.currentData()
        self.balance_presets.clear()

        if n == 2:
            base = _RATIO_2
            # n=2 時，我們只想要灰色為 0 的排列
            presets = [(base[0], 0.0, base[2]), (base[2], 0.0, base[0])]
        else:
            base = _RATIO_3PLUS
            # n>=3 時，取得所有 6 種排列組合
            presets = sorted(list(set(itertools.permutations(base))), reverse=True)
        
        for w, g, b in presets:
            if n == 2:
                name = f"{int(w)}:0:{int(b)}"
            else:
                name = f"{int(w)}:{int(g)}:{int(b)}"
            self.balance_presets.addItem(name, (w, g, b))

        # 嘗試恢復之前選取的索引
        idx = self.balance_presets.findData(current_data)
        if idx >= 0:
            self.balance_presets.setCurrentIndex(idx)
        else:
            self.balance_presets.setCurrentIndex(0)
        self.balance_presets.blockSignals(False)

    def set_balance_preset(self, ratio_wgb: tuple[float, float, float], mark_best: bool = False) -> None:
        best_idx = self._best_preset_index(ratio_wgb)
        self.balance_presets.blockSignals(True)
        self.balance_presets.setCurrentIndex(best_idx)
        self.balance_presets.blockSignals(False)

    def nearest_balance_preset(self, ratio_wgb: tuple[float, float, float]) -> tuple[float, float, float]:
        preset = self.balance_presets.itemData(self._best_preset_index(ratio_wgb))
        return preset if preset is not None else (70.0, 20.0, 10.0)

    def _best_preset_index(self, ratio_wgb: tuple[float, float, float]) -> int:
        best_idx = 0
        best_dist = float("inf")
        for i in range(self.balance_presets.count()):
            preset = self.balance_presets.itemData(i)
            if preset is None:
                continue
            dist = sum((preset[j] - ratio_wgb[j]) ** 2 for j in range(3))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def _reset_logic_settings(self) -> None:
        self.range_slider.set_values(0, 255)
        self.exp_slider.setValue(0)
        self._emit_settings()

    def _reset_display_settings(self) -> None:
        self.display_range_slider.set_values(0, 255)
        self.display_exp_slider.setValue(0)
        self._emit_display_settings()


    def sync_from_settings(self, settings: AppSettings) -> None:
        if settings.levels in _LEVEL_PRESETS:
            self.levels.setCurrentIndex(_LEVEL_PRESETS.index(settings.levels))
        self.range_slider.set_values(settings.min_value, settings.max_value)
        self.exp_slider.setValue(int(round(-settings.exp_value * 100)))
        self.display_range_slider.set_values(settings.display_min_value, settings.display_max_value)
        self.display_exp_slider.setValue(int(round(-settings.display_exp_value * 100)))
        self.enabled_check.setChecked(settings.compare_mode)
        self.blur_slider.setValue(settings.blur_radius)
        self.dither_slider.setValue(settings.dither_strength)
        states = {
            "blur": settings.blur_enabled,
            "dither": settings.dither_enabled,
            "edge": settings.edge_enabled,
            "morph": settings.morph_enabled
        }
        self.order_widget.set_order(settings.process_order, states)
        self.edge_slider.setValue(settings.edge_strength)
        self.edge_mix_slider.setValue(settings.edge_mix)
        self.morph_slider.setValue(min(5, settings.morph_strength))
        self.morph_thresh_slider.setValue(settings.morph_threshold)
        self._current_hotkey = settings.hotkey
        self.hotkey_action.setText(f"設定快捷鍵 ({settings.hotkey})")
        self._emit_display_settings()
        self.update_presets_ui(settings.presets)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            child = self.childAt(event.position().toPoint())
            # Allow dragging on empty space, labels, or the background containers themselves
            if child in (None, self.top_row_widget, self.extra_container) or isinstance(child, QLabel):
                self.drag_started.emit(event.globalPosition().toPoint())
                event.accept()
                return
        super().mousePressEvent(event)


class DualHandleSlider(QWidget):
    range_changed = Signal(int, int)

    def __init__(self, minimum: int, maximum: int, lower: int, upper: int, parent=None) -> None:
        super().__init__(parent)
        self.minimum = minimum
        self.maximum = maximum
        self.lower_value = max(minimum, min(upper - 1, lower))
        self.upper_value = min(maximum, max(lower + 1, upper))
        self._active_handle: str | None = None
        self.setMinimumWidth(180)
        self.setFixedHeight(30)
        self._levels_info: tuple[int, int, int, float] | None = None

    def set_levels(self, levels: int, min_val: int, max_val: int, exp_val: float) -> None:
        self._levels_info = (levels, min_val, max_val, exp_val)
        self.update()

    def set_values(self, lower: int, upper: int) -> None:
        new_lower = max(self.minimum, min(self.maximum - 1, lower))
        new_upper = max(new_lower + 1, min(self.maximum, upper))
        if new_lower == self.lower_value and new_upper == self.upper_value:
            return
        self.lower_value = new_lower
        self.upper_value = new_upper
        self.range_changed.emit(self.lower_value, self.upper_value)
        self.update()

    def _track_rect(self) -> QRect:
        return QRect(10, self.height() // 2 - 3, max(24, self.width() - 20), 6)

    def _value_to_x(self, value: int) -> int:
        track = self._track_rect()
        ratio = (value - self.minimum) / (self.maximum - self.minimum)
        return track.left() + int(ratio * track.width())

    def _x_to_value(self, x: int) -> int:
        track = self._track_rect()
        clamped = min(track.right(), max(track.left(), x))
        ratio = (clamped - track.left()) / track.width()
        return int(round(self.minimum + ratio * (self.maximum - self.minimum)))

    def paintEvent(self, event: QPaintEvent) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        track = self._track_rect()
        painter.setPen(Qt.PenStyle.NoPen)
        
        # 如果有階層資訊，繪製分段背景
        if self._levels_info:
            levels, min_v, max_v, exp_v = self._levels_info
            gamma = 2.0 ** exp_v
            
            # 建立剪裁區域
            painter.save()
            path = QPainterPath()
            path.addRoundedRect(track, 3, 3)
            painter.setClipPath(path)
            
            for k in range(levels):
                # 段落起始與結束比例 (0.0 ~ 1.0)
                # v = min + (max-min) * (norm ^ (1/gamma))
                # 所以 norm = (k/levels)
                # 邊界值 v_k = min + (max-min) * ((k/levels) ^ (1/gamma))
                
                def get_v(norm_val):
                    return min_v + (max_v - min_v) * (norm_val ** (1.0 / gamma))

                v_start = 0 if k == 0 else get_v(k / levels)
                v_end = 255 if k == levels - 1 else get_v((k + 1) / levels)
                
                x_start = self._value_to_x(int(round(v_start)))
                x_end = self._value_to_x(int(round(v_end)))
                
                tone = int(round((k / (levels - 1)) * 255))
                painter.setBrush(QColor(tone, tone, tone))
                painter.drawRect(x_start, track.top(), x_end - x_start, track.height())
            
            painter.restore()
        else:
            painter.setBrush(QColor("#333"))
            painter.drawRoundedRect(track, 3, 3)

        x1 = self._value_to_x(self.lower_value)
        x2 = self._value_to_x(self.upper_value)
        
        # 繪製選取範圍的半透明藍色覆蓋 (避免完全遮擋階層顏色)
        active = QRect(min(x1, x2), track.top(), abs(x2 - x1), track.height())
        painter.setBrush(QColor(46, 123, 246, 80)) # 增加透明度
        painter.drawRoundedRect(active, 3, 3)

        painter.setPen(QColor("#888"))
        for x in (x1, x2):
            painter.setBrush(QColor("#eee"))
            handle = QRect(x - 7, track.center().y() - 10, 14, 20)
            painter.drawRoundedRect(handle, 3, 3)

    def _lower_hit_rect(self) -> QRect:
        x = self._value_to_x(self.lower_value)
        return QRect(x - 15, self.height() // 2 - 13, 30, 26)

    def _upper_hit_rect(self) -> QRect:
        x = self._value_to_x(self.upper_value)
        return QRect(x - 15, self.height() // 2 - 13, 30, 26)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        pos = event.position().toPoint()
        if self._upper_hit_rect().contains(pos):
            self._active_handle = "upper"
            self._move_active_handle(pos.x())
            return
        if self._lower_hit_rect().contains(pos):
            self._active_handle = "lower"
            self._move_active_handle(pos.x())
            return
        x = pos.x()
        lower_x = self._value_to_x(self.lower_value)
        upper_x = self._value_to_x(self.upper_value)
        self._active_handle = "upper" if abs(x - upper_x) <= abs(x - lower_x) else "lower"
        self._move_active_handle(x)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._active_handle is None:
            return
        self._move_active_handle(event.position().toPoint().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        self._active_handle = None

    def _move_active_handle(self, x: int) -> None:
        value = self._x_to_value(x)
        if self._active_handle == "lower":
            self.set_values(value, self.upper_value)
        elif self._active_handle == "upper":
            self.set_values(self.lower_value, value)


class DraggableOrderWidget(QWidget):
    order_changed = Signal(list)
    toggle_requested = Signal(str, bool)

    def __init__(self, order: list[str], states: dict[str, bool], parent=None) -> None:
        super().__init__(parent)
        self._order = list(order)
        self._states = dict(states)
        self._item_map = {
            "blur": "Blur",
            "dither": "Dither",
            "edge": "Edge",
            "morph": "Morph"
        }
        self._colors = {
            "blur": "#4CAF50",
            "dither": "#2196F3",
            "edge": "#FF9800",
            "morph": "#E91E63"
        }
        self._active_idx = -1
        self._drag_x = 0
        self._has_dragged = False
        self.setMinimumHeight(26)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_order(self, order: list[str], states: dict[str, bool]) -> None:
        self._order = list(order)
        self._states = dict(states)
        self.update()

    def _item_rects(self) -> list[QRect]:
        w = self.width() // len(self._order)
        return [QRect(i * w, 0, w - 4, self.height()) for i in range(len(self._order))]

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rects = self._item_rects()
        for i, key in enumerate(self._order):
            if i == self._active_idx: continue
            self._draw_item(painter, rects[i], key)
            
        if self._active_idx != -1:
            w = self.width() // len(self._order)
            drag_rect = QRect(self._drag_x - w//2, 0, w - 4, self.height())
            self._draw_item(painter, drag_rect, self._order[self._active_idx], alpha=180)

    def _draw_item(self, painter, rect, key, alpha=255):
        is_on = self._states.get(key, True)
        if is_on:
            color = QColor(self._colors[key])
        else:
            color = QColor("#444")
            
        color.setAlpha(alpha)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, 4, 4)
        
        painter.setPen(QColor("white") if is_on else QColor("#888"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._item_map[key])

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.position().toPoint()
        rects = self._item_rects()
        for i, r in enumerate(rects):
            if r.contains(pos):
                self._active_idx = i
                self._drag_x = pos.x()
                self._has_dragged = False
                self.update()
                break

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._active_idx == -1: return
        self._drag_x = event.position().toPoint().x()
        self._has_dragged = True
        
        # Check for swap
        w = self.width() // len(self._order)
        new_idx = max(0, min(len(self._order)-1, self._drag_x // w))
        if new_idx != self._active_idx:
            item = self._order.pop(self._active_idx)
            self._order.insert(new_idx, item)
            self._active_idx = new_idx
            self.order_changed.emit(self._order)
            
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._active_idx != -1 and not self._has_dragged:
            key = self._order[self._active_idx]
            new_state = not self._states.get(key, True)
            self._states[key] = new_state
            self.toggle_requested.emit(key, new_state)
            
        self._active_idx = -1
        self.update()
