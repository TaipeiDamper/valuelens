from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QAction, QColor, QMouseEvent, QPainter, QPaintEvent
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
)

from valuelens.config.settings import AppSettings


_LEVEL_PRESETS = [2, 3, 5, 8]
_BALANCE_PRESETS: list[tuple[float, float, float]] = [
    (10.0, 20.0, 70.0),
    (10.0, 70.0, 20.0),
    (20.0, 10.0, 70.0),
    (20.0, 70.0, 10.0),
    (70.0, 10.0, 20.0),
    (70.0, 20.0, 10.0),
]

_PANEL_STYLE = """
QWidget#gl_panel { background: #1e1e22; color: #e0e0e0; border-top: 2px solid #444; }
QLabel { color: #aaa; background: transparent; font-size: 9pt; }
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
    border: 1px solid #333;
    height: 4px;
    background: #333;
    margin: 2px 0;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #eee;
    border: 1px solid #aaa;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #fff;
    border: 1px solid #2e7bf6;
}
"""


class ControlPanel(QWidget):
    settings_changed = Signal(int, int, int, float)
    display_settings_changed = Signal(int, int, float)
    effect_settings_changed = Signal(bool, int, bool, int, bool)
    collapse_toggled = Signal(bool)
    compare_mode_changed = Signal(bool)
    hotkey_changed = Signal(str)
    image_mode_requested = Signal()
    quit_requested = Signal()
    minimize_requested = Signal()
    drag_started = Signal(object)
    auto_balance_raw_requested = Signal()
    auto_balance_target_requested = Signal(tuple)

    def __init__(self, settings: AppSettings, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("gl_panel")
        self.setFixedHeight(136)
        self.setStyleSheet(_PANEL_STYLE)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._current_hotkey = settings.hotkey

        self.levels = QComboBox()
        for v in _LEVEL_PRESETS:
            self.levels.addItem(str(v), v)
        if settings.levels in _LEVEL_PRESETS:
            self.levels.setCurrentIndex(_LEVEL_PRESETS.index(settings.levels))
        else:
            self.levels.addItem(str(settings.levels), settings.levels)
            self.levels.setCurrentIndex(self.levels.count() - 1)
        self.levels.setFixedWidth(68)
        self.levels.setToolTip("階層數量")

        self.enabled_check = QCheckBox("開啟對照")
        self.enabled_check.setChecked(settings.compare_mode)
        self.enabled_check.setToolTip("顯示或隱藏原始參考圖")

        self.range_slider = DualHandleSlider(0, 255, settings.min_value, settings.max_value)
        self.range_slider.setToolTip("亮度上下限 (輸入範圍)")

        self.exp_slider = QSlider(Qt.Orientation.Horizontal)
        self.exp_slider.setRange(-200, 200)
        self.exp_slider.setValue(int(round(-settings.exp_value * 100)))
        self.exp_slider.setInvertedAppearance(False)
        self.exp_slider.setFixedWidth(110)
        self.exp_slider.setToolTip("exp 修正偏移 (左暗右亮)")

        self.display_range_slider = DualHandleSlider(
            0, 255, settings.display_min_value, settings.display_max_value
        )
        self.display_range_slider.setToolTip("顯示亮度上下限 (影像輸出呈現)")

        self.display_exp_slider = QSlider(Qt.Orientation.Horizontal)
        self.display_exp_slider.setRange(-200, 200)
        self.display_exp_slider.setValue(int(round(-settings.display_exp_value * 100)))
        self.display_exp_slider.setInvertedAppearance(False)
        self.display_exp_slider.setFixedWidth(110)
        self.display_exp_slider.setToolTip("顯示 exp (僅影響顯示)")

        self.more_btn = QToolButton()
        self.more_btn.setText("更多")
        self.more_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more_btn.setToolTip("選單")

        more_menu = QMenu(self.more_btn)
        self.image_mode_action = QAction("圖片模式", self)
        self.hotkey_action = QAction(f"設定快捷鍵 ({settings.hotkey})", self)
        self.quit_action = QAction("結束程式", self)
        more_menu.addAction(self.image_mode_action)
        more_menu.addAction(self.hotkey_action)
        more_menu.addSeparator()
        more_menu.addAction(self.quit_action)
        self.more_btn.setMenu(more_menu)

        self.min_btn = QToolButton()
        self.min_btn.setObjectName("gl_winbtn")
        self.min_btn.setText("－")
        self.min_btn.setToolTip("最小化")
        self.min_btn.setFixedWidth(34)

        self.close_btn = QToolButton()
        self.close_btn.setObjectName("gl_closebtn")
        self.close_btn.setProperty("class", "gl_winbtn")
        self.close_btn.setText("×")
        self.close_btn.setToolTip("結束")
        self.close_btn.setFixedWidth(34)
        self.close_btn.setStyleSheet("QToolButton { border: none; padding: 0; font-weight: bold; }")

        self.balance_raw_btn = QToolButton()
        self.balance_raw_btn.setText("自動平衡")
        self.balance_raw_btn.setToolTip("根據目前原始畫面自動平衡")
        self.balance_target_btn = QToolButton()
        self.balance_target_btn.setText("目標平衡")
        self.balance_target_btn.setToolTip("根據選定比例進行平衡優化")

        self.preset_group = QButtonGroup(self)
        self.preset_group.setExclusive(True)
        self.preset_buttons = []
        for i, (white, gray, black) in enumerate(_BALANCE_PRESETS):
            name = f"{int(white)}:{int(gray)}:{int(black)}"
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setStyleSheet("padding: 2px 6px; font-size: 9pt;")
            self.preset_group.addButton(btn, i)
            self.preset_buttons.append(btn)
        self.preset_buttons[0].setChecked(True)

        self.logic_reset_btn = QToolButton()
        self.logic_reset_btn.setText("Reset")
        self.logic_reset_btn.setToolTip("重置第一階段 (Logic) 參數")
        self.display_reset_btn = QToolButton()
        self.display_reset_btn.setText("Reset")
        self.display_reset_btn.setToolTip("重置第二階段 (Display) 參數")

        self.blur_check = QCheckBox("平滑 Blur")
        self.blur_check.setChecked(settings.blur_enabled)
        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setRange(0, 50)
        self.blur_slider.setValue(settings.blur_radius)
        self.blur_slider.setFixedWidth(110)
        
        self.dither_check = QCheckBox("遞色 Dither")
        self.dither_check.setChecked(settings.dither_enabled)
        self.dither_slider = QSlider(Qt.Orientation.Horizontal)
        self.dither_slider.setRange(0, 100)
        self.dither_slider.setValue(settings.dither_strength)
        self.dither_slider.setFixedWidth(110)

        self.order_btn = QToolButton()
        self.order_btn.setCheckable(True)
        self.order_btn.setChecked(getattr(settings, 'dither_first', False))
        self.order_btn.setText("先平滑後遞色" if not getattr(settings, 'dither_first', False) else "先遞色後平滑")
        self.order_btn.setToolTip("點擊切換平滑與遞色兩階段的處理順序")
        self.order_btn.toggled.connect(self._on_order_toggled)

        self.collapse_btn = QToolButton()
        self.collapse_btn.setText("▲")
        self.collapse_btn.setToolTip("收合/展開設定面板")
        self.collapse_btn.setCheckable(True)
        self.collapse_btn.setChecked(False)
        self.collapse_btn.toggled.connect(self._on_collapse_toggled)

        preset_layout = QHBoxLayout()
        preset_layout.setSpacing(2)
        for btn in self.preset_buttons:
            preset_layout.addWidget(btn)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(8, 8, 8, 4)
        top_row.setSpacing(10)
        top_row.addWidget(QLabel("階層"))
        top_row.addWidget(self.levels)
        top_row.addWidget(self.enabled_check)
        top_row.addSpacing(10)
        top_row.addLayout(preset_layout)
        top_row.addWidget(self.balance_raw_btn)
        top_row.addWidget(self.balance_target_btn)
        top_row.addStretch(1)
        top_row.addWidget(self.collapse_btn)
        top_row.addWidget(self.more_btn)
        top_row.addSpacing(10)
        top_row.addWidget(self.min_btn)
        top_row.addWidget(self.close_btn)

        row2 = QHBoxLayout()
        row2.setContentsMargins(8, 0, 8, 2)
        row2.setSpacing(8)
        row2.addWidget(QLabel("Logic Range"))
        row2.addWidget(self.range_slider, 2)
        row2.addWidget(QLabel("Logic exp"))
        row2.addWidget(self.exp_slider)
        row2.addWidget(self.logic_reset_btn)

        row3 = QHBoxLayout()
        row3.setContentsMargins(8, 0, 8, 2)
        row3.setSpacing(8)
        row3.addWidget(self.blur_check)
        row3.addWidget(self.blur_slider, 2)
        row3.addWidget(self.order_btn)
        row3.addWidget(self.dither_check)
        row3.addWidget(self.dither_slider, 2)
        row3.addStretch(1)

        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(8, 0, 8, 4)
        bottom_row.setSpacing(8)
        bottom_row.addWidget(QLabel("Display Range"))
        bottom_row.addWidget(self.display_range_slider, 2)
        bottom_row.addWidget(QLabel("Display exp"))
        bottom_row.addWidget(self.display_exp_slider)
        bottom_row.addWidget(self.display_reset_btn)

        self.extra_container = QWidget()
        extra_layout = QVBoxLayout(self.extra_container)
        extra_layout.setContentsMargins(0, 0, 0, 0)
        extra_layout.setSpacing(0)
        extra_layout.addLayout(row2)
        extra_layout.addLayout(row3)
        extra_layout.addLayout(bottom_row)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(top_row)
        layout.addWidget(self.extra_container)

        self.levels.currentIndexChanged.connect(self._emit_settings)
        self.range_slider.range_changed.connect(self._on_range_change)
        self.exp_slider.valueChanged.connect(self._emit_settings)
        self.display_range_slider.range_changed.connect(self._on_display_range_change)
        self.display_exp_slider.valueChanged.connect(self._emit_display_settings)
        self.enabled_check.toggled.connect(self.compare_mode_changed.emit)
        self.blur_check.toggled.connect(self._emit_effect_settings)
        self.blur_slider.valueChanged.connect(self._emit_effect_settings)
        self.dither_check.toggled.connect(self._emit_effect_settings)
        self.dither_slider.valueChanged.connect(self._emit_effect_settings)
        self.image_mode_action.triggered.connect(self.image_mode_requested.emit)
        self.hotkey_action.triggered.connect(self._prompt_hotkey)
        self.quit_action.triggered.connect(self.quit_requested.emit)
        self.min_btn.clicked.connect(self.minimize_requested.emit)
        self.close_btn.clicked.connect(self.quit_requested.emit)
        self.balance_raw_btn.clicked.connect(self._request_raw_auto_balance)
        self.balance_target_btn.clicked.connect(self._request_target_auto_balance)
        self.logic_reset_btn.clicked.connect(self._reset_logic_settings)
        self.display_reset_btn.clicked.connect(self._reset_display_settings)

    def _on_collapse_toggled(self, checked: bool) -> None:
        if checked:
            self.extra_container.hide()
            self.setFixedHeight(36)
            self.collapse_btn.setText("▼")
        else:
            self.extra_container.show()
            self.setFixedHeight(136)
            self.collapse_btn.setText("▲")
        self.collapse_toggled.emit(checked)

    def _current_levels(self) -> int:
        data = self.levels.currentData()
        return int(data) if data is not None else _LEVEL_PRESETS[0]

    def _on_range_change(self, min_value: int, max_value: int) -> None:
        self._emit_settings()

    def _emit_settings(self, *_) -> None:
        self.settings_changed.emit(
            self._current_levels(),
            self.range_slider.lower_value,
            self.range_slider.upper_value,
            -self.exp_slider.value() / 100.0,
        )

    def _on_display_range_change(self, min_value: int, max_value: int) -> None:
        self._emit_display_settings()

    def _emit_display_settings(self, *_) -> None:
        self.display_settings_changed.emit(
            self.display_range_slider.lower_value,
            self.display_range_slider.upper_value,
            -self.display_exp_slider.value() / 100.0,
        )

    def _on_order_toggled(self, checked: bool) -> None:
        self.order_btn.setText("先遞色後平滑" if checked else "先平滑後遞色")
        self._emit_effect_settings()

    def _emit_effect_settings(self, *_) -> None:
        self.effect_settings_changed.emit(
            self.blur_check.isChecked(),
            self.blur_slider.value(),
            self.dither_check.isChecked(),
            self.dither_slider.value(),
            self.order_btn.isChecked(),
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
        idx = self.preset_group.checkedId()
        idx = max(0, min(len(_BALANCE_PRESETS) - 1, idx))
        self.auto_balance_target_requested.emit(_BALANCE_PRESETS[idx])

    def _request_raw_auto_balance(self) -> None:
        self.auto_balance_raw_requested.emit()

    def set_balance_preset(self, ratio_wgb: tuple[float, float, float], mark_best: bool = False) -> None:
        best_idx = 0
        best_dist = float("inf")
        for idx, preset in enumerate(_BALANCE_PRESETS):
            dist = sum((preset[i] - ratio_wgb[i]) ** 2 for i in range(3))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        for idx, btn in enumerate(self.preset_buttons):
            w, g, b = _BALANCE_PRESETS[idx]
            base_name = f"{int(w)}:{int(g)}:{int(b)}"
            is_best = mark_best and idx == best_idx
            btn.setText(f"{base_name}*" if is_best else base_name)
            btn.setProperty("best", "true" if is_best else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        
        if 0 <= best_idx < len(self.preset_buttons):
            self.preset_buttons[best_idx].setChecked(True)

    def nearest_balance_preset(self, ratio_wgb: tuple[float, float, float]) -> tuple[float, float, float]:
        best_idx = 0
        best_dist = float("inf")
        for idx, preset in enumerate(_BALANCE_PRESETS):
            dist = sum((preset[i] - ratio_wgb[i]) ** 2 for i in range(3))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return _BALANCE_PRESETS[best_idx]

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
        self.blur_check.setChecked(settings.blur_enabled)
        self.blur_slider.setValue(settings.blur_radius)
        self.dither_check.setChecked(settings.dither_enabled)
        self.dither_slider.setValue(settings.dither_strength)
        self.order_btn.setChecked(getattr(settings, 'dither_first', False))
        self._current_hotkey = settings.hotkey
        self.hotkey_action.setText(f"設定快捷鍵 ({settings.hotkey})")

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            child = self.childAt(event.position().toPoint())
            if child is None or isinstance(child, QLabel):
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
        painter.setBrush(QColor("#333"))
        painter.drawRoundedRect(track, 3, 3)

        x1 = self._value_to_x(self.lower_value)
        x2 = self._value_to_x(self.upper_value)
        active = QRect(min(x1, x2), track.top(), abs(x2 - x1), track.height())
        painter.setBrush(QColor("#2e7bf6"))
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
