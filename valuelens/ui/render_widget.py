from __future__ import annotations

from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QPainter, QColor, QPixmap
from PySide6.QtWidgets import QWidget

class RenderWidget(QWidget):
    """
    獨立的渲染畫布 (Render Canvas)
    負責將處理後與原始的畫面渲染到畫面上，並繪製灰階比例等圖層。
    """
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self._parent = parent
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        
        # 取得視窗幾何資訊
        lens = self._parent._lens_rect()
        compare = self._parent._compare_rect()
        
        painter.setClipRect(self.rect())
        
        if self._parent._frame.isNull():
            painter.fillRect(lens, Qt.GlobalColor.black)
        else:
            painter.drawPixmap(lens.topLeft(), self._parent._frame)
            
        if self._parent._compare_mode and not compare.isNull():
            if self._parent._raw_frame.isNull():
                painter.fillRect(compare, Qt.GlobalColor.black)
            else:
                painter.drawPixmap(compare.topLeft(), self._parent._raw_frame)
                
            painter.setPen(Qt.GlobalColor.darkGray)
            divider_x = lens.right() + (self._parent._compare_gap // 2) + 1
            painter.drawLine(divider_x, lens.y(), divider_x, lens.bottom())
        
        # 繪製外框
        painter.setPen(QColor(100, 100, 100, 180))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        
        painter.setPen(Qt.GlobalColor.white)
        painter.drawRect(lens.adjusted(0, 0, -1, -1))
        
        if self._parent._show_distribution:
            self._draw_distribution_overlay(painter, lens, self._parent._processed_distribution_pct)
            
        if self._parent._compare_mode and not compare.isNull():
            painter.drawRect(compare.adjusted(0, 0, -1, -1))
            if self._parent._show_distribution:
                self._draw_distribution_overlay(painter, compare, self._parent._raw_distribution_pct)
                
        # 繪製手動按鈕
        if not self._parent._rect_compare_bw.isNull():
            self._draw_manual_button(painter, self._parent._rect_compare_bw, "黑白", self._parent.settings.compare_bw)
        if not self._parent._rect_global_calc.isNull():
            self._draw_manual_button(painter, self._parent._rect_global_calc, "計算全圖", self._parent._use_global_calc)

    def _draw_manual_button(self, painter: QPainter, rect: QRect, text: str, checked: bool) -> None:
        bg_color = QColor(46, 123, 246) if checked else QColor(0, 0, 0, 150)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.GlobalColor.white)
        painter.setBrush(bg_color)
        painter.drawRoundedRect(rect, 4, 4)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_distribution_overlay(self, painter: QPainter, rect: QRect, values: list[float]) -> None:
        display_values = list(reversed(values))
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
