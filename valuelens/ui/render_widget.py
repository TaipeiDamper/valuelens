from __future__ import annotations

from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QPainter, QColor, QPixmap
from PySide6.QtWidgets import QWidget

class RenderWidget(QWidget):
    """
    獨立的渲染畫布 (Render Canvas)
    負責將處理後與原始的畫面渲染到畫面上，並繪製灰階比例等圖層。
    """
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.lens_rect = QRect()
        self.compare_rect = QRect()
        self.frame = QPixmap()
        self.raw_frame = QPixmap()
        self.compare_mode = False
        self.compare_gap = 6
        self.show_distribution = True
        self.processed_distribution_pct = []
        self.raw_distribution_pct = []
        self.rect_compare_bw = QRect()
        self.compare_bw = False
        self.rect_global_calc = QRect()
        self.use_global_calc = False

    def update_data(self, **kwargs) -> None:
        """更新渲染所需資料並觸發重繪"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.update()
        
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        
        painter.setClipRect(self.rect())
        
        if self.frame.isNull():
            painter.fillRect(self.lens_rect, Qt.GlobalColor.black)
        else:
            painter.drawPixmap(self.lens_rect.topLeft(), self.frame)
            
        if self.compare_mode and not self.compare_rect.isNull():
            if self.raw_frame.isNull():
                painter.fillRect(self.compare_rect, Qt.GlobalColor.black)
            else:
                painter.drawPixmap(self.compare_rect.topLeft(), self.raw_frame)
                
            painter.setPen(Qt.GlobalColor.darkGray)
            divider_x = self.lens_rect.right() + (self.compare_gap // 2) + 1
            painter.drawLine(divider_x, self.lens_rect.y(), divider_x, self.lens_rect.bottom())
        
        # 繪製外框
        painter.setPen(QColor(100, 100, 100, 180))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        
        painter.setPen(Qt.GlobalColor.white)
        painter.drawRect(self.lens_rect.adjusted(0, 0, -1, -1))
        
        if self.show_distribution:
            self._draw_distribution_overlay(painter, self.lens_rect, self.processed_distribution_pct)
            
        if self.compare_mode and not self.compare_rect.isNull():
            painter.drawRect(self.compare_rect.adjusted(0, 0, -1, -1))
            if self.show_distribution:
                self._draw_distribution_overlay(painter, self.compare_rect, self.raw_distribution_pct)
                
        # 繪製手動按鈕
        if not self.rect_compare_bw.isNull():
            self._draw_manual_button(painter, self.rect_compare_bw, "黑白", self.compare_bw)
        if not self.rect_global_calc.isNull():
            self._draw_manual_button(painter, self.rect_global_calc, "計算全圖", self.use_global_calc)

    def _draw_manual_button(self, painter: QPainter, rect: QRect, text: str, checked: bool) -> None:
        bg_color = QColor(46, 123, 246) if checked else QColor(0, 0, 0, 150)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.GlobalColor.white)
        painter.setBrush(bg_color)
        painter.drawRoundedRect(rect, 4, 4)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_distribution_overlay(self, painter: QPainter, rect: QRect, values: list[float]) -> None:
        if not values:
            return
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
