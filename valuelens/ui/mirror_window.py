from __future__ import annotations

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPainter, QPixmap, QResizeEvent
from PySide6.QtWidgets import QWidget


class MirrorWindow(QWidget):
    """
    錄製專用鏡像視窗。
    這是一個標準的視窗（非透傳、非無邊框），方便錄影軟體（如 OBS）直接擷取。
    它會接收主視窗的渲染結果並等比例顯示。
    """
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ValueLens - Recording Mirror")
        self.resize(1024, 640)
        self.setMinimumSize(320, 240)
        self._pixmap = QPixmap()
        
        # 錄製視窗通常不需要置頂，除非使用者有需求
        # self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

    def update_frame(self, pixmap: QPixmap) -> None:
        """更新顯示的影像內容。"""
        self._pixmap = pixmap
        self.update()

    def paintEvent(self, event) -> None:
        if self._pixmap.isNull():
            painter = QPainter(self)
            painter.fillRect(self.rect(), Qt.GlobalColor.black)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # 背景填黑
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        
        # 等比例縮放繪製
        s = self._pixmap.size()
        s.scale(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        
        target_rect = s
        x = (self.width() - s.width()) // 2
        y = (self.height() - s.height()) // 2
        
        painter.drawPixmap(x, y, s.width(), s.height(), self._pixmap)

    def closeEvent(self, event) -> None:
        # 當錄製視窗關閉時，可以通知主視窗同步狀態（如果需要）
        super().closeEvent(event)
