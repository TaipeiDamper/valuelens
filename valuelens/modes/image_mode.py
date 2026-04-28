from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QGuiApplication, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from valuelens.core.quantize import quantize_gray
from valuelens.core.qt_image import bgr_to_qpixmap, qimage_to_bgr

from valuelens.core.engine import ImageProcessWorker
from valuelens.config.settings import AppSettings

class ImageModeDialog(QDialog):
    def __init__(self, settings: AppSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ValueLens - 圖片模式")
        self.resize(900, 600)

        self.settings = settings
        self.worker = ImageProcessWorker(self)
        self.worker.finished.connect(self._on_calc_finished)

        self._source: Optional[np.ndarray] = None
        self._result: Optional[np.ndarray] = None
        self._import_callback = None

        self.preview = QLabel("請貼上圖片 (Ctrl+V) 或從主視窗匯入 / 拖放檔案至此")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(700, 450)
        self.preview.setStyleSheet("background:#222;color:#ccc;border:1px solid #555;")

        self.open_btn = QPushButton("開啟圖片")
        self.import_btn = QPushButton("匯入目前畫面")
        self.apply_btn = QPushButton("套用目前參數")
        self.copy_btn = QPushButton("複製結果")
        self.save_btn = QPushButton("儲存結果")

        self.open_btn.clicked.connect(self.open_file)
        self.import_btn.clicked.connect(self.import_current_window)
        self.apply_btn.clicked.connect(self.apply_filter)
        self.copy_btn.clicked.connect(self.copy_result)
        self.save_btn.clicked.connect(self.save_result)

        buttons = QHBoxLayout()
        buttons.addWidget(self.open_btn)
        buttons.addWidget(self.import_btn)
        buttons.addWidget(self.apply_btn)
        buttons.addWidget(self.copy_btn)
        buttons.addWidget(self.save_btn)

        root = QVBoxLayout(self)
        root.addWidget(self.preview)
        root.addLayout(buttons)

        paste_action = QAction(self)
        paste_action.setShortcut(QKeySequence("Ctrl+V"))
        paste_action.triggered.connect(self.paste_image)
        self.addAction(paste_action)

    def set_import_callback(self, callback) -> None:
        self._import_callback = callback

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇圖片", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            QMessageBox.warning(self, "錯誤", "無法讀取該圖片")
            return
        self._source = image
        self._result = None
        self.preview.setPixmap(bgr_to_qpixmap(image).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def paste_image(self) -> None:
        clipboard = QGuiApplication.clipboard()
        qimg = clipboard.image()
        if qimg.isNull():
            QMessageBox.information(self, "提示", "剪貼簿中沒有圖片")
            return
        self._source = qimage_to_bgr(qimg)
        self._result = None
        self.preview.setPixmap(bgr_to_qpixmap(self._source).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def import_current_window(self) -> None:
        if self._import_callback is None:
            QMessageBox.information(self, "提示", "目前無法匯入圖片")
            return
        frame = self._import_callback()
        if frame is None:
            QMessageBox.information(self, "提示", "目前沒有活動中的覆蓋視窗")
            return
        self._source = frame
        self._result = None
        self.apply_filter()

    def apply_filter(self) -> None:
        if self._source is None:
            QMessageBox.information(self, "提示", "請先開啟或貼上圖片")
            return
        
        self.apply_btn.setText("計算中...")
        self.apply_btn.setEnabled(False)
        self.worker.process_frame(self._source.copy(), self.settings)

    def _on_calc_finished(self, logic_quantized: np.ndarray, logic_indices: np.ndarray, edges: object, t_computed: float) -> None:
        self.apply_btn.setText("套用目前參數")
        self.apply_btn.setEnabled(True)
        
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
            self._result = final_bgr
        else:
            self._result = cv2.cvtColor(logic_quantized, cv2.COLOR_GRAY2BGR)

        self.preview.setPixmap(bgr_to_qpixmap(self._result).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def copy_result(self) -> None:
        if self._result is None:
            QMessageBox.information(self, "提示", "請先套用參數")
            return
        qpixmap = bgr_to_qpixmap(self._result)
        QGuiApplication.clipboard().setPixmap(qpixmap)

    def save_result(self) -> None:
        if self._result is None:
            QMessageBox.information(self, "提示", "請先套用參數")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "輸出圖片", "valuelens_output.png", "PNG (*.png);;JPEG (*.jpg)"
        )
        if not path:
            return
        cv2.imwrite(path, self._result)
