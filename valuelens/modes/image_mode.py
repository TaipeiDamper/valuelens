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


class ImageModeDialog(QDialog):
    def __init__(
        self, levels: int, min_value: int, max_value: int, exp_value: float,
        blur_enabled: bool = False, blur_radius: int = 0,
        dither_enabled: bool = False, dither_strength: int = 0,
        process_order: list[str] = ("blur", "dither", "edge", "morph"),
        morph_enabled: bool = False, morph_strength: int = 1,
        morph_threshold: int = 35,
        parent=None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("ValueLens - 圖片模式")
        self.resize(900, 600)

        self.levels = levels
        self.min_value = min_value
        self.max_value = max_value
        self.exp_value = exp_value
        self.blur_enabled = blur_enabled
        self.blur_radius = blur_radius
        self.dither_enabled = dither_enabled
        self.dither_strength = dither_strength
        self.process_order = list(process_order)
        self.morph_enabled = morph_enabled
        self.morph_strength = morph_strength
        self.morph_threshold = morph_threshold

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

    def set_quantize_settings(
        self, levels: int, min_value: int, max_value: int, exp_value: float,
        blur_enabled: bool = False, blur_radius: int = 0,
        dither_enabled: bool = False, dither_strength: int = 0,
        process_order: list[str] = ("blur", "dither", "edge", "morph"),
        morph_enabled: bool = False, morph_strength: int = 1,
        morph_threshold: int = 35,
    ) -> None:
        self.levels = levels
        self.min_value = min_value
        self.max_value = max_value
        self.exp_value = exp_value
        self.blur_enabled = blur_enabled
        self.blur_radius = blur_radius
        self.dither_enabled = dither_enabled
        self.dither_strength = dither_strength
        self.process_order = list(process_order)
        self.morph_enabled = morph_enabled
        self.morph_strength = morph_strength
        self.morph_threshold = morph_threshold

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
        
        eff_blur = self.blur_radius if self.blur_enabled else 0
        eff_dither = self.dither_strength if self.dither_enabled else 0
        
        self._result = quantize_gray(
            self._source, self.levels, self.min_value, self.max_value, self.exp_value,
            blur_radius=eff_blur,
            dither_strength=eff_dither,
            process_order=self.process_order,
            morph_enabled=self.morph_enabled,
            morph_strength=self.morph_strength,
            morph_threshold=self.morph_threshold
        )
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
