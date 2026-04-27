from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap


def qimage_to_bgr(image: QImage) -> np.ndarray:
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    ptr = converted.bits()
    arr = np.frombuffer(ptr, np.uint8).reshape((converted.height(), converted.width(), 4))
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)


def bgr_to_qimage(image: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)


def gray_to_qimage(image: np.ndarray) -> QImage:
    gray = np.ascontiguousarray(image)
    h, w = gray.shape
    return QImage(gray.data, w, h, gray.strides[0], QImage.Format.Format_Grayscale8)


def bgr_to_qpixmap(image: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(bgr_to_qimage(image).copy())

