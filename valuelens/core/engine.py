import time
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition

from valuelens.core.quantize import quantize_gray_with_indices
from valuelens.config.settings import AppSettings

class ImageProcessWorker(QThread):
    """背景影像處理運算執行緒，負責執行耗時的 quantize_gray_with_indices。"""
    # finished 信號：帶回 logic_quantized, logic_indices, edges, t_computed
    finished = Signal(np.ndarray, np.ndarray, object, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.mutex = QMutex()
        self.cond = QWaitCondition()
        self._busy = False
        self._pending_task = None
        self._is_stopping = False
        self.start()  # 啟動並進入休眠等待

    def is_busy(self) -> bool:
        return self._busy

    def process_frame(self, frame: np.ndarray, settings: AppSettings) -> None:
        """排程一個運算任務，若目前在忙則直接覆蓋暫存任務(永遠只算最新的一張)。"""
        self.mutex.lock()
        if not self._is_stopping:
            self._pending_task = (frame, settings)
            self.cond.wakeOne()
        self.mutex.unlock()

    def stop(self) -> None:
        """優雅且安全地終止執行緒。"""
        self.mutex.lock()
        self._is_stopping = True
        self.cond.wakeOne()
        self.mutex.unlock()
        self.quit()
        self.wait(2000)

    def run(self) -> None:
        while True:
            self.mutex.lock()
            while self._pending_task is None and not self._is_stopping:
                self.cond.wait(self.mutex)
                
            if self._is_stopping:
                self.mutex.unlock()
                break
                
            task = self._pending_task
            self._pending_task = None
            self.mutex.unlock()
            
            self._busy = True
            frame, settings = task
            
            t_start = time.perf_counter()
            
            eff_blur = settings.blur_radius if settings.blur_enabled else 0
            eff_dither = settings.dither_strength if settings.dither_enabled else 0
            eff_edge = settings.edge_strength if settings.edge_enabled else 0
            
            logic_quantized, logic_indices, edges = quantize_gray_with_indices(
                frame,
                settings.levels,
                settings.min_value,
                settings.max_value,
                settings.exp_value,
                display_min=settings.display_min_value,
                display_max=settings.display_max_value,
                display_exp=settings.display_exp_value,
                blur_radius=eff_blur,
                dither_strength=eff_dither,
                edge_strength=eff_edge,
                process_order=settings.process_order,
                morph_enabled=settings.morph_enabled,
                morph_strength=settings.morph_strength,
                morph_threshold=settings.morph_threshold,
            )
            t_computed = time.perf_counter() - t_start
            
            self.finished.emit(logic_quantized, logic_indices, edges, t_computed)
            
        self._busy = False
