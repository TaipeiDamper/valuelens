import time
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition

from valuelens.core.quantize import quantize_gray_with_indices
from valuelens.core.balance import optimize_balance_params
from valuelens.config.settings import AppSettings

class AutoBalanceWorker(QThread):
    finished = Signal(int, int, float, int) # min, max, exp, mode

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.mutex = QMutex()
        self.cond = QWaitCondition()
        self._busy = False
        self._pending_task = None
        self._is_stopping = False
        self.start()

    def is_busy(self) -> bool:
        return self._busy

    def request_balance(self, gray_frame: np.ndarray, target: tuple, current_min: int, current_max: int, current_exp: float, levels_count: int, hysteresis: float, current_mode: int = 0, search_all_modes: bool = False) -> None:
        self.mutex.lock()
        if not self._is_stopping:
            self._pending_task = (gray_frame, target, current_min, current_max, current_exp, levels_count, hysteresis, current_mode, search_all_modes)
            self.cond.wakeOne()
        self.mutex.unlock()

    def stop(self) -> None:
        self.mutex.lock()
        self._is_stopping = True
        self.cond.wakeOne()
        self.mutex.unlock()
        self.quit()
        self.wait(3000)

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
            gray_frame, target, current_min, current_max, current_exp, levels_count, hysteresis, current_mode, search_all_modes = task
            
            best_params = optimize_balance_params(
                gray_frame, target, current_min, current_max, current_exp, levels_count, hysteresis, current_mode, search_all_modes
            )
            
            self.finished.emit(best_params[0], best_params[1], best_params[2], best_params[3])
            self._busy = False


class BufferManager:
    """記憶體緩衝池，用於重複使用相同大小的 NumPy 矩陣。"""
    def __init__(self):
        self._pool: dict[tuple, np.ndarray] = {}

    def get_buffer(self, shape: tuple, dtype: np.dtype) -> np.ndarray:
        key = (shape, dtype)
        if key not in self._pool:
            self._pool[key] = np.zeros(shape, dtype=dtype)
        return self._pool[key]

    def clear(self):
        self._pool.clear()

class ImageProcessWorker(QThread):
    """背景影像處理運算執行緒，負責執行耗時的 quantize_gray_with_indices。"""
    # (量化圖, 索引圖, 邊緣圖, 真實比例計數, 運算時間)
    finished = Signal(np.ndarray, np.ndarray, object, np.ndarray, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.mutex = QMutex()
        self.cond = QWaitCondition()
        self._busy = False
        self._pending_task = None
        self._is_stopping = False
        self.buffer_manager = BufferManager()
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
        self.wait(3000)

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
            
            # --- 執行量化核心 (V2 帶有真實統計) ---
            logic_quantized, logic_indices, edges, true_counts = quantize_gray_with_indices(
                frame,
                settings.levels,
                settings.min_value,
                settings.max_value,
                settings.exp_value,
                curve_mode=settings.curve_mode,
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
                palette=getattr(settings, 'custom_palette', None),
                edge_mix=settings.edge_mix if settings.edge_enabled else 0,
                edge_color=settings.edge_color,
                buffer_manager=self.buffer_manager,
            )
            t_computed = time.perf_counter() - t_start
            
            self.finished.emit(logic_quantized, logic_indices, edges, true_counts, t_computed)
            self._busy = False
class CaptureWorker(QThread):
    """專職擷取的執行緒，避免 mss 阻塞主執行緒。"""
    frame_ready = Signal(np.ndarray, np.ndarray, float, float) # color, gray, timestamp, cap_time

    def __init__(self, frame_source, settings):
        super().__init__()
        self.frame_source = frame_source
        self.settings = settings
        self._is_running = True
        self._ctx = None
        self.mutex = QMutex()
        
        # 背景專用偵測器
        from valuelens.core.scene_detector import RandomSceneDetector
        self.detector = RandomSceneDetector(threshold=settings.scene_threshold)
        self._last_full_sync_ts = 0.0
        self._idle_streak = 0

    def update_context(self, ctx):
        self.mutex.lock()
        self._ctx = ctx
        self.mutex.unlock()

    def stop(self):
        self._is_running = False
        self.wait(1000)

    def update_threshold(self, threshold: float):
        """同步更新背景偵測器的門檻。"""
        self.mutex.lock()
        if hasattr(self, 'detector'):
            self.detector.threshold = threshold
        self.mutex.unlock()

    def run(self):
        # 讓底層 CaptureService 在這個新的執行緒中建立專屬的 native capture 資源
        if hasattr(self.frame_source, "capture") and hasattr(self.frame_source.capture, "bind_to_current_thread"):
            self.frame_source.capture.bind_to_current_thread()

        try:
            while self._is_running:
                self.mutex.lock()
                ctx = self._ctx
                self.mutex.unlock()

                if ctx:
                    t0 = time.perf_counter()
                    frame, gray = self.frame_source.get_frame(ctx)
                    cap_time = (time.perf_counter() - t0) * 1000
                    
                    if frame is not None and gray is not None:
                        # 在背景直接判斷是否需要更新
                        is_changed, _ = self.detector.detect_change(gray)
                        
                        # 檢查是否需要強制同步 (心跳)
                        mon_now = time.monotonic()
                        is_timeout = (mon_now - self._last_full_sync_ts > self.settings.sync_timeout_s)
                        
                        if is_changed or is_timeout:
                            self._last_full_sync_ts = mon_now
                            self._idle_streak = 0
                            self.frame_ready.emit(frame, gray, t0, cap_time)
                        else:
                            self._idle_streak = min(self._idle_streak + 1, 30)
                
                # Adaptive capture pacing: static scenes use slower polling to reduce CPU.
                base_interval = max(0.001, self.settings.refresh_ms / 1000.0)
                if self._idle_streak >= 12:
                    sleep_s = min(0.10, base_interval * 3.0)
                elif self._idle_streak >= 6:
                    sleep_s = min(0.06, base_interval * 2.0)
                else:
                    sleep_s = base_interval
                time.sleep(sleep_s)
        finally:
            if hasattr(self.frame_source, "capture") and hasattr(self.frame_source.capture, "close"):
                self.frame_source.capture.close()
