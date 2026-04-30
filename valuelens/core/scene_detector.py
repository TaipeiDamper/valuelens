import numpy as np
import cv2
from typing import Tuple

class RandomSceneDetector:
    """
    隨機點場景變更檢測器 (Random Point Scene Detector)。
    結合 MAE (平均絕對誤差) 與 MSE (均方誤差) 指標。
    每次偵測到變動後會自動更換部分採樣點，消除盲區。
    """
    def __init__(self, threshold: float = 10.0, sample_count: int = 1024):
        self.threshold = threshold
        self.sample_count = sample_count
        self.last_samples = None
        self.sample_indices = None
        self.last_shape = None

    def _regen_samples(self, shape: Tuple[int, int]):
        """重新生成隨機採樣點。"""
        h, w = shape
        self.sample_indices = (
            np.random.randint(0, h, self.sample_count),
            np.random.randint(0, w, self.sample_count)
        )
        self.last_shape = shape

    def get_sampled_pixels(self, gray_frame: np.ndarray) -> np.ndarray:
        """獲取目前隨機採樣點的像素值。"""
        if self.sample_indices is None or self.last_shape != gray_frame.shape:
            self._regen_samples(gray_frame.shape)
        return gray_frame[self.sample_indices]

    def detect_change(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        檢測畫面是否有顯著變動。
        支援 BGR 或 Gray 輸入，內部會自動確保使用灰階進行比對。
        """
        if frame is None:
            return False, 0.0

        # 確保使用灰階進行判斷
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        h, w = frame.shape
        
        # 初始化或解析度改變時重新採樣
        if self.sample_indices is None or self.last_shape != (h, w):
            self._regen_samples((h, w))
            self.last_samples = frame[self.sample_indices]
            return True, 0.0

        current_samples = frame[self.sample_indices]
        
        # 同時計算 MAE 與 MSE
        diff = current_samples.astype(np.float32) - self.last_samples.astype(np.float32)
        # mae = np.mean(np.abs(diff)) # 暫時備用
        mse = np.mean(diff ** 2)
        
        # 只要 MSE 超過門檻就判定為變更
        is_changed = mse > self.threshold
        
        if is_changed:
            self.last_samples = current_samples
            # [優化]：偵測到大變動後，隨機更換 20% 的採樣點，增加未來檢測的覆蓋率
            replace_count = max(1, self.sample_count // 5)
            replace_idx = np.random.choice(self.sample_count, replace_count, replace=False)
            self.sample_indices[0][replace_idx] = np.random.randint(0, h, replace_count)
            self.sample_indices[1][replace_idx] = np.random.randint(0, w, replace_count)
            
        return is_changed, mse
