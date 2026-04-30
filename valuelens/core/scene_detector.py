import numpy as np

class RandomSceneDetector:
    """
    隨機點場景變更檢測器 (Random Point Scene Detector)。
    在全圖隨機採樣固定數量的像素點進行比對，消除井字網格的死角，且運算量極低。
    """
    def __init__(self, threshold: float = 20.0, sample_count: int = 256):
        self.threshold = threshold
        self.sample_count = sample_count
        self.coords = None  # 隨機座標快取
        self.last_signature = None

    def _get_coords(self, h, w):
        """生成並快取隨機座標"""
        if self.coords is None or self.coords_shape != (h, w):
            # 使用固定種子確保同一個解析度下的採樣點是固定的，方便比對
            rng = np.random.default_rng(42)
            y = rng.integers(0, h, size=self.sample_count)
            x = rng.integers(0, w, size=self.sample_count)
            self.coords = (y, x)
            self.coords_shape = (h, w)
        return self.coords

    def get_sampled_pixels(self, gray_frame: np.ndarray) -> np.ndarray:
        """獲取隨機採樣點的像素值"""
        h, w = gray_frame.shape[:2]
        y_idx, x_idx = self._get_coords(h, w)
        return gray_frame[y_idx, x_idx]

    def detect_change(self, gray_frame: np.ndarray) -> tuple[bool, float]:
        """
        比對隨機採樣點的差異，返回 (是否超過變更門檻, 變動強度)。
        """
        if gray_frame is None or gray_frame.size == 0:
            return False, 0.0
            
        h, w = gray_frame.shape[:2]
        y_idx, x_idx = self._get_coords(h, w)
        
        # 取得採樣像素值
        current_pixels = gray_frame[y_idx, x_idx]
        
        # 首次初始化
        if self.last_signature is None:
            self.last_signature = current_pixels.copy()
            return True, 255.0
            
        # 計算平均絕對誤差 (MAE)
        diff = np.abs(current_pixels.astype(np.float32) - self.last_signature.astype(np.float32))
        mae = float(np.mean(diff))
        
        is_changed = mae > self.threshold
        
        # 累積偵測：只有變動顯著時才更新基準點
        if is_changed:
            self.last_signature = current_pixels.copy()
            
        return is_changed, mae
