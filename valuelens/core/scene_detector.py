import numpy as np

class GridSceneDetector:
    """
    井字網格場景變更檢測器 (Grid Scene Detector)。
    僅抽樣畫面的特定網格線上像素（如 1/3, 2/3 處），以極低運算量比對畫面背景是否發生劇烈變更。
    """
    def __init__(self, threshold: float = 30.0, grid_count: int = 2):
        self.threshold = threshold
        self.grid_count = grid_count
        self.last_signature = None

    def extract_grid_pixels(self, gray_frame: np.ndarray) -> np.ndarray:
        """
        擷取畫面上縱橫網格線條的像素。
        """
        if gray_frame is None or gray_frame.size == 0:
            return np.array([], dtype=np.uint8)
            
        h, w = gray_frame.shape[:2]
        
        # 計算動態網格的切分點
        y_coords = [int(h * (i + 1) / (self.grid_count + 1)) for i in range(self.grid_count)]
        x_coords = [int(w * (i + 1) / (self.grid_count + 1)) for i in range(self.grid_count)]
        
        pixel_strips = []
        
        # 1. 擷取水平線的像素
        for y in y_coords:
            if 0 <= y < h:
                pixel_strips.append(gray_frame[y, :].flatten())
                
        # 2. 擷取垂直線的像素
        for x in x_coords:
            if 0 <= x < w:
                pixel_strips.append(gray_frame[:, x].flatten())
                
        if not pixel_strips:
            return np.array([], dtype=np.uint8)
            
        return np.concatenate(pixel_strips)

    def detect_change(self, gray_frame: np.ndarray) -> bool:
        """
        比對當前畫面網格與快取的差異，返回是否超過變更門檻。
        """
        if gray_frame is None or gray_frame.size == 0:
            return False
            
        current_pixels = self.extract_grid_pixels(gray_frame)
        if current_pixels.size == 0:
            return False
            
        # 首次初始化
        if self.last_signature is None or self.last_signature.shape != current_pixels.shape:
            self.last_signature = current_pixels.copy()
            return True 
            
        # 快速計算平均絕對誤差 (MAE)
        diff = np.abs(current_pixels.astype(np.float32) - self.last_signature.astype(np.float32))
        mae = float(np.mean(diff))
        
        # 更新快取簽章
        self.last_signature = current_pixels.copy()
        
        # 超過門檻即判定場景發生切換
        return mae > self.threshold
