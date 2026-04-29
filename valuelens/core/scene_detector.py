import numpy as np
import random

class GridSceneDetector:
    """
    動態混合網格場景檢測器。
    並行使用「傳統井字網格」與「西洋棋主教隨機對角線網格」。
    在視窗大小改變時快取密度偏移量，並在畫面變動或自動平衡時動態擾動基準點。
    """
    def __init__(self, threshold: float = 30.0, grid_count: int = 2):
        self.threshold = threshold
        self.grid_count = grid_count
        self.last_signature = None
        
        # 為了主教隨機網格設計的快取與基準點
        self.last_w = 0
        self.last_h = 0
        self.cx = -1
        self.cy = -1
        self.bishop_offsets = []  # 存放相對偏移量的快取

    def shuffle_center(self, w: int, h: int) -> None:
        """隨機調整主教網格的交叉基準點。"""
        if w <= 0 or h <= 0:
            return
        # 讓基準點保持在視窗內部（保留 10% 邊距避免過於貼邊）
        self.cx = random.randint(max(1, int(w * 0.1)), max(2, int(w * 0.9)))
        self.cy = random.randint(max(1, int(h * 0.1)), max(2, int(h * 0.9)))

    def extract_grid_pixels(self, gray_frame: np.ndarray) -> np.ndarray:
        """
        並行擷取：1. 縱橫網格線像素 2. 主教隨機對角線像素
        """
        if gray_frame is None or gray_frame.size == 0:
            return np.array([], dtype=np.uint8)
            
        h, w = gray_frame.shape[:2]
        
        # --- 階段 A: 視窗大小改變時，重新計算快取偏移量 ---
        if w != self.last_w or h != self.last_h:
            self.last_w = w
            self.last_h = h
            self.shuffle_center(w, h)
            
            # 根據 grid_count 做三種密度步進 (Step) 的動態轉折對應
            if self.grid_count <= 2:
                step = 12  # 密度低
            elif self.grid_count <= 6:
                step = 6   # 密度中
            else:
                step = 3   # 密度高
                
            self.bishop_offsets = []
            max_dim = max(w, h)
            
            # 四個主教斜向方向：右上、右下、左下、左上
            directions = [(1, -1), (1, 1), (-1, 1), (-1, -1)]
            for dx, dy in directions:
                path = []
                for i in range(1, max_dim, step):
                    path.append((dx * i, dy * i))
                self.bishop_offsets.append(path)
                
        # 如果尚未初始化基準點，則補上
        if self.cx == -1 or self.cy == -1:
            self.shuffle_center(w, h)

        pixel_strips = []
        
        # --- 階段 B: 傳統棋盤網格提取 ---
        y_coords = [int(h * (i + 1) / (self.grid_count + 1)) for i in range(self.grid_count)]
        x_coords = [int(w * (i + 1) / (self.grid_count + 1)) for i in range(self.grid_count)]
        
        for y in y_coords:
            if 0 <= y < h:
                pixel_strips.append(gray_frame[y, :].flatten())
                
        for x in x_coords:
            if 0 <= x < w:
                pixel_strips.append(gray_frame[:, x].flatten())
                
        # --- 階段 C: 西洋棋主教斜向抽樣提取 ---
        bishop_pixels = []
        for path in self.bishop_offsets:
            for dx, dy in path:
                px = self.cx + dx
                py = self.cy + dy
                if 0 <= px < w and 0 <= py < h:
                    bishop_pixels.append(gray_frame[py, px])
                    
        if bishop_pixels:
            pixel_strips.append(np.array(bishop_pixels, dtype=np.uint8))
            
        if not pixel_strips:
            return np.array([], dtype=np.uint8)
            
        return np.concatenate(pixel_strips)

    def detect_change(self, gray_frame: np.ndarray) -> bool:
        """
        比對當前畫面特徵與快取的差異，超過門檻回傳 True 並重洗基準點。
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
        
        # 超過門檻即判定場景發生切換，重洗基準點
        if mae > self.threshold:
            h, w = gray_frame.shape[:2]
            self.shuffle_center(w, h)
            return True
            
        return False

