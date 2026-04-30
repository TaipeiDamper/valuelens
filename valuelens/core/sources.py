import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class FrameContext:
    """提供影像來源擷取時所需的視窗或環境資訊"""
    def __init__(self, view_rect: tuple[int, int, int, int], dpr: float, 
                 phys_rect: Optional[tuple[int, int, int, int]] = None,
                 hwnd: Optional[int] = None,
                 panel_height: int = 0):
        self.view_rect = view_rect  # (x, y, w, h)
        self.dpr = dpr
        self.phys_rect = phys_rect
        self.hwnd = hwnd
        self.panel_height = panel_height

class IFrameSource(ABC):
    """影像來源策略介面"""
    
    @abstractmethod
    def get_frame(self, ctx: FrameContext) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """回傳 (color_frame, gray_frame)"""
        pass
        
    @property
    @abstractmethod
    def is_static(self) -> bool:
        """是否為靜態影像模式"""
        pass
        
    # TODO: [CLEANUP] 目前無任何模組呼叫此平移方法，屬於架構預留。
    def pan(self, dx: float, dy: float) -> None:
        """平移畫面 (預設不處理)"""
        pass
        
    # TODO: [CLEANUP] 目前無任何模組呼叫此縮放方法，屬於架構預留。
    def zoom(self, delta: float, pivot_x: float, pivot_y: float) -> None:
        """縮放畫面 (預設不處理)"""
        pass
        
    # TODO: [CLEANUP] 目前無任何模組呼叫此重置方法，屬於架構預留。
    def reset_view(self) -> None:
        """重置縮放與平移 (預設不處理)"""
        pass

class LiveScreenSource(IFrameSource):
    """即時螢幕擷取來源"""
    def __init__(self, capture_service):
        self.capture = capture_service
        self.last_raw_frame = None
        
    @property
    def is_static(self) -> bool:
        return False
        
    def get_frame(self, ctx: FrameContext) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        x, y, w, h = ctx.view_rect
        dpr = ctx.dpr
        
        if ctx.phys_rect is not None:
            win_x, win_y, win_pw, win_ph = ctx.phys_rect
            panel_h_phys = int(round(ctx.panel_height * dpr))
            cap_x, cap_y = win_x, win_y + panel_h_phys
            cap_w, cap_h = win_pw, max(1, win_ph - panel_h_phys)
        else:
            cap_x = int(round(x * dpr))
            cap_y = int(round(y * dpr))
            cap_w = max(1, int(round(w * dpr)))
            cap_h = max(1, int(round(h * dpr)))
            
        frame = self.capture.capture_region(cap_x, cap_y, cap_w, cap_h)
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.last_raw_frame = frame
            return frame, gray
        return None, None

class StaticImageSource(IFrameSource):
    """靜態影像來源 (支援縮放與平移)"""
    def __init__(self, source_image: np.ndarray, source_type: str = "frozen"):
        self.source_image = source_image
        self.source_type = source_type
        self.zoom_factor = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        
    @property
    def is_static(self) -> bool:
        return True
        
    # TODO: [CLEANUP] 目前無任何模組呼叫此平移方法，屬於架構預留。
    def pan(self, dx: float, dy: float) -> None:
        self.pan_offset_x += dx
        self.pan_offset_y += dy
        
    # TODO: [CLEANUP] 目前無任何模組呼叫此縮放方法，屬於架構預留。
    def zoom(self, delta: float, pivot_x: float, pivot_y: float) -> None:
        old_zoom = self.zoom_factor
        self.zoom_factor *= delta
        
        # 保持縮放中心點不動
        self.pan_offset_x = pivot_x - (pivot_x - self.pan_offset_x) * (self.zoom_factor / old_zoom)
        self.pan_offset_y = pivot_y - (pivot_y - self.pan_offset_y) * (self.zoom_factor / old_zoom)
        
    # TODO: [CLEANUP] 目前無任何模組呼叫此重置方法，屬於架構預留。
    def reset_view(self) -> None:
        self.zoom_factor = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        
    def get_frame(self, ctx: FrameContext) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        vx, vy, vw, vh = ctx.view_rect
        dpr = ctx.dpr
        
        sh, sw = self.source_image.shape[:2]
        view_w_phys = max(1, int(round(vw * dpr)))
        view_h_phys = max(1, int(round(vh * dpr)))
        
        view_long = max(view_w_phys, view_h_phys)
        img_long = max(sw, sh)
        min_zoom = view_long / (1.5 * img_long)
        if self.zoom_factor < min_zoom:
            self.zoom_factor = min_zoom

        vsw = sw * self.zoom_factor
        vsh = sh * self.zoom_factor
        
        if vsw > view_w_phys:
            self.pan_offset_x = max(0, min(self.pan_offset_x, vsw - view_w_phys))
        else:
            self.pan_offset_x = (vsw - view_w_phys) / 2
            
        if vsh > view_h_phys:
            self.pan_offset_y = max(0, min(self.pan_offset_y, vsh - view_h_phys))
        else:
            self.pan_offset_y = (vsh - view_h_phys) / 2
            
        sx1 = max(0, int(round(self.pan_offset_x / self.zoom_factor)))
        sy1 = max(0, int(round(self.pan_offset_y / self.zoom_factor)))
        sx2 = min(sw, int(round((self.pan_offset_x + view_w_phys) / self.zoom_factor)))
        sy2 = min(sh, int(round((self.pan_offset_y + view_h_phys) / self.zoom_factor)))
        
        crop = self.source_image[sy1:sy2, sx1:sx2]
        
        if crop.size > 0:
            target_w = int(round((sx2 - sx1) * self.zoom_factor))
            target_h = int(round((sy2 - sy1) * self.zoom_factor))
            if target_w > 0 and target_h > 0:
                resized_crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                frame = np.full((view_h_phys, view_w_phys, 3), 34, dtype=np.uint8)
                dx = max(0, int(round(-self.pan_offset_x)))
                dy = max(0, int(round(-self.pan_offset_y)))
                fh, fw = resized_crop.shape[:2]
                
                jh = min(fh, view_h_phys - dy)
                jw = min(fw, view_w_phys - dx)
                if jh > 0 and jw > 0:
                    frame[dy:dy+jh, dx:dx+jw] = resized_crop[:jh, :jw]
                
                gray = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2GRAY)
                return frame, gray
                
        # 失敗或無效範圍
        frame = np.full((view_h_phys, view_w_phys, 3), 34, dtype=np.uint8)
        return frame, None
