import os

file_path = "valuelens/ui/overlay_window.py"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 定義要刪除的整塊字串 (從 866 到 882 行的內容)
# 我們直接用大區塊匹配來替換成空字串
target_block = """            # 1. 座標與參數準備
            rect = self._lens_rect()
            if rect.width() <= 0 or rect.height() <= 0: return
            dpr = self.devicePixelRatioF()
            
            # --- 核心邏輯：決定影像來源 ---
            ctx = FrameContext(
                view_rect=(rect.x(), rect.y(), rect.width(), rect.height()),
                dpr=dpr,
                phys_rect=self._physical_window_rect(),
                hwnd=self._hwnd(),
                panel_height=self._panel_height
            )
            # 2. 獲取畫面並檢查變動 (使用整合後的 GridSceneDetector)
            frame, gray_frame = self.frame_source.get_frame(ctx)
            t_captured = time.perf_counter()
            
            if frame is None or frame.size == 0 or gray_frame is None: 
                return
            
            self._last_gray_frame = gray_frame"""

if target_block in content:
    new_content = content.replace(target_block, "")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Success: Block found and removed.")
else:
    print("Error: Target block NOT found. White-space issue?")
    # 嘗試模糊匹配
    import re
    # 匹配從 rect = self._lens_rect() 到 self._last_gray_frame 的所有內容
    pattern = r'            # 1\. 座標與參數準備.*?self\._last_gray_frame'
    new_content, count = re.subn(pattern, "", content, flags=re.DOTALL)
    if count > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Success: Block removed via Regex (count: {count}).")
    else:
        print("Final Error: Regex also failed.")
