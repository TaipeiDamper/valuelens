import cv2
import time
import numpy as np
import os
import sys

# 加入專案路徑
sys.path.append(os.getcwd())

def debug_video_performance():
    video_path = "value_test.mp4"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    from valuelens.core.quantize import quantize_gray_with_indices
    from valuelens.config.settings import AppSettings
    
    settings = AppSettings()
    settings.levels = 3
    settings.blur_enabled = True
    settings.blur_radius = 10
    settings.edge_enabled = True
    settings.morph_enabled = True
    
    print(f"[FPS-Debug] Analyzing performance with {video_path}...")
    
    frame_counts = 0
    while frame_counts < 30:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        t1 = time.perf_counter()
        
        # 模擬 Engine.py 的展開調用方式
        eff_blur = settings.blur_radius if settings.blur_enabled else 0
        eff_dither = settings.dither_strength if settings.dither_enabled else 0
        eff_edge = settings.edge_strength if settings.edge_enabled else 0
        
        logic_q, logic_idx, edges, counts = quantize_gray_with_indices(
            frame, # 雖然叫 bgr，但 Engine 傳的是 gray 或是 frame
            settings.levels,
            min_value=settings.min_value,
            max_value=settings.max_value,
            exp_value=settings.exp_value,
            display_min=settings.display_min_value,
            display_max=settings.display_max_value,
            display_exp=settings.display_exp_value,
            blur_radius=eff_blur,
            dither_strength=eff_dither,
            edge_strength=eff_edge,
            process_order=settings.process_order,
            morph_enabled=settings.morph_enabled,
            morph_strength=settings.morph_strength,
            morph_threshold=settings.morph_threshold
        )
        t2 = time.perf_counter()
        
        calc_time = (t2 - t1) * 1000
        print(f"Frame {frame_counts:02d}: Calc: {calc_time:5.1f}ms (Level: {settings.levels})")
        frame_counts += 1
        
    cap.release()

if __name__ == "__main__":
    debug_video_performance()
