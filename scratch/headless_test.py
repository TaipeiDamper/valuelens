import cv2
import numpy as np
import os
import sys

# 將專案路徑加入路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from valuelens.core.quantize import quantize_gray_with_indices

def run_headless_test(video_path="test_pattern.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open test video.")
        return

    frame_idx = 0
    print(f"{'Frame':<8} | {'Black %':<10} | {'Gray %':<10} | {'White %':<10} | {'Status'}")
    print("-" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 轉換為灰階
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 模擬 3 階量化設定
        # levels=3, min=20, max=235 (排除極端值)
        out, indices, edges = quantize_gray_with_indices(
            frame, 
            levels=3, 
            min_value=64,   # 設定在 128 左右的中間點
            max_value=192,
            exp_value=0.0
        )
        
        # 計算比例
        h, w = indices.shape
        total_pixels = h * w
        counts = np.bincount(indices.ravel(), minlength=3)
        pcts = (counts / total_pixels) * 100
        
        status = ""
        # 在前 150 幀 (Static Bars) 進行精確校驗
        if frame_idx < 150:
            # 預期各 33.3%
            if all(30 < p < 36 for p in pcts):
                status = "PASS (Calibration OK)"
            else:
                status = f"FAIL (Expected 33%, got {pcts[0]:.1f}/{pcts[1]:.1f}/{pcts[2]:.1f})"
        
        if frame_idx % 30 == 0:  # 每秒輸出一次結果
            print(f"{frame_idx:<8} | {pcts[0]:>8.2f}% | {pcts[1]:>8.2f}% | {pcts[2]:>8.2f}% | {status}")
            
        frame_idx += 1

    cap.release()
    print("\nTest Complete.")

if __name__ == "__main__":
    run_headless_test()
