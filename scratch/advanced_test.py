import cv2
import numpy as np
import os
import sys

# 將專案路徑加入路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from valuelens.core.quantize import quantize_gray_with_indices

def run_advanced_test(video_path="test_pattern.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open test video.")
        return

    # 模擬自動平衡的參數
    min_val, max_val = 64, 192
    target_pct = 33.3
    
    print(f"{'Frame':<8} | {'Mode':<15} | {'Order':<8} | {'Black %':<10} | {'Gray %':<10} | {'White %':<10}")
    print("-" * 75)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- 測試 A: 自動平衡追蹤 (僅在 150-300 幀) ---
        if 150 <= frame_idx <= 300:
            # 簡易模擬自動平衡：根據目前亮度調整 min/max
            avg_brightness = np.mean(frame)
            # 讓區間跟著亮度跑
            min_val = max(0, int(avg_brightness - 64))
            max_val = min(255, int(avg_brightness + 64))
            mode_name = "Auto-Tracking"
        else:
            mode_name = "Fixed-Param"

        # --- 測試 B: 濾鏡順序對比 ---
        orders = ["blur", "dither", "edge", "morph"] # 標準
        
        # 跑一次標準運算
        out, indices, _ = quantize_gray_with_indices(
            frame, levels=3, min_value=min_val, max_value=max_val, exp_value=0.0,
            process_order=orders
        )
        
        counts = np.bincount(indices.ravel(), minlength=3)
        pcts = (counts / indices.size) * 100
        
        if frame_idx % 30 == 0:
            print(f"{frame_idx:<8} | {mode_name:<15} | {''.join([s[0] for s in orders]):<8} | {pcts[0]:>8.2f}% | {pcts[1]:>8.2f}% | {pcts[2]:>8.2f}%")
            
            # 如果是測試 B 的關鍵幀，我們跑另一個順序對比
            if frame_idx == 30: # 在 Static Bars 期間對比
                alt_order = ["edge", "dither", "blur", "morph"]
                _, alt_indices, _ = quantize_gray_with_indices(
                    frame, levels=3, min_value=min_val, max_value=max_val, exp_value=0.0,
                    process_order=alt_order
                )
                alt_counts = np.bincount(alt_indices.ravel(), minlength=3)
                alt_pcts = (alt_counts / alt_indices.size) * 100
                print(f"{' ':8} | {'Compare Order':<15} | {''.join([s[0] for s in alt_order]):<8} | {alt_pcts[0]:>8.2f}% | {alt_pcts[1]:>8.2f}% | {alt_pcts[2]:>8.2f}% (<- 順序對比例的影響)")

        frame_idx += 1

    cap.release()
    print("\nAdvanced Test Complete.")

if __name__ == "__main__":
    run_advanced_test()
