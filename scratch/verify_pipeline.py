import cv2
import numpy as np
import json
import os
import sys

# 將專案路徑加入路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from valuelens.core.quantize import quantize_gray_with_indices
from valuelens.core.scene_detector import GridSceneDetector

def run_verification(video_path="test_pattern.mp4", meta_path="test_metadata.json"):
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    cap = cv2.VideoCapture(video_path)
    detector = GridSceneDetector(threshold=20.0, grid_count=3)
    
    print(f"{'Frame':<6} | {'True (B/G/W)':<22} | {'Calc (B/G/W)':<22} | {'Error %':<8} | {'Detect'}")
    print("-" * 85)

    frame_idx = 0
    correct_detections = 0
    total_error = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        meta = metadata[frame_idx]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. 測試變動偵測 (Scene Detection)
        is_detected, mae = detector.detect_change(gray)
        true_changed = meta["is_changed"]
        
        if is_detected == true_changed:
            detect_status = "OK"
            correct_detections += 1
        else:
            detect_status = "MISS" if true_changed else "FALSE"

        # 2. 測試比例計算 (Distribution)
        # 使用與生成器相同的閥值 (64/192) 以確保理論一致性
        out, indices, _ = quantize_gray_with_indices(
            frame, levels=3, min_value=64, max_value=192, exp_value=0.0
        )
        
        counts = np.bincount(indices.ravel(), minlength=3)
        calc_pcts = (counts / indices.size) * 100
        true_pcts = [meta["black_pct"], meta["gray_pct"], meta["white_pct"]]
        
        # 計算平均誤差
        error = np.mean(np.abs(calc_pcts - true_pcts))
        total_error += error

        if frame_idx % 15 == 0:
            true_str = f"{true_pcts[0]:.1f}/{true_pcts[1]:.1f}/{true_pcts[2]:.1f}"
            calc_str = f"{calc_pcts[0]:.1f}/{calc_pcts[1]:.1f}/{calc_pcts[2]:.1f}"
            phase = meta["phase"]
            print(f"{frame_idx:<6} | {phase:<12} | {true_str:<12} | {calc_str:<12} | {error:>7.2f}% | {detect_status} (MAE:{mae:.1f})")
            
        frame_idx += 1

    cap.release()
    print("-" * 85)
    print(f"Average Distribution Error: {total_error / frame_idx:.4f}%")
    print(f"Scene Detection Accuracy: {(correct_detections / frame_idx) * 100:.2f}%")

if __name__ == "__main__":
    run_verification()
