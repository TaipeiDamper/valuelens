import cv2
import numpy as np
import json
import os
import sys

# 將專案路徑加入路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from valuelens.core.quantize import quantize_gray_with_indices
from valuelens.core.scene_detector import RandomSceneDetector

def run_verification(video_path="test_pattern.mp4", meta_path="test_metadata.json"):
    if not os.path.exists(video_path):
        print(f"[Skip] {video_path} not found.")
        return
        
    print(f"\n[Verify] Checking {video_path}...")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    cap = cv2.VideoCapture(video_path)
    from valuelens.core.scene_detector import RandomSceneDetector
    detector = RandomSceneDetector(threshold=10.0, sample_count=256)
    
    print(f"{'Frame':<6} | {'Phase':<12} | {'True (B/G/W)':<12} | {'Calc (B/G/W)':<12} | {'Error %':<8} | {'Detect'}")
    print("-" * 85)

    frame_idx = 0
    correct_detections = 0
    total_error = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx >= len(metadata):
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
        # 使用與生成器相同的閥值 (64/192)
        # 注意：關閉所有濾鏡 (process_order=[])，以確保與原始 Metadata 對齊
        out, indices, edges, true_counts = quantize_gray_with_indices(
            frame, levels=3, min_value=64, max_value=192, exp_value=0.0,
            process_order=[]
        )
        
        # 使用背景回傳的 true_counts (這已經是降採樣優化後的結果)
        calc_pcts = (true_counts / np.sum(true_counts)) * 100
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
    # 跑兩個 Video Test
    run_verification("color_test.mp4", "color_test_meta.json")
    run_verification("value_test.mp4", "value_test_meta.json")
