import os
import sys

# 將專案路徑加入路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import time
import json

def create_value_test(output_path="value_test.mp4", duration_sec=10, fps=30):
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    metadata = []
    num_frames = duration_sec * fps
    
    # 隨機噪點與井字圖案的混合背景
    grid_size = 128
    
    for f in range(num_frames):
        t = f / fps
        # 建立基礎畫布
        frame = np.zeros((height, width), dtype=np.uint8)
        
        # 1. 隨機噪點色塊 (每秒變換)
        if f % fps == 0:
            noise_block = np.random.randint(0, 256, (height // grid_size, width // grid_size), dtype=np.uint8)
            noise_frame = cv2.resize(noise_block, (width, height), interpolation=cv2.INTER_NEAREST)
            frame = cv2.addWeighted(frame, 0, noise_frame, 1.0, 0)
        else:
            # 維持上一格噪點但加入微小擾動
            frame = noise_frame
            
        # 2. 漸變背景 (動態移動)
        gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
        shift = int(t * 100) % width
        gradient = np.roll(gradient, shift, axis=1)
        frame = cv2.addWeighted(frame, 0.7, gradient, 0.3, 0)

        # 3. 井字格線 (測試隨機點命中率)
        for x in range(0, width, grid_size):
            cv2.line(frame, (x, 0), (x, height), 255, 1)
        for y in range(0, height, grid_size):
            cv2.line(frame, (0, y), (width, y), 255, 1)

        # 轉回 BGR 格式
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(bgr_frame)
        
        # 紀錄 Ground Truth
        from valuelens.core.quantize import get_quantization_lut
        lut = get_quantization_lut(3, 64, 192, 0.0)
        true_indices = cv2.LUT(frame, lut)
        counts = np.bincount(true_indices.ravel(), minlength=3)
        
        black_pct = float(counts[0] / frame.size * 100)
        gray_pct = float(counts[1] / frame.size * 100)
        white_pct = float(counts[2] / frame.size * 100)
        is_changed = (f % fps == 0) or (f % 5 == 0) # 每 5 幀變一次
        
        metadata.append({
            "frame": f,
            "avg_brightness": float(np.mean(frame)),
            "timestamp": t,
            "phase": f // fps,
            "black_pct": black_pct,
            "gray_pct": gray_pct,
            "white_pct": white_pct,
            "is_changed": is_changed
        })

    out.release()
    with open("value_test_meta.json", "w") as f_meta:
        json.dump(metadata, f_meta)
    print(f"Value test generated: {output_path}")

if __name__ == "__main__":
    create_value_test()
