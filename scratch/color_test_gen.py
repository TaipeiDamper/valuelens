import cv2
import numpy as np
import time
import json
import os

def create_color_stress_test(output_path="color_test.mp4", duration_sec=10, fps=30):
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    metadata = []
    
    print(f"Generating color stress test: {output_path}...")
    
    num_frames = duration_sec * fps
    
    # 預先準備隨機粒子數據
    num_particles = 50
    particles = []
    for _ in range(num_particles):
        particles.append({
            "pos": np.array([np.random.randint(0, width), np.random.randint(0, height)], dtype=np.float32),
            "vel": np.random.uniform(-5, 5, 2),
            "color": [np.random.randint(0, 256) for _ in range(3)],
            "size": np.random.randint(20, 80)
        })

    for f in range(num_frames):
        # 建立畫布 (深灰色背景)
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        
        t = f / fps
        phase = (f // (fps * 2)) % 4  # 每 2 秒換一個主要模式
        
        # --- 背景層：動態漸變 ---
        if phase == 0: # 彩色掃描線
            for y in range(0, height, 10):
                c = (int(127 + 127 * np.sin(t + y/100)), int(127 + 127 * np.cos(t + y/100)), 150)
                cv2.line(frame, (0, y), (width, y), c, 1)
        
        elif phase == 1: # 旋轉色相環
            center = (width // 2, height // 2)
            for angle in range(0, 360, 2):
                rad = np.deg2rad(angle + t * 50)
                end_x = int(center[0] + 300 * np.cos(rad))
                end_y = int(center[1] + 300 * np.sin(rad))
                # 簡單模擬色相
                color = [0, 0, 0]
                color[angle % 3] = 200
                cv2.line(frame, center, (end_x, end_y), tuple(color), 2)

        # --- 中間層：物理粒子 (測試隨機點偵測與邊緣) ---
        for p in particles:
            p["pos"] += p["vel"]
            # 邊界反彈
            if p["pos"][0] <= 0 or p["pos"][0] >= width: p["vel"][0] *= -1
            if p["pos"][1] <= 0 or p["pos"][1] >= height: p["vel"][1] *= -1
            
            # 繪製發光粒子
            cv2.circle(frame, tuple(p["pos"].astype(int)), p["size"], p["color"], -1)
            cv2.circle(frame, tuple(p["pos"].astype(int)), p["size"], (255, 255, 255), 2) # 白邊測試 Edge

        # --- 前景層：高頻噪點格線 (測試 Dither 與 Morph) ---
        if f % 2 == 0: # 閃爍測試
            grid_size = 20
            for x in range(0, width, grid_size):
                color = (0, 255, 0) if (x // grid_size) % 2 == 0 else (0, 0, 255)
                cv2.line(frame, (x, 0), (x, height), color, 1)

        # --- UI 資訊欄 (左下角) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_v = np.mean(gray)
        cv2.putText(frame, f"Frame: {f} | Time: {t:.2f}s | AvgV: {avg_v:.1f}", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)
        
        # 紀錄真相 (Ground Truth)
        metadata.append({
            "frame": f,
            "avg_brightness": float(avg_v),
            "phase": int(phase)
        })

    out.release()
    
    # 儲存 Metadata
    with open("color_test_meta.json", "w") as f_meta:
        json.dump(metadata, f_meta)
    
    print("Done! Video saved to color_test.mp4")

if __name__ == "__main__":
    create_color_stress_test()
