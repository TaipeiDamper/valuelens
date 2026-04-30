import cv2
import numpy as np
import json
import random

def generate_test_video(output_path="test_pattern.mp4", meta_path="test_metadata.json", width=1280, height=720, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    metadata = []
    
    def add_frame(img, is_changed, phase_name):
        nonlocal frame_idx
        # 計算真實比例
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        b = np.mean(gray < 64) * 100
        g = np.mean((gray >= 64) & (gray < 192)) * 100
        w = np.mean(gray >= 192) * 100
        
        metadata.append({
            "frame": frame_idx,
            "phase": phase_name,
            "black_pct": b,
            "gray_pct": g,
            "white_pct": w,
            "is_changed": is_changed
        })
        out.write(img)
        frame_idx += 1

    frame_idx = 0
    
    # --- Phase 1: Static Bars (3s) ---
    print("Phase 1: Static Bars...")
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :width//3] = [0, 0, 0]
    frame[:, width//3:2*width//3] = [128, 128, 128]
    frame[:, 2*width//3:] = [255, 255, 255]
    for _ in range(3 * fps):
        add_frame(frame, False, "Calibration")

    # --- Phase 2: Brightness Sweep (3s) ---
    print("Phase 2: Brightness Sweep...")
    for i in range(3 * fps):
        val = int((i / (3 * fps)) * 255)
        frame = np.full((height, width, 3), val, dtype=np.uint8)
        add_frame(frame, True, "Global Sweep")

    # --- Phase 3: Smooth Gradient (3s) ---
    print("Phase 3: Smooth Gradient...")
    for i in range(3 * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        offset = int((i / (3 * fps)) * width)
        for x in range(width):
            val = ((x + offset) % width) / width * 255
            frame[:, x] = [int(val)] * 3
        add_frame(frame, True, "Gradient")

    # --- Phase 4: Chaos Grids (3s) ---
    print("Phase 4: Chaos Grids...")
    grid_size = 128
    cols, rows = width // grid_size, height // grid_size
    grid_colors = [random.choice([0, 128, 255]) for _ in range(rows * cols)]
    for i in range(3 * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        is_changed = False
        if i % 2 == 0:
            idx = random.randint(0, len(grid_colors)-1)
            grid_colors[idx] = random.choice([0, 128, 255])
            is_changed = True
        for r in range(rows):
            for c in range(cols):
                color = grid_colors[r * cols + c]
                cv2.rectangle(frame, (c*grid_size, r*grid_size), ((c+1)*grid_size, (r+1)*grid_size), (color, color, color), -1)
        add_frame(frame, is_changed, "Chaos")

    out.release()
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print("All phases generated.")

if __name__ == "__main__":
    generate_test_video()
