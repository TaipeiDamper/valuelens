import time
import mss
import numpy as np

def test_raw_capture_speed():
    sct = mss.mss()
    # 測試一個標準的 1280x720 區域
    monitor = {"left": 100, "top": 100, "width": 1280, "height": 720}
    
    print(f"[Capture-Test] Starting raw MSS speed test (1280x720)...")
    
    times = []
    for i in range(50):
        t1 = time.perf_counter()
        shot = sct.grab(monitor)
        # 測試轉換速度
        img = np.frombuffer(shot.bgra, dtype=np.uint8).reshape((shot.height, shot.width, 4))
        t2 = time.perf_counter()
        
        times.append((t2 - t1) * 1000)
        if i % 10 == 0:
            print(f"  - Frame {i}: {times[-1]:.2f}ms")
            
    avg = sum(times) / len(times)
    print(f"\n[Capture-Test] Average Capture Time: {avg:.2f}ms")
    print(f"[Capture-Test] Theoretical Max FPS: {1000/avg:.2f}")

if __name__ == "__main__":
    test_raw_capture_speed()
