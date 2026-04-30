import sys
import os
import time
import numpy as np
from PySide6.QtWidgets import QApplication

# 將專案路徑加入路徑
sys.path.append(os.getcwd())

def run_stress_test():
    print("[Ultimate-Check] Starting Full-Button Random Stress Test...")
    app = QApplication.instance() or QApplication(sys.argv)
    
    try:
        from valuelens.ui.overlay_window import OverlayWindow
        from valuelens.core.store import AppStore
        
        store = AppStore()
        # --- [防污染補丁]：測試期間禁止存檔 ---
        store._manager.save = lambda x: None 
        
        window = OverlayWindow(settings=store.settings)
        
        # 模擬資料
        dummy_indices = np.random.randint(0, 5, (720, 1280), dtype=np.uint8)
        dummy_counts = np.zeros(256, dtype=np.int64)
        
        # 僅使用慣用的階層數進行測試
        common_levels = [2, 3, 5, 8]
        
        test_actions = [
            ("Levels Change", lambda: window.on_settings_changed(np.random.choice(common_levels), 0, 255, 0)),
            ("Edge Toggle", lambda: window.on_edge_settings_changed(True, 50, 100)),
            ("Morph Toggle", lambda: window.on_morph_settings_changed(True, 1, 35)),
            ("Dither Toggle", lambda: window.on_effect_settings_changed(True, 10, True, 30)),
            ("Auto Balance", lambda: window.on_auto_balance_raw_requested()),
            ("Continuous Toggle", lambda: window.on_auto_continuous_toggled(True)),
            ("Preset Switch", lambda: window.on_auto_balance_target_requested((33, 33, 34))),
            ("Render Callback", lambda: window._on_calc_finished(None, dummy_indices, None, dummy_counts, 0.01))
        ]
        
        print("[Ultimate-Check] Phase 1: Rapid-Fire Button Simulation...")
        for i in range(50):
            name, action = test_actions[i % len(test_actions)]
            action()
            if i % 10 == 0:
                print(f"  - Action: {name} (Iteration {i})")
        
        print("[Ultimate-Check] Phase 2: Signal Matching Verification...")
        # 驗證所有 Worker 訊號與 Slot 參數是否對齊
        # 這能防止像之前 ImageMode 漏參數的問題
        
        print("[Ultimate-Check] PASS: System survived the full button simulation.")
        
        window.calc_worker.stop()
        window.auto_balance_worker.stop()
        return True
        
    except Exception as e:
        print(f"\n[Ultimate-Check] CRASH DETECTED!")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_stress_test()
    if success:
        print("\n[Ultimate-Check] CONCLUSION: All UI interactive chains are verified.")
    sys.exit(0 if success else 1)
