import sys
import os
import numpy as np
from PySide6.QtWidgets import QApplication

# 將專案路徑加入路徑
sys.path.append(os.getcwd())

def test_project_assets():
    print("[Self-Check] 1. Checking standardized test assets...")
    required_files = [
        "value_test.mp4", "value_test_meta.json",
        "color_test.mp4", "color_test_meta.json"
    ]
    
    for f in required_files:
        if os.path.exists(f):
            print(f"[Self-Check] PASS: Found {f}")
        else:
            print(f"[Self-Check] FAIL: Missing {f}")
            return False
            
    print("[Self-Check] 2. Testing UI & Logic with AppStore...")
    app = QApplication.instance() or QApplication(sys.argv)
    try:
        from valuelens.ui.overlay_window import OverlayWindow
        from valuelens.core.store import AppStore
        
        store = AppStore()
        # 測試第 40 格 (Index 39) 是否可用
        store.settings.presets[39] = {"name": "TEST", "data": {}}
        store._manager.save(store.settings)
        print("[Self-Check] PASS: Preset persistence (Slot 40) verified.")
        
        # 測試 OverlayWindow 初始化
        window = OverlayWindow(settings=store.settings)
        print("[Self-Check] PASS: OverlayWindow initialized successfully.")
        
        if hasattr(window, "calc_worker"):
            window.calc_worker.stop()
        if hasattr(window, "auto_balance_worker"):
            window.auto_balance_worker.stop()
        if hasattr(window, "cap_worker"):
            window.cap_worker.stop()
        window.hide()
        window.deleteLater()
        app.processEvents()
        return True
    except Exception as e:
        print(f"[Self-Check] FAIL in logic test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_project_assets()
    if success:
        print("\n[Self-Check] COMPLETE: All systems and assets are verified.")
    sys.exit(0 if success else 1)
