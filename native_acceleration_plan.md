# ValueLens C++ 原生加速計畫

本計畫旨在透過 **C++ (PyBind11)** 重構 ValueLens 的核心運算與擷取段，以解決 Python 在即時像素處理時的效能瓶頸，同時確保代碼具備高度靈活性與回退機制。

---

## 1. 核心技術架構：雙軌並行 (Hybrid Strategy)

為了確保穩定性，我們採用「偵測並切換」的邏輯。如果原生擴充模組存在，則啟動加速引擎；否則自動退回 Python 實作。

### 結構示意：
```text
valuelens/
├── native/                  # C++ 原生層
│   ├── core.cpp             # 像素處理與 DirectX 擷取實作
│   └── setup.py             # 編譯指令檔
├── core/
│   └── quantize.py          # 銜接層：負責 import 並執行 fallback 邏輯
└── main.py
```

---

## 2. 優雅降級實作 (Graceful Degradation)

在 Python 銜接層，我們使用 `try...except` 結構包裹原生模組：

```python
# valuelens/core/quantize.py

try:
    # 嘗試載入由 C++ 編譯出來的 .pyd 模組
    import valuelens_native as native
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False
    print("[ValueLens] Native engine NOT found, using Python fallback.")

def quantize_gray_with_indices(frame, ...):
    if HAS_NATIVE:
        # 呼叫 C++ 核心，速度提升 5~10 倍
        return native.process_frame(frame, levels, min_v, max_v, exp_v)
    else:
        # 原本的 Python + NumPy + OpenCV 實作
        return old_python_implementation(frame, ...)
```

---

## 3. C++ 核心優化點 (Roadmap)

### 第一階段：運算加速 (Pixel Processing)
*   **目標**：將量化（Quantization）、Exp 映射、以及與 N-Level 分佈相關的統計移至 C++。
*   **關鍵技術**：
    *   **OpenMP**：利用多核 CPU 分開處理圖像的各個掃描線。
    *   **SIMD (AVX2)**：在支援的 CPU 上進行向量化加速。

### 第二階段：擷取加速 (Zero-Copy Capture)
*   **目標**：取代 `mss`，改用 **DirectX Desktop Duplication API**。
*   **關鍵技術**：
    *   直接讀取顯存，減少 CPU 負擔。
    *   在 C++ 內部完成「擷取後立即運算」，只回傳最終結果給顯示層。

---

## 4. 編譯與依賴說明

### 依賴項：
1.  **pybind11**：作為 Python 與 C++ 的溝通橋樑。
2.  **MSVC (Visual Studio)**：Windows 端的編譯環境。

### 編譯流程：
1.  進入 `valuelens/native` 資料夾。
2.  執行編譯命令：
    ```powershell
    pip install .
    ```
3.  編譯完成後，系統會生成一個 `valuelens_native.pyd` 檔案，Python 即可直接 import。

---

## 5. 計畫優勢總結

*   **極低入侵性**：即便未來移除 `native` 資料夾，主程式依然可以透過 Python 慢速版正常執行。
*   **極致效能**：解決 4K 螢幕下處理高動態畫面時的掉幀問題。
*   **開發便利**：核心邏輯寫一次 C++，未來多個版本（或不同專案）皆可複用此高性能 DLL。

---
*文件草擬於：2026-04-25*
