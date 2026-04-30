# ValueLens

ValueLens 是一個 Windows 練習工具，提供最上層灰階覆蓋視窗與圖片處理模式。

## 主要功能

- 最上層視窗覆蓋，可即時查看灰階效果
- 支援多種灰階演算法與色階調整
- 可即時調整灰階對比與 `min/max` 範圍
- 快捷鍵即時開關與截圖 (預設 `Ctrl+Alt+G`)
- 圖片處理模式
  - 從剪貼簿匯入 (`Ctrl+V`)
  - 調整圖片色階
  - 儲存處理後的圖片
  - 複製圖片到剪貼簿
  - 支援拖放

## 如何運行

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## 依賴與加速選項

`requirements.txt` 包含執行與測試 ValueLens 所需的主要 Python 套件：

- `PySide6`：桌面 UI。
- `numpy` / `opencv-python`：影像處理與矩陣運算。
- `mss`：預設 Windows 螢幕擷取後端。
- `keyboard`：全域快捷鍵。
- `pytest`：回歸測試。
- `pybind11`：編譯 native 加速模組時使用。

### Native 加速

ValueLens 支援可選的 `valuelens_native` native 模組，用於加速高解析度 pixel mapping。若本機可編譯 C++ extension，建議安裝：

```powershell
pip install -e .\valuelens\native
```

目前 native 模組為可選加速：如果編譯或載入失敗，程式會回到 Python / OpenCV 路徑，不影響基本功能。Windows 需要可用的 Visual Studio Build Tools / MSVC C++ 編譯器。

### 可選擷取後端

預設擷取後端仍是 `mss`。`dxcam` 與 `bettercam` 只用於擷取 benchmark A/B 評估，沒有被列為必要依賴。

若要比較 DXGI 類擷取速度，可手動安裝其中一個：

```powershell
pip install dxcam
# 或
pip install bettercam
```

未安裝時，`scripts\benchmark_capture_pipeline.py` 會顯示 skip，不影響主程式。

## 測試與 Benchmark

常用驗證命令：

```powershell
python -m pytest
python scripts\native_smoke.py
python scripts\benchmark_quantize.py
python scripts\benchmark_capture_pipeline.py
python scripts\benchmark_balance.py
```

目前效能觀察重點：

- 擷取層主要瓶頸是 `mss.grab` 與 BGR materialize/copy。
- 量化層主要瓶頸原本是 `color_map`，已加入 native color map 原型降低高解析度成本。
- `morph` 場景仍會因 mask 套用與 fallback 路徑較重，屬於後續優化重點。

## 打包成 EXE

```powershell
pip install pyinstaller
.\build_windows.ps1
```

輸出檔案：`dist\ValueLens.exe`

## 預設快捷鍵

- 開關覆蓋：`Ctrl+Alt+G`
- 圖片模式：`Ctrl+Alt+I`
- 結束程式：`Ctrl+Alt+Q`
- 隱藏介面：`Ctrl+Alt+P`
