# ValueLens

ValueLens 是一個 Windows 視覺練習工具，提供最上層即時影像覆蓋、灰階/色階量化、可調式濾鏡管線、圖片模式與自動平衡功能。

核心設計目標：

- 即時觀察螢幕或圖片的明暗/色階結構。
- 透過 `min/max`、偏差曲線與階層數調整輸入映射。
- 支援可排序的濾鏡管線，方便比較不同處理順序。
- 在高解析度下以背景 worker、場景偵測與 native color map 降低 UI 阻塞。

## 主要功能

- 最上層鏡片視窗，可即時處理螢幕區域。
- 支援 `2 / 3 / 5 / 8` 階量化。
- 支援三種輸入曲線：
  - `Gamma`
  - `Sigmoid`
  - `Log`
- 支援輸入範圍與輸出範圍分離調整。
- 支援可拖曳排序的處理管線：
  - `blur`
  - `dither`
  - `edge`
  - `morph`
- 支援自動平衡：
  - 重設平衡：依原始畫面尋找較合適的 W:G:B preset。
  - Auto continuous：持續追蹤畫面變化並平滑更新參數。
- 支援 preset：
  - 啟動 preset
  - 上次狀態
  - 20 個手動 slot
- 快捷鍵即時開關與截圖。
- 圖片處理模式：
  - 從剪貼簿匯入 (`Ctrl+V`)
  - 調整圖片色階
  - 儲存處理後的圖片
  - 複製圖片到剪貼簿
  - 支援拖放

## 專案結構

```text
main.py                         # 應用入口
valuelens/app.py                # QApplication 與 OverlayWindow 啟動
valuelens/config/settings.py    # AppSettings 與設定讀寫
valuelens/core/
  capture_service.py            # bettercam/mss 擷取與視窗隱身設定
  sources.py                    # Live screen / static image frame source
  engine.py                     # CaptureWorker / ImageProcessWorker / AutoBalanceWorker
  quantize.py                   # 濾鏡管線、LUT、量化、native color map fallback
  balance.py                    # W:G:B 分佈與 auto balance 參數搜尋
  scene_detector.py             # 隨機採樣場景變動偵測
valuelens/ui/
  overlay_window.py             # 主視窗、worker 接線、auto balance orchestration
  control_panel.py              # 控制面板、preset、濾鏡排序與滑桿
  render_widget.py              # 覆蓋畫面繪製
valuelens/native/               # pybind11 native acceleration module
scripts/                        # smoke test 與 benchmark 工具
tests/                          # pytest 回歸測試
```

## 如何運行

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

若要啟用 native 加速，請在同一個環境中執行：

```powershell
pip install -e .\valuelens\native
```

## 依賴與加速選項

`requirements.txt` 包含執行與測試 ValueLens 所需的主要 Python 套件：

- `PySide6`：桌面 UI。
- `numpy` / `opencv-python`：影像處理與矩陣運算。
- `mss`：穩定 fallback 螢幕擷取後端。
- `keyboard`：全域快捷鍵。
- `pytest`：回歸測試。
- `pybind11`：編譯 native 加速模組時使用。

### Native 加速

ValueLens 支援可選的 `valuelens_native` native 模組，用於加速高解析度 pixel mapping。若本機可編譯 C++ extension，建議安裝：

```powershell
pip install -e .\valuelens\native
```

目前 native 模組為可選加速：如果編譯或載入失敗，程式會回到 Python / OpenCV 路徑，不影響基本功能。Windows 需要可用的 Visual Studio Build Tools / MSVC C++ 編譯器。

目前 native 模組主要用於：

- `indices + color LUT -> RGB output` 的 color map。
- 量化分佈輔助計算。

不支援或不適合的情境會自動 fallback 到 Python / OpenCV 路徑。

### 可選擷取後端

Windows 上若安裝 `bettercam`，ValueLens 會優先使用它作為 live capture backend；初始化或擷取失敗時會自動 fallback 到 `mss`。可用環境變數強制選擇：

```powershell
$env:VALUELENS_CAPTURE_BACKEND="mss"       # 強制使用 mss
$env:VALUELENS_CAPTURE_BACKEND="bettercam" # 優先 bettercam，失敗 fallback mss
$env:VALUELENS_CAPTURE_BACKEND="auto"      # 預設
```

若要比較 DXGI 類擷取速度，可手動安裝其中一個：

```powershell
pip install dxcam
# 或
pip install bettercam
```

未安裝時，主程式與 `scripts\benchmark_capture_pipeline.py` 都會回到 `mss` / 顯示 skip，不影響基本功能。

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
- 量化層主要瓶頸原本是 `color_map`，已加入 native color map 降低高解析度成本。
- `edge` 場景主要成本在 Canny。
- `morph` 場景仍會因 mask 套用與 fallback 路徑較重，屬於後續優化重點。

## 架構閱讀順序

第一次閱讀建議順序：

1. [`pipeline.md`](pipeline.md)：理解整體資料流。
2. `valuelens/core/engine.py`：理解三個 worker 如何解耦 capture / compute / balance。
3. `valuelens/core/quantize.py`：理解濾鏡管線與輸出組裝。
4. `valuelens/core/balance.py`：理解 W:G:B 分佈與自動平衡。
5. `valuelens/ui/overlay_window.py`：理解 UI 如何觸發與接收背景結果。
6. `valuelens/ui/control_panel.py`：理解使用者操作與信號。

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
