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
