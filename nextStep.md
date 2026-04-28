非同步背景計算 (done)


🗺️ ValueLens 重構地圖與執行進度
✅ [階段一] 數學邏輯獨立解耦 (已完成)
剛才完成的工作：我已經把 overlay_window.py 裡面超過 200 行的演算法（包含直方圖分析、WGB 比例換算、最佳化參數搜尋）全部拔除，搬移到了全新建立的 valuelens/core/balance.py 模組中。
測試項目：請您現在測試**「自動平衡按鈕 (天平圖示)」以及底部的「色階分佈比例條」**是否仍能正常顯示與運作。如果一切正常，代表最核心的運算模組化已成功。
⏳ [階段二] 建立全域非同步引擎 (下一步)
即將進行：把 overlay_window.py 裡的 ImageProcessWorker 獨立出來變成 valuelens/core/engine.py。
效益：讓未來所有的視窗或輸出功能，都能直接呼叫它來算圖，不會卡死主畫面。
✅ [階段三] 圖片模式 (ImageMode) 現代化 (已完成)
即將進行：翻新 valuelens/modes/image_mode.py。把它老舊的「同步算圖」改接上 [階段二] 的引擎，並且引入您最新的 Edge 參數。
效益：處理再大張的圖片，視窗也不會凍結當機了。
⏳ [階段四] 全域狀態總線 (State Store)
即將進行：建立統一的 valuelens/core/store.py。把滿天飛的 Qt Signals (高達 10 幾條) 砍掉，統一使用單向資料流同步。
效益：解除 control_panel 與 overlay_window 極高的耦合度。
⏳ [階段五] 渲染畫布抽離
即將進行：把 paintEvent 跟各種框線、色塊繪製，抽離為專用的 RenderWidget。
效益：未來新增各種輔助標線時，再也不用擔心不小心搞壞核心的邏輯變數。

---
### 📌 待處理與優化事項

#### 🔴 高優先級 (High Priority)
- [x] **優化核心演算法以跨越鞍點**：方案 B 已實作。
- [x] **防止持續平衡的畫面震盪**：設計臨界點保護，避免色調在特定灰階分佈時，參數激烈跳動造成畫面閃爍。
- [x] **UI 算圖效能與大圖卡頓排查**：已修正 Profiling 測速公式，排除了背景 Queue 造成的「偽高延遲」錯覺，並確認目前 `QPixmap` 在 CPU 上的效能負荷在正常範圍內。

#### 🟡 中優先級 (Medium Priority)
- [x] **修正背景線程在退出時的殘留**：已於 `engine.py` 實作 `stop()` 機制，並安全介入 `OverlayWindow.closeEvent`。
- [x] **滑鼠穿透與點擊事件委派安全化**：已於 `RenderWidget` 設定 `WA_TransparentForMouseEvents=True`，徹底確保全畫面疊加時事件能安全穿透至 `OverlayWindow`。