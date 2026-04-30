# ValueLens 開發錯誤紀錄 (Error Log)

## ❌ 錯誤 1：MSS 跨執行緒調用失敗 (AttributeError)
*   **改錯的東西**：直接在 `CaptureWorker` 背景執行緒中使用在主執行緒建立的 `mss` 實例。
*   **預計的情況**：背景執行緒能正常抓圖。
*   **發生的錯誤**：`AttributeError: '_thread._local' object has no attribute 'srcdc'`。
*   **教訓**：`mss` 在 Windows 上是 Thread-local 的。
*   **修改方式**：在 `CaptureWorker.run()` 中呼叫 `bind_to_current_thread()` 重新建立資源。

## ❌ 錯誤 2：變數名稱不一致 (NameError)
*   **改錯的東西**：在重構 `_on_capture_finished` 時，參數改名為 `t_captured_val` 但後段程式碼仍使用 `t_captured`。
*   **發生的錯誤**：`NameError: name 't_captured' is not defined`，導致 1 FPS 的現象持續存在。
*   **修改方式**：統一變數名稱，並移除冗餘的重複擷取代碼。

## ❌ 錯誤 3：沉重的渲染阻塞 (grab 調用)
*   **改錯的東西**：在渲染迴圈中調用 `self.grab()` 來同步「錄製視窗」。
*   **預計的情況**：錄製視窗能看到畫面。
*   **實際的提升**：移除後，Render 耗時從 **108ms 降至 <10ms**。UI 流暢度大幅提升。
*   **修改方式**：直接傳遞現有的 `QPixmap` 緩存。

## ❌ 錯誤 4：缺乏統計優化 (4K 效能懲罰)
*   **改錯的東西**：在 4K 全解析度下計算灰階分佈統計。
*   **預計的改善**：透過降採樣 (Downsample) 進行統計。
*   **實際的提升**：`Calc` 耗時在 4K 環境下可節省約 20-40ms。
## ❌ 錯誤 5：場景偵測過於嚴格 (1 FPS 誤判)
*   **改錯的東西**：使用 `RandomSceneDetector` 阻擋 Compute 層運算。
*   **發生的錯誤**：在畫面變動微小時，MAE 門檻未觸發，導致影像更新被鎖死在 1s 一次的超時刷新。
*   **預計的改善**：為了極速體驗，目前強制將 `should_calc` 設為 `True`。
*   **教訓**：在追求高 FPS 的場景下，變動偵測應作為可選的「省電模式」，而非強制守門員。
