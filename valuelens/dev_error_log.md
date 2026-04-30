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
## ❌ 錯誤 6：OpenCV LUT 通道不匹配 (Assertion Failed)
*   **改錯的東西**：嘗試用 `cv2.LUT` 將單通道的索引矩陣 (Indices) 直接映射到三通道的 BGR 查表。
*   **發生的錯誤**：OpenCV 報錯 `(lutcn == cn || lutcn == 1)` 斷言失敗。單通道輸入不支援三通道輸出。
*   **教訓**：在進行「單通道轉多通道」的映射時，NumPy 的進階索引 (`lut[indices]`) 比 `cv2.LUT` 更靈活且不容易報錯。
*   **修改方式**：將 `cv2.LUT` 替換為 `full_lut_bgr[ctx.indices]`。

## ❌ 錯誤 7：冗餘的顏色空間轉換 (Double Conversion)
*   **改錯的東西**：在主執行緒同時呼叫 `cv2.cvtColor` 與 `bgr_to_qimage` (內部含 `cvtColor`)。
*   **發生的錯誤**：對 4K 影像進行了兩次重複的 RGB 轉換，浪費 UI 執行緒資源。
*   **教訓**：影像應該在背景執行緒一次性轉為最終格式（RGB），主執行緒只負責顯示。
*   **修改方式**：將 `cvtColor` 移至 `quantize_gray_with_indices` 結尾，並新增 `rgb_to_qimage` 輔助函式。
