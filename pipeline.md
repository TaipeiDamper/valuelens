# ValueLens Pipeline 核心架構設計 (V2 實作版)

這份文件定義了影像處理的兩大核心分層：**Scan 層**（負責感知與決策）與 **Compute 層**（負責重度運算）。

---

## 🟢 第一階段：Scan 層 (Scanning & Decision)
> **運行位置**：主執行緒 (UI Thread)
> **頻率**：~60Hz (高頻率監控)

### 0. 擷取層 (Capture Layer)
*   **方法**：使用 MSS / WinAPI 進行快速畫面擷取。
*   **輸出**：`raw_bgr` 影像陣列。

### 1. 偵測與決策層 (Detection & Decision)
*   **1-1. 變動偵測 (Scene Detection) —— [目前設為全速模式]**
    *   **現狀**：為了極致的反應速度，目前強制設為 `should_calc = True`。
    *   **未來擴充**：可透過 `RandomSceneDetector` 進行 MAE 門檻判定來達到省電目的。
*   **1-2. 超時校正 (Sync Timeout)**
    *   **機制**：若超過 1.0s 未觸發運算，強迫執行一次完整的 Compute pass，防止數據漂移或狀態滯後。

### 2. 異步渲染層 (Async Render)
*   **邏輯**：即便 Compute 層仍在忙碌，Scan 層也會立即調用 `canvas.update()`，使用最新的 `raw_bgr`（直通模式）或上一幀的 `logic_indices`（量化模式）進行繪製，確保 UI 不卡死。

---

## 🔵 第二階段：Compute 層 (Heavy Computation)
> **運行位置**：背景執行緒 (Worker Thread)
> **頻率**：按需執行 (盡力而為)

### 3. 統計與濾鏡管線 (Statistics & Filters)
*   **原則**：**統計前移 (Pre-Filter Stats)**。
*   **流程**：
    1.  **True Counts 統計**：在任何濾鏡套用前，先計算真實的 256 階灰階分佈。
    2.  **影像預處理**：套用 Blur, Dither, Morph 等重度濾鏡。
    3.  **量化運算**：將預處理後的影像壓縮為 N 階 (2, 3, 5, 8...) 的 `logic_indices`。

### 4. 數據反饋 (Feedback)
*   **輸出**：`logic_indices` (量化後的索引矩陣)、`true_counts` (原始統計數據)。
*   **訊號**：透過異步訊號發送回 Scan 層，觸發 UI 更新。
