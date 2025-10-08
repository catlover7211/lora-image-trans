# 影像串流範例

此專案示範如何透過序列埠在電腦與微控制器（如 Arduino/ESP32）之間傳送影像幀。

## 專案結構

- `capture.py`：發送端。負責擷取攝影機影像、壓縮成 JPEG，並透過自訂的幀協定送出。
- `main.py`：接收端。從序列埠讀取幀資料、驗證 CRC32、解碼 JPEG 並顯示畫面。
- `protocol.py`：共享的幀協定工具，包含位元組填充、CRC 驗證與序列幀讀寫邏輯。
- `tests/test_protocol.py`：簡單的單元測試，確保幀協定的基本行為正確。

## 需求

- Python 3.10+
- OpenCV (`opencv-python`)
- PySerial (`pyserial`)
- NumPy (`numpy`)

可使用 `pip` 安裝：

```bash
python -m pip install opencv-python numpy pyserial
```

## 使用方式

1. **啟動接收端**（建議先啟動）：
   ```bash
   python main.py
   ```
2. **啟動發送端**：
   ```bash
   python capture.py
   ```

按下 `q` 或 `Ctrl+C` 可結束程式。

## 單元測試

執行內建的協定測試：

```bash
python -m unittest tests/test_protocol.py
```

## 注意事項

- 預設會自動偵測第一個可用的序列埠。若環境中有多個裝置，可依需求調整 `protocol.auto_detect_serial_port`。
- 影像預設縮放為 `80x60` 並以低品質 JPEG 壓縮，以降低傳輸負載。可依硬體能力調整 `capture.py` 中的設定。
- 協定採用位元組填充與 CRC32 校驗，避免控制碼衝突並提高傳輸可靠度。
