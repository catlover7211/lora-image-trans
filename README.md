# 影像串流範例

此專案示範如何透過序列埠在電腦與微控制器（如 Arduino/ESP32）之間傳送影像幀。

## 專案結構

- `capture.py`：發送端。擷取攝影機影像、以 H.264 編碼並透過自訂幀協定傳送。
- `main.py`：接收端。從序列埠讀取幀資料、驗證 CRC32、解碼 H.264 並顯示畫面。
- `protocol.py`：共享的幀協定工具，負責 ASCII 封包、CRC 驗證、分段 ACK 與降級處理。
- `h264_codec.py`：封裝 PyAV 的 H.264 編碼與解碼流程。
- `tests/test_protocol.py`：簡單的單元測試，確保幀協定的基本行為正確。

## 需求

- Python 3.10+
- OpenCV (`opencv-python`)
- PySerial (`pyserial`)
- NumPy (`numpy`)
- PyAV (`av`) — 需系統已安裝 FFmpeg/Libx264

可使用 `pip` 安裝：

```bash
python -m pip install opencv-python numpy pyserial av
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
- 影像預設縮放後以 H.264 編碼，透過 P-frame 只回傳變動內容，並定期插入 I-frame 以保持同步。
- 協定採用 ASCII 框架與 CRC32 校驗，並支援逐段 ACK（初始階段可自動降級為無 ACK 模式，以防止接收端尚未就緒時阻塞）。
