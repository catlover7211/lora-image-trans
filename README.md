# 影像串流範例

此專案示範如何透過序列埠在電腦與微控制器（如 Arduino/ESP32）之間傳送影像幀。

## 專案結構

- `capture.py`：發送端。擷取攝影機影像、以 H.264/H.265/AV1/JPEG 或自訂 Wavelet 編碼並透過自訂幀協定傳送。
- `main.py`：接收端。從序列埠讀取幀資料、驗證 CRC32、解碼 H.264/H.265/AV1/JPEG 與 Wavelet 並顯示畫面。
- `protocol.py`：共享的幀協定工具，負責 ASCII 封包、CRC 驗證、分段 ACK 與降級處理。
- `h264_codec.py`：封裝 PyAV 的 H.264/H.265/AV1 編碼與解碼流程，並提供 JPEG 與自訂 Wavelet 編解碼器。
- `image_settings.py`：集中管理影像尺寸、位元率、關鍵幀與動態偵測等調校參數，修改此檔即可快速調整影像品質。
- `tests/test_protocol.py`：簡單的單元測試，確保幀協定的基本行為正確。

## 需求

- Python 3.10+
- OpenCV (`opencv-python`)
- PySerial (`pyserial`)
- NumPy (`numpy`)
- PyAV (`av`) — 需系統已安裝 FFmpeg 及 libx264 / libx265 / libaom-av1

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

   常用參數：

   - `--codec`：選擇 `h264` / `h265` / `av1` / `jpeg` / `wavelet`；H.265、AV1 與 Wavelet 壓縮率較佳，JPEG 兼具高相容性。
   - `--bitrate`：調整目標位元率，降低可節省頻寬。
   - `--keyframe-interval`：設定關鍵幀間隔，數值越大表示較少完整畫面。
   - `--motion-threshold`：畫面變化門檻，變化低於此值時跳過傳送。
   - `--max-idle`：最多可允許多久不傳送，超過則強制送一幀以維持同步。
   - `--jpeg-quality`：當選擇 `jpeg` 編碼時控制壓縮品質（1-100）。

按下 `q` 或 `Ctrl+C` 可結束程式。

## 單元測試

執行內建的協定測試：

```bash
python -m unittest tests/test_protocol.py
```

## 注意事項

- 預設會自動偵測第一個可用的序列埠。若環境中有多個裝置，可依需求調整 `protocol.auto_detect_serial_port`。
- 影像預設縮放後以 H.265 編碼（可改為 H.264、AV1、JPEG 或 Wavelet），利用差分幀減少資料量並定期插入 I-frame 以保持同步。
- 若系統未安裝 libx265 / libaom-av1 或裝置效能不足，可在啟動發送端時加入 `--codec h264` 改回 H.264 以維持相容性。
- 協定採用 ASCII 框架與 CRC32 校驗，並支援逐段 ACK（初始階段可自動降級為無 ACK 模式，以防止接收端尚未就緒時阻塞）。
