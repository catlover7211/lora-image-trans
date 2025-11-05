# 專案重構遷移指南

本專案已進行重大重構，從舊的架構遷移到新的簡化架構。

## 架構變更

### 舊架構
```
電腦 <--序列埠--> ESP32 <--LoRa--> ESP32 <--序列埠--> 電腦
```
- 使用 H.264/H.265/AV1 等視訊編碼
- ASCII 編碼的複雜協定
- 多種編碼器選項（YOLO、Wavelet、Contour）

### 新架構
```
樹莓派 <--序列埠--> ESP32 <--LoRa--> ESP32 <--序列埠--> 電腦
```
- 專注於 JPEG 和壓縮感知 (CS) 編碼
- 二進位幀協定，更高效
- 清晰的模組化設計

## 檔案對應

### 舊檔案 → 新檔案

| 舊檔案 | 新檔案 | 說明 |
|--------|--------|------|
| `main.py` | `pc/receiver.py` | PC 接收端 |
| `capture.py` | `raspberry_pi/sender.py` | 發送端主程式 |
| `protocol.py` | `common/protocol.py` | 通訊協定（完全重寫）|
| `h264_codec.py` | `raspberry_pi/jpeg_encoder.py`<br>`raspberry_pi/cs_encoder.py`<br>`pc/jpeg_decoder.py`<br>`pc/cs_decoder.py` | 編解碼器（簡化）|
| `image_settings.py` | `common/config.py` | 設定參數 |
| `arduino/esp32_lora_transceiver/` | `arduino/esp32_sender/`<br>`arduino/esp32_receiver/` | ESP32 程式（分離為發送/接收）|

### 移除的功能

以下功能在新架構中已移除，因為它們對 CCTV 應用不是必需的：

- H.264/H.265/AV1 視訊編碼
- YOLO 物件偵測
- Wavelet 編碼
- Contour 編碼
- 複雜的 ACK 機制
- ASCII Base64 編碼

### 保留/新增的功能

- ✅ JPEG 編碼（保留並簡化）
- ✅ 壓縮感知 (CS) 編碼（新增）
- ✅ 串列通訊（簡化）
- ✅ CRC 校驗（改用 CRC16）
- ✅ 二進位協定（更高效）

## 遷移步驟

### 1. 更新 Python 環境

新架構只需要基本套件：

```bash
pip install numpy opencv-python pyserial
```

不再需要：
- ~~PyAV (`av`)~~
- ~~torch (for YOLO)~~

### 2. 更新硬體配置

**舊配置**：兩個相同的 ESP32 運行相同程式

**新配置**：兩個 ESP32 運行不同程式
- 發送端：上傳 `arduino/esp32_sender/esp32_sender.ino`
- 接收端：上傳 `arduino/esp32_receiver/esp32_receiver.ino`

### 3. 更新使用方式

**舊方式**：
```bash
# 接收端
python main.py --codec h265

# 發送端
python capture.py --codec h265 --bitrate 400000
```

**新方式**：
```bash
# PC 接收端
cd pc
python receiver.py

# Raspberry Pi 發送端
cd raspberry_pi
python sender.py --codec jpeg --jpeg-quality 85 --fps 10
```

### 4. 參數對應

| 舊參數 | 新參數 | 說明 |
|--------|--------|------|
| `--codec h264/h265/av1` | `--codec jpeg` | 使用 JPEG |
| `--codec wavelet` | `--codec cs` | 使用壓縮感知 |
| `--bitrate` | `--jpeg-quality` | JPEG 模式的品質控制 |
| `--wavelet-quant` | `--cs-rate` | CS 模式的採樣率 |
| `--width/--height` | `--width/--height` | 相同 |
| `--camera-index` | `--camera` | 相同 |

## 測試新架構

### 1. 測試協定

```bash
python -m unittest tests.test_new_protocol -v
```

### 2. 測試本地（不需要硬體）

如果你只有一台電腦，可以使用虛擬串列埠工具測試：

**Linux**:
```bash
socat -d -d pty,raw,echo=0 pty,raw,echo=0
```

然後在不同終端機執行 sender 和 receiver，指定對應的虛擬埠。

### 3. 完整系統測試

1. 設定發送端 ESP32（連接 Raspberry Pi）
2. 設定接收端 ESP32（連接 PC）
3. 啟動 PC 接收端
4. 啟動 Raspberry Pi 發送端
5. 觀察影像傳輸

## 效能比較

### 舊架構（H.264）
- 編碼複雜度：高
- 傳輸效率：中（ASCII 編碼）
- 延遲：中
- 資料量：小（視訊壓縮）

### 新架構（JPEG）
- 編碼複雜度：低
- 傳輸效率：高（二進位）
- 延遲：低
- 資料量：中

### 新架構（CS）
- 編碼複雜度：中
- 傳輸效率：高（二進位）
- 延遲：低
- 資料量：小（壓縮感知）

## 建議設定

### 高品質（網路狀況良好）
```bash
python sender.py --codec jpeg --jpeg-quality 95 --width 640 --height 480 --fps 15
```

### 平衡（預設）
```bash
python sender.py --codec jpeg --jpeg-quality 85 --width 320 --height 240 --fps 10
```

### 低頻寬（網路狀況不佳）
```bash
python sender.py --codec cs --cs-rate 0.2 --width 160 --height 120 --fps 5
```

## 常見問題

### Q: 為什麼移除 H.264 編碼？
A: H.264 雖然壓縮率高，但編碼複雜度高，不適合 Raspberry Pi Zero 等低階設備。JPEG 編碼快速且支援度高，CS 編碼則提供更好的壓縮率。

### Q: 舊的程式還能用嗎？
A: 舊程式保留在專案中，但建議使用新架構。新架構更簡單、更可靠、更容易維護。

### Q: 如何選擇 JPEG 還是 CS？
A: 
- JPEG：適合彩色影像、高品質需求、即時性要求高
- CS：適合灰階影像、低頻寬環境、可接受輕微品質損失

### Q: 可以在電腦之間傳輸嗎？
A: 可以。將 `raspberry_pi/sender.py` 在任何有 Python 和攝影機的電腦上執行即可。

## 回退到舊版本

如果需要使用舊版本，可以：

```bash
git checkout <舊版本的 commit hash>
```

或直接使用舊的 Python 檔案（它們仍保留在專案中）。

## 取得幫助

- 查看 `NEW_README.md` 獲取完整使用說明
- 查看各目錄下的 `README.md` 獲取詳細說明
- 提交 Issue 到 GitHub

## 未來計畫

- [ ] 新增 ACK 機制（可選）
- [ ] 支援多幀合併
- [ ] 新增更多壓縮演算法
- [ ] 改善錯誤恢復機制
- [ ] 新增效能監控
