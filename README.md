# LoRa 影像傳輸系統

## 系統架構

```
樹莓派 (Raspberry Pi) → ESP32 → LoRa → LoRa → ESP32 → 電腦 (PC)
    [發送端]              [中繼]  [無線]  [中繼]     [接收端]
```

## 專案結構

```
├── common/                      # 共用模組
│   ├── config.py               # 設定參數
│   └── protocol.py             # 通訊協定
│
├── raspberry_pi/               # Raspberry Pi 發送端
│   ├── sender.py               # 主程式
│   ├── camera_capture.py       # 攝影機擷取
│   ├── jpeg_encoder.py         # JPEG 編碼器
│   ├── cs_encoder.py           # 壓縮感知編碼器
│   └── serial_comm.py          # 串列通訊
│
├── pc/                         # PC 接收端
│   ├── receiver.py             # 主程式
│   ├── jpeg_decoder.py         # JPEG 解碼器
│   ├── cs_decoder.py           # 壓縮感知解碼器
│   └── serial_comm.py          # 串列通訊
│
└── arduino/                    # ESP32 Arduino 程式
    ├── esp32_sender/           # 發送端中繼器
    │   └── esp32_sender.ino
    └── esp32_receiver/         # 接收端中繼器
        └── esp32_receiver.ino
```

## 功能特點

### 運作模式

1. **CCTV 模式（連續視訊）**
   - 連續擷取並傳輸視訊串流
   - 可調整 FPS（每秒幀數）
   - 適用於即時監控
   - 預設解析度：128x72
   - 預設 JPEG 品質：85

2. **照片模式（高清照片）**
   - 擷取並傳輸單張高品質照片
   - 高解析度：640x480
   - 高 JPEG 品質：95
   - 確保照片高清
   - 支援預覽與儲存

### 編碼方式

1. **JPEG 編碼**
   - 標準 JPEG 壓縮
   - 可調整品質 (1-100)
   - 相容性高
   - 適合彩色影像

2. **壓縮感知 (CS) 編碼**
   - 基於 DCT 的區塊壓縮感知
   - 採樣率可調整 (0.0-1.0)
   - 資料量更小
   - 適合灰階影像

### 通訊協定

幀格式：`START|TYPE|LENGTH|DATA|CRC|END`

- **START**: 起始標記 (0xAA 0x55)
- **TYPE**: 資料類型 (0x01=JPEG, 0x02=CS)
- **LENGTH**: 資料長度 (2 bytes, big-endian)
- **DATA**: 影像資料
- **CRC**: CRC16 校驗碼 (2 bytes, big-endian)
- **END**: 結束標記 (0x55 0xAA)

## 硬體需求

### Raspberry Pi 端
- Raspberry Pi (任何型號，建議 3B+ 或更新)
- USB 攝影機或 Pi Camera
- USB 轉串列模組 (連接 ESP32)

### ESP32 中繼器
- ESP32 開發板 × 2
- ATK-LORA-01 或相容的 LoRa 模組 × 2
- 連接線

**ESP32 接線：**
- GPIO16 (RX2) → LoRa TX
- GPIO17 (TX2) → LoRa RX
- GND → LoRa GND
- 3.3V/5V → LoRa VCC (依模組規格)

### PC 端
- 電腦 (Windows/Linux/macOS)
- USB 轉串列模組 (連接 ESP32)
- Python 3.8+

## 軟體需求

### Raspberry Pi / PC

```bash
pip install numpy opencv-python pyserial
```

### Arduino IDE

1. 安裝 ESP32 開發板支援
2. 選擇開發板：ESP32 Dev Module
3. 設定上傳速度：115200

## 使用方式

### 1. 設定 ESP32 中繼器

**發送端 ESP32：**
```bash
# 使用 Arduino IDE 上傳 arduino/esp32_sender/esp32_sender.ino
# 連接：Raspberry Pi USB ↔ ESP32 USB
#       ESP32 UART2 ↔ LoRa 模組
```

**接收端 ESP32：**
```bash
# 使用 Arduino IDE 上傳 arduino/esp32_receiver/esp32_receiver.ino
# 連接：LoRa 模組 ↔ ESP32 UART2
#       ESP32 USB ↔ PC USB
```

### 2. 啟動接收端 (PC)

**CCTV 模式（連續視訊）：**
```bash
cd pc
python receiver.py [--port /dev/ttyUSB0]
```

**照片模式（單張高清照片）：**
```bash
cd pc
python receiver.py --mode photo [--save photo.jpg]
```

**參數：**
- `--mode`: 運作模式 `cctv` 或 `photo`（預設：cctv）
- `--port`: 指定串列埠（可選，會自動偵測）
- `--save`: 儲存接收到的照片（僅照片模式）

### 3. 啟動發送端 (Raspberry Pi)

**CCTV 模式 - JPEG 編碼：**
```bash
cd raspberry_pi
python sender.py --mode cctv --codec jpeg --jpeg-quality 85 --fps 10 [--preview]
```

**CCTV 模式 - 壓縮感知 (CS) 編碼：**
```bash
cd raspberry_pi
python sender.py --mode cctv --codec cs --cs-rate 0.3 --fps 10 [--preview]
```

**照片模式（高清單張照片）：**
```bash
cd raspberry_pi
python sender.py --mode photo [--preview]
```

**參數：**
- `--mode`: 運作模式 `cctv`（連續視訊）或 `photo`（單張高清照片）（預設：cctv）
- `--port`: 指定串列埠（可選）
- `--camera`: 攝影機索引（預設：0）
- `--width`: 影像寬度（CCTV 模式預設：128，照片模式預設：640）
- `--height`: 影像高度（CCTV 模式預設：72，照片模式預設：480）
- `--codec`: 編碼方式 `jpeg` 或 `cs`（預設：jpeg）
- `--jpeg-quality`: JPEG 品質 1-100（CCTV 模式預設：85，照片模式預設：95）
- `--cs-rate`: CS 採樣率 0.0-1.0（預設：0.05）
- `--cs-block`: CS 區塊大小（預設：8）
- `--fps`: 目標 FPS，僅用於 CCTV 模式（預設：10）
- `--inter-frame-delay`: 幀間延遲秒數（預設：0.005），用於防止接收端緩衝溢位
- `--preview`: 顯示預覽視窗
- `--chunk-delay-ms`: 逐區塊固定延遲（毫秒），預設 0（啟用自動流量控制視窗）

### 4. 流量控制與診斷

- **動態流量控制**：發送端 ESP32 會定期送出 `[FC] backlog=...` 文字訊息，Raspberry Pi 上的 `serial_comm.py` 會在背景執行緒解析並依 backlog 自動調整 chunk 大小與幀間延遲（仍維持 115200 bps）。
- **自訂邊界**：若仍希望固定延遲，可透過 `--chunk-delay-ms` 或 `--inter-frame-delay` 覆寫；設定為 0 代表完全交給自動調節。
- **接收端緩衝監控**：PC 端 `receiver.py` 新增 `--debug-buffer` 參數，可在串流時輸出序列緩衝使用率，便於判斷是否需要降低 FPS/解析度。
- **最佳化提示**：當 Raspberry Pi 觀察到 backlog 過大時會自動放慢幀速；若長時間 backlog 為 0，可酌情降低 `--inter-frame-delay` 以最多化吞吐。

## 設定調整

在 `common/config.py` 中可調整：

```python
# 串列通訊設定
BAUD_RATE = 115200          # 波特率
SERIAL_TIMEOUT = 1.0        # 逾時時間

# 影像設定
DEFAULT_WIDTH = 80          # 預設寬度
DEFAULT_HEIGHT = 45         # 預設高度
DEFAULT_JPEG_QUALITY = 85   # JPEG 品質

# 壓縮感知設定
CS_MEASUREMENT_RATE = 0.05  # CS 採樣率
CS_BLOCK_SIZE = 8           # CS 區塊大小

# 緩衝設定
MAX_FRAME_SIZE = 65535      # 最大幀大小
CHUNK_SIZE = 240            # LoRa 傳輸區塊大小

# 流量控制設定
INTER_FRAME_DELAY = 0.05    # 幀間延遲（秒），防止接收端緩衝溢位
```

## 效能調校

### CCTV 模式調校

#### 增加傳輸速度
1. 降低解析度：`--width 160 --height 120`
2. 降低 JPEG 品質：`--jpeg-quality 70`
3. 使用 CS 編碼並降低採樣率：`--codec cs --cs-rate 0.2`
4. 提高 FPS：`--fps 15`

#### 改善影像品質
1. 提高解析度：`--width 640 --height 480`
2. 提高 JPEG 品質：`--jpeg-quality 95`
3. 提高 CS 採樣率：`--cs-rate 0.5`
4. 降低 FPS：`--fps 5`

### 照片模式調校

照片模式已針對高品質進行最佳化（640x480，品質 95），但您可以：

#### 進一步提升品質
```bash
python sender.py --mode photo --width 1280 --height 720 --jpeg-quality 98
```

#### 降低傳輸時間（犧牲品質）
```bash
python sender.py --mode photo --width 320 --height 240 --jpeg-quality 90
```

## 故障排除

### 無法找到串列埠
```bash
# Linux
ls /dev/ttyUSB* /dev/ttyACM*

# 檢查權限
sudo chmod 666 /dev/ttyUSB0

# 或加入使用者到 dialout 群組
sudo usermod -a -G dialout $USER
```

### 影像傳輸延遲
1. 降低解析度
2. 降低品質或採樣率
3. 確認 LoRa 模組設定（頻寬、擴頻因子）
4. 檢查串列埠緩衝區大小

### CRC 錯誤
1. 檢查接線是否正確
2. 降低傳輸速度
3. 縮短 LoRa 傳輸距離
4. 檢查電源供應是否穩定

### 幀丟失率高 / ESP32 緩衝區溢位
本系統採用改良的緩衝架構，將幀重組邏輯從 ESP32 移至 PC 端：

1. **ESP32 接收端作為中繼器**
   - 不進行幀驗證或緩衝
   - 僅需 512 bytes 記憶體（相對於舊版的 65KB）
   - 避免記憶體溢位問題

2. **PC 端處理幀重組**
   - 使用 PC 的大容量記憶體（最高 100KB 緩衝）
   - 能夠處理高速資料流和複雜的幀偵測

3. **幀間延遲流量控制**
   - 發送端自動在每個幀之間加入 50ms 延遲（預設）
   - 防止接收端緩衝區溢位
   - 可透過 `--inter-frame-delay` 參數調整
   - 如果接收端出現 "Invalid frame" 警告，可增加延遲：`--inter-frame-delay 0.1`
   - 如果傳輸速度太慢，可減少延遲：`--inter-frame-delay 0.02`

4. **效能提升**
   - 大幅降低幀丟失率
   - 提高系統可靠性
   - ESP32 記憶體使用量減少 99%

如果仍然遇到問題，請檢查：
- LoRa 模組設定是否正確
- 串列埠連接是否穩定
- PC 端程式是否正常運作
- 嘗試增加 `--inter-frame-delay` 參數值

## 授權

本專案採用 MIT 授權條款。

## 貢獻

歡迎提交 Issue 和 Pull Request。
