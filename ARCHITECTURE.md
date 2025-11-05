# 系統架構說明

## 整體架構圖

```
┌─────────────────┐     ┌─────────────┐     ┌──────────┐     ┌──────────┐     ┌─────────────┐     ┌─────────────────┐
│   Raspberry Pi  │     │   ESP32 #1  │     │  LoRa #1 │     │  LoRa #2 │     │   ESP32 #2  │     │       PC        │
│   (發送端)      │────▶│  (中繼器)   │────▶│  (發送)  │◀───▶│  (接收)  │────▶│  (中繼器)   │────▶│   (接收端)      │
│                 │ USB │             │UART2│          │ 無線 │          │UART2│             │ USB │                 │
│  sender.py      │     │esp32_sender │     │          │      │          │     │esp32_receiver│    │  receiver.py    │
└─────────────────┘     └─────────────┘     └──────────┘     └──────────┘     └─────────────┘     └─────────────────┘
```

## 資料流向

### 發送端流程

```
1. Camera Capture
   └─▶ camera_capture.py
       └─▶ 擷取影像 (OpenCV)

2. Image Encoding
   ├─▶ jpeg_encoder.py (JPEG模式)
   │   └─▶ cv2.imencode() → JPEG bytes
   │
   └─▶ cs_encoder.py (CS模式)
       └─▶ DCT + 採樣 + 量化 → CS bytes

3. Protocol Framing
   └─▶ protocol.encode_frame()
       └─▶ START + TYPE + LENGTH + DATA + CRC + END

4. Serial Transmission
   └─▶ serial_comm.send()
       └─▶ 分塊傳輸 → ESP32

5. ESP32 Forwarding
   └─▶ esp32_sender.ino
       └─▶ USB Serial → UART2 → LoRa
```

### 接收端流程

```
1. LoRa Reception
   └─▶ LoRa 模組接收無線訊號

2. ESP32 Forwarding
   └─▶ esp32_receiver.ino
       └─▶ UART2 → USB Serial → PC

3. Serial Reception
   └─▶ serial_comm.receive_frame()
       └─▶ 緩衝、同步、幀偵測

4. Protocol Decoding
   └─▶ protocol.decode_frame()
       └─▶ 驗證 START/END、檢查 CRC、解析資料

5. Image Decoding
   ├─▶ jpeg_decoder.py (JPEG模式)
   │   └─▶ cv2.imdecode() → BGR image
   │
   └─▶ cs_decoder.py (CS模式)
       └─▶ 反量化 + IDCT → BGR image

6. Display
   └─▶ cv2.imshow() → 顯示視窗
```

## 協定層次結構

```
┌────────────────────────────────────────────────────────────┐
│                     應用層 (Application)                    │
│  - sender.py / receiver.py                                 │
│  - JPEG/CS encoding/decoding                               │
└────────────────────────────────────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────────┐
│                    協定層 (Protocol)                        │
│  - Frame structure: START|TYPE|LENGTH|DATA|CRC|END         │
│  - CRC16 checksum                                          │
│  - Type identification (JPEG/CS)                           │
└────────────────────────────────────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────────┐
│                    傳輸層 (Transport)                       │
│  - Serial communication (115200 bps)                       │
│  - Chunked transmission (240 bytes)                        │
│  - Buffer management                                       │
└────────────────────────────────────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────────┐
│                     物理層 (Physical)                       │
│  - USB Serial (Raspberry Pi ↔ ESP32 ↔ PC)                 │
│  - UART2 (ESP32 ↔ LoRa)                                    │
│  - LoRa wireless (433/868/915 MHz)                         │
└────────────────────────────────────────────────────────────┘
```

## 模組依賴關係

### Raspberry Pi 端

```
sender.py
  ├─▶ camera_capture.py
  │     └─▶ opencv (cv2)
  │
  ├─▶ jpeg_encoder.py
  │     └─▶ opencv (cv2)
  │
  ├─▶ cs_encoder.py
  │     └─▶ opencv (cv2)
  │     └─▶ numpy
  │
  ├─▶ serial_comm.py
  │     └─▶ pyserial
  │
  └─▶ common/
        ├─▶ config.py
        └─▶ protocol.py
```

### PC 端

```
receiver.py
  ├─▶ jpeg_decoder.py
  │     └─▶ opencv (cv2)
  │
  ├─▶ cs_decoder.py
  │     └─▶ opencv (cv2)
  │     └─▶ numpy
  │
  ├─▶ serial_comm.py
  │     └─▶ pyserial
  │
  └─▶ common/
        ├─▶ config.py
        └─▶ protocol.py
```

### ESP32 端

```
esp32_sender.ino
  └─▶ HardwareSerial (ESP32 內建)

esp32_receiver.ino
  └─▶ HardwareSerial (ESP32 內建)
```

## 幀格式詳解

```
┌──────┬──────┬────────┬──────────┬──────┬──────┐
│START │ TYPE │ LENGTH │   DATA   │ CRC  │ END  │
├──────┼──────┼────────┼──────────┼──────┼──────┤
│2 byte│1 byte│ 2 byte │ Variable │2 byte│2 byte│
│0xAA55│ 0x01 │ 0x0100 │  [...]   │0x1234│0x55AA│
│      │ JPEG │ 256    │          │      │      │
└──────┴──────┴────────┴──────────┴──────┴──────┘

總計: 9 + len(DATA) bytes
```

### TYPE 值定義

- `0x01` - JPEG 編碼
- `0x02` - 壓縮感知 (CS) 編碼
- `0x03-0xFF` - 保留供未來使用

### CRC16 計算範圍

```
CRC = CRC16(TYPE + LENGTH + DATA)
不包含 START 和 END 標記
```

## 錯誤處理機制

### 發送端

```
camera_capture.py
  └─▶ 失敗 → 重試讀取 → 記錄錯誤

encoder (JPEG/CS)
  └─▶ 失敗 → 跳過此幀 → 繼續下一幀

protocol.encode_frame()
  └─▶ 資料過大 → ValueError → 記錄錯誤

serial_comm.send()
  └─▶ 失敗 → 記錄錯誤 → 累積錯誤計數
```

### 接收端

```
serial_comm.receive_frame()
  ├─▶ 找不到 START → 捨棄資料
  ├─▶ 找不到 END → 等待更多資料
  └─▶ 緩衝區溢位 → 重置緩衝區

protocol.decode_frame()
  ├─▶ 標記錯誤 → 返回 None
  ├─▶ CRC 錯誤 → 返回 None
  └─▶ 長度不符 → 返回 None

decoder (JPEG/CS)
  └─▶ 解碼失敗 → 跳過此幀 → 顯示上一幀
```

### ESP32 中繼器

```
esp32_sender.ino
  ├─▶ 緩衝區溢位 → 重新同步到 FRAME_START
  └─▶ 記錄錯誤訊息

esp32_receiver.ino
  ├─▶ 緩衝區溢位 → 重新同步到 FRAME_START
  └─▶ 記錄錯誤訊息
```

## 效能考量

### 瓶頸分析

1. **攝影機擷取** (~30-50ms @ 320x240)
   - 受限於攝影機硬體

2. **JPEG 編碼** (~10-30ms @ 320x240)
   - 可透過降低品質或解析度優化

3. **CS 編碼** (~20-50ms @ 320x240)
   - 可透過降低採樣率或區塊大小優化

4. **串列傳輸** (~10-50ms，視資料量)
   - 受限於 115200 bps 波特率
   - 可考慮提高波特率（921600）

5. **LoRa 傳輸** (變動，取決於距離和設定)
   - 主要瓶頸
   - 可調整 LoRa 參數（頻寬、擴頻因子）

### 優化建議

**高速模式** (10-15 FPS):
```python
--width 160 --height 120 --jpeg-quality 70
```

**平衡模式** (5-10 FPS):
```python
--width 320 --height 240 --jpeg-quality 85
```

**高品質模式** (2-5 FPS):
```python
--width 640 --height 480 --jpeg-quality 95
```

**低頻寬模式** (CS):
```python
--codec cs --cs-rate 0.2 --width 160 --height 120
```

## 擴充性設計

### 新增編碼器

1. 建立編碼器類別（繼承介面）
2. 實作 `encode()` 方法
3. 在 `common/config.py` 定義新的 TYPE
4. 更新發送端和接收端程式

### 新增功能模組

- 動態品質調整
- 多幀合併
- 錯誤重傳
- ACK 確認
- 壓縮前處理
- 影像增強

### 協定版本化

保留 TYPE 範圍用於協定版本：
- `0x00-0x7F` - 資料類型
- `0x80-0xFF` - 控制訊息

## 安全性考量

### 資料完整性

- CRC16 校驗確保資料未損壞
- 幀標記確保同步

### 未來改進

- 加密傳輸（AES）
- 認證機制
- 防重放攻擊
- 資料簽章
