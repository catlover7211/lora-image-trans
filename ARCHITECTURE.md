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

2. ESP32 Forwarding (中繼模式)
   └─▶ esp32_receiver.ino
       └─▶ UART2 → 批次讀取 → 直接轉發 → USB Serial → PC
       └─▶ 不進行幀驗證或重組（避免 ESP32 記憶體溢位）

3. Serial Reception and Frame Reconstruction (PC端)
   └─▶ serial_comm.receive_frame()
       └─▶ 從原始位元流緩衝、同步、幀偵測
       └─▶ 使用 PC 記憶體進行大容量緩衝（最高 100KB）

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
  ├─▶ 找不到 START → 保留最後一個位元組 (處理分割標記)
  ├─▶ 找不到 END → 等待更多資料
  ├─▶ 緩衝區溢位 → 重置緩衝區
  └─▶ 從 START 標記之後開始搜尋 END (避免假標記)

protocol.decode_frame()
  ├─▶ 標記錯誤 → 返回 None
  ├─▶ CRC 錯誤 → 返回 None
  ├─▶ 長度不符 → 返回 None
  └─▶ 空資料 → 返回 None

decoder (JPEG/CS)
  └─▶ 解碼失敗 → 跳過此幀 → 顯示上一幀

receiver.py
  ├─▶ 詳細錯誤日誌 (顯示幀長度)
  ├─▶ 錯誤率統計 (百分比)
  └─▶ 成功率報告
```

### ESP32 中繼器

```
esp32_sender.ino
  ├─▶ 緩衝區溢位 → 重新同步到 FRAME_START
  ├─▶ 記錄錯誤訊息
  └─▶ 統計資訊 (每60秒輸出)

esp32_receiver.ino (中繼模式)
  ├─▶ 簡單資料轉發：LoRa → USB Serial
  ├─▶ 不進行幀驗證或緩衝（避免記憶體溢位）
  ├─▶ 批次轉發以提升效率
  └─▶ 統計資訊 (每60秒輸出轉發位元組數)
```

## 緩衝架構設計

### 緩衝策略

本系統採用**分層緩衝**策略，在不同層級進行適當的資料暫存：

1. **ESP32 發送端** (esp32_sender.ino)
   - 緩衝區大小：4KB
   - 功能：接收來自 Raspberry Pi 的完整幀並轉發
   - 若緩衝區溢位，會重新同步到下一個幀起始標記

2. **ESP32 接收端** (esp32_receiver.ino) - **中繼模式**
   - 緩衝區大小：512 bytes（僅用於批次轉發）
   - 功能：簡單的資料中繼器，不進行幀驗證或重組
   - **重要**：避免在 ESP32 上進行大量緩衝，防止記憶體溢位
   - 立即將接收到的資料轉發到 PC

3. **PC 接收端** (serial_comm.py)
   - 緩衝區大小：動態，最高 100KB
   - 功能：從原始位元流進行幀重組和驗證
   - 利用 PC 的大容量記憶體處理複雜的幀偵測邏輯
   - 能夠處理分割的標記、不完整的幀和雜訊資料

### 設計優勢

**舊架構問題**：
- ESP32 接收端嘗試緩衝和驗證完整幀（需要最高 65KB 記憶體）
- 在高速資料傳輸時容易發生記憶體溢位
- 導致大量幀丟失（如問題描述中的 667 dropped vs 39 forwarded）

**新架構優勢**：
- ESP32 只作為簡單中繼器，記憶體需求降低至 512 bytes
- 所有複雜的幀處理邏輯移至 PC 端
- PC 有充足的記憶體可以緩衝多個完整幀
- 提高系統可靠性和資料吞吐量
- 降低幀丟失率

## 吞吐量優化藍圖（2025Q4）

為了在現有架構上擠出更高的有效頻寬，以下為雙端（ESP32 與 Python）協同的優化規劃，後續實作會依序落地：

### 1. 串列層升級與參數化

- **波特率升級**：`common/config.py` 內預設 BAUD_RATE 從 115200 調高至 921600（硬體允許時），並允許 CLI 參數覆寫，確保 Raspberry Pi ↔ ESP32 以及 ESP32 ↔ PC 的串列層不再成為瓶頸。
- **Chunk 與視窗常數**：新增 `USB_TX_WINDOW`、`LORA_TX_WINDOW`、`FLOW_CTRL_WATERMARK` 等常數，集中管理 USB/LoRa 緩衝大小與水位線，讓 Arduino 與 Python 可以共用同一組設定，避免 magic number 分散各處。
- **自適應 chunk size**：`raspberry_pi/serial_comm.py` 根據 ESP32 回報的可用視窗動態調整 chunk（128~1024 bytes），取代固定 500 bytes，兼顧延遲與吞吐。

### 2. ESP32 發送端（`esp32_sender.ino`）

- **雙緩衝 + state machine**：以 2 × 4096 bytes 的 DMA-friendly buffer 輪替，`Serial.readBytes()` 直接填滿當前 buffer，切換時透過非阻塞 `LoRaSerial.write()` 將資料推進；`Serial.flush()` 僅保留在錯誤/重置情境。
- **LoRa TX 追蹤**：利用 `LoRaSerial.availableForWrite()` 計算剩餘視窗，實作簡單的 credit counter。每當視窗回升至水位線以上時，回傳一個 `CTRL_CREDIT` 訊息（ASCII 行或 0xF0 控制幀）到 Raspberry Pi，告知可再送 N bytes。
- **錯誤自動恢復**：維持既有的 FRAME_START 重新同步邏輯，但改寫為「收到 `CTRL_RESET` 指令即可手動清空」，供開發測試時遠端觸發。

### 3. ESP32 接收端（`esp32_receiver.ino`）

- **增大中繼緩衝**：改用 2048 bytes ring buffer，並採用 `while (LoRaSerial.available() && Serial.availableForWrite())` 的雙向推進，減少 `delayMicroseconds()` 依賴。
- **USB 傳輸優化**：改用 `Serial.write()` + `Serial.availableForWrite()` 分批送出，搭配 `yield()` 讓 FreeRTOS scheduler 處理背景工作，避免長時間佔用 CPU。
- **統計回覆**：每秒輸出 JSON 風格統計（bytes、溫度、重試），讓 PC 端 Python 可以解析並決定後續是否調整 frame pacing。

### 4. Python 端（Raspberry Pi 與 PC）

- **Raspberry Pi `SerialComm`**：
   - 新增背景 reader 解析 ESP32 `CTRL_*` 訊息，維護 `credits` 計數；`send()` 會在 credits 耗盡時阻塞（或等待條件），取代硬式 `inter_frame_delay`。
   - 支援 `--target-bitrate` / `--max-latency` 參數，讓 `sender.py` 可以依場景調整品質或 fps。
- **PC `SerialComm`**：
   - 保留現有背景緩衝，額外提供簡易 API 讓上層得知串列 backlog（`get_buffer_level()`），並在視窗不足時回傳建議延遲給使用者。
   - 解析來自 ESP32 接收端的統計訊息，寫入日誌或顯示在 CLI UI。

### 5. 驗證與量測

- **回歸測試**：沿用 `tests/test_inter_frame_delay.py`、`tests/test_relay_mode.py`，新增 mock credit provider 以測試 flow control 邏輯。
- **端到端量測腳本**：提供 `examples/jitter_benchmark.py`，自動收集 FPS、平均延遲、drop 率並輸出 CSV。
- **Fallback 設計**：若硬體無法支援高波特率或控制訊息，CLI 提供 `--legacy-mode` 參數，將行為退回舊有固定 chunk/delay 策略。

此藍圖將作為後續程式調整的依據：先落實設定檔與 ESP32 儲傳邏輯，再更新 Python 端流程，最後以文件與測試收尾。

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
