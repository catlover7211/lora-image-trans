# ESP32 LoRa 影像串流系統整合說明

本文件說明 Python 程式端與 Arduino/ESP32 韌體的整合架構。

## 系統架構

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Python     │ USB  │   ESP32      │ LoRa │   ESP32      │ USB  │   Python     │
│  (發送端)    │<---->│   (中繼器)   │<---->│   (中繼器)   │<---->│  (接收端)    │
│  capture.py  │      │   Arduino    │      │   Arduino    │      │   main.py    │
└──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
     ▲                                                                     │
     │                                                                     │
     └─────────────────────── protocol.py ───────────────────────────────┘
```

## 通訊協定層級

### 第一層：幀協定 (Frame Protocol)

**實作位置**: `protocol.py`

**幀格式**:
```
FRAME <length> <crc> <base64_data>\n
```

**欄位說明**:
- `FRAME`: 固定前綴，用於同步
- `<length>`: base64 編碼後的資料長度（十進位）
- `<crc>`: CRC32 校驗碼（8 位十六進位）
- `<base64_data>`: Base64 編碼的 payload
- `\n`: 行結束符

**範例**:
```
FRAME 320 a3f5c2d1 SGVsbG8gV29ybGQhIFRoaXMgaXMgYSB0ZXN0IG1lc3NhZ2UuCg==\n
```

### 第二層：Payload 協定

**實作位置**: `h264_codec.py`

**Payload 結構**:
```
[flags: 1 byte][data: variable length]
```

**Flags 位元定義**:
- `0x01`: Keyframe
- `0x02`: Config
- `0x04`: HEVC/H.265
- `0x08`: AV1
- `0x10`: Wavelet
- `0x20`: JPEG
- `0x40`: Contour
- `0x80`: YOLO

### 第三層：編碼資料

**實作位置**: `h264_codec.py`

根據 codec 類型，data 欄位包含：
- **H.264/H.265/AV1**: NAL units (見 `h264_codec.py` 的 `H264Encoder`/`H264Decoder`)
- **JPEG**: JPEG 檔案資料 (見 `h264_codec.py` 的 `JPEGEncoder`)
- **Wavelet**: 自訂壓縮資料 (見 `h264_codec.py` 的 `WaveletEncoder`/`WaveletDecoder`)
- **Contour**: 傅立葉係數 (見 `h264_codec.py` 的 `ContourEncoder`/`ContourDecoder`)
- **YOLO**: 邊界框資料 (見 `h264_codec.py` 的 `DetectionEncoder`/`DetectionDecoder`)

## Arduino/ESP32 韌體角色

### 功能定位

ESP32 韌體扮演**透明中繼器**的角色：
- **不解析**幀內容
- **不修改**資料
- 僅負責在 USB Serial 和 LoRa 之間**轉發**完整幀

### 關鍵設計決策

1. **行緩衝處理**: 以 `\n` 作為訊息邊界
2. **透明轉發**: 保持協定層級分離
3. **雙向通訊**: 同時處理雙向資料流
4. **錯誤恢復**: 緩衝區溢出時自動重新同步

### 為什麼不在 Arduino 上解析？

1. **資源限制**: ESP32 記憶體和處理能力有限
2. **維護性**: 協定變更不需要更新韌體
3. **簡單性**: 降低韌體複雜度和故障點
4. **效能**: 避免額外的處理延遲

## 資料流程

### 發送端 (capture.py)

```python
1. 擷取影像
   ↓
2. 編碼 (H264Encoder/JPEGEncoder/WaveletEncoder/...)
   ↓
3. 建立 EncodedChunk (flags + data)
   ↓
4. 建立 Frame (FRAME + length + crc + base64)
   ↓
5. 分割成 chunks (預設 240 bytes)
   ↓
6. 透過 Serial 發送到 ESP32
```

### ESP32 中繼器 (arduino/)

```
USB Serial → 逐字節接收 → 累積到 \n → LoRa Serial
   ↑                                           ↓
   ←───────────────── 批次轉發 ←───────────────
```

### 接收端 (main.py)

```python
1. 從 Serial 接收資料
   ↓
2. 解析 Frame (驗證 CRC)
   ↓
3. Base64 解碼得到 payload
   ↓
4. 解析 EncodedChunk (flags + data)
   ↓
5. 解碼影像 (H264Decoder/...)
   ↓
6. 顯示畫面
```

## ACK 機制 (可選)

### 何時使用 ACK

- 需要可靠傳輸時
- 網路品質不佳時
- 長距離傳輸時

### ACK 流程

```
發送端                    接收端
  │                          │
  ├──> Chunk 1 (240 bytes) ─>│
  │                          │
  │<──────── ACK ────────────┤
  │                          │
  ├──> Chunk 2 (240 bytes) ─>│
  │                          │
  │<──────── ACK ────────────┤
  │                          │
  ⋮                          ⋮
```

### ACK 訊息格式

```
ACK\n
```

**特性**:
- 簡單的 4 字節訊息
- 由接收端自動發送
- 發送端等待 ACK 後再發送下一個 chunk
- 支援自動降級（未收到 ACK 時切換為無 ACK 模式）

## 設定參數對照

| Python 參數 | 預設值 | Arduino 對應 | 說明 |
|------------|--------|-------------|------|
| `BAUD_RATE` | 115200 | `Serial.begin(115200)` | 序列埠波特率 |
| `chunk_size` | 240 | - | Python 端分割大小 |
| `LINE_TERMINATOR` | `\n` | `incoming_byte == '\n'` | 訊息邊界 |
| `ACK_MESSAGE` | `ACK\n` | 透明轉發 | ACK 訊息 |

## 除錯建議

### Python 端

啟用詳細日誌:
```bash
python capture.py --verbose
python main.py --verbose
```

### Arduino 端

取消註解除錯輸出:
```cpp
Serial.print("Forwarded to LoRa, size: ");
Serial.println(usb_buffer_pos);
```

### 常見問題診斷

1. **無法連線**:
   - 檢查波特率設定
   - 確認 COM port 選擇正確
   - 驗證硬體接線

2. **資料損壞**:
   - 檢查 CRC 錯誤訊息
   - 嘗試啟用 ACK 模式
   - 降低發送速率

3. **傳輸延遲**:
   - 調整 chunk_size
   - 減少 inter_chunk_delay
   - 停用 ACK（如果網路穩定）

## 效能調校

### 增加傳輸速度

1. **增大 chunk_size**: `--chunk-size 480`
2. **停用 ACK**: `--no-ack`
3. **降低畫質**: `--bitrate 200000`
4. **減少解析度**: 在 `image_settings.py` 調整

### 增加可靠性

1. **啟用 ACK**: `--ack`
2. **增加重試**: 調整 `ack_timeout`
3. **降低速率**: 減少 `chunk_size`
4. **使用錯誤更正**: 調整 LoRa 模組設定

## 參考文件

- **Python 端**:
  - `README.md`: 主要使用說明
  - `protocol.py`: 協定實作
  - `h264_codec.py`: 編解碼器
  - `main.py`: 接收端
  - `capture.py`: 發送端

- **Arduino 端**:
  - `arduino/README.md`: Arduino 使用說明
  - `arduino/CHANGES.md`: 調整說明
  - `arduino/esp32_lora_transceiver/`: 韌體程式碼

## 授權

請參考主專案的授權條款。
