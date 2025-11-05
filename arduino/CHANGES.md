# Arduino 程式碼調整說明

## 主要變更

此文件說明根據當前 `protocol.py`、`main.py` 和 `h264_codec.py` 對舊 Arduino 程式碼所做的調整。

## 協定分析

### Python 端協定實作 (protocol.py)

當前協定使用以下格式：

```
FRAME <length> <crc> <base64_data>\n
```

**重要發現：**
- **沒有獨立的 SYNC_MARKER**：幀格式本身已包含同步機制
- `FRAME_PREFIX` = "FRAME"
- `FIELD_SEPARATOR` = " " (空格)
- `LINE_TERMINATOR` = "\n" (換行符)
- 預設 `BAUD_RATE` = 115200
- 預設 `chunk_size` = 240 bytes
- 支援可選的 chunk-level ACK: `ACK\n`

### 幀建立流程 (protocol.py:119-129)

```python
def build_frame(self, payload: bytes) -> tuple[bytes, FrameStats]:
    """Construct the ASCII framed byte stream and statistics for *payload*."""
    encoded = base64.b64encode(payload).decode("ascii")
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    header = FIELD_SEPARATOR.join(
        (FRAME_PREFIX, str(len(encoded)), f"{crc:08x}")
    )
    frame_str = header + FIELD_SEPARATOR + encoded + LINE_TERMINATOR
    frame_bytes = frame_str.encode("ascii")
    stats = FrameStats(payload_size=len(payload), stuffed_size=len(encoded), crc=crc)
    return frame_bytes, stats
```

### 幀發送流程 (protocol.py:136-158)

```python
def send_frame(self, ser: SerialLike, payload: bytes) -> FrameStats:
    """Send *payload* through the provided serial connection."""
    frame_bytes, stats = self.build_frame(payload)
    for chunk in self.iter_chunks(frame_bytes):
        ser.write(chunk)
        if self.inter_chunk_delay:
            time.sleep(self.inter_chunk_delay)
        if not self.use_chunk_ack:
            continue
        # ... ACK handling ...
    ser.flush()
    return stats
```

## 舊程式碼 vs 新程式碼

### 主要差異

| 項目 | 舊程式碼 | 新程式碼 | 說明 |
|------|----------|----------|------|
| SYNC_MARKER 假設 | 註解中提到 Python 已加入 SYNC_MARKER | 移除誤導性註解 | 實際上沒有獨立的 SYNC_MARKER |
| 協定說明 | 簡單提及透明傳輸 | 詳細的協定格式註解 | 明確說明幀格式和協定特性 |
| 功能說明 | 基本轉發功能 | 完整的協定說明 | 包含幀格式、ACK、chunk 等細節 |
| 除錯輸出 | 已註解的簡單日誌 | 更詳細的可選日誌 | 提供更好的除錯資訊 |
| setup() 輸出 | 基本資訊 | 詳細的協定和配置資訊 | 包含幀格式、波特率等 |

### 保持不變的部分

以下核心功能**完全保持不變**，因為原始實作已經正確：

1. **緩衝區大小**：
   - `USB_BUFFER_SIZE = 4096`
   - `LORA_BUFFER_SIZE = 512`

2. **USB -> LoRa 轉發邏輯**：
   - 逐字節讀取
   - 遇到 `\n` 時轉發完整訊息
   - 緩衝區溢出處理

3. **LoRa -> USB 轉發邏輯**：
   - 批次讀取
   - 直接轉發

4. **硬體配置**：
   - RXD2 = GPIO16
   - TXD2 = GPIO17
   - 波特率 = 115200

## 為何舊程式碼已經正確

舊程式碼的核心轉發邏輯已經完全符合協定需求：

1. **行緩衝處理**：以 `\n` 作為訊息邊界，這與 `LINE_TERMINATOR` 一致
2. **透明轉發**：Arduino 不需要解析幀內容，只需轉發完整訊息
3. **雙向通訊**：同時處理 USB->LoRa 和 LoRa->USB
4. **緩衝區管理**：適當的緩衝區大小和溢出處理

## 調整內容總結

新程式碼主要改進了**文件和註解**，而非核心邏輯：

1. ✅ **修正註解**：移除對不存在的 SYNC_MARKER 的提及
2. ✅ **增強文件**：詳細說明實際的幀協定格式
3. ✅ **改進啟動訊息**：提供更多協定和配置資訊
4. ✅ **統一術語**：使用與 Python 程式碼一致的術語
5. ✅ **新增 README**：提供完整的設定和使用說明

## 驗證清單

- [x] 波特率一致 (115200)
- [x] 行結束符處理 (`\n`)
- [x] 支援 ACK 訊息轉發 (`ACK\n`)
- [x] 適當的緩衝區大小
- [x] 雙向透明轉發
- [x] 錯誤處理和重新同步
- [x] 註解準確反映實際協定

## 使用建議

1. **直接使用新程式碼**：新版本有更好的文件說明
2. **檢查 ATK-LORA-01 設定**：確保設定為透明傳輸模式
3. **波特率匹配**：ESP32、LoRa 模組和 Python 程式都使用 115200
4. **可選除錯**：需要時可取消註解除錯輸出
5. **測試 ACK 模式**：可以測試有/無 ACK 的情況

## 參考資料

- `protocol.py`: 幀協定實作
- `main.py`: 接收端邏輯
- `h264_codec.py`: 編碼/解碼實作
- `arduino/README.md`: Arduino 韌體使用說明
