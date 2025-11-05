# Arduino/ESP32 韌體

此目錄包含用於 LoRa 影像串流的 ESP32 韌體程式碼。

## 硬體需求

- **ESP32 開發板**（例如：ESP32-DevKitC）
- **ATK-LORA-01 模組**（或其他相容的 LoRa 模組）
- **連接線**

## 硬體連接

將 ESP32 與 ATK-LORA-01 模組連接如下：

| ESP32 | ATK-LORA-01 |
|-------|-------------|
| GPIO16 (RXD2) | TX |
| GPIO17 (TXD2) | RX |
| GND | GND |
| 3.3V/5V | VCC |

## 軟體需求

- **Arduino IDE** 1.8.x 或更高版本
- **ESP32 開發板支援**

### 安裝 ESP32 開發板支援

1. 開啟 Arduino IDE
2. 前往 **檔案 > 偏好設定**
3. 在「額外的開發板管理員網址」中加入：
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
4. 前往 **工具 > 開發板 > 開發板管理員**
5. 搜尋「ESP32」並安裝

## 上傳韌體

1. 開啟 Arduino IDE
2. 載入 `esp32_lora_transceiver/esp32_lora_transceiver.ino`
3. 選擇正確的開發板：**工具 > 開發板 > ESP32 Arduino > ESP32 Dev Module**
4. 選擇正確的序列埠：**工具 > 序列埠**
5. 點擊「上傳」按鈕

## 設定 ATK-LORA-01 模組

在使用前，ATK-LORA-01 模組需要設定為**透明傳輸模式**。請參考 ATK-LORA-01 的使用手冊進行設定。

常見設定參數：
- **波特率**: 115200
- **傳輸模式**: 透明傳輸
- **頻率**: 433MHz 或 868MHz（依地區法規）
- **發射功率**: 依需求調整
- **空中速率**: 依需求調整（影響傳輸距離與速度）

## 通訊協定

此韌體配合 Python 端的 `protocol.py` 實現的 ASCII 幀協定：

### 幀格式
```
FRAME <length> <crc> <base64_data>\n
```

- **FRAME**: 幀前綴標記
- **length**: base64 編碼後的資料長度
- **crc**: CRC32 校驗碼（十六進位）
- **base64_data**: Base64 編碼的 payload
- **\n**: 行結束符

### ACK 訊息（可選）
```
ACK\n
```

當啟用 chunk-level ACK 時，接收端會在每個 chunk 後發送 ACK。

## 運作原理

1. **USB -> LoRa**: 從 USB Serial 接收完整幀（以 `\n` 結束），直接轉發到 LoRa 模組
2. **LoRa -> USB**: 從 LoRa 模組接收資料，直接轉發到 USB Serial

韌體本身不解析幀內容，僅負責透明轉發，所有協定邏輯由 Python 端處理。

## 緩衝區設定

- **USB_BUFFER_SIZE**: 4096 bytes（足以容納單個完整幀）
- **LORA_BUFFER_SIZE**: 512 bytes（批次轉發提升效率）

## 除錯

韌體包含可選的除錯日誌。要啟用除錯輸出，請取消註解 `.ino` 檔案中的相關 `Serial.print()` 行：

```cpp
// 取消註解以啟用除錯
Serial.print("Forwarded to LoRa, size: ");
Serial.println(usb_buffer_pos);
```

## 注意事項

1. **波特率一致性**: ESP32、ATK-LORA-01 和 Python 端的波特率必須一致（預設 115200）
2. **透明傳輸模式**: ATK-LORA-01 必須設定為透明傳輸模式，否則會干擾資料轉發
3. **電源供應**: LoRa 模組發射時耗電較大，請確保電源供應充足
4. **天線連接**: 使用前請確保天線已正確連接，避免損壞 LoRa 模組

## 故障排除

### 無法上傳韌體
- 確認 ESP32 已正確連接到電腦
- 嘗試按住 ESP32 的 BOOT 按鈕再上傳
- 確認已選擇正確的序列埠

### 無法通訊
- 檢查 ESP32 與 ATK-LORA-01 的接線
- 確認 ATK-LORA-01 已設定為透明傳輸模式
- 檢查波特率設定是否一致
- 使用序列埠監控器查看除錯訊息

### 資料遺失或錯誤
- 檢查 LoRa 模組的空中速率設定
- 減少發送速率或增加 chunk 延遲
- 檢查電源供應是否穩定
- 考慮啟用 ACK 模式提升可靠性
