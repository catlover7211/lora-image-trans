# ESP32 LoRa Sender (發送端中繼器)

## 功能說明

此程式運行在發送端的 ESP32 上，負責：
1. 接收來自 Raspberry Pi 的影像資料（透過 USB Serial）
2. 將資料轉發到 LoRa 模組（透過 UART2）

## 硬體連接

```
Raspberry Pi <--USB Serial--> ESP32 <--UART2--> LoRa 模組 (ATK-LORA-01)
```

### ESP32 接腳配置

| ESP32 Pin | 連接對象 | 說明 |
|-----------|---------|------|
| USB       | Raspberry Pi | USB Serial 通訊 |
| GPIO 16 (RX2) | LoRa TX | UART2 接收 |
| GPIO 17 (TX2) | LoRa RX | UART2 發送 |
| GND       | LoRa GND | 共地 |
| 3.3V      | LoRa VCC | 電源（依 LoRa 模組規格，可能需要 5V）|

## 工作原理

1. **接收階段**：從 USB Serial 逐位元組接收資料
2. **緩衝處理**：將資料累積到緩衝區
3. **幀偵測**：偵測完整幀（以 `0x55 0xAA` 結束）
4. **轉發階段**：將完整幀透過 UART2 發送到 LoRa 模組

## 緩衝區大小

- **USB_BUFFER_SIZE**: 4096 bytes - 接收來自 Raspberry Pi 的資料
- **LORA_BUFFER_SIZE**: 512 bytes - 暫存區（未使用）

## 上傳步驟

1. 開啟 Arduino IDE
2. 選擇開發板：`Tools > Board > ESP32 Dev Module`
3. 選擇序列埠：`Tools > Port > [你的 ESP32 序列埠]`
4. 設定上傳速度：`Tools > Upload Speed > 115200`
5. 點擊「上傳」按鈕

## 除錯

程式啟動後會透過 USB Serial 輸出狀態訊息：
- `ESP32 LoRa Sender Started` - 程式啟動
- `Waiting for data from Raspberry Pi...` - 等待資料
- `ERROR: USB buffer overflow, resetting` - 緩衝區溢位（重新同步）

若要啟用詳細除錯，取消註解程式中的除錯輸出：
```cpp
// Serial.print("Forwarded frame to LoRa: ");
// Serial.print(usb_buffer_pos);
// Serial.println(" bytes");
```

## 注意事項

1. **波特率一致**：確保 Raspberry Pi、ESP32、LoRa 模組使用相同波特率 (115200)
2. **電源供應**：確保 LoRa 模組有足夠電源（某些模組需要 5V）
3. **接線正確**：TX 接 RX，RX 接 TX
4. **LoRa 設定**：LoRa 模組需預先設定為透明傳輸模式

## LoRa 模組設定

使用 AT 命令設定 ATK-LORA-01（參考模組手冊）：
```
AT+MODE=1        # 設定為透明傳輸模式
AT+ADDR=1        # 設定位址
AT+CHANNEL=23    # 設定頻道
AT+BAUDRATE=9    # 設定波特率為 115200
```

確保發送端和接收端的 LoRa 模組設定相同。
