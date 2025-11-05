# ESP32 LoRa Receiver (接收端中繼器)

## 功能說明

此程式運行在接收端的 ESP32 上，負責：
1. 接收來自 LoRa 模組的影像資料（透過 UART2）
2. 將資料轉發到 PC（透過 USB Serial）

## 硬體連接

```
LoRa 模組 (ATK-LORA-01) <--UART2--> ESP32 <--USB Serial--> PC
```

### ESP32 接腳配置

| ESP32 Pin | 連接對象 | 說明 |
|-----------|---------|------|
| GPIO 16 (RX2) | LoRa TX | UART2 接收 |
| GPIO 17 (TX2) | LoRa RX | UART2 發送 |
| USB       | PC | USB Serial 通訊 |
| GND       | LoRa GND | 共地 |
| 3.3V      | LoRa VCC | 電源（依 LoRa 模組規格，可能需要 5V）|

## 工作原理

1. **接收階段**：從 UART2 批次讀取來自 LoRa 模組的資料
2. **緩衝處理**：將資料累積到緩衝區
3. **幀偵測**：偵測完整幀（以 `0x55 0xAA` 結束）
4. **轉發階段**：將完整幀透過 USB Serial 發送到 PC

## 緩衝區大小

- **LORA_BUFFER_SIZE**: 512 bytes - 接收來自 LoRa 模組的資料
- **USB_BUFFER_SIZE**: 4096 bytes - 累積完整幀後發送到 PC

## 上傳步驟

1. 開啟 Arduino IDE
2. 選擇開發板：`Tools > Board > ESP32 Dev Module`
3. 選擇序列埠：`Tools > Port > [你的 ESP32 序列埠]`
4. 設定上傳速度：`Tools > Upload Speed > 115200`
5. 點擊「上傳」按鈕

## 除錯

程式啟動後會透過 USB Serial 輸出狀態訊息：
- `ESP32 LoRa Receiver Started` - 程式啟動
- `Waiting for data from LoRa...` - 等待資料
- `ERROR: USB buffer overflow, resetting` - 緩衝區溢位（重新同步）

若要啟用詳細除錯，取消註解程式中的除錯輸出：
```cpp
// Serial.print("\nForwarded frame to PC: ");
// Serial.print(usb_buffer_pos);
// Serial.println(" bytes");
```

**注意**：啟用除錯輸出會在資料流中插入訊息，可能干擾 PC 端的接收。建議只在測試時啟用。

## 注意事項

1. **波特率一致**：確保 LoRa 模組、ESP32、PC 使用相同波特率 (115200)
2. **電源供應**：確保 LoRa 模組有足夠電源（某些模組需要 5V）
3. **接線正確**：TX 接 RX，RX 接 TX
4. **LoRa 設定**：LoRa 模組需預先設定為透明傳輸模式
5. **天線連接**：確保 LoRa 模組已連接天線

## LoRa 模組設定

使用 AT 命令設定 ATK-LORA-01（參考模組手冊）：
```
AT+MODE=1        # 設定為透明傳輸模式
AT+ADDR=2        # 設定位址（與發送端不同）
AT+CHANNEL=23    # 設定頻道（與發送端相同）
AT+BAUDRATE=9    # 設定波特率為 115200
```

確保發送端和接收端的 LoRa 模組頻道設定相同。

## 效能最佳化

1. **批次讀取**：程式使用批次讀取提高效率
2. **最小延遲**：使用 `delayMicroseconds(500)` 減少延遲
3. **緩衝管理**：智慧型緩衝區管理避免溢位
