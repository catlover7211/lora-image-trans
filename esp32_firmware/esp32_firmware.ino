#include <HardwareSerial.h>

// ESP32 UART2 - 連接 ATK-LORA-01 模組
constexpr int RXD2 = 16;
constexpr int TXD2 = 17;

// 緩衝區大小常數
constexpr size_t USB_BUFFER_SIZE = 4096;
constexpr size_t LORA_BUFFER_SIZE = 512;

HardwareSerial LoRaSerial(2);

// 使用固定大小的緩衝區避免記憶體碎片化
uint8_t usb_buffer[USB_BUFFER_SIZE];
size_t usb_buffer_pos = 0;

// LoRa 轉發緩衝區
uint8_t lora_buffer[LORA_BUFFER_SIZE];

void setup() {
  Serial.begin(115200);
  LoRaSerial.begin(115200, SERIAL_8N1, RXD2, TXD2);

  // 設定一個合理的超時，避免無限等待
  Serial.setTimeout(10); 
  LoRaSerial.setTimeout(10);

  Serial.println("ESP32 LoRa Transceiver Ready (ATK-LORA-01)");
  Serial.println("Note: ATK-LORA-01 should be pre-configured in transparent transmission mode");
  Serial.print("USB Buffer: ");
  Serial.print(USB_BUFFER_SIZE);
  Serial.print(" bytes, LoRa Buffer: ");
  Serial.print(LORA_BUFFER_SIZE);
  Serial.println(" bytes");
}

void loop() {
  // 1. 從 USB Serial 讀取數據並轉發到 LoRa
  // Python 端的 protocol.py 已經在 build_frame() 中加入了 SYNC_MARKER
  // 所以這裡不需要再次添加，直接轉發即可
  while (Serial.available() > 0) {
    uint8_t incoming_byte = Serial.read();
    
    // 檢查緩衝區是否有空間
    if (usb_buffer_pos < USB_BUFFER_SIZE) {
      usb_buffer[usb_buffer_pos++] = incoming_byte;
      
      // 如果讀到了換行符，代表一個完整的封包已接收
      if (incoming_byte == '\n') {
        // 直接轉發整個封包到 LoRa（封包已包含 SYNC_MARKER 和完整的幀數據）
        LoRaSerial.write(usb_buffer, usb_buffer_pos);
        LoRaSerial.flush();
        
        // 除錯日誌（可選）
        // Serial.print("Forwarded packet, size: ");
        // Serial.println(usb_buffer_pos);
        
        // 重置緩衝區位置，準備接收下一個封包
        usb_buffer_pos = 0;
      }
    } else {
      // 緩衝區溢出 - 重置並記錄錯誤
      Serial.println("ERROR: USB buffer overflow, resetting");
      usb_buffer_pos = 0;
      // 快速清空序列埠緩衝區直到換行符以重新同步
      // 批次讀取以提升效率
      while (Serial.available() > 0) {
        if (Serial.read() == '\n') {
          break;
        }
      }
    }
  }

  // 2. 將 LoRa 的資料批次轉回 USB Serial（提升效率）
  if (LoRaSerial.available() > 0) {
    size_t bytes_read = LoRaSerial.readBytes(lora_buffer, LORA_BUFFER_SIZE);
    if (bytes_read > 0) {
      Serial.write(lora_buffer, bytes_read);
      Serial.flush();
    }
  }

  // 短暫延遲，讓出 CPU（使用微秒延遲減少延遲）
  delayMicroseconds(500);
}