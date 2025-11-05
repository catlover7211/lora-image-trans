/*
 * ESP32 LoRa Sender (發送端中繼器)
 * 
 * 此程式負責接收來自 Raspberry Pi 的影像資料並透過 LoRa 模組轉發。
 * 
 * 硬體連接：
 * - ESP32 USB Serial <-> Raspberry Pi
 * - ESP32 UART2 (GPIO16 RX, GPIO17 TX) <-> LoRa 模組 (ATK-LORA-01)
 * 
 * 資料流向：
 * Raspberry Pi -> USB Serial -> ESP32 -> UART2 -> LoRa 模組
 */

#include <HardwareSerial.h>

// 定義 ESP32 的 UART2 引腳 (連接 LoRa 模組)
#define RXD2 16
#define TXD2 17

// 緩衝區大小
#define USB_BUFFER_SIZE 4096
#define LORA_BUFFER_SIZE 512

// 使用 ESP32 的第二個串口連接 LoRa 模組
HardwareSerial LoRaSerial(2);

// 緩衝區
uint8_t usb_buffer[USB_BUFFER_SIZE];
size_t usb_buffer_pos = 0;
uint8_t lora_buffer[LORA_BUFFER_SIZE];

// 幀標記
const uint8_t FRAME_START[] = {0xAA, 0x55};
const uint8_t FRAME_END[] = {0x55, 0xAA};

void setup() {
  // 初始化 USB Serial (連接 Raspberry Pi)
  Serial.begin(115200);
  
  // 初始化 LoRa Serial (連接 LoRa 模組)
  LoRaSerial.begin(115200, SERIAL_8N1, RXD2, TXD2);

  // 設定合理的超時
  Serial.setTimeout(10);
  LoRaSerial.setTimeout(10);

  // 等待序列埠穩定
  delay(1000);

  Serial.println("ESP32 LoRa Sender Started");
  Serial.println("Waiting for data from Raspberry Pi...");
  Serial.print("USB Buffer: ");
  Serial.print(USB_BUFFER_SIZE);
  Serial.print(" bytes, LoRa Buffer: ");
  Serial.print(LORA_BUFFER_SIZE);
  Serial.println(" bytes");
}

void loop() {
  // ========================================================================
  // 1. 從 USB Serial 接收資料並轉發到 LoRa
  // ========================================================================
  while (Serial.available() > 0) {
    uint8_t incoming_byte = Serial.read();
    
    // 檢查緩衝區空間
    if (usb_buffer_pos < USB_BUFFER_SIZE) {
      usb_buffer[usb_buffer_pos++] = incoming_byte;
      
      // 檢查是否收到完整幀 (以 FRAME_END 標記結束)
      if (usb_buffer_pos >= 2) {
        if (usb_buffer[usb_buffer_pos - 2] == FRAME_END[0] && 
            usb_buffer[usb_buffer_pos - 1] == FRAME_END[1]) {
          
          // 完整幀接收完畢，轉發到 LoRa
          LoRaSerial.write(usb_buffer, usb_buffer_pos);
          LoRaSerial.flush();
          
          // 可選的除錯輸出（取消註解以啟用）
          // Serial.print("Forwarded frame to LoRa: ");
          // Serial.print(usb_buffer_pos);
          // Serial.println(" bytes");
          
          // 重置緩衝區
          usb_buffer_pos = 0;
        }
      }
    } else {
      // 緩衝區溢位 - 尋找下一個 FRAME_START 重新同步
      Serial.println("ERROR: USB buffer overflow, resetting");
      usb_buffer_pos = 0;
      
      // 清空序列埠緩衝區直到找到 FRAME_START
      bool found_start = false;
      while (Serial.available() > 0 && !found_start) {
        uint8_t b1 = Serial.read();
        if (b1 == FRAME_START[0] && Serial.available() > 0) {
          uint8_t b2 = Serial.read();
          if (b2 == FRAME_START[1]) {
            usb_buffer[0] = b1;
            usb_buffer[1] = b2;
            usb_buffer_pos = 2;
            found_start = true;
          }
        }
      }
    }
  }

  // 短暫延遲避免過度佔用 CPU
  delayMicroseconds(500);
}
