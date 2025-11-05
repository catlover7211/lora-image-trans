/*
 * ESP32 LoRa Receiver (接收端中繼器)
 * 
 * 此程式負責接收來自 LoRa 模組的影像資料並透過 USB Serial 轉發到電腦。
 * 
 * 硬體連接：
 * - ESP32 UART2 (GPIO16 RX, GPIO17 TX) <-> LoRa 模組 (ATK-LORA-01)
 * - ESP32 USB Serial <-> 電腦 (PC)
 * 
 * 資料流向：
 * LoRa 模組 -> UART2 -> ESP32 -> USB Serial -> PC
 */

#include <HardwareSerial.h>

// 定義 ESP32 的 UART2 引腳 (連接 LoRa 模組)
#define RXD2 16
#define TXD2 17

// 緩衝區大小
#define LORA_BUFFER_SIZE 512
// Reserve enough space for the largest frame (JPEG/CS) plus protocol overhead
// MAX_FRAME_SIZE is 65535 bytes, so 7000000 keeps a safety margin.
#define USB_BUFFER_SIZE 7000000

// 使用 ESP32 的第二個串口連接 LoRa 模組
HardwareSerial LoRaSerial(2);

// 緩衝區
uint8_t lora_buffer[LORA_BUFFER_SIZE];
uint8_t usb_buffer[USB_BUFFER_SIZE];
size_t usb_buffer_pos = 0;

// 幀標記
const uint8_t FRAME_START[] = {0xAA, 0x55};
const uint8_t FRAME_END[] = {0x55, 0xAA};

void setup() {
  // 初始化 USB Serial (連接電腦)
  Serial.begin(115200);
  
  // 初始化 LoRa Serial (連接 LoRa 模組)
  LoRaSerial.begin(115200, SERIAL_8N1, RXD2, TXD2);

  // 設定合理的超時
  Serial.setTimeout(10);
  LoRaSerial.setTimeout(10);

  // 等待序列埠穩定
  delay(1000);

  Serial.println("ESP32 LoRa Receiver Started");
  Serial.println("Waiting for data from LoRa...");
  Serial.print("LoRa Buffer: ");
  Serial.print(LORA_BUFFER_SIZE);
  Serial.print(" bytes, USB Buffer: ");
  Serial.print(USB_BUFFER_SIZE);
  Serial.println(" bytes");
}

void loop() {
  // ========================================================================
  // 1. 從 LoRa 接收資料並轉發到 USB Serial
  // ========================================================================
  if (LoRaSerial.available() > 0) {
    // 批次讀取以提升效率
    size_t bytes_read = LoRaSerial.readBytes(lora_buffer, LORA_BUFFER_SIZE);
    
    if (bytes_read > 0) {
      // 將接收到的資料加入 USB 緩衝區
      for (size_t i = 0; i < bytes_read; i++) {
        if (usb_buffer_pos < USB_BUFFER_SIZE) {
          usb_buffer[usb_buffer_pos++] = lora_buffer[i];
          
          // 檢查是否收到完整幀 (以 FRAME_END 標記結束)
          if (usb_buffer_pos >= 2) {
            if (usb_buffer[usb_buffer_pos - 2] == FRAME_END[0] && 
                usb_buffer[usb_buffer_pos - 1] == FRAME_END[1]) {
              
              // 完整幀接收完畢，轉發到 USB Serial
              Serial.write(usb_buffer, usb_buffer_pos);
              Serial.flush();
              
              // 可選的除錯輸出（取消註解以啟用）
              // Serial.print("\nForwarded frame to PC: ");
              // Serial.print(usb_buffer_pos);
              // Serial.println(" bytes");
              
              // 重置緩衝區
              usb_buffer_pos = 0;
            }
          }
        } else {
          // 緩衝區溢位 - 尋找下一個 FRAME_START 重新同步
          Serial.println("\nERROR: USB buffer overflow, resetting");
          usb_buffer_pos = 0;
          
          // 尋找 FRAME_START
          bool found_start = false;
          for (size_t j = i; j < bytes_read - 1 && !found_start; j++) {
            if (lora_buffer[j] == FRAME_START[0] && lora_buffer[j + 1] == FRAME_START[1]) {
              usb_buffer[0] = lora_buffer[j];
              usb_buffer[1] = lora_buffer[j + 1];
              usb_buffer_pos = 2;
              i = j + 1;
              found_start = true;
            }
          }
          if (!found_start) {
            break;  // 跳出當前批次，等待下一次讀取
          }
        }
      }
    }
  }

  // 短暫延遲避免過度佔用 CPU
  delayMicroseconds(500);
}
