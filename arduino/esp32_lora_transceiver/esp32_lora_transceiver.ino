/*
 * ESP32 LoRa 影像串流中繼器 (ATK-LORA-01)
 * 
 * 此程式負責在 USB Serial 與 LoRa 模組之間轉發資料。
 * 配合 protocol.py 的 ASCII 幀協定：
 * 
 * 幀格式: FRAME <length> <crc> <base64_data>\n
 * - FRAME_PREFIX: "FRAME"
 * - FIELD_SEPARATOR: " " (空格)
 * - LINE_TERMINATOR: "\n" (換行符)
 * - 每個完整幀以 \n 結束
 * 
 * 協定特性：
 * - ASCII base64 編碼的 payload
 * - CRC32 校驗
 * - 支援分段 ACK (可選)
 * - 預設 chunk_size: 240 bytes
 * - 預設 baud_rate: 115200
 * 
 * 硬體連接：
 * - ESP32 UART2 RX (GPIO16) -> ATK-LORA-01 TX
 * - ESP32 UART2 TX (GPIO17) -> ATK-LORA-01 RX
 * - ATK-LORA-01 需預先設定為透明傳輸模式
 */

#include <HardwareSerial.h>

// ESP32 UART2 - 連接 ATK-LORA-01 模組
constexpr int RXD2 = 16;
constexpr int TXD2 = 17;

// 緩衝區大小常數
// USB buffer 需足夠容納單個完整幀 (包含標頭、base64 編碼資料和結束符)
// 根據 protocol.py: max_payload_size 預設 1920*1080，經 base64 編碼後約 4/3 倍
// 加上標頭開銷，4096 bytes 足以應付大多數單幀分段
constexpr size_t USB_BUFFER_SIZE = 4096;
constexpr size_t LORA_BUFFER_SIZE = 512;

HardwareSerial LoRaSerial(2);

// 使用固定大小的緩衝區避免記憶體碎片化
uint8_t usb_buffer[USB_BUFFER_SIZE];
size_t usb_buffer_pos = 0;

// LoRa 轉發緩衝區
uint8_t lora_buffer[LORA_BUFFER_SIZE];

void setup() {
  // 初始化 USB Serial (連接電腦)
  Serial.begin(115200);
  
  // 初始化 LoRa Serial (連接 ATK-LORA-01)
  // 波特率需與 protocol.py 的 BAUD_RATE 一致 (115200)
  LoRaSerial.begin(115200, SERIAL_8N1, RXD2, TXD2);

  // 設定合理的超時，避免無限等待
  Serial.setTimeout(10); 
  LoRaSerial.setTimeout(10);

  Serial.println("ESP32 LoRa Transceiver Ready (ATK-LORA-01)");
  Serial.println("Protocol: ASCII frame with base64 encoding");
  Serial.println("Frame format: FRAME <length> <crc> <base64_data>\\n");
  Serial.println("Note: ATK-LORA-01 should be pre-configured in transparent transmission mode");
  Serial.print("USB Buffer: ");
  Serial.print(USB_BUFFER_SIZE);
  Serial.print(" bytes, LoRa Buffer: ");
  Serial.print(LORA_BUFFER_SIZE);
  Serial.println(" bytes");
  Serial.println("Baud Rate: 115200");
}

void loop() {
  // ========================================================================
  // 1. 從 USB Serial 讀取資料並轉發到 LoRa
  // ========================================================================
  // Python 端的 protocol.py 在 build_frame() 中已經建立完整的幀：
  // "FRAME <length> <crc> <base64_data>\n"
  // 這裡只需接收完整幀（以 \n 結束）並轉發
  
  while (Serial.available() > 0) {
    uint8_t incoming_byte = Serial.read();
    
    // 檢查緩衝區是否有空間
    if (usb_buffer_pos < USB_BUFFER_SIZE) {
      usb_buffer[usb_buffer_pos++] = incoming_byte;
      
      // 如果讀到換行符，代表一個完整的幀或 ACK 訊息已接收
      if (incoming_byte == '\n') {
        // 直接轉發整個訊息到 LoRa
        // (幀已包含完整的 "FRAME <length> <crc> <base64>\n" 格式)
        LoRaSerial.write(usb_buffer, usb_buffer_pos);
        LoRaSerial.flush();
        
        // 除錯日誌（可選，取消註解以啟用）
        // Serial.print("Forwarded to LoRa, size: ");
        // Serial.println(usb_buffer_pos);
        
        // 重置緩衝區位置，準備接收下一個訊息
        usb_buffer_pos = 0;
      }
    } else {
      // 緩衝區溢出 - 重置並記錄錯誤
      Serial.println("ERROR: USB buffer overflow, resetting");
      usb_buffer_pos = 0;
      
      // 快速清空序列埠緩衝區直到換行符以重新同步
      while (Serial.available() > 0) {
        if (Serial.read() == '\n') {
          break;
        }
      }
    }
  }

  // ========================================================================
  // 2. 從 LoRa 接收資料並轉發到 USB Serial
  // ========================================================================
  // LoRa 端接收的資料可能是：
  // - 完整幀：FRAME <length> <crc> <base64_data>\n
  // - ACK 訊息：ACK\n (如果啟用 chunk ACK)
  // 批次讀取以提升效率
  
  if (LoRaSerial.available() > 0) {
    size_t bytes_read = LoRaSerial.readBytes(lora_buffer, LORA_BUFFER_SIZE);
    if (bytes_read > 0) {
      // 將接收到的資料轉發到 USB Serial
      Serial.write(lora_buffer, bytes_read);
      Serial.flush();
      
      // 除錯日誌（可選，取消註解以啟用）
      // Serial.print("Forwarded from LoRa, size: ");
      // Serial.println(bytes_read);
    }
  }

  // 短暫延遲，讓出 CPU（使用微秒延遲減少延遲）
  // 500us = 0.5ms，在 115200 baud 下約可傳輸 5.75 bytes
  delayMicroseconds(500);
}
