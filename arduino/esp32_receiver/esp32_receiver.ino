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
 * 
 * 架構說明：
 * ESP32 現在作為簡單的資料中繼器，不進行幀驗證或緩衝。
 * 所有幀重組和驗證邏輯都在 PC 端進行，以避免 ESP32 記憶體溢位。
 */

#include <HardwareSerial.h>

// 定義 ESP32 的 UART2 引腳 (連接 LoRa 模組)
#define RXD2 16
#define TXD2 17

// 緩衝區大小 - 只需要小型緩衝區用於批次轉發
#define RELAY_BUFFER_SIZE 512

// 使用 ESP32 的第二個串口連接 LoRa 模組
HardwareSerial LoRaSerial(2);

// 緩衝區
uint8_t relay_buffer[RELAY_BUFFER_SIZE];

// Statistics
unsigned long bytes_forwarded = 0;
unsigned long last_stats_time = 0;

void setup() {
  // 初始化 USB Serial (連接電腦)
  Serial.setTxBufferSize(4096);
  Serial.setRxBufferSize(1024);
  Serial.begin(115200);
  
  // 初始化 LoRa Serial (連接 LoRa 模組)
  LoRaSerial.setRxBufferSize(4096);
  LoRaSerial.begin(115200, SERIAL_8N1, RXD2, TXD2);

  // 設定合理的超時
  Serial.setTimeout(10);
  LoRaSerial.setTimeout(10);

  // 等待序列埠穩定
  delay(1000);

  Serial.println("ESP32 LoRa Receiver Started (Relay Mode)");
  Serial.println("Waiting for data from LoRa...");
  Serial.print("Relay Buffer: ");
  Serial.print(RELAY_BUFFER_SIZE);
  Serial.println(" bytes");
  Serial.println("Frame reconstruction handled by PC");
}

void loop() {
  // ========================================================================
  // 簡單的資料中繼：從 LoRa 接收資料並立即轉發到 USB Serial
  // ========================================================================
  if (LoRaSerial.available() > 0) {
    // 批次讀取以提升效率
    size_t bytes_read = LoRaSerial.readBytes(relay_buffer, RELAY_BUFFER_SIZE);

    if (bytes_read > 0) {
      // 直接轉發到 PC，不進行幀驗證或緩衝
      Serial.write(relay_buffer, bytes_read);
      Serial.flush();
      bytes_forwarded += bytes_read;
    }
  }

  // Print statistics every 60 seconds
  unsigned long current_time = millis();
  if (current_time - last_stats_time >= 60000) {
    Serial.print("\n--- Stats: Bytes forwarded=");
    Serial.print(bytes_forwarded);
    Serial.println(" ---");
    last_stats_time = current_time;
  }

  // 短暫延遲避免過度佔用 CPU
  delayMicroseconds(500);
}
