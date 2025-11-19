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

// 緩衝區大小 - 以環形緩衝平衡 USB 與 LoRa 速率
#define RELAY_BUFFER_SIZE 4096
#define USB_WRITE_GUARD_US 30

// 使用 ESP32 的第二個串口連接 LoRa 模組
HardwareSerial LoRaSerial(2);

// 環形緩衝
uint8_t relay_buffer[RELAY_BUFFER_SIZE];
size_t relay_head = 0;
size_t relay_tail = 0;
size_t relay_fill = 0;

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
  Serial.setTimeout(0);
  LoRaSerial.setTimeout(0);

  // 等待序列埠穩定
  delay(1000);

  Serial.println("ESP32 LoRa Receiver Started (Relay Mode)");
  Serial.println("Waiting for data from LoRa...");
  Serial.print("Relay Buffer: ");
  Serial.print(RELAY_BUFFER_SIZE);
  Serial.println(" bytes");
  Serial.println("Frame reconstruction handled by PC");
}

inline size_t relay_free() {
  return RELAY_BUFFER_SIZE - relay_fill;
}

void pump_from_lora() {
  while (LoRaSerial.available() > 0 && relay_free() > 0) {
    relay_buffer[relay_head] = LoRaSerial.read();
    relay_head = (relay_head + 1) % RELAY_BUFFER_SIZE;
    relay_fill++;
  }
}

void pump_to_usb() {
  size_t writable = Serial.availableForWrite();
  while (relay_fill > 0 && writable > 0) {
    size_t contiguous = RELAY_BUFFER_SIZE - relay_tail;
    size_t chunk = relay_fill < contiguous ? relay_fill : contiguous;
    if (chunk > writable) {
      chunk = writable;
    }
    if (chunk == 0) {
      break;
    }
    Serial.write(relay_buffer + relay_tail, chunk);
    relay_tail = (relay_tail + chunk) % RELAY_BUFFER_SIZE;
    relay_fill -= chunk;
    writable -= chunk;
    bytes_forwarded += chunk;
  }
}

void loop() {
  pump_from_lora();
  pump_to_usb();

  if (relay_fill == RELAY_BUFFER_SIZE) {
    // 以丟棄最舊資料的方式回復，避免完全阻塞
    relay_tail = (relay_tail + RELAY_BUFFER_SIZE / 4) % RELAY_BUFFER_SIZE;
    relay_fill -= RELAY_BUFFER_SIZE / 4;
  }

  // Print statistics every 60 seconds
  /*
  unsigned long current_time = millis();
  if (current_time - last_stats_time >= 60000) {
    Serial.print("\n--- Stats: Bytes forwarded=");
    Serial.print(bytes_forwarded);
    Serial.println(" ---");
    last_stats_time = current_time;
  }
  */

  if (relay_fill == 0 && LoRaSerial.available() == 0) {
    delayMicroseconds(200);
  } else {
    delayMicroseconds(USB_WRITE_GUARD_US);
  }
}
