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
#define MAX_FRAME_SIZE 65535
#define USB_BUFFER_SIZE (MAX_FRAME_SIZE + 16)

// 使用 ESP32 的第二個串口連接 LoRa 模組
HardwareSerial LoRaSerial(2);

// 緩衝區
uint8_t lora_buffer[LORA_BUFFER_SIZE];
uint8_t frame_buffer[USB_BUFFER_SIZE];

enum FrameState {
  SeekingStart,
  CollectingFrame
};

FrameState frame_state = SeekingStart;
size_t frame_buffer_pos = 0;
size_t expected_frame_length = 0;
size_t start_match = 0;

// 幀標記
const uint8_t FRAME_START[] = {0xAA, 0x55};
const uint8_t FRAME_END[] = {0x55, 0xAA};

void reset_frame_state() {
  frame_state = SeekingStart;
  frame_buffer_pos = 0;
  expected_frame_length = 0;
  start_match = 0;
}

void try_resync_with_byte(uint8_t byte) {
  if (byte == FRAME_START[0]) {
    frame_buffer[0] = byte;
    start_match = 1;
  }
}

void process_byte(uint8_t byte) {
  if (frame_state == SeekingStart) {
    if (start_match == 0) {
      if (byte == FRAME_START[0]) {
        frame_buffer[0] = byte;
        start_match = 1;
      }
    } else {  // start_match == 1
      if (byte == FRAME_START[1]) {
        frame_buffer[1] = byte;
        frame_buffer_pos = 2;
        frame_state = CollectingFrame;
        expected_frame_length = 0;
        start_match = 0;
      } else if (byte == FRAME_START[0]) {
        frame_buffer[0] = byte;
        start_match = 1;
      } else {
        start_match = 0;
      }
    }
    return;
  }

  // Collecting frame payload
  if (frame_buffer_pos >= USB_BUFFER_SIZE) {
    Serial.println("\nWARN: Frame buffer overflow, dropping");
    reset_frame_state();
    try_resync_with_byte(byte);
    return;
  }

  frame_buffer[frame_buffer_pos++] = byte;

  // Once header is complete determine expected length
  if (frame_buffer_pos == 5 && expected_frame_length == 0) {
    size_t payload_length = (static_cast<size_t>(frame_buffer[3]) << 8) | frame_buffer[4];
    if (payload_length == 0 || payload_length > MAX_FRAME_SIZE) {
      Serial.println("\nWARN: Invalid frame length, dropping");
      reset_frame_state();
      try_resync_with_byte(byte);
      return;
    }
    expected_frame_length = payload_length + 9;  // full frame including markers
  }

  if (expected_frame_length > 0 && frame_buffer_pos == expected_frame_length) {
    bool has_valid_end =
        frame_buffer[frame_buffer_pos - 2] == FRAME_END[0] &&
        frame_buffer[frame_buffer_pos - 1] == FRAME_END[1];

    if (has_valid_end) {
      Serial.write(frame_buffer, frame_buffer_pos);
      Serial.flush();
    } else {
      Serial.println("\nWARN: Frame end mismatch, dropping");
    }

    reset_frame_state();
  }
}

void setup() {
  reset_frame_state();
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
      for (size_t i = 0; i < bytes_read; i++) {
        process_byte(lora_buffer[i]);
      }
    }
  }

  // 短暫延遲避免過度佔用 CPU
  delayMicroseconds(500);
}
