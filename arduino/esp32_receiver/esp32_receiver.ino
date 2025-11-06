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
#include <cstring>

// 定義 ESP32 的 UART2 引腳 (連接 LoRa 模組)
#define RXD2 16
#define TXD2 17

// 緩衝區大小
#define LORA_BUFFER_SIZE 512
#define MAX_FRAME_SIZE 65535
#define FRAME_MIN_SIZE 9
#define USB_BUFFER_SIZE (MAX_FRAME_SIZE + FRAME_MIN_SIZE + 8)

// 使用 ESP32 的第二個串口連接 LoRa 模組
HardwareSerial LoRaSerial(2);

// 緩衝區
uint8_t lora_buffer[LORA_BUFFER_SIZE];
uint8_t frame_buffer[USB_BUFFER_SIZE];
size_t frame_buffer_pos = 0;

// 幀標記
const uint8_t FRAME_START[] = {0xAA, 0x55};
const uint8_t FRAME_END[] = {0x55, 0xAA};

void reset_frame_buffer() {
  frame_buffer_pos = 0;
}

void shift_frame_buffer(size_t offset) {
  if (offset == 0 || frame_buffer_pos == 0) {
    return;
  }
  if (offset >= frame_buffer_pos) {
    frame_buffer_pos = 0;
    return;
  }
  memmove(frame_buffer, frame_buffer + offset, frame_buffer_pos - offset);
  frame_buffer_pos -= offset;
}

int find_frame_start() {
  if (frame_buffer_pos < 2) {
    return -1;
  }
  for (size_t i = 0; i < frame_buffer_pos - 1; ++i) {
    if (frame_buffer[i] == FRAME_START[0] && frame_buffer[i + 1] == FRAME_START[1]) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

void forward_complete_frames() {
  while (true) {
    if (frame_buffer_pos < FRAME_MIN_SIZE) {
      return;
    }

    int start_idx = find_frame_start();
    if (start_idx < 0) {
      // Keep last byte only if it could be the beginning of FRAME_START
      uint8_t last_byte = frame_buffer[frame_buffer_pos - 1];
      if (last_byte == FRAME_START[0]) {
        frame_buffer[0] = last_byte;
        frame_buffer_pos = 1;
      } else {
        frame_buffer_pos = 0;
      }
      return;
    }

    if (start_idx > 0) {
      shift_frame_buffer(static_cast<size_t>(start_idx));
      continue;
    }

    if (frame_buffer_pos < 5) {
      return;
    }

    size_t payload_length = (static_cast<size_t>(frame_buffer[3]) << 8) | frame_buffer[4];
    if (payload_length == 0 || payload_length > MAX_FRAME_SIZE) {
      Serial.println("\nWARN: Invalid frame length, skipping");
      shift_frame_buffer(1);
      continue;
    }

    size_t frame_length = payload_length + FRAME_MIN_SIZE;
    if (frame_length > USB_BUFFER_SIZE) {
      Serial.println("\nWARN: Frame larger than buffer, dropping");
      reset_frame_buffer();
      return;
    }

    if (frame_buffer_pos < frame_length) {
      return;
    }

    if (frame_buffer[frame_length - 2] != FRAME_END[0] ||
        frame_buffer[frame_length - 1] != FRAME_END[1]) {
      Serial.println("\nWARN: Frame end mismatch, searching next start");
      shift_frame_buffer(1);
      continue;
    }

    Serial.write(frame_buffer, frame_length);
    Serial.flush();
    shift_frame_buffer(frame_length);
  }
}

void setup() {
  reset_frame_buffer();
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
      if (frame_buffer_pos + bytes_read > USB_BUFFER_SIZE) {
        size_t overflow = frame_buffer_pos + bytes_read - USB_BUFFER_SIZE;
        Serial.println("\nWARN: Local frame buffer overflow, discarding oldest data");
        shift_frame_buffer(overflow + 1);
      }

      memcpy(frame_buffer + frame_buffer_pos, lora_buffer, bytes_read);
      frame_buffer_pos += bytes_read;
      forward_complete_frames();
    }
  }

  // 短暫延遲避免過度佔用 CPU
  delayMicroseconds(500);
}
