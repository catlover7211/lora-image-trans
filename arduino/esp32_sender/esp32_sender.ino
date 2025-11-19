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

// 串流管線參數
#define PIPE_BUFFER_SIZE 256
#define LORA_WRITE_GUARD_US 40
#define FLOW_REPORT_INTERVAL_MS 250
#define FRAME_GUARD_SIZE 65550  // MAX_FRAME_SIZE(65535) + marker/CRC margin

// 使用 ESP32 的第二個串口連接 LoRa 模組
HardwareSerial LoRaSerial(2);

// 串流緩衝
uint8_t pipe_buffer[PIPE_BUFFER_SIZE];
size_t pipe_fill = 0;

// 狀態機
bool in_frame = false;
uint8_t start_match = 0;
uint8_t end_history[2] = {0, 0};
uint32_t frame_bytes = 0;

// Statistics
unsigned long frames_sent = 0;
unsigned long frame_drops = 0;
unsigned long last_flow_report = 0;
unsigned long last_stats_time = 0;

// 幀標記
const uint8_t FRAME_START[] = {0xAA, 0x55};
const uint8_t FRAME_END[] = {0x55, 0xAA};

void reset_frame_state(bool count_drop) {
  if (count_drop) {
    frame_drops++;
  }
  in_frame = false;
  start_match = 0;
  frame_bytes = 0;
  end_history[0] = 0;
  end_history[1] = 0;
  pipe_fill = 0;
}

void flush_pipe() {
  if (pipe_fill == 0) {
    return;
  }
  while (LoRaSerial.availableForWrite() < pipe_fill) {
    delayMicroseconds(LORA_WRITE_GUARD_US);
  }
  LoRaSerial.write(pipe_buffer, pipe_fill);
  pipe_fill = 0;
}

inline void emit_byte(uint8_t b) {
  pipe_buffer[pipe_fill++] = b;
  if (pipe_fill == PIPE_BUFFER_SIZE) {
    flush_pipe();
  }
}

void report_flow_control() {
  unsigned long now = millis();
  if (now - last_flow_report < FLOW_REPORT_INTERVAL_MS) {
    return;
  }
  last_flow_report = now;
  const size_t backlog = Serial.available();
  const size_t lora_free = LoRaSerial.availableForWrite();
  const size_t printable = 48;  // 約略輸出長度
  while (Serial.availableForWrite() < printable) {
    delay(1);
  }
  Serial.print(F("[FC] backlog="));
  Serial.print(backlog);
  Serial.print(F(",loraFree="));
  Serial.print(lora_free);
  Serial.print(F(",frames="));
  Serial.print(frames_sent);
  Serial.print(F(",drops="));
  Serial.println(frame_drops);
}

void setup() {
  // 初始化 USB Serial (連接 Raspberry Pi)
  Serial.setTxBufferSize(1024);
  Serial.setRxBufferSize(4096);
  Serial.begin(115200);
  
  // 初始化 LoRa Serial (連接 LoRa 模組)
  LoRaSerial.setTxBufferSize(4096);
  LoRaSerial.setRxBufferSize(1024);
  LoRaSerial.begin(115200, SERIAL_8N1, RXD2, TXD2);

  // 設定合理的超時
  Serial.setTimeout(0);
  LoRaSerial.setTimeout(0);

  // 等待序列埠穩定
  delay(1000);

  Serial.println("ESP32 LoRa Sender Started (streaming mode)");
  Serial.println("Frame reconstruction handled upstream");
}

void loop() {
  while (Serial.available() > 0) {
    uint8_t incoming_byte = Serial.read();

    if (!in_frame) {
      if (start_match == 0) {
        start_match = (incoming_byte == FRAME_START[0]) ? 1 : 0;
      } else { // start_match == 1
        if (incoming_byte == FRAME_START[1]) {
          // 正式進入幀狀態，寫出 START 標記
          in_frame = true;
          frame_bytes = 2;
          emit_byte(FRAME_START[0]);
          emit_byte(FRAME_START[1]);
          end_history[0] = FRAME_START[0];
          end_history[1] = FRAME_START[1];
        }
        start_match = (incoming_byte == FRAME_START[0]) ? 1 : 0;
      }
      continue;
    }

    emit_byte(incoming_byte);
    frame_bytes++;
    end_history[0] = end_history[1];
    end_history[1] = incoming_byte;

    if (frame_bytes > FRAME_GUARD_SIZE) {
      Serial.println("[WARN] Frame dropped: exceeded guard size");
      reset_frame_state(true);
      continue;
    }

    if (end_history[0] == FRAME_END[0] && end_history[1] == FRAME_END[1]) {
      flush_pipe();
      frames_sent++;
      reset_frame_state(false);
    }
  }

  report_flow_control();

  unsigned long current_time = millis();
  if (current_time - last_stats_time >= 60000) {
    Serial.print("\n--- Stats: Sent=");
    Serial.print(frames_sent);
    Serial.print(", Drops=");
    Serial.print(frame_drops);
    Serial.println(" ---");
    last_stats_time = current_time;
  }

  if (!in_frame) {
    delayMicroseconds(200);
  } else {
    yield();
  }
}
