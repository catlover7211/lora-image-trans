#include <HardwareSerial.h>

// ESP32 UART2 - 連接 ATK-LORA-01 模組
constexpr int RXD2 = 16;
constexpr int TXD2 = 17;

HardwareSerial LoRaSerial(2);

// 用於緩存從 USB Serial 讀取的數據
String usb_buffer = "";

void setup() {
  Serial.begin(115200);
  LoRaSerial.begin(115200, SERIAL_8N1, RXD2, TXD2);

  // 設定一個合理的超時，避免 readStringUntil 無限等待
  Serial.setTimeout(10); 
  LoRaSerial.setTimeout(10);

  usb_buffer.reserve(4096); // 為緩衝區預留足夠的空間，減少記憶體重分配

  Serial.println("ESP32 LoRa Transceiver Ready (ATK-LORA-01)");
  Serial.println("Note: ATK-LORA-01 should be pre-configured in transparent transmission mode");
}

void loop() {
  // 1. 從 USB Serial 讀取數據並轉發到 LoRa
  // Python 端的 protocol.py 已經在 build_frame() 中加入了 SYNC_MARKER
  // 所以這裡不需要再次添加，直接轉發即可
  if (Serial.available() > 0) {
    char incoming_char = Serial.read();
    usb_buffer += incoming_char;

    // 如果讀到了換行符，代表一個完整的封包已接收
    if (incoming_char == '\n') {
      // 檢查緩衝區是否以同步標記開頭（0xDE 0xAD 0xBE 0xEF = "\xDE\xAD\xBE\xEF"）
      // 注意：SYNC_MARKER 是二進制數據，已經由 Python 端的 build_frame() 添加
      // 我們只需要將完整的封包轉發給 LoRa 模組
      
      // 直接轉發整個封包（已包含 SYNC_MARKER）
      LoRaSerial.print(usb_buffer);
      
      // 除錯日誌（可選）
      // Serial.print("Forwarded packet, size: ");
      // Serial.println(usb_buffer.length());
      
      // 清空緩衝區，準備接收下一個封包
      usb_buffer = "";
    }
  }

  // 2. 將 LoRa 的資料即時轉回 USB Serial
  while (LoRaSerial.available() > 0) {
    int ch = LoRaSerial.read();
    if (ch >= 0) {
      Serial.write(static_cast<uint8_t>(ch));
    }
  }

  // 短暫延遲，讓出 CPU
  delay(1);
}