#include <HardwareSerial.h>

// ESP32 UART2
constexpr int RXD2 = 16;
constexpr int TXD2 = 17;

HardwareSerial LoRaSerial(2);

// 與 Python 端 protocol.py 中定義的 SYNC_MARKER 保持一致
const uint8_t SYNC_MARKER[] = {0xDE, 0xAD, 0xBE, 0xEF};

// 用於緩存從 USB Serial 讀取的數據
String usb_buffer = "";

void setup() {
  Serial.begin(115200);
  LoRaSerial.begin(115200, SERIAL_8N1, RXD2, TXD2);

  // 設定一個合理的超時，避免 readStringUntil 無限等待
  Serial.setTimeout(10); 
  LoRaSerial.setTimeout(10);

  usb_buffer.reserve(4096); // 為緩衝區預留足夠的空間，減少記憶體重分配

  Serial.println("LoRa Transceiver with Sync Marker Ready...");
}

void loop() {
  // 1. 從 USB Serial 讀取數據，直到換行符，代表一個完整的 FRAME 封包
  if (Serial.available() > 0) {
    char incoming_char = Serial.read();
    usb_buffer += incoming_char;

    // 如果讀到了換行符，代表一個完整的封包已接收
    if (incoming_char == '\n') {
      // 檢查緩衝區是否以 "FRAME" 開頭，確保是我們想要的封包
      if (usb_buffer.startsWith("FRAME")) {
        // 首先，發送同步標記
        LoRaSerial.write(SYNC_MARKER, sizeof(SYNC_MARKER));
        
        // 然後，發送整個封包內容
        LoRaSerial.print(usb_buffer);
        
        // 可以在這裡印出日誌，方便除錯
        // Serial.print("Forwarded packet with sync marker, size: ");
        // Serial.println(usb_buffer.length());
      }
      
      // 清空緩衝區，準備接收下一個封包
      usb_buffer = "";
    }
  }

  // 2. 將 LoRa 的資料即時轉回 USB Serial (這部分邏輯不變)
  while (LoRaSerial.available() > 0) {
    int ch = LoRaSerial.read();
    if (ch >= 0) {
      Serial.write(static_cast<uint8_t>(ch));
    }
  }

  // 短暫延遲，讓出 CPU
  delay(1);
}