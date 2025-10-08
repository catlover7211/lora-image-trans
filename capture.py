import cv2
import serial
import time
from typing import Any, Optional

from protocol import BAUD_RATE, FrameProtocol, auto_detect_serial_port

# 影像設定
FRAME_WIDTH = 80  # 寬度 (像素)
FRAME_HEIGHT = 60  # 高度 (像素)
JPEG_QUALITY = 10  # JPEG 壓縮品質 (0-100，數值越低壓縮越高)
TRANSMIT_INTERVAL = 0.1  # 傳送間隔（秒）
CAMERA_INDEX = 0  # 預設使用的攝影機索引


def open_serial_connection() -> Optional[serial.Serial]:
    """偵測並開啟第一個可用的序列埠。"""
    port = auto_detect_serial_port()
    if not port:
        print("錯誤: 找不到任何可用的序列埠。")
        print("請確認您的微控制器 (Arduino/ESP32) 已連接。")
        return None

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
    except serial.SerialException as exc:
        print(f"錯誤: 無法打開序列埠 {port}。")
        print(f"詳細資訊: {exc}")
        return None

    print(f"成功打開序列埠: {port} @ {BAUD_RATE} bps")
    return ser


def prepare_payload(frame) -> tuple[bytes, Any]:
    """將影像轉為灰階、縮放並輸出 JPEG 位元串。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (FRAME_WIDTH, FRAME_HEIGHT))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    ok, encoded = cv2.imencode('.jpg', resized, encode_param)
    if not ok:
        raise ValueError('無法將影像編碼為 JPEG。')
    return encoded.tobytes(), resized


def main() -> None:
    """主函式，用於擷取、壓縮並透過序列埠傳送影像。"""
    ser = open_serial_connection()
    if ser is None:
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("錯誤: 無法打開攝影機。請確認攝影機已連接且驅動正常。")
        ser.close()
        return

    protocol = FrameProtocol()

    print("攝影機已啟動，開始擷取與傳送影像...")
    print("按下 'q' 鍵或 Ctrl+C 停止程式。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("錯誤: 無法讀取影像幀。")
                break

            try:
                payload, preview_frame = prepare_payload(frame)
            except ValueError as exc:
                print(f"影像處理失敗: {exc}")
                continue

            try:
                stats = protocol.send_frame(ser, payload)
                print(
                    f"成功傳送一幀影像，原始大小: {stats.payload_size} bytes, "
                    f"填充後大小: {stats.stuffed_size} bytes, CRC: {stats.crc:08x}"
                )
            except serial.SerialException as exc:
                print(f"序列埠寫入錯誤: {exc}")
                break

            cv2.imshow('CCTV Preview (Press q to quit)', preview_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(TRANSMIT_INTERVAL)

    except KeyboardInterrupt:
        print("\n程式被使用者中斷。")
    finally:
        print("正在關閉程式並釋放資源...")
        cap.release()
        cv2.destroyAllWindows()
        ser.close()
        print("程式已關閉。")


if __name__ == '__main__':
    main()