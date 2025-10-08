import cv2
import numpy as np
import serial
from typing import Optional

from protocol import BAUD_RATE, FrameProtocol, auto_detect_serial_port

MAX_PAYLOAD_SIZE = 128 * 1024  # 允許的最大影像資料大小（反填充後）
WINDOW_TITLE = 'Received CCTV (Press q to quit)'


def open_serial_connection() -> Optional[serial.Serial]:
    """偵測並開啟第一個可用的序列埠。"""
    port = auto_detect_serial_port()
    if not port:
        print("錯誤: 找不到任何可用的序列埠。")
        print("請確認您的微控制器 (Arduino/ESP32) 已連接。")
        return None

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.2)
    except serial.SerialException as exc:
        print(f"錯誤: 無法打開序列埠 {port}。")
        print(f"詳細資訊: {exc}")
        return None

    print(f"成功打開序列埠: {port} @ {BAUD_RATE} bps")
    return ser


def decode_frame(payload: bytes) -> Optional[np.ndarray]:
    """將 JPEG 位元串解碼為灰階影像。"""
    if not payload:
        return None
    np_arr = np.frombuffer(payload, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)


def main() -> None:
    """主函式，用於從序列埠接收、解碼並顯示影像。"""
    ser = open_serial_connection()
    if ser is None:
        return

    protocol = FrameProtocol(max_payload_size=MAX_PAYLOAD_SIZE)

    print("已連接序列埠，開始等待接收影像...")
    print("按下 'q' 鍵或 Ctrl+C 停止程式。")

    last_error_reported: Optional[str] = None

    try:
        while True:
            frame = protocol.receive_frame(ser, block=True)
            if frame is None:
                error = protocol.last_error
                if error and error != last_error_reported:
                    print(f"接收失敗: {error}")
                    last_error_reported = error
                continue

            if frame.stats.payload_size == 0:
                print("警告: 收到空白幀，忽略。")
                continue

            image = decode_frame(frame.payload)
            if image is None:
                print("錯誤: 無法解碼影像。資料可能已損毀。")
                continue

            print(
                f"成功接收並解碼一幀影像，反填充大小: {frame.stats.payload_size} bytes, "
                f"填充後大小: {frame.stats.stuffed_size} bytes, CRC: {frame.stats.crc:08x}"
            )

            last_error_reported = None

            cv2.imshow(WINDOW_TITLE, image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n程式被使用者中斷。")
    finally:
        print("正在關閉程式並釋放資源...")
        cv2.destroyAllWindows()
        ser.close()
        print("程式已關閉。")


if __name__ == '__main__':
    main()