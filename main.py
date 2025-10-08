import cv2
import serial
from typing import Optional

from h264_codec import EncodedChunk, H264Decoder
from protocol import BAUD_RATE, FrameProtocol, FrameStats, auto_detect_serial_port

MAX_PAYLOAD_SIZE = 1920 * 1080  # 允許的最大影像資料大小（反填充後）
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


def main() -> None:
    """主函式，用於從序列埠接收、解碼並顯示影像。"""
    ser = open_serial_connection()
    if ser is None:
        return

    protocol = FrameProtocol(max_payload_size=MAX_PAYLOAD_SIZE, use_chunk_ack=True)
    decoder = H264Decoder()
    pending_stats: list[FrameStats] = []
    pending_fragments = 0

    print("已連接序列埠，開始等待接收影像...")
    print("按下 'q' 鍵或 Ctrl+C 停止程式。")

    last_error_reported: Optional[str] = None

    try:
        while True:
            framed = protocol.receive_frame(ser, block=True)
            if framed is None:
                error = protocol.last_error
                if error and error != last_error_reported:
                    print(f"接收失敗: {error}")
                    last_error_reported = error
                continue

            pending_stats.append(framed.stats)
            pending_fragments += 1

            try:
                chunk = EncodedChunk.from_payload(framed.payload)
            except ValueError as exc:
                print(f"接收失敗: {exc}")
                pending_stats.clear()
                pending_fragments = 0
                continue

            decoded_frames = list(decoder.decode(chunk))

            if not decoded_frames:
                continue

            total_payload = sum(stat.payload_size for stat in pending_stats)
            total_ascii = sum(stat.stuffed_size for stat in pending_stats)
            fragment_count = pending_fragments
            pending_stats.clear()
            pending_fragments = 0

            image = decoded_frames[-1]
            if image is None:
                print("錯誤: 無法解碼影像。資料可能已損毀。")
                continue

            print(
                f"成功接收並解碼一幀影像，封包大小: {total_payload} bytes, "
                f"ASCII 編碼大小: {total_ascii} bytes, 分片數: {fragment_count}"
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