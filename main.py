import argparse
import queue
import threading
import time
from typing import Optional

import cv2
import serial

from h264_codec import EncodedChunk, H264Decoder
from image_settings import DEFAULT_IMAGE_SETTINGS
from protocol import BAUD_RATE, FrameProtocol, FrameStats, auto_detect_serial_port

MAX_PAYLOAD_SIZE = 1920 * 1080  # 允許的最大影像資料大小（反填充後）
WINDOW_TITLE = 'Received CCTV (Press q to quit)'
DEFAULT_RX_BUFFER = DEFAULT_IMAGE_SETTINGS.rx_buffer_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Receive CCTV frames over serial and display them.')
    parser.add_argument('--no-ack', action='store_true', help='停用 chunk 級 ACK。與傳送端設定一致才有作用。')
    parser.add_argument('--rx-buffer', type=int, default=DEFAULT_RX_BUFFER, help='接收端佇列容量 (幀數，預設: %(default)s)')
    return parser.parse_args()


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
    args = parse_args()
    ser = open_serial_connection()
    if ser is None:
        return

    protocol = FrameProtocol(
        max_payload_size=MAX_PAYLOAD_SIZE,
        use_chunk_ack=not args.no_ack,
    )
    decoder = H264Decoder()

    rx_buffer_size = max(1, args.rx_buffer)
    frame_queue: "queue.Queue[tuple[EncodedChunk, FrameStats]]" = queue.Queue(maxsize=rx_buffer_size)
    stop_event = threading.Event()
    pending_stats: list[FrameStats] = []
    pending_fragments = 0

    print("已連接序列埠，開始等待接收影像...")
    print("按下 'q' 鍵或 Ctrl+C 停止程式。")

    current_codec: Optional[str] = None
    dropped_chunks = 0
    last_error_reported: Optional[str] = None

    def receiver_worker() -> None:
        nonlocal last_error_reported, dropped_chunks
        while not stop_event.is_set():
            try:
                framed = protocol.receive_frame(ser, block=False)
            except serial.SerialException as exc:
                if not stop_event.is_set():
                    print(f"接收錯誤: {exc}")
                time.sleep(0.1)
                continue

            if framed is None:
                error = protocol.last_error
                if error and error != last_error_reported:
                    print(f"接收失敗: {error}")
                    last_error_reported = error
                time.sleep(0.005)
                continue

            last_error_reported = None
            try:
                chunk = EncodedChunk.from_payload(framed.payload)
            except ValueError as exc:
                print(f"接收失敗: {exc}")
                continue

            while not stop_event.is_set():
                try:
                    frame_queue.put((chunk, framed.stats), timeout=0.1)
                    break
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                        frame_queue.task_done()
                    except queue.Empty:
                        pass
                    try:
                        frame_queue.put_nowait((chunk, framed.stats))
                    except queue.Full:
                        dropped_chunks += 1
                        print(f"警告: 接收緩衝區已滿，無法加入最新片段 (累積捨棄: {dropped_chunks})")
                        break
                    dropped_chunks += 1
                    print(f"警告: 接收緩衝區已滿，已捨棄最舊片段以插入新片段 (累積捨棄: {dropped_chunks})")
                    break

    receiver_thread = threading.Thread(target=receiver_worker, name="LoRaReceiver", daemon=True)
    receiver_thread.start()

    try:
        while not stop_event.is_set():
            try:
                chunk, stats = frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            decoded_frames: list = []
            try:
                pending_stats.append(stats)
                pending_fragments += 1

                if chunk.is_config and chunk.codec != current_codec:
                    current_codec = chunk.codec
                    print(f"收到編碼器設定: {current_codec.upper()} (extradata {len(chunk.data)} bytes)")

                decoded_frames = list(decoder.decode(chunk))
            except RuntimeError as exc:
                print(f"解碼失敗: {exc}")
                pending_stats.clear()
                pending_fragments = 0
                continue
            finally:
                frame_queue.task_done()

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
                f"封包大小: {total_payload} bytes, "
                f"ASCII 編碼大小: {total_ascii} bytes, 分片數: {fragment_count}"
            )

            cv2.imshow(WINDOW_TITLE, image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    except KeyboardInterrupt:
        print("\n程式被使用者中斷。")
    finally:
        print("正在關閉程式並釋放資源...")
        stop_event.set()
        receiver_thread.join(timeout=2)
        cv2.destroyAllWindows()
        ser.close()
        print("程式已關閉。")


if __name__ == '__main__':
    main()