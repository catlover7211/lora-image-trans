"""Video frame receiver for serial communication.

This module receives, decodes, and displays video frames transmitted over serial.
It supports multiple codecs (H.264, H.265, AV1, JPEG, Wavelet, Contour, YOLO, CS)
and includes features like:
- Multi-threaded reception for smooth playback
- Error reporting with cool-down to prevent console flooding
- Queue-based buffering to handle transmission irregularities
- CRC validation for data integrity
"""
import argparse
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np
import serial

from h264_codec import EncodedChunk, H264Decoder
from image_settings import DEFAULT_IMAGE_SETTINGS
from protocol import BAUD_RATE, FrameProtocol, FrameStats, auto_detect_serial_port

MAX_PAYLOAD_SIZE = 1920 * 1080
"""Maximum allowed image data size (after unstuffing)."""

WINDOW_TITLE = 'Received CCTV (Press q to quit)'
"""Window title for the display window."""

DEFAULT_RX_BUFFER = DEFAULT_IMAGE_SETTINGS.rx_buffer_size
"""Default receive buffer size in frames."""

ERROR_COOLDOWN_SECONDS = 5.0
"""Minimum seconds between reporting identical errors."""


def create_receiver_worker(
    protocol: FrameProtocol,
    ser: serial.Serial,
    frame_queue: "queue.Queue[tuple[EncodedChunk, FrameStats]]",
    stop_event: threading.Event,
) -> tuple[callable, dict]:
    """Create a receiver worker function with shared state.
    
    This function creates a worker that continuously receives frames from the serial
    port and places them in a queue for processing. It implements error cool-down
    to prevent console flooding with repetitive error messages.
    
    Args:
        protocol: The FrameProtocol instance for decoding frames.
        ser: The serial port to read from.
        frame_queue: Queue for passing received frames to the main thread.
        stop_event: Event to signal the worker to stop.
    
    Returns:
        A tuple of (worker_function, state_dict) where state_dict contains
        worker state like error tracking and dropped chunk counts.
    """
    state = {
        'last_error_reported': None,
        'last_error_time': 0.0,
        'error_cooldown': ERROR_COOLDOWN_SECONDS,
        'dropped_chunks': 0
    }
    
    def receiver_worker() -> None:
        """Worker thread that receives frames from serial port."""
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
                if error:
                    current_time = time.time()
                    # Only report if it's a different error or enough time has passed
                    if (error != state['last_error_reported'] or 
                        current_time - state['last_error_time'] >= state['error_cooldown']):
                        print(f"接收失敗: {error}")
                        state['last_error_reported'] = error
                        state['last_error_time'] = current_time
                time.sleep(0.005)
                continue

            state['last_error_reported'] = None
            
            try:
                chunk = EncodedChunk.from_payload(framed.payload)
            except ValueError as exc:
                print(f"接收失敗: {exc}")
                continue

            _enqueue_chunk(frame_queue, chunk, framed.stats, stop_event, state)
    
    return receiver_worker, state


def _enqueue_chunk(
    frame_queue: "queue.Queue[tuple[EncodedChunk, FrameStats]]",
    chunk: EncodedChunk,
    stats: FrameStats,
    stop_event: threading.Event,
    state: dict
) -> None:
    """Enqueue a chunk, dropping old frames if queue is full."""
    while not stop_event.is_set():
        try:
            frame_queue.put((chunk, stats), timeout=0.1)
            break
        except queue.Full:
            # Try to drop oldest frame and insert new one
            try:
                frame_queue.get_nowait()
                frame_queue.task_done()
            except queue.Empty:
                pass
            
            try:
                frame_queue.put_nowait((chunk, stats))
            except queue.Full:
                state['dropped_chunks'] += 1
                print(f"警告: 接收緩衝區已滿，無法加入最新片段 (累積捨棄: {state['dropped_chunks']})")
                break
            
            state['dropped_chunks'] += 1
            print(f"警告: 接收緩衝區已滿，已捨棄最舊片段以插入新片段 (累積捨棄: {state['dropped_chunks']})")
            break


def process_frame(
    chunk: EncodedChunk,
    stats: FrameStats,
    decoder: H264Decoder,
    current_codec: Optional[str],
    pending_stats: list[FrameStats],
    pending_fragments: int
) -> tuple[list, Optional[str], int]:
    """Process a single frame chunk and return decoded frames."""
    pending_stats.append(stats)
    pending_fragments += 1

    if chunk.is_config and chunk.codec != current_codec:
        current_codec = chunk.codec
        print(f"收到編碼器設定: {current_codec.upper()} (extradata {len(chunk.data)} bytes)")

    try:
        decoded_frames = list(decoder.decode(chunk))
    except RuntimeError as exc:
        print(f"解碼失敗: {exc}")
        pending_stats.clear()
        pending_fragments = 0
        decoded_frames = []

    return decoded_frames, current_codec, pending_fragments


def display_frame(
    image: np.ndarray,
    pending_stats: list[FrameStats],
    pending_fragments: int,
    window_title: str
) -> tuple[list, int]:
    """Display a frame and print statistics."""
    total_payload = sum(stat.payload_size for stat in pending_stats)
    total_ascii = sum(stat.stuffed_size for stat in pending_stats)
    fragment_count = pending_fragments
    
    # Calculate compression ratio
    compression_ratio = (total_ascii / total_payload * 100) if total_payload > 0 else 0
    
    print(
        f"封包: {total_payload} bytes, "
        f"編碼後: {total_ascii} bytes "
        f"(膨脹率: {compression_ratio:.1f}%), "
        f"分片: {fragment_count}"
    )
    
    cv2.imshow(window_title, image)
    
    return [], 0  # Clear pending stats and fragments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Receive CCTV frames over serial and display them.')
    parser.add_argument('--rx-buffer', type=int, default=DEFAULT_RX_BUFFER, help='接收端佇列容量 (幀數，預設: %(default)s)')
    parser.add_argument('--lenient', action='store_true', help='寬鬆模式：忽略部分長度/CRC 驗證，盡量嘗試解碼。')
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
        lenient=args.lenient,
    )
    decoder = H264Decoder()

    _print_connection_info(args.lenient)

    # Setup queue and worker thread
    rx_buffer_size = max(1, args.rx_buffer)
    frame_queue: "queue.Queue[tuple[EncodedChunk, FrameStats]]" = queue.Queue(maxsize=rx_buffer_size)
    stop_event = threading.Event()
    
    receiver_worker, _ = create_receiver_worker(protocol, ser, frame_queue, stop_event)
    receiver_thread = threading.Thread(target=receiver_worker, name="LoRaReceiver", daemon=True)
    receiver_thread.start()

    # Main processing loop
    try:
        _main_processing_loop(frame_queue, decoder, stop_event)
    except KeyboardInterrupt:
        print("\n程式被使用者中斷。")
    finally:
        _cleanup(stop_event, receiver_thread, ser)


def _print_connection_info(lenient: bool) -> None:
    """Print connection and configuration information."""
    print(f"串流 ACK 模式: 停用 (與 ESP32 韌體相容)")
    print(f"接收容錯模式: {'寬鬆' if lenient else '嚴格'}")
    print("已連接序列埠，開始等待接收影像...")
    print("按下 'q' 鍵或 Ctrl+C 停止程式。")


def _main_processing_loop(
    frame_queue: "queue.Queue[tuple[EncodedChunk, FrameStats]]",
    decoder: H264Decoder,
    stop_event: threading.Event
) -> None:
    """Main loop for processing received frames."""
    current_codec: Optional[str] = None
    pending_stats: list[FrameStats] = []
    pending_fragments = 0

    while not stop_event.is_set():
        try:
            chunk, stats = frame_queue.get(timeout=0.2)
        except queue.Empty:
            # Keep window responsive even when no frames are available
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
            continue

        try:
            decoded_frames, current_codec, pending_fragments = process_frame(
                chunk, stats, decoder, current_codec, pending_stats, pending_fragments
            )
        finally:
            frame_queue.task_done()

        if not decoded_frames:
            # Keep window responsive even when decoding produces no frames
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
            continue

        image = decoded_frames[-1]
        if image is None:
            print("錯誤: 無法解碼影像。資料可能已損毀。")
            # Keep window responsive even when decoding fails
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
            continue

        pending_stats, pending_fragments = display_frame(
            image, pending_stats, pending_fragments, WINDOW_TITLE
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break


def _cleanup(
    stop_event: threading.Event,
    receiver_thread: threading.Thread,
    ser: serial.Serial
) -> None:
    """Clean up resources before exiting."""
    print("正在關閉程式並釋放資源...")
    stop_event.set()
    receiver_thread.join(timeout=2)
    cv2.destroyAllWindows()
    ser.close()
    print("程式已關閉。")


if __name__ == '__main__':
    main()