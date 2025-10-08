import argparse
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import serial

from protocol import BAUD_RATE, FrameProtocol, FrameStats, auto_detect_serial_port


DEFAULT_FRAME_WIDTH = 160
DEFAULT_FRAME_HEIGHT = 90
DEFAULT_JPEG_QUALITY = 5
DEFAULT_TRANSMIT_INTERVAL = 10.0
DEFAULT_CAMERA_INDEX = 0
DEFAULT_SERIAL_TIMEOUT = 1.0
WINDOW_TITLE = 'CCTV Preview (Press q to quit)'


@dataclass(frozen=True)
class EncoderConfig:
    width: int = DEFAULT_FRAME_WIDTH
    height: int = DEFAULT_FRAME_HEIGHT
    quality: int = DEFAULT_JPEG_QUALITY
    color_conversion: Optional[int] = cv2.COLOR_BGR2GRAY

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError('影像尺寸必須為正整數。')
        if not (0 <= self.quality <= 100):
            raise ValueError('JPEG 壓縮品質必須介於 0 到 100 之間。')


@dataclass(frozen=True)
class TransmissionConfig:
    interval: float = DEFAULT_TRANSMIT_INTERVAL

    def __post_init__(self) -> None:
        if self.interval < 0:
            raise ValueError('傳輸間隔不得為負數。')


class FrameEncoder:
    """負責影像預處理與 JPEG 編碼。"""

    def __init__(self, config: EncoderConfig) -> None:
        self.config = config
        self._encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.quality]

    def encode(self, frame: Any) -> tuple[bytes, Any]:
        processed = frame
        if self.config.color_conversion is not None:
            processed = cv2.cvtColor(processed, self.config.color_conversion)
        resized = cv2.resize(processed, (self.config.width, self.config.height), interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode('.jpg', resized, self._encode_param)
        if not success:
            raise ValueError('無法將影像編碼為 JPEG。')
        return encoded.tobytes(), resized


class FrameTransmitter:
    """包裝 FrameProtocol，負責節流與統計回報。"""

    def __init__(self, serial_port: serial.Serial, *, protocol: FrameProtocol, config: TransmissionConfig) -> None:
        self.serial_port = serial_port
        self.protocol = protocol
        self.config = config
        self._last_sent: Optional[float] = None

    def send(self, payload: bytes) -> FrameStats:
        if self.config.interval > 0 and self._last_sent is not None:
            elapsed = time.monotonic() - self._last_sent
            remaining = self.config.interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
        stats = self.protocol.send_frame(self.serial_port, payload)
        self._last_sent = time.monotonic()
        return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Capture webcam frames and transmit them over serial.')
    parser.add_argument('--port', help='指定序列埠，預設會自動偵測。')
    parser.add_argument('--camera-index', type=int, default=DEFAULT_CAMERA_INDEX, help='攝影機索引 (預設: %(default)s)')
    parser.add_argument('--width', type=int, default=DEFAULT_FRAME_WIDTH, help='輸出影像寬度 (預設: %(default)s)')
    parser.add_argument('--height', type=int, default=DEFAULT_FRAME_HEIGHT, help='輸出影像高度 (預設: %(default)s)')
    parser.add_argument('--quality', type=int, default=DEFAULT_JPEG_QUALITY, help='JPEG 壓縮品質 0-100 (預設: %(default)s)')
    parser.add_argument('--color-mode', choices=('gray', 'bgr'), default='gray', help='影像編碼顏色模式 (預設: %(default)s)')
    parser.add_argument('--interval', type=float, default=DEFAULT_TRANSMIT_INTERVAL, help='幀與幀之間的最小秒數 (預設: %(default)s)')
    parser.add_argument('--serial-timeout', type=float, default=DEFAULT_SERIAL_TIMEOUT, help='序列埠 timeout 秒數 (預設: %(default)s)')
    return parser.parse_args()


def open_serial_connection(port: Optional[str], *, timeout: float) -> Optional[serial.Serial]:
    """開啟指定或自動偵測到的序列埠。"""
    target_port = port or auto_detect_serial_port()
    if not target_port:
        print('錯誤: 找不到任何可用的序列埠。')
        print('請確認您的微控制器 (Arduino/ESP32) 已連接。')
        return None

    try:
        ser = serial.Serial(target_port, BAUD_RATE, timeout=timeout)
    except serial.SerialException as exc:
        print(f'錯誤: 無法打開序列埠 {target_port}。')
        print(f'詳細資訊: {exc}')
        return None

    print(f'成功打開序列埠: {target_port} @ {BAUD_RATE} bps (timeout={timeout}s)')
    return ser


def main() -> None:
    """主流程：擷取影像、編碼並透過序列埠傳送。"""
    args = parse_args()

    try:
        encoder = FrameEncoder(
            EncoderConfig(
                width=args.width,
                height=args.height,
                quality=args.quality,
                color_conversion=None if args.color_mode == 'bgr' else cv2.COLOR_BGR2GRAY,
            )
        )
    except ValueError as exc:
        print(f'無效的編碼參數: {exc}')
        return

    try:
        transmitter_config = TransmissionConfig(interval=max(args.interval, 0.0))
    except ValueError as exc:
        print(f'無效的傳輸參數: {exc}')
        return

    ser = open_serial_connection(args.port, timeout=max(args.serial_timeout, 0.0))
    if ser is None:
        return

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print('錯誤: 無法打開攝影機。請確認攝影機已連接且驅動正常。')
        ser.close()
        return

    protocol = FrameProtocol(
        use_chunk_ack=True,
        ack_timeout=max(args.serial_timeout, 0.0),
        initial_skip_acks=1,
    )
    transmitter = FrameTransmitter(ser, protocol=protocol, config=transmitter_config)

    print('攝影機已啟動，開始擷取與傳送影像...')
    print("按下 'q' 鍵或 Ctrl+C 停止程式。")

    frame_counter = 0
    loop_start = time.monotonic()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('錯誤: 無法讀取影像幀。')
                break

            try:
                payload, preview_frame = encoder.encode(frame)
            except ValueError as exc:
                print(f'影像處理失敗: {exc}')
                continue

            try:
                stats = transmitter.send(payload)
            except (serial.SerialException, TimeoutError) as exc:
                print(f'序列埠寫入錯誤: {exc}')
                break

            frame_counter += 1
            elapsed = time.monotonic() - loop_start
            fps = frame_counter / elapsed if elapsed > 0 else 0.0
            print(
                f'成功傳送一幀影像，原始大小: {stats.payload_size} bytes, '
                f'ASCII 編碼大小: {stats.stuffed_size} bytes, CRC: {stats.crc:08x}, '
                f'平均頻率: {fps:.2f} fps'
            )

            #cv2.imshow(WINDOW_TITLE, preview_frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

    except KeyboardInterrupt:
        print('\n程式被使用者中斷。')
    finally:
        print('正在關閉程式並釋放資源...')
        cap.release()
        cv2.destroyAllWindows()
        ser.close()
        print('程式已關閉。')


if __name__ == '__main__':
    main()