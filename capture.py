import argparse
import time
from dataclasses import dataclass
from typing import Any, Optional, cast

import cv2
import numpy as np
import serial

from h264_codec import EncodedChunk, H264Encoder, VideoCodec
from image_settings import DEFAULT_IMAGE_SETTINGS, color_conversion
from protocol import BAUD_RATE, FrameProtocol, FrameStats, auto_detect_serial_port


IMAGE_DEFAULTS = DEFAULT_IMAGE_SETTINGS
DEFAULT_CAMERA_INDEX = 0
DEFAULT_SERIAL_TIMEOUT = 1.0
WINDOW_TITLE = 'CCTV Preview (Press q to quit)'
MIN_USEFUL_PAYLOAD = 64  # bytes; skip extremely tiny predict-only frames to避免灰畫面


@dataclass(frozen=True)
class EncoderConfig:
    width: int = IMAGE_DEFAULTS.width
    height: int = IMAGE_DEFAULTS.height
    bitrate: int = IMAGE_DEFAULTS.target_bitrate
    keyframe_interval: int = IMAGE_DEFAULTS.keyframe_interval
    color_conversion: Optional[int] = color_conversion(IMAGE_DEFAULTS.color_mode)
    codec: VideoCodec = IMAGE_DEFAULTS.codec

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError('影像尺寸必須為正整數。')
        if self.bitrate <= 0:
            raise ValueError('位元率必須為正整數。')
        if self.keyframe_interval <= 0:
            raise ValueError('關鍵幀間隔必須為正整數。')


@dataclass(frozen=True)
class TransmissionConfig:
    interval: float = IMAGE_DEFAULTS.transmit_interval

    def __post_init__(self) -> None:
        if self.interval < 0:
            raise ValueError('傳輸間隔不得為負數。')


class FrameEncoder:
    """負責影像預處理與視訊編碼。"""

    def __init__(self, config: EncoderConfig, *, fps: float) -> None:
        self.config = config
        self.encoder = H264Encoder(
            width=config.width,
            height=config.height,
            fps=fps,
            bitrate=config.bitrate,
            keyframe_interval=max(1, config.keyframe_interval),
            codec=config.codec,
        )

    def force_keyframe(self) -> None:
        self.encoder.force_keyframe()

    def encode(self, frame: Any) -> tuple[list[EncodedChunk], Any, int]:
        processed = frame
        if self.config.color_conversion is not None:
            processed = cv2.cvtColor(processed, self.config.color_conversion)
        resized = cv2.resize(processed, (self.config.width, self.config.height), interpolation=cv2.INTER_AREA)

        if resized.ndim == 2:
            bgr_frame = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        else:
            bgr_frame = resized

        chunks = self.encoder.encode(bgr_frame)
        raw_size = bgr_frame.size * bgr_frame.itemsize
        return chunks, resized, raw_size


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
    parser.add_argument('--width', type=int, default=IMAGE_DEFAULTS.width, help='輸出影像寬度 (預設: %(default)s)')
    parser.add_argument('--height', type=int, default=IMAGE_DEFAULTS.height, help='輸出影像高度 (預設: %(default)s)')
    parser.add_argument('--color-mode', choices=('gray', 'bgr'), default=IMAGE_DEFAULTS.color_mode, help='影像編碼顏色模式 (預設: %(default)s)')
    parser.add_argument('--codec', choices=('h264', 'h265'), default=IMAGE_DEFAULTS.codec, help='選擇影像編碼器 (預設: %(default)s)')
    parser.add_argument('--interval', type=float, default=IMAGE_DEFAULTS.transmit_interval, help='幀與幀之間的最小秒數 (預設: %(default)s)')
    parser.add_argument('--camera-fps', type=float, default=None, help='嘗試設定攝影機的擷取 FPS (<=0 表示維持裝置預設)')
    parser.add_argument('--serial-timeout', type=float, default=DEFAULT_SERIAL_TIMEOUT, help='序列埠 timeout 秒數 (預設: %(default)s)')
    parser.add_argument('--bitrate', type=int, default=IMAGE_DEFAULTS.target_bitrate, help='H.264 目標位元率 (bps，預設: %(default)s)')
    parser.add_argument('--keyframe-interval', type=int, default=IMAGE_DEFAULTS.keyframe_interval, help='關鍵幀間隔 (預設: %(default)s)')
    parser.add_argument('--motion-threshold', type=float, default=IMAGE_DEFAULTS.motion_threshold, help='平均灰階變化門檻，低於此值則跳過傳送 (負值表示停用，預設: %(default)s)')
    parser.add_argument('--max-idle', type=float, default=IMAGE_DEFAULTS.max_idle_seconds, help='允許最長未送出秒數，超過則強制送一幀 (<=0 表示不限，預設: %(default)s)')
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

    capture_fps = args.camera_fps if args.camera_fps and args.camera_fps > 0 else None
    fps_hint = 30.0
    if args.interval > 0:
        fps_hint = max(fps_hint, 1.0 / args.interval)
    if capture_fps:
        fps_hint = max(fps_hint, capture_fps)

    try:
        encoder = FrameEncoder(
            EncoderConfig(
                width=args.width,
                height=args.height,
                bitrate=args.bitrate,
                keyframe_interval=args.keyframe_interval,
                color_conversion=color_conversion(args.color_mode),
                codec=cast(VideoCodec, args.codec),
            ),
            fps=fps_hint,
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
    if capture_fps:
        if not cap.set(cv2.CAP_PROP_FPS, capture_fps):
            print(f'警告: 無法將攝影機 FPS 設為 {capture_fps}，將採用裝置預設值。')
        else:
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f'攝影機 FPS 目標: {capture_fps}，裝置回報: {actual_fps:.2f}')
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
    codec_label = args.codec.upper()
    loop_start = time.monotonic()
    motion_threshold = args.motion_threshold
    max_idle = max(args.max_idle, 0.0)
    previous_gray: Optional[np.ndarray] = None
    skipped_frames = 0
    last_sent_time = time.monotonic()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('錯誤: 無法讀取影像幀。')
                break

            try:
                chunks, preview_frame, raw_size = encoder.encode(frame)
            except ValueError as exc:
                print(f'影像處理失敗: {exc}')
                continue

            if not chunks:
                continue

            contains_config = any(chunk.is_config for chunk in chunks)
            non_config_bytes = sum(len(chunk.data) for chunk in chunks if not chunk.is_config)
            if (
                non_config_bytes > 0
                and non_config_bytes <= MIN_USEFUL_PAYLOAD
                and not any(chunk.is_keyframe for chunk in chunks)
            ):
                # 極小的預測幀通常只包含雜訊，直接跳過維持上一張畫面
                skipped_frames += 1
                if isinstance(current_gray, np.ndarray):
                    previous_gray = current_gray
                continue

            current_gray = preview_frame
            if isinstance(current_gray, np.ndarray) and current_gray.ndim == 3:
                current_gray = cv2.cvtColor(current_gray, cv2.COLOR_BGR2GRAY)

            if isinstance(current_gray, np.ndarray):
                current_gray = current_gray.astype(np.uint8)

            now = time.monotonic()
            if (
                motion_threshold >= 0
                and isinstance(current_gray, np.ndarray)
                and previous_gray is not None
                and not contains_config
            ):
                diff_value = float(cv2.absdiff(previous_gray, current_gray).mean())
                idle_duration = now - last_sent_time
                if diff_value < motion_threshold and (max_idle <= 0 or idle_duration < max_idle):
                    skipped_frames += 1
                    previous_gray = current_gray
                    continue
                if diff_value < motion_threshold and max_idle > 0 and idle_duration >= max_idle:
                    encoder.force_keyframe()

            if isinstance(current_gray, np.ndarray):
                previous_gray = current_gray

            encoded_size = 0
            framed_size = 0
            ascii_size = 0
            chunks_sent = 0
            keyframe_sent = False

            try:
                for chunk in chunks:
                    payload = chunk.to_payload()
                    encoded_size += len(chunk.data)
                    framed_size += len(payload)
                    stats = transmitter.send(payload)
                    ascii_size += stats.stuffed_size
                    chunks_sent += 1
                    last_sent_time = time.monotonic()
                    keyframe_sent = keyframe_sent or chunk.is_keyframe
            except (serial.SerialException, TimeoutError) as exc:
                print(f'序列埠寫入錯誤: {exc}')
                break

            frame_counter += 1
            elapsed = time.monotonic() - loop_start
            fps = frame_counter / elapsed if elapsed > 0 else 0.0
            print(
                f'成功傳送一幀影像，原始估計大小: {raw_size} bytes, '
                f'{codec_label} 編碼大小: {encoded_size} bytes, 封裝後大小: {framed_size} bytes, '
                f'ASCII 編碼大小: {ascii_size} bytes, 分片數: {chunks_sent}, '
                f'是否關鍵幀: {"是" if keyframe_sent else "否"}, '
                f'跳過累積: {skipped_frames}, 平均頻率: {fps:.2f} fps'
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