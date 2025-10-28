"""Shared serial video frame protocol utilities using ASCII framing."""
from __future__ import annotations

import base64
import glob
import platform
import time
import zlib
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, Sequence, runtime_checkable

import serial  # type: ignore

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
BAUD_RATE = 115_200
FRAME_PREFIX = "FRAME"
FIELD_SEPARATOR = " "
LINE_TERMINATOR = "\n"
DEFAULT_CHUNK_SIZE = 220
DEFAULT_INTER_CHUNK_DELAY = 0  # seconds
DEFAULT_MAX_PAYLOAD_SIZE = 1920 * 1080  # 128 KB (raw payload)
ACK_MESSAGE = b"ACK\n"
DEFAULT_ACK_TIMEOUT = 2
DEFAULT_INITIAL_ACK_SKIP = 1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameStats:
    """Metadata describing a single transmitted frame."""

    payload_size: int
    stuffed_size: int  # Encoded ASCII size
    crc: int


@dataclass(frozen=True)
class Frame:
    """Container for a received frame."""

    payload: bytes
    stats: FrameStats


# ---------------------------------------------------------------------------
# Serial helpers
# ---------------------------------------------------------------------------
def list_serial_ports() -> Sequence[str]:
    """Return a best-effort list of candidate serial port names for this OS."""
    system = platform.system()
    if system == "Windows":
        return [f"COM{i}" for i in range(2, 257)]
    if system == "Linux":
        return [f"/dev/ttyUSB{i}" for i in range(8)] + [f"/dev/ttyACM{i}" for i in range(8)]
    if system == "Darwin":
        return tuple(glob.glob("/dev/tty.usbserial*")) + tuple(glob.glob("/dev/tty.usbmodem*"))
    return tuple()


def auto_detect_serial_port() -> Optional[str]:
    """Attempt to locate the first available serial port by probing the OS list."""
    for port in list_serial_ports():
        try:
            probe = serial.Serial(port)
        except (OSError, serial.SerialException):
            continue
        else:
            probe.close()
            return port
    return None


# ---------------------------------------------------------------------------
# Frame protocol implementation
# ---------------------------------------------------------------------------
@runtime_checkable
class SerialLike(Protocol):
    def write(self, data, /) -> int | None: ...

    def read(self, size: int = ...) -> bytes: ...

    def flush(self) -> None: ...


class FrameProtocol:
    """Utility responsible for framing, validation, and transport orchestration."""

    def __init__(
        self,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        inter_chunk_delay: float = DEFAULT_INTER_CHUNK_DELAY,
        max_payload_size: int = DEFAULT_MAX_PAYLOAD_SIZE,
        use_chunk_ack: bool = False,
        ack_timeout: float = DEFAULT_ACK_TIMEOUT,
        initial_skip_acks: int = DEFAULT_INITIAL_ACK_SKIP,
        lenient: bool = False,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必須為正整數")
        self.chunk_size = chunk_size
        self.inter_chunk_delay = max(inter_chunk_delay, 0.0)
        self.max_payload_size = max_payload_size
        # Upper bound for encoded ASCII length (base64 expands ~4/3)
        self.max_encoded_size = int(max_payload_size * 4 / 3) + 16
        self.use_chunk_ack = use_chunk_ack
        self.ack_timeout = max(0.0, ack_timeout)
        self._acks_to_skip = max(0, initial_skip_acks)
        self._ack_established = False
        self._last_error: Optional[str] = None
        self._pending_ascii: bytearray = bytearray()
        self._lenient = lenient

    # -------------------------- Encoding helpers --------------------------
    def build_frame(self, payload: bytes) -> tuple[bytes, FrameStats]:
        """Construct the ASCII framed byte stream and statistics for *payload*."""
        encoded = base64.b64encode(payload).decode("ascii")
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        header = FIELD_SEPARATOR.join(
            (FRAME_PREFIX, str(len(encoded)), f"{crc:08x}")
        )
        frame_str = header + FIELD_SEPARATOR + encoded + LINE_TERMINATOR
        frame_bytes = frame_str.encode("ascii")
        stats = FrameStats(payload_size=len(payload), stuffed_size=len(encoded), crc=crc)
        return frame_bytes, stats

    def iter_chunks(self, frame: bytes) -> Iterable[bytes]:
        """Split a frame into chunks suitable for streaming over the serial port."""
        for index in range(0, len(frame), self.chunk_size):
            yield frame[index:index + self.chunk_size]

    def send_frame(self, ser: SerialLike, payload: bytes) -> FrameStats:
        """Send *payload* through the provided serial connection."""
        frame_bytes, stats = self.build_frame(payload)
        for chunk in self.iter_chunks(frame_bytes):
            ser.write(chunk)
            if self.inter_chunk_delay:
                time.sleep(self.inter_chunk_delay)
            if not self.use_chunk_ack:
                continue
            if self._acks_to_skip > 0:
                self._acks_to_skip -= 1
                continue
            if self._wait_for_ack(ser):
                self._ack_established = True
                continue
            if not self._ack_established:
                # 尚未建立 ACK 聯繫，降級為無 ACK 模式避免阻塞
                self.use_chunk_ack = False
                self._last_error = "尚未收到 ACK，已降級為無 ACK 模式"
                continue
            raise TimeoutError("等待 ACK 時間逾時")
        ser.flush()
        return stats

    @property
    def last_error(self) -> Optional[str]:
        """Return the most recent receive error, if any."""
        return self._last_error

    # -------------------------- Decoding helpers --------------------------
    def receive_frame(self, ser: SerialLike, *, block: bool = True) -> Optional[Frame]:
        """Attempt to read and validate a single ASCII framed payload from *ser*."""
        line = self._read_line(ser, block=block)
        if line is None:
            self._last_error = None if not block else "未收到任何資料"
            return None

        try:
            text = line.decode("ascii").rstrip("\r\n")
        except UnicodeDecodeError:
            self._last_error = "資料非 ASCII 編碼"
            return None

        if not text:
            self._last_error = "收到空白訊息"
            return None

        parts = text.split(FIELD_SEPARATOR, 3)
        if len(parts) != 4:
            self._last_error = "欄位數不正確"
            return None

        prefix, length_str, crc_str, encoded = parts
        if prefix != FRAME_PREFIX:
            self._last_error = "開頭標記錯誤"
            return None

        try:
            encoded_length = int(length_str)
        except ValueError:
            self._last_error = "長度欄位不是整數"
            return None

        if encoded_length != len(encoded):
            if encoded_length < len(encoded):
                overflow = encoded[encoded_length:]
                encoded = encoded[:encoded_length]
                self._pending_ascii[:0] = overflow.encode("ascii")
            msg = f"長度不符，宣告 {encoded_length} 實際 {len(encoded)}"
            if not self._lenient:
                self._last_error = msg
                return None
            self._last_error = msg + " (lenient: 繼續嘗試解碼)"
            encoded_length = len(encoded)

        if encoded_length > self.max_encoded_size:
            self._last_error = "編碼資料超過上限"
            return None

        try:
            expected_crc = int(crc_str, 16)
        except ValueError:
            self._last_error = "CRC 欄位格式錯誤"
            return None

        try:
            payload = base64.b64decode(encoded.encode("ascii"), validate=True)
        except Exception:
            self._last_error = "Base64 解碼失敗"
            return None

        if not payload:
            self._last_error = "解碼後為空資料"
            return None

        if len(payload) > self.max_payload_size:
            self._last_error = "解碼資料超過大小限制"
            return None

        crc = zlib.crc32(payload) & 0xFFFFFFFF
        if crc != expected_crc:
            msg = "CRC 驗證失敗"
            if not self._lenient:
                self._last_error = msg
                return None
            self._last_error = msg + " (lenient: 忽略驗證)"

        stats = FrameStats(
            payload_size=len(payload),
            stuffed_size=len(encoded),
            crc=crc,
        )
        self._last_error = None
        return Frame(payload=payload, stats=stats)

    # -------------------------- Internal helpers --------------------------
    def _wait_for_ack(self, ser: SerialLike) -> bool:
        deadline = time.monotonic() + self.ack_timeout if self.ack_timeout > 0 else None
        buffer = bytearray()
        while True:
            if deadline is not None and time.monotonic() > deadline:
                return False
            chunk = ser.read(1)
            if chunk:
                buffer.extend(chunk)
                if len(buffer) > len(ACK_MESSAGE):
                    del buffer[:-len(ACK_MESSAGE)]
                if buffer.endswith(ACK_MESSAGE):
                    return True
                continue
            time.sleep(0.001)

    def _read_line(self, ser: SerialLike, *, block: bool) -> Optional[bytes]:
        bytes_since_ack = 0
        while True:
            newline_index = self._pending_ascii.find(b"\n")
            if newline_index != -1:
                line = bytes(self._pending_ascii[:newline_index + 1])
                del self._pending_ascii[:newline_index + 1]
                if self.use_chunk_ack and bytes_since_ack > 0:
                    ser.write(ACK_MESSAGE)
                    ser.flush()
                    bytes_since_ack = 0
                return line
            chunk = ser.read(1)
            if chunk:
                self._pending_ascii.extend(chunk)
                if self.use_chunk_ack:
                    bytes_since_ack += 1
                    if bytes_since_ack >= self.chunk_size:
                        ser.write(ACK_MESSAGE)
                        ser.flush()
                        bytes_since_ack = 0
                if len(self._pending_ascii) > self.max_encoded_size + 64:
                    self._last_error = "資料行過長"
                    self._pending_ascii.clear()
                    return None
                continue
            if not block:
                return None
            time.sleep(0.001)