"""Shared serial video frame protocol utilities using ASCII framing.

This module provides a robust protocol for transmitting video frames over serial connections.
The protocol features:
- ASCII-based framing with base64 encoding for binary data
- CRC32 checksums for data integrity
- Sync markers for reliable frame boundary detection
- Configurable lenient mode for lossy connections
"""
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
"""Default baud rate for serial communication."""

FRAME_PREFIX = "FRAME"
"""ASCII prefix for frame identification."""

SYNC_MARKER = b"\xDE\xAD\xBE\xEF"
"""Binary sync marker for frame boundary detection."""

FIELD_SEPARATOR = " "
"""Separator between frame header fields."""

LINE_TERMINATOR = "\n"
"""Frame terminator character."""

DEFAULT_CHUNK_SIZE = 220
"""Default size for chunking frames during transmission."""

DEFAULT_INTER_CHUNK_DELAY = 0  # seconds
"""Default delay between chunks (0 = no delay)."""

DEFAULT_MAX_PAYLOAD_SIZE = 1920 * 1080  # 128 KB (raw payload)
"""Maximum allowed payload size to prevent memory issues."""

DEFAULT_SYNC_TIMEOUT = 2.0  # seconds
"""Default timeout for synchronization when blocking."""


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
    """Protocol handler for reliable frame transmission over serial connections.
    
    This class manages the encoding, transmission, and reception of frames using:
    - Base64 encoding for ASCII-safe transmission
    - CRC32 checksums for data integrity validation
    - Sync markers for reliable frame boundary detection
    - Optional lenient mode for handling lossy connections
    
    The protocol is designed to work with the ESP32 firmware without ACK support.
    """

    def __init__(
        self,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        inter_chunk_delay: float = DEFAULT_INTER_CHUNK_DELAY,
        max_payload_size: int = DEFAULT_MAX_PAYLOAD_SIZE,
        lenient: bool = False,
    ) -> None:
        """Initialize the frame protocol handler.
        
        Args:
            chunk_size: Size of chunks for splitting frames during transmission.
            inter_chunk_delay: Delay in seconds between chunk transmissions.
            max_payload_size: Maximum allowed payload size in bytes.
            lenient: If True, tolerate some length/CRC mismatches (for lossy connections).
        
        Raises:
            ValueError: If chunk_size is not positive.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size 必須為正整數")
        self.chunk_size = chunk_size
        self.inter_chunk_delay = max(inter_chunk_delay, 0.0)
        self.max_payload_size = max_payload_size
        # Upper bound for encoded ASCII length (base64 expands ~4/3)
        self.max_encoded_size = int(max_payload_size * 4 / 3) + 16
        self._last_error: Optional[str] = None
        self._pending_ascii: bytearray = bytearray()
        self._lenient = lenient
        self._synced = False

    # -------------------------- Encoding helpers --------------------------
    def build_frame(self, payload: bytes) -> tuple[bytes, FrameStats]:
        """Construct the ASCII framed byte stream and statistics for *payload*.
        
        Frame format (compatible with ESP32 transparent transmission):
        1. SYNC_MARKER (4 bytes): \\xDE\\xAD\\xBE\\xEF - binary sync marker
        2. Header: "FRAME <length> <crc32_hex>"
        3. Payload: base64-encoded binary data
        4. LINE_TERMINATOR: "\\n"
        
        The ESP32 firmware forwards the entire frame (including SYNC_MARKER)
        to LoRa without modification, ensuring reliable frame boundary detection
        even with lossy transmission.
        """
        encoded = base64.b64encode(payload).decode("ascii")
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        header = FIELD_SEPARATOR.join(
            (FRAME_PREFIX, str(len(encoded)), f"{crc:08x}")
        )
        frame_str = header + FIELD_SEPARATOR + encoded + LINE_TERMINATOR
        # 在整個封包前加上同步標記（ESP32 會原封不動轉發）
        frame_bytes = SYNC_MARKER + frame_str.encode("ascii")
        stats = FrameStats(payload_size=len(payload), stuffed_size=len(encoded), crc=crc)
        return frame_bytes, stats

    def iter_chunks(self, frame: bytes) -> Iterable[bytes]:
        """Split a frame into chunks suitable for streaming over the serial port.
        
        Chunking prevents overwhelming the ESP32 USB serial buffer and allows
        for controlled transmission rates when inter_chunk_delay is set.
        """
        for index in range(0, len(frame), self.chunk_size):
            yield frame[index:index + self.chunk_size]

    def send_frame(self, ser: SerialLike, payload: bytes) -> FrameStats:
        """Send *payload* through the provided serial connection.
        
        The payload is encoded, chunked, and sent with optional inter-chunk
        delays. The ESP32 firmware accumulates chunks until it receives a
        newline, then forwards the complete frame to LoRa.
        """
        frame_bytes, stats = self.build_frame(payload)
        for chunk in self.iter_chunks(frame_bytes):
            ser.write(chunk)
            if self.inter_chunk_delay:
                time.sleep(self.inter_chunk_delay)
        ser.flush()
        return stats

    @property
    def use_ack(self) -> bool:
        """Return whether ACK mode is enabled.
        
        The ESP32 firmware does not support ACK mode, so this always returns False.
        """
        return False

    @property
    def last_error(self) -> Optional[str]:
        """Return the most recent receive error, if any."""
        return self._last_error

    # -------------------------- Decoding helpers --------------------------
    def receive_frame(self, ser: SerialLike, *, block: bool = True, timeout: Optional[float] = None) -> Optional[Frame]:
        """Attempt to read and validate a single ASCII framed payload from *ser*."""
        sync_timeout = timeout if timeout is not None else (DEFAULT_SYNC_TIMEOUT if block else 0)

        if not self._synced and not self._synchronize(ser, timeout=sync_timeout):
            self._set_error_if_blocking("無法同步到封包標記 (超時)", block, sync_timeout)
            return None

        line_timeout = 0.2 if block else 0
        line = self._read_line(ser, block=block, timeout=line_timeout)
        if line is None:
            self._set_error_if_blocking("同步後讀取資料超時", block, line_timeout)
            return None

        return self._parse_and_validate_frame(line)

    def _set_error_if_blocking(self, error_msg: str, block: bool, timeout: float) -> None:
        """Set error message only if blocking with timeout, otherwise clear it."""
        if block and timeout > 0:
            self._last_error = error_msg
        else:
            self._last_error = None

    def _parse_and_validate_frame(self, line: bytes) -> Optional[Frame]:
        """Parse and validate a frame line, returning Frame or None on error."""
        try:
            text = line.decode("ascii").rstrip("\r\n")
        except UnicodeDecodeError:
            self._last_error = "資料非 ASCII 編碼"
            self._resync()
            return None

        if not text:
            self._last_error = "收到空白訊息"
            self._resync()
            return None

        parts = text.split(FIELD_SEPARATOR, 3)
        if len(parts) != 4 or parts[0] != FRAME_PREFIX:
            self._last_error = "欄位數不正確" if len(parts) != 4 else "開頭標記錯誤"
            self._resync()
            return None

        _, length_str, crc_str, encoded = parts
        
        encoded_length = self._parse_length(length_str)
        if encoded_length is None:
            return None

        encoded = self._handle_length_mismatch(encoded, encoded_length, text)
        if encoded is None:
            return None

        if encoded_length > self.max_encoded_size:
            self._last_error = "編碼資料超過上限"
            self._resync()
            return None

        expected_crc = self._parse_crc(crc_str)
        if expected_crc is None:
            return None

        payload = self._decode_payload(encoded)
        if payload is None:
            return None

        if not self._validate_crc(payload, expected_crc):
            return None

        self._last_error = None
        stats = FrameStats(payload_size=len(payload), stuffed_size=len(encoded), crc=expected_crc)
        self._resync()
        return Frame(payload=payload, stats=stats)

    def _parse_length(self, length_str: str) -> Optional[int]:
        """Parse length field, return None on error."""
        try:
            return int(length_str)
        except ValueError:
            self._last_error = "長度欄位不是整數"
            self._resync()
            return None

    def _parse_crc(self, crc_str: str) -> Optional[int]:
        """Parse CRC field, return None on error."""
        try:
            return int(crc_str, 16)
        except ValueError:
            self._last_error = "CRC 欄位格式錯誤"
            self._resync()
            return None

    def _decode_payload(self, encoded: str) -> Optional[bytes]:
        """Decode base64 payload, return None on error."""
        try:
            return base64.b64decode(encoded.encode("ascii"), validate=True)
        except Exception:
            self._last_error = "Base64 解碼失敗"
            self._resync()
            return None

    def _validate_crc(self, payload: bytes, expected_crc: int) -> bool:
        """Validate CRC, return False on mismatch."""
        actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            self._last_error = f"CRC 校驗失敗 (預期 {expected_crc:08x}, 實際 {actual_crc:08x})"
            self._resync()
            return False
        return True

    def _handle_length_mismatch(self, encoded: str, encoded_length: int, text: str) -> Optional[str]:
        """Handle cases where declared length doesn't match actual length."""
        actual_length = len(encoded)
        
        if encoded_length > actual_length:
            # Data incomplete, put back in buffer
            self._pending_ascii[:0] = text.encode("ascii")
            self._last_error = "資料不足，等待更多數據"
            return None

        if encoded_length < actual_length:
            # Data overflow, split and put extra back in buffer
            overflow = encoded[encoded_length:]
            encoded = encoded[:encoded_length]
            self._pending_ascii[:0] = overflow.encode("ascii")
            self._last_error = f"資料溢出，宣告 {encoded_length} 實際 {actual_length} (已處理)"

        # Strict mode: lengths must match exactly
        if not self._lenient and encoded_length != actual_length:
            self._last_error = f"長度不符，宣告 {encoded_length} 實際 {actual_length}"
            self._resync()
            return None
        
        # Lenient mode: trust declared length
        if encoded_length != actual_length:
            encoded = encoded[:encoded_length]
            self._last_error = f"長度不符，宣告 {encoded_length} 實際 {actual_length} (已修正)"
        else:
            self._last_error = None

        return encoded

    def _read_serial_chunk(self, ser: SerialLike, *, wait: bool) -> bytes:
        """Read a small chunk from the serial port respecting blocking preference.
        
        Optimized to read larger chunks when data is available to reduce
        system call overhead and improve throughput. Maximum read size is
        capped to prevent memory issues.
        """
        # Maximum read size to prevent excessive memory allocation
        MAX_READ_SIZE = 8192  # 8KB hard limit
        
        max_bytes = self.chunk_size
        available = 0
        try:
            available = int(getattr(ser, "in_waiting", 0))
        except (AttributeError, TypeError, ValueError):
            available = 0

        if not wait and available <= 0:
            return b""

        if available > 0:
            # Read up to double the chunk size when more data is available,
            # but never exceed MAX_READ_SIZE
            max_bytes = max(1, min(self.chunk_size * 2, available, MAX_READ_SIZE))

        try:
            return ser.read(max_bytes)
        except Exception:
            return b""

    def _resync(self, *, clear_pending: bool = False) -> None:
        """Reset sync state so the next frame search starts from a clean boundary."""
        self._synced = False
        if clear_pending:
            self._pending_ascii.clear()

    def _synchronize(self, ser: SerialLike, timeout: Optional[float] = None) -> bool:
        """Search for the sync marker or FRAME prefix in the input buffer with a timeout.
        
        This method tries to find either:
        1. Binary SYNC_MARKER (for clean transmissions)
        2. ASCII "FRAME" prefix (for LoRa transmissions where binary data gets corrupted)
        
        Optimized to handle large buffers more efficiently and reduce
        memory allocations during synchronization.
        """
        if self._synced:
            return True

        marker_len = len(SYNC_MARKER)
        buffer = bytearray(self._pending_ascii)
        self._pending_ascii.clear()
        start_time = time.monotonic()
        wait_for_data = timeout is None or timeout > 0
        
        # Maximum buffer size before trimming (2x encoded size for safety)
        max_buffer_size = self.max_encoded_size * 2

        while True:
            # Try to find sync marker (binary or ASCII FRAME prefix)
            if self._try_find_sync_marker(buffer, marker_len):
                return True

            # Check timeout
            if timeout is not None and (time.monotonic() - start_time) > timeout:
                self._pending_ascii[:0] = buffer
                return False

            # Read more data
            chunk = self._read_serial_chunk(ser, wait=wait_for_data)
            if not chunk:
                if not wait_for_data:
                    self._pending_ascii[:0] = buffer
                    return False
                time.sleep(0.001)
                continue

            buffer.extend(chunk)
            
            # Prevent buffer from growing too large
            # Only trim when significantly over limit to reduce operations
            if len(buffer) > max_buffer_size + self.chunk_size:
                # Keep most recent data, discarding older data that didn't contain sync marker
                excess = len(buffer) - max_buffer_size
                del buffer[:excess]

    def _try_find_sync_marker(self, buffer: bytearray, marker_len: int) -> bool:
        """Try to find sync marker or FRAME prefix in buffer, updating state if found.
        
        Tries two methods:
        1. Look for binary SYNC_MARKER (preferred for clean transmissions)
        2. Look for ASCII "FRAME " prefix (fallback for LoRa where binary gets corrupted)
        """
        # Try binary SYNC_MARKER first
        idx = buffer.find(SYNC_MARKER)
        if idx != -1:
            del buffer[:idx + marker_len]
            self._pending_ascii.extend(buffer)
            self._synced = True
            return True
        
        # Fallback: Try to find ASCII "FRAME " prefix (for LoRa transmissions)
        frame_prefix_bytes = FRAME_PREFIX.encode("ascii") + b" "
        idx = buffer.find(frame_prefix_bytes)
        if idx != -1:
            # Found "FRAME " - sync to the start of this frame
            del buffer[:idx]
            self._pending_ascii.extend(buffer)
            self._synced = True
            return True
        
        return False

    def _read_line(self, ser: SerialLike, *, block: bool = True, timeout: Optional[float] = None) -> Optional[bytes]:
        """Read the next newline-terminated ASCII frame payload."""
        start_time = time.monotonic()

        while True:
            newline_index = self._pending_ascii.find(b"\n")
            if newline_index != -1:
                line = bytes(self._pending_ascii[:newline_index + 1])
                del self._pending_ascii[:newline_index + 1]
                return line

            if not block:
                chunk = self._read_serial_chunk(ser, wait=False)
                if chunk:
                    self._pending_ascii.extend(chunk)
                    continue
                return None

            if timeout is not None and (time.monotonic() - start_time) > timeout:
                return None

            chunk = self._read_serial_chunk(ser, wait=True)
            if chunk:
                self._pending_ascii.extend(chunk)
                continue

            time.sleep(0.001)