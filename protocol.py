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
SYNC_MARKER = b"\xDE\xAD\xBE\xEF"  # 新增同步標記
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
        # 在整個封包前加上同步標記
        frame_bytes = SYNC_MARKER + frame_str.encode("ascii")
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
    def receive_frame(self, ser: SerialLike, *, block: bool = True, timeout: Optional[float] = None) -> Optional[Frame]:
        """Attempt to read and validate a single ASCII framed payload from *ser*."""
        # For blocking calls, use the ACK timeout as the base, otherwise non-blocking.
        sync_timeout = timeout if timeout is not None else (self.ack_timeout if block else 0)
        
        synced = self._synchronize(ser, timeout=sync_timeout)
        if not synced:
            # In non-blocking mode, failure to sync just means no data is ready.
            # Only set an error if we were actually blocking and timed out.
            if block and sync_timeout > 0:
                self._last_error = "無法同步到封包標記 (超時)"
            else:
                self._last_error = None # No error, just no data
            return None

        # After sync, the rest of the line should arrive quickly. Use a short timeout.
        line_timeout = 0.2 if block else 0
        line = self._read_line(ser, block=block, timeout=line_timeout)
        if line is None:
            # Similar to sync, only report error if we were expecting data.
            if block and line_timeout > 0:
                self._last_error = "同步後讀取資料超時"
            else:
                self._last_error = None # No error, just incomplete data
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

        if encoded_length > len(encoded):
            # 資料不足，可能被截斷。將其放回緩衝區等待更多資料。
            self._pending_ascii[:0] = text.encode("ascii")
            self._last_error = "資料不足，等待更多數據"
            return None

        if encoded_length < len(encoded):
            # 資料過多，分割並將多餘部分放回緩衝區
            overflow = encoded[encoded_length:]
            encoded = encoded[:encoded_length]
            self._pending_ascii[:0] = overflow.encode("ascii")
            self._last_error = f"資料溢出，宣告 {encoded_length} 實際 {len(encoded)} (已處理)"
        
        # 在嚴格模式下，長度必須完全相符
        if not self._lenient and encoded_length != len(encoded):
            msg = f"長度不符，宣告 {encoded_length} 實際 {len(encoded)}"
            self._last_error = msg
            # 當長度不符時，我們假設整個封包都已損毀，因此不將其放回緩衝區
            # 而是直接丟棄，等待下一個有效的 FRAME 開頭
            return None
        
        # 如果宣告長度與實際長度不同，但在寬鬆模式下，我們信任宣告的長度
        if encoded_length != len(encoded):
            encoded = encoded[:encoded_length]
            self._last_error = f"長度不符，宣告 {encoded_length} 實際 {len(encoded)} (已修正)"
        else:
            self._last_error = None  # 成功解析，清除舊錯誤

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

        actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            self._last_error = f"CRC 校驗失敗 (預期 {expected_crc:08x}, 實際 {actual_crc:08x})"
            return None

        stats = FrameStats(
            payload_size=len(payload), stuffed_size=len(encoded), crc=expected_crc
        )
        return Frame(payload=payload, stats=stats)

    def _synchronize(self, ser: SerialLike, timeout: Optional[float] = None) -> bool:
        """Search for the sync marker in the input buffer with a timeout."""
        marker_len = len(SYNC_MARKER)
        search_buffer = bytearray()
        start_time = time.monotonic()

        if self._pending_ascii:
            search_buffer.extend(self._pending_ascii)
            self._pending_ascii.clear()

        while True:
            if timeout is not None and (time.monotonic() - start_time) > timeout:
                self._pending_ascii.extend(search_buffer) # Keep unprocessed data
                return False

            idx = search_buffer.find(SYNC_MARKER)
            if idx != -1:
                self._pending_ascii.extend(search_buffer[idx + marker_len:])
                return True

            # Determine how many bytes to read
            # Set a non-blocking read if timeout is 0
            original_timeout = getattr(ser, 'timeout', None)
            read_timeout = 0 if timeout == 0 else 0.01 # Short poll
            if getattr(ser, 'timeout', -1) != read_timeout:
                try:
                    ser.timeout = read_timeout
                except AttributeError:
                    pass # Not all serial-like objects have a timeout setter

            new_data = ser.read(self.chunk_size)
            
            # Restore original timeout if we changed it
            if original_timeout is not None and getattr(ser, 'timeout', -1) != original_timeout:
                 try:
                    ser.timeout = original_timeout
                 except AttributeError:
                    pass

            if not new_data:
                if timeout == 0: # Non-blocking, no data is not an error
                    self._pending_ascii.extend(search_buffer)
                    return False
                continue # Poll again

            search_buffer.extend(new_data)

            if len(search_buffer) > self.max_encoded_size * 2:
                search_buffer = search_buffer[-marker_len:]

    def _wait_for_ack(self, ser: SerialLike) -> bool:
        """Wait for an ACK message from the remote."""
        if self.ack_timeout <= 0:
            return True
        deadline = time.monotonic() + self.ack_timeout
        buffer = bytearray()
        while True:
            if time.monotonic() > deadline:
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

    def _read_line(self, ser: SerialLike, *, block: bool = True, timeout: Optional[float] = None) -> Optional[bytes]:
        """
        Read from the serial port until a line terminator is found or timeout.
        Handles a persistent buffer (_pending_ascii).
        """
        start_time = time.monotonic()
        
        while True:
            # Check for timeout
            if timeout is not None and (time.monotonic() - start_time) > timeout:
                return None # Timeout occurred

            # Check for line terminator in the existing buffer
            if b'\n' in self._pending_ascii:
                line, self._pending_ascii = self._pending_ascii.split(b'\n', 1)
                return line + b'\n'

            # If non-blocking and no full line, return immediately
            if not block:
                return None

            # Read more data from serial
            # Set a non-blocking read if timeout is 0
            original_timeout = getattr(ser, 'timeout', None)
            read_timeout = 0 if timeout == 0 else 0.01 # Short poll
            if getattr(ser, 'timeout', -1) != read_timeout:
                try:
                    ser.timeout = read_timeout
                except AttributeError:
                    pass

            new_data = ser.read(self.chunk_size)

            # Restore original timeout
            if original_timeout is not None and getattr(ser, 'timeout', -1) != original_timeout:
                 try:
                    ser.timeout = original_timeout
                 except AttributeError:
                    pass

            if new_data:
                self._pending_ascii.extend(new_data)
            elif timeout is None:
                # In true blocking mode (timeout=None), no data might mean closed port
                return None