"""Shared serial video frame protocol utilities."""
from __future__ import annotations

import glob
import platform
import struct
import time
import zlib
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, Sequence, runtime_checkable

import serial  # type: ignore

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
BAUD_RATE = 115_200
START_OF_FRAME = b"\x01"  # SOH
END_OF_FRAME = b"\x04"  # EOT
ESC = b"\x1B"  # ESC (escape)
ACK_MESSAGE = b"ACK"  # Acknowledgement message
HEADER_FORMAT = ">II"  # stuffed_size, crc32 (payload)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
DEFAULT_CHUNK_SIZE = 128
DEFAULT_INTER_CHUNK_DELAY = 0.001  # seconds
DEFAULT_MAX_STUFFED_SIZE = 256 * 1024  # 256 KB
DEFAULT_MAX_PAYLOAD_SIZE = 128 * 1024  # 128 KB (post unstuff)
DEFAULT_ACK_TIMEOUT = 1.0


class AckTimeoutError(RuntimeError):
    """Raised when an expected ACK message is not received within timeout."""

    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameStats:
    """Metadata describing a single transmitted frame."""

    payload_size: int
    stuffed_size: int
    crc: int


@dataclass(frozen=True)
class Frame:
    """Container for a received frame."""

    payload: bytes
    stats: FrameStats


# ---------------------------------------------------------------------------
# Byte stuffing helpers
# ---------------------------------------------------------------------------
def stuff_bytes(data: bytes) -> bytes:
    """Perform byte stuffing so payload doesn't collide with control markers."""
    stuffed = bytearray()
    for byte in data:
        if byte == ESC[0]:
            stuffed.extend(ESC + ESC)
        elif byte == START_OF_FRAME[0]:
            stuffed.extend(ESC + START_OF_FRAME)
        elif byte == END_OF_FRAME[0]:
            stuffed.extend(ESC + END_OF_FRAME)
        else:
            stuffed.append(byte)
    return bytes(stuffed)


def unstuff_bytes(data: bytes) -> bytes:
    """Reverse byte stuffing applied by :func:`stuff_bytes`."""
    unstuffed = bytearray()
    index = 0
    while index < len(data):
        if data[index:index + 1] == ESC:
            # If ESC is the final byte, we ignore it (protocol guarantee prevents this)
            if index + 1 < len(data):
                unstuffed.append(data[index + 1])
            index += 2
        else:
            unstuffed.append(data[index])
            index += 1
    return bytes(unstuffed)


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
    def write(self, data) -> int | None: ...

    def read(self, size: int = ...) -> bytes: ...

    def flush(self) -> None: ...


class FrameProtocol:
    """Utility responsible for framing, validation, and transport orchestration."""

    def __init__(
        self,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        inter_chunk_delay: float = DEFAULT_INTER_CHUNK_DELAY,
        max_stuffed_size: int = DEFAULT_MAX_STUFFED_SIZE,
        max_payload_size: int = DEFAULT_MAX_PAYLOAD_SIZE,
        ack_timeout: float = DEFAULT_ACK_TIMEOUT,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必須為正整數")
        self.chunk_size = chunk_size
        self.inter_chunk_delay = max(inter_chunk_delay, 0.0)
        self.max_stuffed_size = max_stuffed_size
        self.max_payload_size = max_payload_size
        self.ack_timeout = max(ack_timeout, 0.0)

    # -------------------------- Encoding helpers --------------------------
    def build_frame(self, payload: bytes) -> tuple[bytes, bytes, FrameStats]:
        """Construct the header, stuffed payload, and statistics for *payload*."""
        stuffed = stuff_bytes(payload)
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        header = struct.pack(HEADER_FORMAT, len(stuffed), crc)
        stats = FrameStats(payload_size=len(payload), stuffed_size=len(stuffed), crc=crc)
        return header, stuffed, stats

    def iter_chunks(self, frame: bytes) -> Iterable[bytes]:
        """Split a frame into chunks suitable for streaming over the serial port."""
        for index in range(0, len(frame), self.chunk_size):
            yield frame[index:index + self.chunk_size]

    def send_frame(self, ser: SerialLike, payload: bytes) -> FrameStats:
        """Send *payload* through the provided serial connection with ACK flow control."""
        header, stuffed_payload, stats = self.build_frame(payload)

        # Send frame start and header without requiring ACK
        ser.write(START_OF_FRAME)
        ser.write(header)
        ser.flush()

        # Send stuffed payload chunk by chunk, waiting for ACK after each chunk if configured
        offset = 0
        stuffed_length = len(stuffed_payload)
        while offset < stuffed_length:
            chunk = stuffed_payload[offset: offset + self.chunk_size]
            ser.write(chunk)
            ser.flush()

            if self.ack_timeout > 0:
                if not self.wait_for_ack(ser, timeout=self.ack_timeout):
                    raise AckTimeoutError("等待 ACK 超時")

            if self.inter_chunk_delay:
                time.sleep(self.inter_chunk_delay)

            offset += len(chunk)

        # Send frame end marker
        ser.write(END_OF_FRAME)
        ser.flush()
        return stats

    def wait_for_ack(self, ser: SerialLike, timeout: Optional[float] = None) -> bool:
        """Wait for an ACK message from the receiver."""
        effective_timeout = self.ack_timeout if timeout is None else max(timeout, 0.0)
        if effective_timeout <= 0:
            return True

        deadline = time.time() + effective_timeout
        while time.time() < deadline:
            data = ser.read(len(ACK_MESSAGE))
            if data == ACK_MESSAGE:
                return True
            if not data:
                continue
        return False

    # -------------------------- Decoding helpers --------------------------
    def receive_frame(self, ser: SerialLike, *, block: bool = True) -> Optional[Frame]:
        """Attempt to read and validate a single frame from *ser*.

        When *block* is False, the method returns ``None`` immediately if no
        data is available. When True (default), it will continue waiting for the
        next frame until data arrives.
        """
        if not self._await_start_marker(ser, block=block):
            return None

        header = self._read_exact(ser, HEADER_SIZE, block=True)
        if header is None:
            return None

        try:
            stuffed_size, expected_crc = struct.unpack(HEADER_FORMAT, header)
        except struct.error:
            # Corrupted header: resynchronise
            self._discard_until_end(ser)
            return None

        if stuffed_size <= 0 or stuffed_size > self.max_stuffed_size:
            # Invalid size, discard until we reach the end marker
            self._discard_until_end(ser)
            return None

        stuffed_payload = bytearray()
        remaining = stuffed_size
        while remaining > 0:
            expected_size = min(self.chunk_size, remaining)
            chunk = self._read_exact(ser, expected_size, block=True)
            if chunk is None or len(chunk) != expected_size:
                return None

            stuffed_payload.extend(chunk)
            remaining -= expected_size

            if self.ack_timeout > 0:
                ser.write(ACK_MESSAGE)
                ser.flush()

        end_marker = self._read_exact(ser, len(END_OF_FRAME), block=True)
        if end_marker != END_OF_FRAME:
            # Attempt to resynchronise for subsequent frames
            self._discard_until_end(ser)
            return None

        payload = unstuff_bytes(bytes(stuffed_payload))
        if not payload or len(payload) > self.max_payload_size:
            return None

        crc = zlib.crc32(payload) & 0xFFFFFFFF
        if crc != expected_crc:
            return None

        stats = FrameStats(
            payload_size=len(payload),
            stuffed_size=len(stuffed_payload),
            crc=crc,
        )
        return Frame(payload=payload, stats=stats)

    # -------------------------- Internal helpers --------------------------
    def _await_start_marker(self, ser: SerialLike, *, block: bool) -> bool:
        while True:
            byte = ser.read(1)
            if byte == START_OF_FRAME:
                return True
            if not byte:
                if block:
                    continue
                return False

    def _read_exact(self, ser: SerialLike, size: int, *, block: bool) -> Optional[bytes]:
        buffer = bytearray()
        while len(buffer) < size:
            chunk = ser.read(size - len(buffer))
            if chunk:
                buffer.extend(chunk)
                continue
            if not block:
                return None
        return bytes(buffer)

    def _discard_until_end(self, ser: SerialLike) -> None:
        while True:
            byte = ser.read(1)
            if not byte or byte == END_OF_FRAME:
                return