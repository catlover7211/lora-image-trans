"""Shared serial video frame protocol utilities."""
from __future__ import annotations

import glob
import platform
import struct
import time
import zlib
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import serial  # type: ignore

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
BAUD_RATE = 115_200
START_OF_FRAME = b"\x01"  # SOH
END_OF_FRAME = b"\x04"  # EOT
ESC = b"\x1B"  # ESC (escape)
HEADER_FORMAT = ">II"  # stuffed_size, crc32 (payload)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
DEFAULT_CHUNK_SIZE = 128
DEFAULT_INTER_CHUNK_DELAY = 0.001  # seconds
DEFAULT_MAX_STUFFED_SIZE = 256 * 1024  # 256 KB
DEFAULT_MAX_PAYLOAD_SIZE = 128 * 1024  # 128 KB (post unstuff)


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
        return [f"COM{i}" for i in range(1, 257)]
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
class FrameProtocol:
    """Utility responsible for framing, validation, and transport orchestration."""

    def __init__(
        self,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        inter_chunk_delay: float = DEFAULT_INTER_CHUNK_DELAY,
        max_stuffed_size: int = DEFAULT_MAX_STUFFED_SIZE,
        max_payload_size: int = DEFAULT_MAX_PAYLOAD_SIZE,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必須為正整數")
        self.chunk_size = chunk_size
        self.inter_chunk_delay = max(inter_chunk_delay, 0.0)
        self.max_stuffed_size = max_stuffed_size
        self.max_payload_size = max_payload_size

    # -------------------------- Encoding helpers --------------------------
    def build_frame(self, payload: bytes) -> tuple[bytes, FrameStats]:
        """Construct the framed byte stream and statistics for *payload*."""
        stuffed = stuff_bytes(payload)
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        header = struct.pack(HEADER_FORMAT, len(stuffed), crc)
        frame = START_OF_FRAME + header + stuffed + END_OF_FRAME
        stats = FrameStats(payload_size=len(payload), stuffed_size=len(stuffed), crc=crc)
        return frame, stats

    def iter_chunks(self, frame: bytes) -> Iterable[bytes]:
        """Split a frame into chunks suitable for streaming over the serial port."""
        for index in range(0, len(frame), self.chunk_size):
            yield frame[index:index + self.chunk_size]

    def send_frame(self, ser: serial.Serial, payload: bytes) -> FrameStats:
        """Send *payload* through the provided serial connection."""
        frame, stats = self.build_frame(payload)
        for chunk in self.iter_chunks(frame):
            ser.write(chunk)
            if self.inter_chunk_delay:
                time.sleep(self.inter_chunk_delay)
        ser.flush()
        return stats

    # -------------------------- Decoding helpers --------------------------
    def receive_frame(self, ser: serial.Serial, *, block: bool = True) -> Optional[Frame]:
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

        stuffed_payload = self._read_exact(ser, stuffed_size, block=True)
        if stuffed_payload is None:
            return None

        end_marker = self._read_exact(ser, len(END_OF_FRAME), block=True)
        if end_marker != END_OF_FRAME:
            # Attempt to resynchronise for subsequent frames
            self._discard_until_end(ser)
            return None

        payload = unstuff_bytes(stuffed_payload)
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
    def _await_start_marker(self, ser: serial.Serial, *, block: bool) -> bool:
        while True:
            byte = ser.read(1)
            if byte == START_OF_FRAME:
                return True
            if not byte:
                if block:
                    continue
                return False

    def _read_exact(self, ser: serial.Serial, size: int, *, block: bool) -> Optional[bytes]:
        buffer = bytearray()
        while len(buffer) < size:
            chunk = ser.read(size - len(buffer))
            if chunk:
                buffer.extend(chunk)
                continue
            if not block:
                return None
        return bytes(buffer)

    def _discard_until_end(self, ser: serial.Serial) -> None:
        while True:
            byte = ser.read(1)
            if not byte or byte == END_OF_FRAME:
                return