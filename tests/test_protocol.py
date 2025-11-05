import base64
import threading
import time
import unittest
from typing import Optional

from protocol import FIELD_SEPARATOR, FRAME_PREFIX, SYNC_MARKER, Frame, FrameProtocol


class LoopbackSerial:
    """極簡迴路序列埠模擬器，僅用於單元測試。"""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.read_index = 0
        self.timeout: Optional[float] = None

    def write(self, data: bytes) -> int:
        self.buffer.extend(data)
        return len(data)

    def flush(self) -> None:  # pragma: no cover - 無動作需求
        pass

    def read(self, size: int = 1) -> bytes:
        if self.read_index >= len(self.buffer):
            return b""
        end_index = min(self.read_index + size, len(self.buffer))
        chunk = self.buffer[self.read_index:end_index]
        self.read_index = end_index
        return bytes(chunk)

    def rewind(self) -> None:
        self.read_index = 0


class FrameProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = FrameProtocol(inter_chunk_delay=0, chunk_size=32)
        self.loopback = LoopbackSerial()

    def test_send_and_receive_roundtrip(self) -> None:
        payload = bytes(range(256))
        stats = self.protocol.send_frame(self.loopback, payload)
        self.assertEqual(stats.payload_size, len(payload))
        self.assertGreater(stats.stuffed_size, len(payload))
        self.loopback.rewind()
        received = self.protocol.receive_frame(self.loopback)
        self.assertIsNotNone(received)
        assert received  # for mypy / type checkers
        self.assertEqual(received.payload, payload)
        self.assertEqual(received.stats.crc, stats.crc)

    def test_frame_format_ascii(self) -> None:
        payload = b"hello world"
        self.protocol.send_frame(self.loopback, payload)
        # Skip sync marker (first 4 bytes)
        frame_bytes = self.loopback.buffer[len(SYNC_MARKER):]
        frame_line = frame_bytes.decode("ascii")
        self.assertTrue(frame_line.endswith("\n"))
        frame_line = frame_line.strip()
        parts = frame_line.split(FIELD_SEPARATOR, 3)
        self.assertEqual(len(parts), 4)
        prefix, length_str, crc_str, encoded = parts
        self.assertEqual(prefix, FRAME_PREFIX)
        self.assertEqual(int(length_str), len(encoded))
        decoded = base64.b64decode(encoded.encode("ascii"))
        self.assertEqual(decoded, payload)
        self.assertEqual(len(crc_str), 8)


if __name__ == "__main__":
    unittest.main()
