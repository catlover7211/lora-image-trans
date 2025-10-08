import unittest
from typing import Optional

from protocol import FrameProtocol, stuff_bytes, unstuff_bytes


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
        self.protocol = FrameProtocol(inter_chunk_delay=0, chunk_size=32, ack_timeout=0)
        self.loopback = LoopbackSerial()

    def test_byte_stuff_roundtrip(self) -> None:
        payload = bytes([1, 27, 4, 0, 255])
        stuffed = stuff_bytes(payload)
        self.assertNotEqual(payload, stuffed)
        restored = unstuff_bytes(stuffed)
        self.assertEqual(payload, restored)

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


if __name__ == "__main__":
    unittest.main()
