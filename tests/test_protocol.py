import base64
import threading
import unittest
from typing import Optional

from protocol import FIELD_SEPARATOR, FRAME_PREFIX, Frame, FrameProtocol


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


class FullDuplexEndpoint:
    def __init__(self, incoming: bytearray, outgoing: bytearray, lock: threading.Lock) -> None:
        self._incoming = incoming
        self._outgoing = outgoing
        self._lock = lock
        self._read_index = 0

    def write(self, data: bytes) -> int:
        with self._lock:
            self._outgoing.extend(data)
        return len(data)

    def flush(self) -> None:  # pragma: no cover - 無動作需求
        pass

    def read(self, size: int = 1) -> bytes:
        with self._lock:
            if self._read_index >= len(self._incoming):
                return b""
            end_index = min(self._read_index + size, len(self._incoming))
            chunk = self._incoming[self._read_index:end_index]
            self._read_index = end_index
            return bytes(chunk)


def create_full_duplex_pair() -> tuple[FullDuplexEndpoint, FullDuplexEndpoint]:
    lock = threading.Lock()
    ab = bytearray()
    ba = bytearray()
    return (
        FullDuplexEndpoint(ba, ab, lock),
        FullDuplexEndpoint(ab, ba, lock),
    )


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
        frame_line = self.loopback.buffer.decode("ascii")
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

    def test_chunk_ack_roundtrip(self) -> None:
        sender_ser, receiver_ser = create_full_duplex_pair()
        sender_protocol = FrameProtocol(
            inter_chunk_delay=0,
            chunk_size=16,
            use_chunk_ack=True,
            ack_timeout=0.5,
        )
        receiver_protocol = FrameProtocol(
            inter_chunk_delay=0,
            chunk_size=16,
            use_chunk_ack=True,
        )

        payload = bytes(range(64))
        received_frame: dict[str, Optional[Frame]] = {"frame": None}

        def receiver_task() -> None:
            received_frame["frame"] = receiver_protocol.receive_frame(receiver_ser, block=True)

        thread = threading.Thread(target=receiver_task)
        thread.start()
        stats = sender_protocol.send_frame(sender_ser, payload)
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive(), "接收執行緒未正常結束")
        frame = received_frame["frame"]
        self.assertIsNotNone(frame)
        assert frame  # for mypy / type checkers
        self.assertEqual(frame.payload, payload)
        self.assertEqual(frame.stats.crc, stats.crc)


if __name__ == "__main__":
    unittest.main()
