"""Test relay mode buffering improvements.

This test verifies that the PC-side buffering can correctly handle
frame reconstruction from raw data streams (as sent by ESP32 in relay mode).
"""
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.protocol import encode_frame, decode_frame, TYPE_JPEG, TYPE_CS
from common.config import FRAME_START, FRAME_END


class MockSerialBuffer:
    """Mock serial buffer to simulate chunked data reception."""
    
    def __init__(self):
        self.data = bytearray()
        self.read_position = 0
    
    def add_data(self, chunk):
        """Add data to the buffer (simulating ESP32 relay)."""
        self.data.extend(chunk)
    
    def read_chunk(self, size):
        """Read a chunk of data (simulating serial.read())."""
        available = len(self.data) - self.read_position
        actual_size = min(size, available)
        
        if actual_size == 0:
            return bytearray()
        
        chunk = self.data[self.read_position:self.read_position + actual_size]
        self.read_position += actual_size
        return chunk
    
    def available(self):
        """Return number of bytes available."""
        return len(self.data) - self.read_position


class TestRelayMode(unittest.TestCase):
    """Test PC-side buffering with relay mode ESP32."""
    
    def simulate_pc_receive_frame(self, serial_buffer):
        """Simulate the PC-side receive_frame logic."""
        buffer = bytearray()
        
        # Read all available data
        while serial_buffer.available() > 0:
            chunk = serial_buffer.read_chunk(512)  # Simulate 512-byte relay chunks
            buffer.extend(chunk)
        
        # Look for frame start marker
        start_idx = buffer.find(FRAME_START)
        if start_idx == -1:
            # No start marker found
            if len(buffer) > 1:
                buffer = buffer[-1:]
            return None, buffer
        
        # Discard data before start marker
        if start_idx > 0:
            buffer = buffer[start_idx:]
        
        # Need at least minimum frame
        if len(buffer) < 9:
            return None, buffer
        
        # Look for frame end marker
        end_idx = buffer.find(FRAME_END, 5)
        if end_idx == -1:
            # No end marker yet
            return None, buffer
        
        # Extract complete frame
        frame = bytes(buffer[:end_idx + len(FRAME_END)])
        buffer = buffer[end_idx + len(FRAME_END):]
        
        return frame, buffer
    
    def test_single_frame_chunked_delivery(self):
        """Test receiving a single frame in small chunks."""
        test_data = b"Test data for chunked delivery"
        frame = encode_frame(TYPE_JPEG, test_data)
        
        # Simulate ESP32 relay sending frame in 16-byte chunks
        serial_buffer = MockSerialBuffer()
        chunk_size = 16
        
        for i in range(0, len(frame), chunk_size):
            chunk = frame[i:i + chunk_size]
            serial_buffer.add_data(chunk)
        
        # Try to receive frame
        received_frame, _ = self.simulate_pc_receive_frame(serial_buffer)
        
        self.assertIsNotNone(received_frame)
        self.assertEqual(received_frame, frame)
        
        # Decode and verify
        result = decode_frame(received_frame)
        self.assertIsNotNone(result)
        frame_type, data = result
        self.assertEqual(frame_type, TYPE_JPEG)
        self.assertEqual(data, test_data)
    
    def test_multiple_frames_chunked_delivery(self):
        """Test receiving multiple frames in random chunks."""
        test_data1 = b"First frame" * 100  # Larger data
        test_data2 = b"Second frame" * 100
        test_data3 = b"Third frame" * 100
        
        frame1 = encode_frame(TYPE_JPEG, test_data1)
        frame2 = encode_frame(TYPE_CS, test_data2)
        frame3 = encode_frame(TYPE_JPEG, test_data3)
        
        # Combine all frames
        all_data = frame1 + frame2 + frame3
        
        # Simulate relay sending in varying chunk sizes
        serial_buffer = MockSerialBuffer()
        chunk_sizes = [32, 64, 128, 256, 512, 128, 64, 32]
        pos = 0
        
        for chunk_size in chunk_sizes:
            if pos >= len(all_data):
                break
            chunk = all_data[pos:pos + chunk_size]
            serial_buffer.add_data(chunk)
            pos += chunk_size
        
        # Add remaining data
        if pos < len(all_data):
            serial_buffer.add_data(all_data[pos:])
        
        # Receive frames
        buffer = bytearray()
        frames_received = []
        
        # Simulate multiple receive calls
        for _ in range(10):  # Try multiple times to get all frames
            while serial_buffer.available() > 0:
                chunk = serial_buffer.read_chunk(512)
                buffer.extend(chunk)
            
            # Try to extract frames from buffer
            while True:
                start_idx = buffer.find(FRAME_START)
                if start_idx == -1:
                    if len(buffer) > 1:
                        buffer = buffer[-1:]
                    break
                
                if start_idx > 0:
                    buffer = buffer[start_idx:]
                
                if len(buffer) < 9:
                    break
                
                end_idx = buffer.find(FRAME_END, 5)
                if end_idx == -1:
                    break
                
                frame = bytes(buffer[:end_idx + len(FRAME_END)])
                buffer = buffer[end_idx + len(FRAME_END):]
                frames_received.append(frame)
        
        # Should have received all 3 frames
        self.assertEqual(len(frames_received), 3)
        
        # Verify each frame
        result1 = decode_frame(frames_received[0])
        self.assertIsNotNone(result1)
        self.assertEqual(result1[0], TYPE_JPEG)
        self.assertEqual(result1[1], test_data1)
        
        result2 = decode_frame(frames_received[1])
        self.assertIsNotNone(result2)
        self.assertEqual(result2[0], TYPE_CS)
        self.assertEqual(result2[1], test_data2)
        
        result3 = decode_frame(frames_received[2])
        self.assertIsNotNone(result3)
        self.assertEqual(result3[0], TYPE_JPEG)
        self.assertEqual(result3[1], test_data3)
    
    def test_frame_with_garbage_before(self):
        """Test handling garbage data before valid frame (relay mode)."""
        garbage = b"\xFF\x00\xAB\xCD" * 50  # 200 bytes of garbage
        test_data = b"Valid frame data after garbage"
        frame = encode_frame(TYPE_JPEG, test_data)
        
        # Simulate relay sending garbage + frame in chunks
        serial_buffer = MockSerialBuffer()
        all_data = garbage + frame
        
        chunk_size = 64
        for i in range(0, len(all_data), chunk_size):
            chunk = all_data[i:i + chunk_size]
            serial_buffer.add_data(chunk)
        
        # Receive and extract frame
        buffer = bytearray()
        received_frame = None
        
        while serial_buffer.available() > 0:
            chunk = serial_buffer.read_chunk(512)
            buffer.extend(chunk)
        
        # Extract frame
        start_idx = buffer.find(FRAME_START)
        if start_idx >= 0:
            buffer = buffer[start_idx:]
            end_idx = buffer.find(FRAME_END, 5)
            if end_idx >= 0:
                received_frame = bytes(buffer[:end_idx + len(FRAME_END)])
        
        self.assertIsNotNone(received_frame)
        
        # Decode and verify
        result = decode_frame(received_frame)
        self.assertIsNotNone(result)
        frame_type, data = result
        self.assertEqual(data, test_data)
    
    def test_split_start_marker(self):
        """Test handling when START marker is split across chunks."""
        test_data = b"Test data with split start marker"
        frame = encode_frame(TYPE_JPEG, test_data)
        
        serial_buffer = MockSerialBuffer()
        
        # Add some garbage
        serial_buffer.add_data(b"\xFF\x00")
        
        # Add first byte of START marker
        serial_buffer.add_data(frame[:1])
        
        # Add rest of frame
        serial_buffer.add_data(frame[1:])
        
        # Receive
        buffer = bytearray()
        while serial_buffer.available() > 0:
            chunk = serial_buffer.read_chunk(512)
            buffer.extend(chunk)
        
        # Extract frame
        start_idx = buffer.find(FRAME_START)
        self.assertGreaterEqual(start_idx, 0)
        
        buffer = buffer[start_idx:]
        end_idx = buffer.find(FRAME_END, 5)
        self.assertGreaterEqual(end_idx, 0)
        
        received_frame = bytes(buffer[:end_idx + len(FRAME_END)])
        
        # Decode and verify
        result = decode_frame(received_frame)
        self.assertIsNotNone(result)
        frame_type, data = result
        self.assertEqual(data, test_data)
    
    def test_large_frame_relay(self):
        """Test relaying a large frame (near maximum size)."""
        # Create large data (close to max)
        test_data = b"X" * 10000  # 10KB data
        frame = encode_frame(TYPE_JPEG, test_data)
        
        # Simulate relay in 512-byte chunks (typical ESP32 relay buffer)
        serial_buffer = MockSerialBuffer()
        chunk_size = 512
        
        for i in range(0, len(frame), chunk_size):
            chunk = frame[i:i + chunk_size]
            serial_buffer.add_data(chunk)
        
        # Receive
        buffer = bytearray()
        while serial_buffer.available() > 0:
            chunk = serial_buffer.read_chunk(512)
            buffer.extend(chunk)
        
        # Should have received complete frame
        self.assertEqual(len(buffer), len(frame))
        
        # Decode and verify
        result = decode_frame(bytes(buffer))
        self.assertIsNotNone(result)
        frame_type, data = result
        self.assertEqual(len(data), len(test_data))
        self.assertEqual(data, test_data)


if __name__ == '__main__':
    unittest.main()
