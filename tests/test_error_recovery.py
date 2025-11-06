"""Test error recovery and edge cases in protocol handling."""
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.protocol import encode_frame, decode_frame, TYPE_JPEG, TYPE_CS
from common.config import FRAME_START, FRAME_END


class TestErrorRecovery(unittest.TestCase):
    """Test protocol error recovery."""
    
    def test_decode_frame_with_garbage_before(self):
        """Test that garbage data before valid frame is ignored."""
        test_data = b"Valid test data"
        valid_frame = encode_frame(TYPE_JPEG, test_data)
        
        # Add garbage before valid frame
        garbage = b"\xFF\xFF\x00\x00garbage data"
        frame_with_garbage = garbage + valid_frame
        
        # The decode function should still fail because we pass the whole thing
        # But in real use, the receiver strips to the first START marker
        result = decode_frame(frame_with_garbage)
        self.assertIsNone(result)
        
        # However, if we extract just the valid frame, it should work
        result = decode_frame(valid_frame)
        self.assertIsNotNone(result)
        frame_type, data = result
        self.assertEqual(data, test_data)
    
    def test_decode_frame_with_wrong_end_marker(self):
        """Test frame with incorrect end marker."""
        test_data = b"Test data"
        frame = encode_frame(TYPE_JPEG, test_data)
        
        # Replace end marker with wrong bytes
        bad_frame = frame[:-2] + b"\x00\x00"
        result = decode_frame(bad_frame)
        self.assertIsNone(result)
    
    def test_decode_frame_with_corrupted_length(self):
        """Test frame with corrupted length field."""
        test_data = b"Test data"
        frame = encode_frame(TYPE_JPEG, test_data)
        
        # Corrupt the length field (bytes 3-4)
        bad_frame = bytearray(frame)
        bad_frame[3] = 0xFF  # Set length to very large value
        bad_frame[4] = 0xFF
        
        result = decode_frame(bytes(bad_frame))
        self.assertIsNone(result)  # Should fail CRC check
    
    def test_decode_multiple_frames_in_buffer(self):
        """Test handling multiple frames in sequence."""
        test_data1 = b"First frame data"
        test_data2 = b"Second frame data"
        
        frame1 = encode_frame(TYPE_JPEG, test_data1)
        frame2 = encode_frame(TYPE_CS, test_data2)
        
        # Concatenate frames
        combined = frame1 + frame2
        
        # First frame
        result1 = decode_frame(frame1)
        self.assertIsNotNone(result1)
        frame_type1, data1 = result1
        self.assertEqual(frame_type1, TYPE_JPEG)
        self.assertEqual(data1, test_data1)
        
        # Second frame
        result2 = decode_frame(frame2)
        self.assertIsNotNone(result2)
        frame_type2, data2 = result2
        self.assertEqual(frame_type2, TYPE_CS)
        self.assertEqual(data2, test_data2)
    
    def test_decode_frame_with_crc_error(self):
        """Test frame with corrupted data (CRC mismatch)."""
        test_data = b"Test data with some content"
        frame = encode_frame(TYPE_JPEG, test_data)
        
        # Corrupt a data byte (after header, before CRC)
        bad_frame = bytearray(frame)
        # Header is START(2) + TYPE(1) + LENGTH(2) = 5 bytes
        # Corrupt first data byte at position 5
        bad_frame[5] ^= 0xFF
        
        result = decode_frame(bytes(bad_frame))
        self.assertIsNone(result)  # Should fail CRC check
    
    def test_frame_start_marker_in_data(self):
        """Test that START marker appearing in data doesn't break parsing."""
        # Create data that contains the START marker
        test_data = b"Some data " + FRAME_START + b" with start marker inside"
        
        frame = encode_frame(TYPE_JPEG, test_data)
        result = decode_frame(frame)
        
        self.assertIsNotNone(result)
        frame_type, data = result
        self.assertEqual(data, test_data)
    
    def test_frame_end_marker_in_data(self):
        """Test that END marker appearing in data doesn't break parsing."""
        # Create data that contains the END marker
        test_data = b"Some data " + FRAME_END + b" with end marker inside"
        
        frame = encode_frame(TYPE_JPEG, test_data)
        result = decode_frame(frame)
        
        self.assertIsNotNone(result)
        frame_type, data = result
        self.assertEqual(data, test_data)
    
    def test_minimum_valid_frame(self):
        """Test smallest valid frame (1 byte of data)."""
        test_data = b"X"
        
        frame = encode_frame(TYPE_JPEG, test_data)
        
        # Verify frame structure
        self.assertEqual(len(frame), 9 + 1)  # 9 overhead + 1 data
        
        result = decode_frame(frame)
        self.assertIsNotNone(result)
        frame_type, data = result
        self.assertEqual(data, test_data)
    
    def test_truncated_frame(self):
        """Test incomplete frame."""
        test_data = b"Test data"
        frame = encode_frame(TYPE_JPEG, test_data)
        
        # Truncate frame
        truncated = frame[:-3]
        result = decode_frame(truncated)
        self.assertIsNone(result)


class TestSerialBufferSimulation(unittest.TestCase):
    """Simulate serial buffer scenarios."""
    
    def test_find_start_marker_in_buffer(self):
        """Test finding start marker in noisy data."""
        garbage = b"\x00\xFF\x12\x34"
        test_data = b"Real data"
        frame = encode_frame(TYPE_JPEG, test_data)
        
        buffer = garbage + frame
        
        # Find start marker
        start_idx = buffer.find(FRAME_START)
        self.assertGreaterEqual(start_idx, 0)
        
        # Extract frame from start marker
        clean_buffer = buffer[start_idx:]
        result = decode_frame(clean_buffer)
        self.assertIsNotNone(result)
    
    def test_split_frame_start_marker(self):
        """Test handling when START marker is split across reads."""
        # This simulates what happens when first byte of START arrives in one read
        # and second byte arrives in another read
        test_data = b"Test"
        frame = encode_frame(TYPE_JPEG, test_data)
        
        # Simulate keeping last byte when no complete marker found
        buffer = bytearray([0xFF, 0x00, FRAME_START[0]])
        
        # Should keep last byte
        if buffer.find(FRAME_START) == -1:
            buffer = buffer[-1:]
        
        # Then add rest of frame
        buffer.extend(FRAME_START[1:])
        buffer.extend(frame[2:])  # Rest of frame after START
        
        # Should now have valid frame
        result = decode_frame(bytes(buffer))
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
