"""Test protocol encoding and decoding."""
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.protocol import encode_frame, decode_frame, crc16, TYPE_JPEG, TYPE_CS
from common.config import FRAME_START, FRAME_END


class TestProtocol(unittest.TestCase):
    """Test protocol functions."""
    
    def test_crc16(self):
        """Test CRC16 calculation."""
        data = b"Hello, World!"
        crc = crc16(data)
        self.assertIsInstance(crc, int)
        self.assertGreaterEqual(crc, 0)
        self.assertLessEqual(crc, 0xFFFF)
        
        # Same data should give same CRC
        crc2 = crc16(data)
        self.assertEqual(crc, crc2)
        
        # Different data should (usually) give different CRC
        crc3 = crc16(b"Hello, World?")
        self.assertNotEqual(crc, crc3)
    
    def test_encode_decode_jpeg(self):
        """Test encoding and decoding JPEG frame."""
        test_data = b"This is test JPEG data"
        
        # Encode
        frame = encode_frame(TYPE_JPEG, test_data)
        
        # Check frame structure
        self.assertTrue(frame.startswith(FRAME_START))
        self.assertTrue(frame.endswith(FRAME_END))
        
        # Decode
        result = decode_frame(frame)
        self.assertIsNotNone(result)
        
        frame_type, data = result
        self.assertEqual(frame_type, TYPE_JPEG)
        self.assertEqual(data, test_data)
    
    def test_encode_decode_cs(self):
        """Test encoding and decoding CS frame."""
        test_data = b"This is test CS data with more bytes"
        
        # Encode
        frame = encode_frame(TYPE_CS, test_data)
        
        # Check frame structure
        self.assertTrue(frame.startswith(FRAME_START))
        self.assertTrue(frame.endswith(FRAME_END))
        
        # Decode
        result = decode_frame(frame)
        self.assertIsNotNone(result)
        
        frame_type, data = result
        self.assertEqual(frame_type, TYPE_CS)
        self.assertEqual(data, test_data)
    
    def test_decode_invalid_frame(self):
        """Test decoding invalid frames."""
        # Too short
        self.assertIsNone(decode_frame(b"short"))
        
        # Wrong start marker
        self.assertIsNone(decode_frame(b"\xFF\xFF\x01\x00\x05Hello\x00\x00\x55\xAA"))
        
        # Wrong end marker
        frame = encode_frame(TYPE_JPEG, b"test")
        bad_frame = frame[:-2] + b"\xFF\xFF"
        self.assertIsNone(decode_frame(bad_frame))
        
        # Corrupted CRC
        frame = encode_frame(TYPE_JPEG, b"test")
        # Modify a byte in the middle (should fail CRC)
        bad_frame = frame[:5] + bytes([frame[5] ^ 0xFF]) + frame[6:]
        self.assertIsNone(decode_frame(bad_frame))
    
    def test_large_data(self):
        """Test with larger data."""
        # Create 10KB of data
        test_data = b"x" * 10240
        
        # Encode
        frame = encode_frame(TYPE_JPEG, test_data)
        
        # Decode
        result = decode_frame(frame)
        self.assertIsNotNone(result)
        
        frame_type, data = result
        self.assertEqual(frame_type, TYPE_JPEG)
        self.assertEqual(len(data), len(test_data))
        self.assertEqual(data, test_data)
    
    def test_empty_data(self):
        """Test with empty data."""
        test_data = b""
        
        # Encode
        frame = encode_frame(TYPE_JPEG, test_data)
        
        # Decode - empty data should be rejected
        result = decode_frame(frame)
        self.assertIsNone(result)
    
    def test_max_size(self):
        """Test maximum frame size limit."""
        from common.config import MAX_FRAME_SIZE
        
        # Just under max size should work
        test_data = b"x" * (MAX_FRAME_SIZE - 100)
        frame = encode_frame(TYPE_JPEG, test_data)
        result = decode_frame(frame)
        self.assertIsNotNone(result)
        
        # Over max size should raise error
        test_data = b"x" * (MAX_FRAME_SIZE + 100)
        with self.assertRaises(ValueError):
            encode_frame(TYPE_JPEG, test_data)


if __name__ == '__main__':
    unittest.main()
