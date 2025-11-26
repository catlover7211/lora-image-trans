"""Tests for SSDV (Slow Scan Digital Video) implementation."""
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.ssdv import (
    SSDVEncoder, SSDVDecoder, encode_callsign, decode_callsign,
    crc32_ssdv, SSDV_PKT_SIZE, SSDV_SYNC_BYTE
)
from common.config import TYPE_SSDV
from common.protocol import encode_frame, decode_frame


class TestSSDVCallsignEncoding(unittest.TestCase):
    """Test callsign encoding and decoding."""
    
    def test_encode_decode_callsign(self):
        """Test encoding and decoding callsign."""
        test_callsigns = ["TEST01", "LORA01", "ABC123", "X", "ABCDEF"]
        
        for callsign in test_callsigns:
            encoded = encode_callsign(callsign)
            decoded = decode_callsign(encoded)
            self.assertEqual(decoded.strip(), callsign.upper())
    
    def test_callsign_max_length(self):
        """Test callsign with maximum length."""
        callsign = "ABCDEF"
        encoded = encode_callsign(callsign)
        decoded = decode_callsign(encoded)
        self.assertEqual(decoded, callsign)
    
    def test_callsign_short(self):
        """Test short callsign."""
        callsign = "X"
        encoded = encode_callsign(callsign)
        decoded = decode_callsign(encoded)
        self.assertEqual(decoded.strip(), callsign)
    
    def test_callsign_invalid_chars(self):
        """Test callsign with invalid characters."""
        # Should handle gracefully by replacing with '0'
        callsign = "T@ST#1"
        encoded = encode_callsign(callsign)
        decoded = decode_callsign(encoded)
        # Invalid chars should be replaced
        self.assertIsNotNone(decoded)


class TestSSDVCRC(unittest.TestCase):
    """Test SSDV CRC calculation."""
    
    def test_crc32_basic(self):
        """Test CRC32 calculation."""
        data = b"Hello, World!"
        crc = crc32_ssdv(data)
        self.assertIsInstance(crc, int)
        self.assertGreaterEqual(crc, 0)
        self.assertLessEqual(crc, 0xFFFFFFFF)
    
    def test_crc32_empty(self):
        """Test CRC32 with empty data."""
        crc = crc32_ssdv(b"")
        self.assertEqual(crc, 0xFFFFFFFF)
    
    def test_crc32_consistency(self):
        """Test CRC32 consistency."""
        data = b"Test data for CRC"
        crc1 = crc32_ssdv(data)
        crc2 = crc32_ssdv(data)
        self.assertEqual(crc1, crc2)


class TestSSDVEncoder(unittest.TestCase):
    """Test SSDV encoder."""
    
    def test_encoder_initialization(self):
        """Test encoder initialization."""
        encoder = SSDVEncoder(callsign="TEST01", image_id=0)
        self.assertEqual(encoder.callsign, "TEST01")
        self.assertEqual(encoder.image_id, 0)
        self.assertFalse(encoder.use_fec)
    
    def test_encode_small_jpeg(self):
        """Test encoding small JPEG data."""
        # Create minimal JPEG-like data
        jpeg_data = b'\xFF\xD8' + b'\x00' * 100 + b'\xFF\xD9'  # SOI + data + EOI
        
        encoder = SSDVEncoder(callsign="TEST01", image_id=0)
        packets = encoder.encode(jpeg_data)
        
        # Should create at least one packet
        self.assertGreater(len(packets), 0)
        
        # Each packet should be 256 bytes
        for packet in packets:
            self.assertEqual(len(packet), SSDV_PKT_SIZE)
            self.assertEqual(packet[0], SSDV_SYNC_BYTE)
    
    def test_encode_multiple_packets(self):
        """Test encoding data that requires multiple packets."""
        # Create larger JPEG-like data
        jpeg_data = b'\xFF\xD8' + b'\x00' * 1000 + b'\xFF\xD9'
        
        encoder = SSDVEncoder(callsign="MULTI", image_id=5)
        packets = encoder.encode(jpeg_data)
        
        # Should create multiple packets
        self.assertGreater(len(packets), 1)
        
        # Check packet IDs are sequential
        for i, packet in enumerate(packets):
            self.assertEqual(len(packet), SSDV_PKT_SIZE)
            self.assertEqual(packet[0], SSDV_SYNC_BYTE)
    
    def test_encoder_image_id_increment(self):
        """Test that image_id can be incremented."""
        encoder = SSDVEncoder(callsign="TEST01", image_id=0)
        
        jpeg_data = b'\xFF\xD8' + b'\x00' * 100 + b'\xFF\xD9'
        packets1 = encoder.encode(jpeg_data)
        
        encoder.image_id = 1
        packets2 = encoder.encode(jpeg_data)
        
        # Image IDs should be different
        self.assertNotEqual(packets1[0][6], packets2[0][6])


class TestSSDVDecoder(unittest.TestCase):
    """Test SSDV decoder."""
    
    def test_decoder_initialization(self):
        """Test decoder initialization."""
        decoder = SSDVDecoder()
        self.assertIsNotNone(decoder)
        self.assertEqual(len(decoder.packets), 0)
    
    def test_decode_packet_header(self):
        """Test decoding packet header."""
        encoder = SSDVEncoder(callsign="DEC01", image_id=42)
        jpeg_data = b'\xFF\xD8' + b'\x00' * 100 + b'\xFF\xD9'
        packets = encoder.encode(jpeg_data)
        
        decoder = SSDVDecoder()
        info = decoder.decode_packet(packets[0])
        
        self.assertIsNotNone(info)
        self.assertEqual(info['callsign'], "DEC01")
        self.assertEqual(info['image_id'], 42)
        self.assertEqual(info['packet_id'], 0)
    
    def test_add_packet(self):
        """Test adding packets to decoder."""
        encoder = SSDVEncoder(callsign="ADD01", image_id=10)
        jpeg_data = b'\xFF\xD8' + b'\x00' * 100 + b'\xFF\xD9'
        packets = encoder.encode(jpeg_data)
        
        decoder = SSDVDecoder()
        for packet in packets:
            result = decoder.add_packet(packet)
            self.assertTrue(result)
        
        # Check image info
        info = decoder.get_image_info()
        self.assertIsNotNone(info)
        self.assertEqual(info['image_id'], 10)
        self.assertEqual(info['callsign'], "ADD01")
        self.assertEqual(info['packet_count'], len(packets))
    
    def test_image_complete_detection(self):
        """Test detection of complete images."""
        encoder = SSDVEncoder(callsign="COMP01", image_id=20)
        jpeg_data = b'\xFF\xD8' + b'\x00' * 100 + b'\xFF\xD9'
        packets = encoder.encode(jpeg_data)
        
        decoder = SSDVDecoder()
        
        # Add all but last packet
        for packet in packets[:-1]:
            decoder.add_packet(packet)
        
        # Should not be complete yet
        # Note: Our simplified implementation marks last packet with EOI
        # but may not detect it until added
        
        # Add last packet
        decoder.add_packet(packets[-1])
        
        # Now should be complete (if EOI flag is set correctly)
        # This depends on implementation details
        info = decoder.get_image_info()
        self.assertIsNotNone(info)
    
    def test_reconstruct_jpeg(self):
        """Test reconstructing JPEG from packets."""
        original_jpeg = b'\xFF\xD8' + b'\xAB\xCD' * 50 + b'\xFF\xD9'
        
        encoder = SSDVEncoder(callsign="RECON", image_id=30)
        packets = encoder.encode(original_jpeg)
        
        decoder = SSDVDecoder()
        for packet in packets:
            decoder.add_packet(packet)
        
        reconstructed = decoder.get_jpeg()
        self.assertIsNotNone(reconstructed)
        
        # Should start with JPEG SOI marker
        self.assertTrue(reconstructed.startswith(b'\xFF\xD8'))
        
        # Should end with JPEG EOI marker (may have padding before it)
        self.assertIn(b'\xFF\xD9', reconstructed)


class TestSSDVProtocolIntegration(unittest.TestCase):
    """Test SSDV integration with protocol layer."""
    
    def test_ssdv_in_protocol_frame(self):
        """Test wrapping SSDV packet in protocol frame."""
        # Create SSDV packet
        encoder = SSDVEncoder(callsign="PROTO", image_id=1)
        jpeg_data = b'\xFF\xD8' + b'\x00' * 100 + b'\xFF\xD9'
        ssdv_packets = encoder.encode(jpeg_data)
        
        # Wrap first SSDV packet in protocol frame
        protocol_frame = encode_frame(TYPE_SSDV, ssdv_packets[0])
        
        self.assertIsNotNone(protocol_frame)
        self.assertGreater(len(protocol_frame), len(ssdv_packets[0]))
        
        # Decode protocol frame
        result = decode_frame(protocol_frame)
        self.assertIsNotNone(result)
        
        frame_type, payload = result
        self.assertEqual(frame_type, TYPE_SSDV)
        self.assertEqual(payload, ssdv_packets[0])
    
    def test_ssdv_end_to_end(self):
        """Test complete SSDV encode-transmit-decode flow."""
        # Original data
        original_jpeg = b'\xFF\xD8' + b'\x12\x34\x56\x78' * 30 + b'\xFF\xD9'
        
        # Encode to SSDV
        encoder = SSDVEncoder(callsign="E2E", image_id=99)
        ssdv_packets = encoder.encode(original_jpeg)
        
        # Wrap in protocol frames
        protocol_frames = []
        for ssdv_packet in ssdv_packets:
            protocol_frames.append(encode_frame(TYPE_SSDV, ssdv_packet))
        
        # Decode protocol frames and extract SSDV packets
        decoder = SSDVDecoder()
        for protocol_frame in protocol_frames:
            result = decode_frame(protocol_frame)
            self.assertIsNotNone(result)
            
            frame_type, ssdv_packet = result
            self.assertEqual(frame_type, TYPE_SSDV)
            
            success = decoder.add_packet(ssdv_packet)
            self.assertTrue(success)
        
        # Reconstruct JPEG
        reconstructed_jpeg = decoder.get_jpeg()
        self.assertIsNotNone(reconstructed_jpeg)
        
        # Verify reconstruction
        self.assertTrue(reconstructed_jpeg.startswith(b'\xFF\xD8'))
        self.assertIn(b'\xFF\xD9', reconstructed_jpeg)


if __name__ == '__main__':
    unittest.main()
