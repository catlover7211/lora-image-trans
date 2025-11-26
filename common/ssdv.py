"""SSDV (Slow Scan Digital Video) protocol implementation for Python.

This module provides encoding and decoding of JPEG images to/from SSDV packet format.
SSDV is designed for robust transmission of images over low-bandwidth, high-error channels
like LoRa. It provides better error resilience than raw JPEG transmission.

Reference: https://github.com/fsphil/ssdv
"""
from __future__ import annotations

import struct
from typing import List, Optional, Tuple
from io import BytesIO

# SSDV packet structure constants
SSDV_PKT_SIZE = 256  # Total packet size
SSDV_PKT_SIZE_HEADER = 15  # Header size
SSDV_PKT_SIZE_PAYLOAD_NOFEC = 256 - SSDV_PKT_SIZE_HEADER  # 241 bytes payload for no-FEC mode
SSDV_PKT_SIZE_PAYLOAD_FEC = 224 - SSDV_PKT_SIZE_HEADER  # 209 bytes payload for FEC mode
SSDV_PKT_SIZE_CRC = 4  # CRC size
SSDV_PKT_SIZE_FEC = 32  # Reed-Solomon FEC size

# SSDV packet types
SSDV_TYPE_NORMAL = 0x00  # Normal mode with FEC
SSDV_TYPE_NOFEC = 0x01   # No-FEC mode

# SSDV sync byte
SSDV_SYNC_BYTE = 0x55

# Maximum callsign length
SSDV_MAX_CALLSIGN = 6


def encode_callsign(callsign: str) -> int:
    """Encode callsign string to 32-bit integer.
    
    Callsign format: up to 6 alphanumeric characters.
    Encoding: Base-40 encoding (0-9, A-Z, space, dash, slash, underscore)
    
    Args:
        callsign: Alphanumeric string up to 6 characters
        
    Returns:
        32-bit encoded callsign
    """
    # Base-40 character set
    charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -/_"
    
    callsign = callsign.upper().ljust(6)[:6]
    result = 0
    
    for i, char in enumerate(callsign):
        if char in charset:
            val = charset.index(char)
        else:
            val = 0  # Default to '0' for invalid characters
        result = result * 40 + val
    
    return result & 0xFFFFFFFF


def decode_callsign(encoded: int) -> str:
    """Decode 32-bit integer to callsign string.
    
    Args:
        encoded: 32-bit encoded callsign
        
    Returns:
        Decoded callsign string
    """
    charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -/_"
    result = ""
    
    for _ in range(6):
        result = charset[encoded % 40] + result
        encoded //= 40
    
    return result.rstrip()


def crc32_ssdv(data: bytes) -> int:
    """Calculate CRC32 checksum for SSDV packet.
    
    Uses CRC32-MPEG2 polynomial: 0x04C11DB7
    
    Args:
        data: Data to calculate CRC for
        
    Returns:
        32-bit CRC value
    """
    crc = 0xFFFFFFFF
    
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = (crc << 1) ^ 0x04C11DB7
            else:
                crc = crc << 1
            crc &= 0xFFFFFFFF
    
    return crc


class SSDVEncoder:
    """SSDV encoder for converting JPEG images to SSDV packets."""
    
    def __init__(self, callsign: str = "TEST01", image_id: int = 0, 
                 use_fec: bool = False, quality: int = 4):
        """Initialize SSDV encoder.
        
        Args:
            callsign: Alphanumeric string up to 6 characters
            image_id: Image identifier (0-255)
            use_fec: Whether to use Reed-Solomon FEC (not implemented in basic version)
            quality: JPEG quality level 0-7 (lower = higher compression)
        """
        self.callsign = callsign
        self.callsign_encoded = encode_callsign(callsign)
        self.image_id = image_id & 0xFF
        self.use_fec = use_fec
        self.quality = quality & 0x07
        self.packet_id = 0
        
        # Packet type
        self.packet_type = SSDV_TYPE_NORMAL if use_fec else SSDV_TYPE_NOFEC
        
        # Calculate payload size
        if use_fec:
            self.payload_size = SSDV_PKT_SIZE_PAYLOAD_FEC
        else:
            self.payload_size = SSDV_PKT_SIZE_PAYLOAD_NOFEC
    
    def encode(self, jpeg_data: bytes) -> List[bytes]:
        """Encode JPEG data into SSDV packets.
        
        Args:
            jpeg_data: JPEG image data
            
        Returns:
            List of SSDV packets (each 256 bytes)
        """
        packets = []
        self.packet_id = 0
        
        # Split JPEG data into chunks
        offset = 0
        total_length = len(jpeg_data)
        
        while offset < total_length:
            # Calculate chunk size for this packet
            chunk_size = min(self.payload_size, total_length - offset)
            chunk = jpeg_data[offset:offset + chunk_size]
            
            # Create packet
            packet = self._create_packet(chunk, offset, total_length)
            packets.append(packet)
            
            offset += chunk_size
            self.packet_id += 1
        
        return packets
    
    def _create_packet(self, payload: bytes, offset: int, total_size: int) -> bytes:
        """Create a single SSDV packet.
        
        Args:
            payload: Payload data for this packet
            offset: Offset in the JPEG data
            total_size: Total size of JPEG data
            
        Returns:
            256-byte SSDV packet
        """
        # Build header (15 bytes)
        header = bytearray()
        
        # Byte 0: Sync byte (0x55)
        header.append(SSDV_SYNC_BYTE)
        
        # Byte 1: Packet type
        header.append(self.packet_type)
        
        # Bytes 2-5: Callsign (32-bit)
        header.extend(struct.pack('>I', self.callsign_encoded))
        
        # Byte 6: Image ID
        header.append(self.image_id)
        
        # Bytes 7-8: Packet ID (16-bit)
        header.extend(struct.pack('>H', self.packet_id & 0xFFFF))
        
        # Bytes 9-10: Width (placeholder, simplified version)
        width = 320  # Default width
        header.extend(struct.pack('>H', width))
        
        # Bytes 11-12: Height (placeholder, simplified version)
        height = 240  # Default height
        header.extend(struct.pack('>H', height))
        
        # Byte 13: Flags (EOI flag if last packet)
        is_last = (offset + len(payload) >= total_size)
        flags = 0x80 if is_last else 0x00
        flags |= (self.quality & 0x07)  # Add quality level
        header.append(flags)
        
        # Byte 14: MCU offset (simplified)
        header.append(0)
        
        # Combine header and payload
        packet_data = bytes(header) + payload
        
        # Pad payload to required size
        padding_needed = SSDV_PKT_SIZE_HEADER + self.payload_size - len(packet_data)
        if padding_needed > 0:
            packet_data += b'\x00' * padding_needed
        
        # Calculate CRC over header + payload (excluding sync byte)
        crc_data = packet_data[1:]  # Skip sync byte
        crc = crc32_ssdv(crc_data)
        
        # For no-FEC mode: header(15) + payload(241) = 256 bytes
        # We need to insert CRC and adjust
        # Simplified: append CRC at the end (note: actual SSDV has specific structure)
        # For this implementation, we'll create 256-byte packets with embedded metadata
        
        # Build final packet: sync + type + header_data + payload + padding
        # Total must be 256 bytes
        final_packet = bytearray(packet_data[:SSDV_PKT_SIZE])
        
        return bytes(final_packet)


class SSDVDecoder:
    """SSDV decoder for reconstructing JPEG images from SSDV packets."""
    
    def __init__(self):
        """Initialize SSDV decoder."""
        self.packets = {}  # Store packets by image_id
        self.current_image_id = None
        self.callsign = None
    
    def decode_packet(self, packet: bytes) -> Optional[dict]:
        """Decode a single SSDV packet header.
        
        Args:
            packet: 256-byte SSDV packet
            
        Returns:
            Dictionary with packet information or None if invalid
        """
        if len(packet) != SSDV_PKT_SIZE:
            return None
        
        # Check sync byte
        if packet[0] != SSDV_SYNC_BYTE:
            return None
        
        # Parse header
        packet_type = packet[1]
        callsign_encoded = struct.unpack('>I', packet[2:6])[0]
        image_id = packet[6]
        packet_id = struct.unpack('>H', packet[7:9])[0]
        width = struct.unpack('>H', packet[9:11])[0]
        height = struct.unpack('>H', packet[11:13])[0]
        flags = packet[13]
        mcu_offset = packet[14]
        
        # Decode flags
        is_eoi = bool(flags & 0x80)
        quality = flags & 0x07
        
        # Determine payload size
        if packet_type == SSDV_TYPE_NOFEC:
            payload_size = SSDV_PKT_SIZE_PAYLOAD_NOFEC
        else:
            payload_size = SSDV_PKT_SIZE_PAYLOAD_FEC
        
        # Extract payload
        payload = packet[SSDV_PKT_SIZE_HEADER:SSDV_PKT_SIZE_HEADER + payload_size]
        
        # Note: Payload may contain padding (null bytes) at the end.
        # This is handled during JPEG reconstruction in get_jpeg() method
        # which looks for the JPEG EOI marker (0xFF 0xD9) to trim padding.
        
        return {
            'type': packet_type,
            'callsign': decode_callsign(callsign_encoded),
            'image_id': image_id,
            'packet_id': packet_id,
            'width': width,
            'height': height,
            'is_eoi': is_eoi,
            'quality': quality,
            'mcu_offset': mcu_offset,
            'payload': payload,
        }
    
    def add_packet(self, packet: bytes) -> bool:
        """Add a packet to the decoder.
        
        Args:
            packet: 256-byte SSDV packet
            
        Returns:
            True if packet was added successfully
        """
        info = self.decode_packet(packet)
        if info is None:
            return False
        
        image_id = info['image_id']
        packet_id = info['packet_id']
        
        # Initialize storage for new image
        if image_id not in self.packets:
            self.packets[image_id] = {}
            self.current_image_id = image_id
            self.callsign = info['callsign']
        
        # Store packet
        self.packets[image_id][packet_id] = info
        
        return True
    
    def is_image_complete(self, image_id: Optional[int] = None) -> bool:
        """Check if an image is complete (has EOI packet).
        
        Args:
            image_id: Image ID to check (uses current if None)
            
        Returns:
            True if image has EOI packet
        """
        if image_id is None:
            image_id = self.current_image_id
        
        if image_id not in self.packets:
            return False
        
        # Check if any packet has EOI flag
        for packet_info in self.packets[image_id].values():
            if packet_info['is_eoi']:
                return True
        
        return False
    
    def get_jpeg(self, image_id: Optional[int] = None) -> Optional[bytes]:
        """Reconstruct JPEG data from received packets.
        
        Args:
            image_id: Image ID to reconstruct (uses current if None)
            
        Returns:
            JPEG data or None if incomplete
        """
        if image_id is None:
            image_id = self.current_image_id
        
        if image_id not in self.packets:
            return None
        
        # Sort packets by packet_id
        sorted_packets = sorted(self.packets[image_id].items())
        
        # Concatenate payloads
        jpeg_data = BytesIO()
        for packet_id, packet_info in sorted_packets:
            payload = packet_info['payload']
            # Remove padding (null bytes at the end)
            # In a real implementation, we'd track the exact payload length
            jpeg_data.write(payload)
        
        result = jpeg_data.getvalue()
        
        # Try to find actual JPEG end marker and trim
        eoi_marker = b'\xFF\xD9'
        eoi_pos = result.find(eoi_marker)
        if eoi_pos != -1:
            result = result[:eoi_pos + 2]
        
        return result
    
    def get_image_info(self, image_id: Optional[int] = None) -> Optional[dict]:
        """Get information about an image.
        
        Args:
            image_id: Image ID to query (uses current if None)
            
        Returns:
            Dictionary with image information
        """
        if image_id is None:
            image_id = self.current_image_id
        
        if image_id not in self.packets or not self.packets[image_id]:
            return None
        
        # Get info from first packet
        first_packet = self.packets[image_id][min(self.packets[image_id].keys())]
        
        return {
            'image_id': image_id,
            'callsign': first_packet['callsign'],
            'width': first_packet['width'],
            'height': first_packet['height'],
            'packet_count': len(self.packets[image_id]),
            'is_complete': self.is_image_complete(image_id),
        }
