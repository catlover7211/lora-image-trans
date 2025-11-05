"""Communication protocol for LoRa image transmission.

Frame format: START|TYPE|LENGTH|DATA|CRC|END
- START: 0xAA 0x55 (2 bytes)
- TYPE: Frame type (1 byte) - 0x01=JPEG, 0x02=CS
- LENGTH: Data length (2 bytes, big-endian)
- DATA: Image data (variable length)
- CRC: CRC16 checksum (2 bytes, big-endian)
- END: 0x55 0xAA (2 bytes)
"""
from __future__ import annotations

import struct
from typing import Optional, Tuple

from .config import FRAME_START, FRAME_END, TYPE_JPEG, TYPE_CS, MAX_FRAME_SIZE


def crc16(data: bytes) -> int:
    """Calculate CRC16-CCITT checksum."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc = crc << 1
        crc &= 0xFFFF
    return crc


def encode_frame(frame_type: int, data: bytes) -> bytes:
    """Encode data into a protocol frame.
    
    Args:
        frame_type: Type of frame (TYPE_JPEG or TYPE_CS)
        data: Image data to encode
        
    Returns:
        Encoded frame bytes
        
    Raises:
        ValueError: If data is too large
    """
    if len(data) > MAX_FRAME_SIZE:
        raise ValueError(f"Data size {len(data)} exceeds maximum {MAX_FRAME_SIZE}")
    
    # Pack: type (1 byte) + length (2 bytes big-endian) + data
    header = struct.pack('>BH', frame_type, len(data))
    frame_data = header + data
    
    # Calculate CRC
    crc = crc16(frame_data)
    crc_bytes = struct.pack('>H', crc)
    
    # Build complete frame
    frame = FRAME_START + frame_data + crc_bytes + FRAME_END
    return frame


def decode_frame(frame: bytes) -> Optional[Tuple[int, bytes]]:
    """Decode a protocol frame.
    
    Args:
        frame: Complete frame bytes
        
    Returns:
        Tuple of (frame_type, data) or None if invalid
    """
    # Check minimum frame size: START(2) + TYPE(1) + LENGTH(2) + CRC(2) + END(2) = 9 bytes
    if len(frame) < 9:
        return None
    
    # Check frame markers
    if not frame.startswith(FRAME_START):
        return None
    if not frame.endswith(FRAME_END):
        return None
    
    # Extract payload (remove START and END markers)
    payload_with_crc = frame[2:-2]
    
    # Extract CRC
    if len(payload_with_crc) < 2:
        return None
    crc_received = struct.unpack('>H', payload_with_crc[-2:])[0]
    payload = payload_with_crc[:-2]
    
    # Verify CRC
    crc_calculated = crc16(payload)
    if crc_received != crc_calculated:
        return None
    
    # Parse header
    if len(payload) < 3:
        return None
    frame_type, data_length = struct.unpack('>BH', payload[:3])
    
    # Extract data
    data = payload[3:]
    if len(data) != data_length:
        return None
    
    return frame_type, data


def get_frame_type_name(frame_type: int) -> str:
    """Get human-readable name for frame type."""
    if frame_type == TYPE_JPEG:
        return "JPEG"
    elif frame_type == TYPE_CS:
        return "CS"
    else:
        return f"Unknown(0x{frame_type:02X})"
