"""Serial communication module for PC receiver."""
import time
from typing import Optional

import serial
import serial.tools.list_ports

from common.config import BAUD_RATE, SERIAL_TIMEOUT, FRAME_START, FRAME_END, MAX_FRAME_SIZE


class SerialComm:
    """Handles serial communication with ESP32."""
    
    def __init__(self, port: Optional[str] = None, baud_rate: int = BAUD_RATE, 
                 timeout: float = SERIAL_TIMEOUT):
        """Initialize serial communication.
        
        Args:
            port: Serial port name (auto-detect if None)
            baud_rate: Baud rate, default 115200
            timeout: Read timeout in seconds
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self._buffer = bytearray()
    
    def receive_frame(self) -> Optional[bytes]:
        """Receive a complete protocol frame using length-based assembly.

        Returns:
            Complete frame bytes or None if no complete frame available
        """
        if self.ser is None or not self.ser.is_open:
            return None

        try:
            # Read available data
            if self.ser.in_waiting > 0:
                data = self.ser.read(self.ser.in_waiting)
                self._buffer.extend(data)

            # Keep trimming runaway buffers conservatively
            if len(self._buffer) > (MAX_FRAME_SIZE + 4096):
                # Preserve potential partial start marker
                tail = self._buffer[-2:]
                self._buffer.clear()
                self._buffer.extend(tail)

            # Search for a start marker
            start_idx = self._buffer.find(FRAME_START)
            if start_idx == -1:
                # Keep only last byte if it could be the start of FRAME_START
                if len(self._buffer) > 0 and self._buffer[-1:] == FRAME_START[:1]:
                    self._buffer = bytearray(self._buffer[-1:])
                else:
                    self._buffer.clear()
                return None

            # Drop any noise before the start marker
            if start_idx > 0:
                del self._buffer[:start_idx]

            # Need at least START(2) + TYPE(1) + LEN(2)
            if len(self._buffer) < 5:
                return None

            # Parse length from header (big-endian)
            data_len = (self._buffer[3] << 8) | self._buffer[4]
            if data_len == 0 or data_len > MAX_FRAME_SIZE:
                # Invalid length, shift by one and retry next time
                del self._buffer[0]
                return None

            total_len = 2 + 1 + 2 + data_len + 2 + 2  # START + TYPE + LEN + DATA + CRC + END

            # Wait for full frame
            if len(self._buffer) < total_len:
                return None

            # Verify end marker is at the expected position
            if self._buffer[total_len - 2: total_len] != FRAME_END:
                # Not aligned; drop first byte to resync (length is untrusted if corrupted)
                del self._buffer[0]
                return None

            # Extract the frame and remove from buffer
            frame = bytes(self._buffer[:total_len])
            del self._buffer[:total_len]
            return frame

        except serial.SerialException as e:
            print(f"Error receiving data: {e}")
            return None
    def receive_frame(self) -> Optional[bytes]:
        """Receive a complete protocol frame.
        
        Returns:
            Complete frame bytes or None if no complete frame available
        """
        if self.ser is None or not self.ser.is_open:
            return None
        
        try:
            # Read available data
            if self.ser.in_waiting > 0:
                data = self.ser.read(self.ser.in_waiting)
                self._buffer.extend(data)
            
            # Look for frame start marker
            start_idx = self._buffer.find(FRAME_START)
            if start_idx == -1:
                # No start marker found
                # Keep last byte in case it's part of a split start marker
                if len(self._buffer) > 1:
                    self._buffer = self._buffer[-1:]
                return None
            
            # Discard data before start marker
            if start_idx > 0:
                self._buffer = self._buffer[start_idx:]
            
            # Need at least minimum frame to proceed: START(2) + TYPE(1) + LENGTH(2) + CRC(2) + END(2) = 9
            if len(self._buffer) < 9:
                # Prevent buffer from growing too large while waiting
                if len(self._buffer) > 100000:
                    print("Warning: Buffer overflow while waiting for complete frame, resetting")
                    self._buffer.clear()
                return None
            
            # Look for frame end marker
            # Search after the header (START + TYPE + LENGTH = 5 bytes) to avoid false positives
            end_idx = self._buffer.find(FRAME_END, 5)
            if end_idx == -1:
                # No end marker yet, keep buffer and wait for more data
                # Prevent buffer from growing too large
                if len(self._buffer) > 100000:
                    print("Warning: Buffer overflow, resetting")
                    self._buffer.clear()
                return None
            
            # Extract complete frame
            frame = bytes(self._buffer[:end_idx + len(FRAME_END)])
            self._buffer = self._buffer[end_idx + len(FRAME_END):]
            
            return frame
            
        except serial.SerialException as e:
            print(f"Error receiving data: {e}")
            return None
    
    def close(self) -> None:
        """Close serial connection."""
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            self.ser = None
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
