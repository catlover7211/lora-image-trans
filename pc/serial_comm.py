"""Serial communication module for PC receiver."""
import time
from typing import Optional

import serial
import serial.tools.list_ports

from common.config import BAUD_RATE, SERIAL_TIMEOUT, FRAME_START, FRAME_END


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
    
    def find_port(self) -> Optional[str]:
        """Auto-detect available serial port.
        
        Returns:
            Port name or None if not found
        """
        ports = serial.tools.list_ports.comports()
        for port in ports:
            # Prefer USB serial devices
            if 'USB' in port.description or 'ACM' in port.device or 'USB' in port.device:
                return port.device
        
        # Return first available port if no USB device found
        if ports:
            return ports[0].device
        
        return None
    
    def open(self) -> bool:
        """Open serial connection.
        
        Returns:
            True if successful, False otherwise
        """
        if self.port is None:
            self.port = self.find_port()
            if self.port is None:
                print("Error: No serial port found")
                return False
        
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            time.sleep(0.5)  # Wait for connection to stabilize
            # Flush any stale data
            self.ser.reset_input_buffer()
            print(f"Serial port opened: {self.port} @ {self.baud_rate} bps")
            return True
            
        except serial.SerialException as e:
            print(f"Error opening serial port {self.port}: {e}")
            return False
    
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
            end_idx = self._buffer.find(FRAME_END, 2)  # Start search after START marker
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
