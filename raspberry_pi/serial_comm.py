"""Serial communication module for Raspberry Pi sender."""
import time
from typing import Optional

import serial
import serial.tools.list_ports

from common.config import BAUD_RATE, SERIAL_TIMEOUT, CHUNK_SIZE, INTER_FRAME_DELAY


class SerialComm:
    """Handles serial communication with ESP32."""
    
    def __init__(self, port: Optional[str] = None, baud_rate: int = BAUD_RATE, 
                 timeout: float = SERIAL_TIMEOUT, chunk_size: int = CHUNK_SIZE,
                 inter_frame_delay: float = INTER_FRAME_DELAY):
        """Initialize serial communication.
        
        Args:
            port: Serial port name (auto-detect if None)
            baud_rate: Baud rate, default 115200
            timeout: Read timeout in seconds
            chunk_size: Chunk size for transmission
            inter_frame_delay: Delay between frames in seconds
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.inter_frame_delay = inter_frame_delay
        self.ser: Optional[serial.Serial] = None
    
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
                timeout=self.timeout,
                write_timeout=self.timeout
            )
            time.sleep(0.5)  # Wait for connection to stabilize
            print(f"Serial port opened: {self.port} @ {self.baud_rate} bps")
            return True
            
        except serial.SerialException as e:
            print(f"Error opening serial port {self.port}: {e}")
            return False
    
    def send(self, data: bytes) -> bool:
        """Send data via serial port.
        
        Args:
            data: Data bytes to send
            
        Returns:
            True if successful, False otherwise
        """
        if self.ser is None or not self.ser.is_open:
            return False
        
        try:
            # Send data in chunks
            for i in range(0, len(data), self.chunk_size):
                chunk = data[i:i + self.chunk_size]
                self.ser.write(chunk)
                # Use short pacing to avoid overrunning LoRa UART bridge
                self.ser.flush()
                if i + self.chunk_size < len(data):
                    time.sleep(0.003)
            
            # Add inter-frame delay to prevent receiver buffer overflow
            # This delay allows the receiver to process the frame before the next one arrives
            if self.inter_frame_delay > 0:
                time.sleep(self.inter_frame_delay)
            
            return True
            
        except serial.SerialException as e:
            print(f"Error sending data: {e}")
            return False
    
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
