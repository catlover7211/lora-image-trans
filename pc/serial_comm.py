"""Serial communication module for PC receiver.

High-FPS hardening: a background reader thread continuously drains the serial
port into a memory buffer, and the main thread parses complete frames out of
that buffer using a length-driven protocol with start/end validation.
"""
import threading
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
        self._lock = threading.Lock()
        self._rx_thread: Optional[threading.Thread] = None
        self._rx_running = False
        # Keep a generous buffer to absorb bursts: ~4 frames + margin
        self._max_buffer = MAX_FRAME_SIZE * 4 + 65536
    
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
            # Start background reader
            self._rx_running = True
            self._rx_thread = threading.Thread(target=self._reader_loop, name="SerialRX", daemon=True)
            self._rx_thread.start()
            return True
        except serial.SerialException as e:
            print(f"Error opening serial port {self.port}: {e}")
            return False

    def _reader_loop(self) -> None:
        """Continuously read from serial into the buffer."""
        assert self.ser is not None
        # Choose a moderate read size; in_waiting preferred when available
        while self._rx_running:
            try:
                n = self.ser.in_waiting if self.ser is not None else 0
                if n and n > 0:
                    data = self.ser.read(n)
                else:
                    # Small blocking read to reduce spin when quiet
                    data = self.ser.read(256)
                if data:
                    with self._lock:
                        self._buffer.extend(data)
                        # Trim if buffer grows too large
                        if len(self._buffer) > self._max_buffer:
                            # Keep last 2 bytes in case they contain partial start marker
                            tail = bytes(self._buffer[-2:])
                            self._buffer.clear()
                            self._buffer.extend(tail)
                else:
                    # No data arrived within timeout; yield
                    time.sleep(0.001)
            except serial.SerialException as e:
                # On read error, pause briefly but keep loop alive to allow recovery
                print(f"Serial read error: {e}")
                time.sleep(0.05)
    
    def receive_frame(self) -> Optional[bytes]:
        """Receive a complete protocol frame using length-based assembly.

        Returns:
            Complete frame bytes or None if no complete frame available
        """
        if self.ser is None or not self.ser.is_open:
            return None

        try:
            # Work on a snapshot to minimize lock time
            with self._lock:
                buf = bytes(self._buffer)

            # Keep trimming runaway buffers conservatively
            if len(buf) > (MAX_FRAME_SIZE + 4096):
                # Preserve potential partial start marker
                tail = buf[-2:]
                self._buffer.clear()
                self._buffer.extend(tail)
                return None

            # Search for a start marker
            start_idx = buf.find(FRAME_START)
            if start_idx == -1:
                # Keep only last byte if it could be the start of FRAME_START
                if len(buf) > 0 and buf[-1:] == FRAME_START[:1]:
                    with self._lock:
                        self._buffer = bytearray(buf[-1:])
                else:
                    with self._lock:
                        self._buffer.clear()
                return None

            # Drop any noise before the start marker
            if start_idx > 0:
                with self._lock:
                    del self._buffer[:start_idx]
                # Update snapshot after modification
                with self._lock:
                    buf = bytes(self._buffer)

            # Need at least START(2) + TYPE(1) + LEN(2)
            if len(buf) < 5:
                return None

            # Parse length from header (big-endian)
            data_len = (buf[3] << 8) | buf[4]
            if data_len == 0 or data_len > MAX_FRAME_SIZE:
                # Invalid length, shift by one and retry next time
                with self._lock:
                    del self._buffer[0]
                return None

            total_len = 2 + 1 + 2 + data_len + 2 + 2  # START + TYPE + LEN + DATA + CRC + END

            # Wait for full frame
            if len(buf) < total_len:
                return None

            # Verify end marker is at the expected position
            if buf[total_len - 2: total_len] != FRAME_END:
                # Not aligned; drop first byte to resync (length is untrusted if corrupted)
                with self._lock:
                    del self._buffer[0]
                return None

            # Extract the frame and remove from buffer
            frame = buf[:total_len]
            with self._lock:
                del self._buffer[:total_len]
            return frame

        except serial.SerialException as e:
            print(f"Error receiving data: {e}")
            return None
    
    def close(self) -> None:
        """Close serial connection."""
        if self.ser is not None and self.ser.is_open:
            # Stop background reader first
            self._rx_running = False
            if self._rx_thread is not None:
                self._rx_thread.join(timeout=0.5)
                self._rx_thread = None
            self.ser.close()
            self.ser = None
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
