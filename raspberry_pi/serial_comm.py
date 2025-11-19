"""Serial communication module for Raspberry Pi sender.

Adjustable pacing between chunks and frames to tune throughput vs gap.
"""
import threading
import time
from typing import Optional

import serial
import serial.tools.list_ports

from common.config import BAUD_RATE, SERIAL_TIMEOUT, CHUNK_SIZE, INTER_FRAME_DELAY


class SerialComm:
    """Handles serial communication with ESP32."""
    
    def __init__(self, port: Optional[str] = None, baud_rate: int = BAUD_RATE, 
                 timeout: float = SERIAL_TIMEOUT, chunk_size: int = CHUNK_SIZE,
                 inter_frame_delay: float = INTER_FRAME_DELAY,
                 chunk_delay_s: float = 0.003):
        """Initialize serial communication.
        
        Args:
            port: Serial port name (auto-detect if None)
            baud_rate: Baud rate, default 115200
            timeout: Read timeout in seconds
            chunk_size: Chunk size for transmission
            inter_frame_delay: Delay between frames in seconds
            chunk_delay_s: Fixed delay between chunks in seconds (0 for none)
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.inter_frame_delay = inter_frame_delay
        self.ser: Optional[serial.Serial] = None
        self.chunk_delay_s = max(0.0, float(chunk_delay_s))
        self._flow_thread: Optional[threading.Thread] = None
        self._flow_running = False
        self._flow_lock = threading.Lock()
        self._backlog = 0
        self._lora_free = 0
        self._adaptive_delay = inter_frame_delay
        self._adaptive_chunk = chunk_size
        self._last_flow_log = 0.0
    
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
            self._start_flow_monitor()
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
            with self._flow_lock:
                adjusted_chunk = self._adaptive_chunk if self._adaptive_chunk else self.chunk_size
                chunk_span = max(1, min(self.chunk_size, adjusted_chunk))
                dynamic_delay = max(0.0, self._adaptive_delay)

            # Send data in chunks
            for i in range(0, len(data), chunk_span):
                chunk = data[i:i + chunk_span]
                self.ser.write(chunk)
                # Only add optional gap between chunks if configured
                if i + chunk_span < len(data) and self.chunk_delay_s > 0:
                    time.sleep(self.chunk_delay_s)

            # Ensure all bytes are pushed after the frame
            self.ser.flush()
            
            # Add inter-frame delay to prevent receiver buffer overflow
            # This delay allows the receiver to process the frame before the next one arrives
            if dynamic_delay > 0:
                time.sleep(dynamic_delay)
            
            return True
            
        except serial.SerialException as e:
            print(f"Error sending data: {e}")
            return False
    
    def close(self) -> None:
        """Close serial connection."""
        self._stop_flow_monitor()
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            self.ser = None

    def _start_flow_monitor(self) -> None:
        if self._flow_running:
            return
        self._flow_running = True
        self._flow_thread = threading.Thread(target=self._flow_monitor_loop, name="SerialFlow", daemon=True)
        self._flow_thread.start()

    def _stop_flow_monitor(self) -> None:
        if not self._flow_running:
            return
        self._flow_running = False
        if self._flow_thread is not None:
            self._flow_thread.join(timeout=0.5)
            self._flow_thread = None

    def _flow_monitor_loop(self) -> None:
        while self._flow_running:
            if self.ser is None or not self.ser.is_open:
                time.sleep(0.1)
                continue
            try:
                line = self.ser.readline()
                if not line:
                    continue
                if line.startswith(b"[FC]"):
                    text = line.decode('ascii', errors='ignore')
                    self._handle_flow_line(text)
            except serial.SerialException:
                time.sleep(0.2)

    def _handle_flow_line(self, line: str) -> None:
        payload = line.split(']', 1)[-1]
        stats = {}
        for token in payload.split(','):
            if '=' not in token:
                continue
            key, value = token.split('=', 1)
            try:
                stats[key.strip()] = int(value.strip())
            except ValueError:
                continue
        backlog = stats.get('backlog', 0)
        lora_free = stats.get('loraFree', 0)
        with self._flow_lock:
            self._backlog = backlog
            self._lora_free = lora_free
            self._adaptive_chunk = self._compute_chunk_size(backlog, lora_free)
            self._adaptive_delay = self._compute_dynamic_delay(backlog)

    def _compute_dynamic_delay(self, backlog: int) -> float:
        base = self.inter_frame_delay
        if backlog > 3500:
            return base + 0.015
        if backlog > 2000:
            return base + 0.008
        if backlog < 200:
            return max(0.0, base - 0.002)
        return base

    def _compute_chunk_size(self, backlog: int, lora_free: int) -> int:
        chunk = self.chunk_size
        if backlog > 3500 or lora_free < 128:
            chunk = min(chunk, 256)
        elif backlog > 2000 or lora_free < 256:
            chunk = min(chunk, 384)
        elif backlog < 500 and lora_free > 512:
            chunk = min(max(chunk, 512), 768)
        return max(128, chunk)

    def get_flow_metrics(self) -> dict:
        with self._flow_lock:
            return {
                'backlog': self._backlog,
                'lora_free': self._lora_free,
                'chunk': self._adaptive_chunk,
                'delay': self._adaptive_delay,
            }
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
