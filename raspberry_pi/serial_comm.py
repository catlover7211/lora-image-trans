"""Serial communication module for Raspberry Pi sender.

Adjustable pacing between chunks and frames to tune throughput vs gap.
"""
import threading
import time
from typing import Optional

import serial
import serial.tools.list_ports

from common.config import BAUD_RATE, SERIAL_TIMEOUT, CHUNK_SIZE, INTER_FRAME_DELAY, FRAME_START, FRAME_END, MAX_FRAME_SIZE

BLOCKED_PORTS = {"/dev/cu.usbserial-10"}


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
        
        # Flow control and Reader state
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_running = False
        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._max_buffer = MAX_FRAME_SIZE * 2 + 4096
        
        # Flow control metrics
        self._backlog = 0
        self._lora_free = 0
        self._adaptive_delay = inter_frame_delay
        self._adaptive_chunk = chunk_size
    
    def find_port(self) -> Optional[str]:
        """Auto-detect available serial port.
        
        Returns:
            Port name or None if not found
        """
        ports = serial.tools.list_ports.comports()
        for port in ports:
            # Prefer USB serial devices
            if port.device in BLOCKED_PORTS:
                continue
            if 'USB' in port.description or 'ACM' in port.device or 'USB' in port.device:
                return port.device
        
        # Return first available port if no USB device found
        if ports:
            for port in ports:
                if port.device in BLOCKED_PORTS:
                    continue
                return port.device
        
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
            self._start_reader()
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
            with self._lock:
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
            if dynamic_delay > 0:
                time.sleep(dynamic_delay)
            
            return True
            
        except serial.SerialException as e:
            print(f"Error sending data: {e}")
            return False
    
    def close(self) -> None:
        """Close serial connection."""
        self._stop_reader()
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            self.ser = None

    def _start_reader(self) -> None:
        if self._reader_running:
            return
        self._reader_running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, name="SerialReader", daemon=True)
        self._reader_thread.start()

    def _stop_reader(self) -> None:
        if not self._reader_running:
            return
        self._reader_running = False
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=0.5)
            self._reader_thread = None

    def _reader_loop(self) -> None:
        """Continuously read from serial into buffer and process flow control."""
        while self._reader_running:
            if self.ser is None or not self.ser.is_open:
                time.sleep(0.1)
                continue
            try:
                n = self.ser.in_waiting
                if n > 0:
                    data = self.ser.read(n)
                    if data:
                        with self._lock:
                            self._buffer.extend(data)
                            # Trim if buffer grows too large
                            if len(self._buffer) > self._max_buffer:
                                tail = bytes(self._buffer[-2:])
                                self._buffer.clear()
                                self._buffer.extend(tail)
                        
                        # Process flow control lines immediately
                        self._process_flow_control()
                else:
                    time.sleep(0.001)
            except serial.SerialException:
                time.sleep(0.2)
            except Exception as e:
                print(f"Reader error: {e}")
                time.sleep(0.2)

    def _process_flow_control(self) -> None:
        """Extract and process [FC] lines from buffer."""
        with self._lock:
            # Look for [FC]...newline
            while True:
                try:
                    start_idx = self._buffer.find(b'[FC]')
                    if start_idx == -1:
                        break
                    
                    newline_idx = self._buffer.find(b'\n', start_idx)
                    if newline_idx == -1:
                        break
                    
                    # Extract line
                    line_bytes = self._buffer[start_idx:newline_idx+1]
                    
                    # Remove ONLY the FC line from buffer (preserve data before it)
                    del self._buffer[start_idx:newline_idx+1]
                    
                    # Process line
                    try:
                        text = line_bytes.decode('ascii', errors='ignore').strip()
                        self._handle_flow_line(text)
                    except Exception:
                        pass
                except Exception:
                    break

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

    def receive_frame(self) -> Optional[bytes]:
        """Receive a complete protocol frame using length-based assembly.

        Returns:
            Complete frame bytes or None if no complete frame available
        """
        if self.ser is None or not self.ser.is_open:
            return None

        try:
            with self._lock:
                buf = bytes(self._buffer)

            # Search for a start marker
            start_idx = buf.find(FRAME_START)
            if start_idx == -1:
                return None

            # Need at least START(2) + TYPE(1) + LEN(2)
            if len(buf) < start_idx + 5:
                return None

            # Parse length from header (big-endian)
            # buf[start_idx+3] is MSB, buf[start_idx+4] is LSB
            data_len = (buf[start_idx+3] << 8) | buf[start_idx+4]
            
            # Basic sanity check on length
            if data_len > MAX_FRAME_SIZE:
                # Invalid length, discard start marker and retry
                with self._lock:
                    if len(self._buffer) > start_idx:
                        del self._buffer[:start_idx+1]
                return None

            total_len = 2 + 1 + 2 + data_len + 2 + 2  # START + TYPE + LEN + DATA + CRC + END

            # Wait for full frame
            if len(buf) < start_idx + total_len:
                return None

            # Verify end marker
            end_idx = start_idx + total_len
            if buf[end_idx - 2: end_idx] != FRAME_END:
                # Corrupted frame, discard start marker
                with self._lock:
                    if len(self._buffer) > start_idx:
                        del self._buffer[:start_idx+1]
                return None

            # Extract frame
            frame = buf[start_idx:end_idx]
            
            # Remove from buffer
            with self._lock:
                del self._buffer[:end_idx]
                
            return frame

        except Exception as e:
            print(f"Error receiving frame: {e}")
            return None

    def get_flow_metrics(self) -> dict:
        with self._lock:
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
