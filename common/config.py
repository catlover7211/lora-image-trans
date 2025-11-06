"""Configuration settings for LoRa image transmission system."""

# Serial communication settings
BAUD_RATE = 115200
SERIAL_TIMEOUT = 1.0

# Protocol settings
FRAME_START = b'\xAA\x55'
FRAME_END = b'\x55\xAA'
TYPE_JPEG = 0x01
TYPE_CS = 0x02  # Compressed Sensing

# Image settings
DEFAULT_WIDTH = 16*3
DEFAULT_HEIGHT = 9*3
DEFAULT_JPEG_QUALITY = 85

# Compressed Sensing settings
CS_MEASUREMENT_RATE = 0.1  # 10% sampling rate
CS_BLOCK_SIZE = 32  # 32x32 pixel blocks

# Buffer settings
MAX_FRAME_SIZE = 65535  # Maximum 2^16-1 bytes per frame
CHUNK_SIZE = 5000  # Bytes per chunk for LoRa transmission

# Flow control settings
INTER_FRAME_DELAY = 0.005  # Delay between frames in seconds (5ms) to prevent receiver buffer overflow

# Display settings
WINDOW_TITLE_SENDER = 'CCTV Sender (Press q to quit)'
WINDOW_TITLE_RECEIVER = 'CCTV Receiver (Press q to quit)'
