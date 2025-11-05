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
CS_BLOCK_SIZE = 8

# Buffer settings
MAX_FRAME_SIZE = 700000  # Maximum 2^16-1 bytes per frame
CHUNK_SIZE = 220  # Bytes per chunk for LoRa transmission

# Display settings
WINDOW_TITLE_SENDER = 'CCTV Sender (Press q to quit)'
WINDOW_TITLE_RECEIVER = 'CCTV Receiver (Press q to quit)'
