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
DEFAULT_WIDTH = 16*8
DEFAULT_HEIGHT = 9*8
DEFAULT_JPEG_QUALITY = 85

# Compressed Sensing settings
CS_MEASUREMENT_RATE = 0.008  # 1% sampling rate
CS_BLOCK_SIZE = 8  # 16x16 pixel blocks

# Buffer settings
# Protocol LENGTH field is 2 bytes (uint16), so max payload is 65535 bytes.
MAX_FRAME_SIZE = 65535
CHUNK_SIZE = 500  # Bytes per chunk for LoRa transmission

# Flow control settings
INTER_FRAME_DELAY = 0.005  # Delay between frames in seconds (5ms) to prevent receiver buffer overflow

# Mode settings
MODE_CCTV = 'cctv'  # Continuous video streaming mode
MODE_PHOTO = 'photo'  # Single high-quality photo mode

# Display settings
WINDOW_TITLE_SENDER = 'CCTV Sender (Press q to quit)'
WINDOW_TITLE_RECEIVER = 'CCTV Receiver (Press q to quit)'
WINDOW_TITLE_PHOTO_SENDER = 'Photo Sender'
WINDOW_TITLE_PHOTO_RECEIVER = 'Photo Receiver'

# Photo mode settings
PHOTO_WIDTH = 640  # Higher resolution for photo mode
PHOTO_HEIGHT = 480
PHOTO_JPEG_QUALITY = 95  # Higher quality for photo mode
