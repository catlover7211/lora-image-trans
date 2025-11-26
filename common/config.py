"""Configuration settings for LoRa image transmission system."""

# Serial communication settings
BAUD_RATE = 115200
SERIAL_TIMEOUT = 1.0

# Protocol settings
FRAME_START = b'\xAA\x55'
FRAME_END = b'\x55\xAA'
TYPE_JPEG = 0x01
TYPE_CS = 0x02  # Compressed Sensing
TYPE_SSDV = 0x03  # SSDV (Slow Scan Digital Video)

# Image settings
DEFAULT_WIDTH = 16*10
DEFAULT_HEIGHT = 9*10
DEFAULT_JPEG_QUALITY = 85

# Compressed Sensing settings
CS_MEASUREMENT_RATE = 0.012  # 1% sampling rate
CS_BLOCK_SIZE = 16  # 16x16 pixel blocks

# Buffer settings
# Protocol LENGTH field is 2 bytes (uint16), so max payload is 65535 bytes.
MAX_FRAME_SIZE = 65535
CHUNK_SIZE = 500  # Bytes per chunk for LoRa transmission

# Flow control settings
INTER_FRAME_DELAY = 0  # Delay between frames in seconds (5ms) to prevent receiver buffer overflow

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

# SSDV settings
SSDV_CALLSIGN = "LORA01"  # Default callsign for SSDV packets
SSDV_IMAGE_ID = 0  # Starting image ID
SSDV_MAX_IMAGE_ID = 256  # Maximum image ID (wraps around)
SSDV_PACKET_DELAY = 0.15  # Delay between SSDV packets (150ms) for LoRa stability
SSDV_WIDTH = 16*120  # SSDV recommended resolution
SSDV_HEIGHT = 9*120
SSDV_QUALITY = 2  # SSDV quality level 0-7 (4 = balanced)
