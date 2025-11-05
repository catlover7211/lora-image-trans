"""
Simple example demonstrating JPEG encoding/decoding without hardware.

This script simulates the sender and receiver in a single program
for testing purposes.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from raspberry_pi.jpeg_encoder import JPEGEncoder
from pc.jpeg_decoder import JPEGDecoder
from common.protocol import encode_frame, decode_frame, TYPE_JPEG


def create_test_image(width=320, height=240):
    """Create a test image with gradient and text."""
    # Create gradient background
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            image[i, j] = [
                int(255 * i / height),           # Blue channel
                int(255 * j / width),            # Green channel
                int(255 * (i + j) / (height + width))  # Red channel
            ]
    
    # Add text
    cv2.putText(image, 'LoRa CCTV Test', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, 'JPEG Encoding', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add a circle
    cv2.circle(image, (width // 2, height // 2), 50, (0, 255, 255), 3)
    
    return image


def main():
    """Run the example."""
    print("=" * 60)
    print("LoRa CCTV JPEG Encoding/Decoding Example")
    print("=" * 60)
    
    # Create test image
    print("\n1. Creating test image...")
    original_image = create_test_image()
    print(f"   Image size: {original_image.shape[1]}x{original_image.shape[0]}")
    
    # Initialize encoder and decoder
    print("\n2. Initializing encoder and decoder...")
    encoder = JPEGEncoder(quality=85)
    decoder = JPEGDecoder()
    print("   JPEG quality: 85")
    
    # Encode image
    print("\n3. Encoding image to JPEG...")
    encoded_data = encoder.encode(original_image)
    if encoded_data is None:
        print("   ERROR: Failed to encode image")
        return
    print(f"   Encoded size: {len(encoded_data)} bytes")
    print(f"   Compression ratio: {original_image.size / len(encoded_data):.2f}:1")
    
    # Build protocol frame
    print("\n4. Building protocol frame...")
    protocol_frame = encode_frame(TYPE_JPEG, encoded_data)
    print(f"   Frame size: {len(protocol_frame)} bytes")
    print(f"   Overhead: {len(protocol_frame) - len(encoded_data)} bytes")
    
    # Simulate transmission (in real system, this would go through ESP32 and LoRa)
    print("\n5. Simulating transmission...")
    print("   (In real system: Raspberry Pi → ESP32 → LoRa → LoRa → ESP32 → PC)")
    received_frame = protocol_frame  # Simulated perfect transmission
    
    # Decode protocol frame
    print("\n6. Decoding protocol frame...")
    result = decode_frame(received_frame)
    if result is None:
        print("   ERROR: Failed to decode frame")
        return
    
    frame_type, decoded_data = result
    print(f"   Frame type: {'JPEG' if frame_type == TYPE_JPEG else 'Unknown'}")
    print(f"   Data size: {len(decoded_data)} bytes")
    
    # Verify data integrity
    if decoded_data == encoded_data:
        print("   ✓ Data integrity verified (CRC passed)")
    else:
        print("   ✗ Data integrity check failed")
        return
    
    # Decode JPEG image
    print("\n7. Decoding JPEG image...")
    decoded_image = decoder.decode(decoded_data)
    if decoded_image is None:
        print("   ERROR: Failed to decode JPEG")
        return
    print(f"   Decoded image size: {decoded_image.shape[1]}x{decoded_image.shape[0]}")
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    print("\n8. Calculating image quality metrics...")
    mse = np.mean((original_image.astype(float) - decoded_image.astype(float)) ** 2)
    if mse > 0:
        psnr = 10 * np.log10(255 ** 2 / mse)
        print(f"   PSNR: {psnr:.2f} dB")
    else:
        print("   PSNR: Infinite (identical images)")
    
    # Display images
    print("\n9. Displaying images...")
    print("   Press any key to close windows")
    
    # Create side-by-side comparison
    comparison = np.hstack([original_image, decoded_image])
    
    # Add labels
    label_original = comparison.copy()
    cv2.putText(label_original, 'Original', (10, 220), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(label_original, 'Decoded', (330, 220), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('LoRa CCTV - JPEG Encoding Example', label_original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
