"""Simple SSDV encoding and decoding example.

This example demonstrates how to:
1. Load a JPEG image
2. Encode it into SSDV packets
3. Decode SSDV packets back to JPEG
4. Display the result
"""
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from common.ssdv import SSDVEncoder, SSDVDecoder


def create_test_image(width=320, height=240):
    """Create a test image with some patterns."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some colorful patterns
    # Gradient background
    for y in range(height):
        for x in range(width):
            image[y, x] = [
                int(255 * x / width),
                int(255 * y / height),
                128
            ]
    
    # Add some shapes
    cv2.circle(image, (width // 2, height // 2), 50, (255, 0, 0), -1)
    cv2.rectangle(image, (50, 50), (150, 150), (0, 255, 0), 3)
    cv2.putText(image, "SSDV Test", (width // 2 - 80, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image


def main():
    """Main function."""
    print("=" * 70)
    print("SSDV Encoding and Decoding Example")
    print("=" * 70)
    
    # Create test image
    print("\n1. Creating test image...")
    image = create_test_image(320, 240)
    print(f"   Image size: {image.shape}")
    
    # Show original image
    cv2.imshow("Original Image", image)
    print("   Displaying original image... Press any key to continue")
    cv2.waitKey(0)
    cv2.destroyWindow("Original Image")
    
    # Encode to JPEG
    print("\n2. Encoding to JPEG...")
    ret, jpeg_buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    jpeg_data = jpeg_buffer.tobytes()
    print(f"   JPEG size: {len(jpeg_data)} bytes")
    
    # Encode to SSDV
    print("\n3. Encoding to SSDV packets...")
    encoder = SSDVEncoder(callsign="TEST01", image_id=0, quality=4)
    ssdv_packets = encoder.encode(jpeg_data)
    print(f"   Generated {len(ssdv_packets)} SSDV packets")
    print(f"   Total SSDV size: {len(ssdv_packets) * 256} bytes")
    print(f"   Overhead: {(len(ssdv_packets) * 256 - len(jpeg_data)) / len(jpeg_data) * 100:.1f}%")
    
    # Show packet info
    print("\n4. Packet information:")
    print(f"   Packet size: 256 bytes (fixed)")
    print(f"   First packet sync byte: 0x{ssdv_packets[0][0]:02X}")
    print(f"   First packet type: 0x{ssdv_packets[0][1]:02X}")
    
    # Decode SSDV packets
    print("\n5. Decoding SSDV packets...")
    decoder = SSDVDecoder()
    
    # Simulate packet reception
    packets_received = 0
    for i, packet in enumerate(ssdv_packets):
        if decoder.add_packet(packet):
            packets_received += 1
            if (i + 1) % 10 == 0 or (i + 1) == len(ssdv_packets):
                print(f"   Received {i+1}/{len(ssdv_packets)} packets")
    
    print(f"   Successfully decoded {packets_received} packets")
    
    # Get image info
    image_info = decoder.get_image_info()
    if image_info:
        print("\n6. Image information:")
        print(f"   Callsign: {image_info['callsign']}")
        print(f"   Image ID: {image_info['image_id']}")
        print(f"   Resolution: {image_info['width']}x{image_info['height']}")
        print(f"   Packet count: {image_info['packet_count']}")
        print(f"   Complete: {image_info['is_complete']}")
    
    # Reconstruct JPEG
    print("\n7. Reconstructing JPEG...")
    reconstructed_jpeg = decoder.get_jpeg()
    if reconstructed_jpeg:
        print(f"   Reconstructed JPEG size: {len(reconstructed_jpeg)} bytes")
        
        # Decode JPEG
        reconstructed_image = cv2.imdecode(
            np.frombuffer(reconstructed_jpeg, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        
        if reconstructed_image is not None:
            print(f"   Reconstructed image size: {reconstructed_image.shape}")
            
            # Show reconstructed image
            cv2.imshow("Reconstructed Image", reconstructed_image)
            print("\n   Displaying reconstructed image... Press any key to close")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Calculate similarity (simple MSE)
            if image.shape == reconstructed_image.shape:
                mse = np.mean((image.astype(float) - reconstructed_image.astype(float)) ** 2)
                print(f"\n8. Image quality metrics:")
                print(f"   MSE: {mse:.2f}")
                if mse < 100:
                    print(f"   Quality: Excellent (almost identical)")
                elif mse < 500:
                    print(f"   Quality: Good")
                elif mse < 1000:
                    print(f"   Quality: Fair")
                else:
                    print(f"   Quality: Poor")
        else:
            print("   Error: Failed to decode reconstructed JPEG")
    else:
        print("   Error: Failed to reconstruct JPEG")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
