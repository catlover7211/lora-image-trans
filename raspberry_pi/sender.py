"""Main sender application for Raspberry Pi.

This application captures images from camera, encodes them using JPEG or 
Compressed Sensing (CS), and transmits via serial port to ESP32.
"""
import argparse
import time
from typing import Optional

import cv2

from camera_capture import CameraCapture
from jpeg_encoder import JPEGEncoder
from cs_encoder import CSEncoder
from serial_comm import SerialComm
import sys
sys.path.insert(0, '..')
from common.protocol import encode_frame, TYPE_JPEG, TYPE_CS
from common.config import (
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_JPEG_QUALITY,
    WINDOW_TITLE_SENDER, CS_MEASUREMENT_RATE, CS_BLOCK_SIZE
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Raspberry Pi CCTV Sender')
    parser.add_argument('--port', type=str, help='Serial port (auto-detect if not specified)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH, help=f'Image width (default: {DEFAULT_WIDTH})')
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT, help=f'Image height (default: {DEFAULT_HEIGHT})')
    parser.add_argument('--codec', type=str, choices=['jpeg', 'cs'], default='jpeg',
                        help='Encoding method: jpeg or cs (Compressed Sensing) (default: jpeg)')
    parser.add_argument('--jpeg-quality', type=int, default=DEFAULT_JPEG_QUALITY,
                        help=f'JPEG quality 1-100 (default: {DEFAULT_JPEG_QUALITY})')
    parser.add_argument('--cs-rate', type=float, default=CS_MEASUREMENT_RATE,
                        help=f'CS measurement rate 0.0-1.0 (default: {CS_MEASUREMENT_RATE})')
    parser.add_argument('--cs-block', type=int, default=CS_BLOCK_SIZE,
                        help=f'CS block size (default: {CS_BLOCK_SIZE})')
    parser.add_argument('--fps', type=float, default=10.0, help='Target FPS (default: 10.0)')
    parser.add_argument('--preview', action='store_true', help='Show preview window')
    return parser.parse_args()


def main():
    """Main application loop."""
    args = parse_args()
    
    print("=" * 60)
    print("Raspberry Pi CCTV Sender")
    print("=" * 60)
    print(f"Codec: {args.codec.upper()}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Target FPS: {args.fps}")
    
    # Initialize camera
    camera = CameraCapture(camera_index=args.camera, width=args.width, height=args.height)
    if not camera.open():
        print("Failed to open camera")
        return
    
    # Initialize encoder
    if args.codec == 'jpeg':
        encoder = JPEGEncoder(quality=args.jpeg_quality)
        frame_type = TYPE_JPEG
        print(f"JPEG Quality: {args.jpeg_quality}")
    else:  # cs
        encoder = CSEncoder(measurement_rate=args.cs_rate, block_size=args.cs_block)
        frame_type = TYPE_CS
        print(f"CS Measurement Rate: {args.cs_rate}")
        print(f"CS Block Size: {args.cs_block}")
    
    # Initialize serial communication
    serial_comm = SerialComm(port=args.port)
    if not serial_comm.open():
        print("Failed to open serial port")
        camera.close()
        return
    
    print("=" * 60)
    print("System initialized successfully")
    print("Press 'q' in preview window or Ctrl+C to quit")
    print("=" * 60)
    
    frame_interval = 1.0 / args.fps if args.fps > 0 else 0.0
    last_frame_time = 0.0
    frame_count = 0
    error_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Control frame rate
            current_time = time.time()
            if frame_interval > 0:
                elapsed = current_time - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                    current_time = time.time()
            
            # Capture frame
            frame = camera.capture()
            if frame is None:
                error_count += 1
                print("Warning: Failed to capture frame")
                time.sleep(0.1)
                continue
            
            # Show preview if requested
            if args.preview:
                cv2.imshow(WINDOW_TITLE_SENDER, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuit requested by user")
                    break
            
            # Encode frame
            encoded_data = encoder.encode(frame)
            if encoded_data is None:
                error_count += 1
                print("Warning: Failed to encode frame")
                continue
            
            # Build protocol frame
            try:
                protocol_frame = encode_frame(frame_type, encoded_data)
            except ValueError as e:
                error_count += 1
                print(f"Warning: Failed to build frame: {e}")
                continue
            
            # Send via serial
            if not serial_comm.send(protocol_frame):
                error_count += 1
                print("Warning: Failed to send frame")
                continue
            
            # Update statistics
            frame_count += 1
            last_frame_time = current_time
            
            # Print status every 10 frames
            if frame_count % 10 == 0:
                elapsed_total = current_time - start_time
                actual_fps = frame_count / elapsed_total if elapsed_total > 0 else 0
                print(f"Frames: {frame_count}, "
                      f"Encoded size: {len(encoded_data)} bytes, "
                      f"Frame size: {len(protocol_frame)} bytes, "
                      f"FPS: {actual_fps:.2f}, "
                      f"Errors: {error_count}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        camera.close()
        serial_comm.close()
        if args.preview:
            cv2.destroyAllWindows()
        
        # Print final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print("\n" + "=" * 60)
        print("Session Statistics")
        print("=" * 60)
        print(f"Total frames: {frame_count}")
        print(f"Total errors: {error_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        print("=" * 60)


if __name__ == '__main__':
    main()
