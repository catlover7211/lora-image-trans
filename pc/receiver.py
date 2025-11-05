"""Main receiver application for PC.

This application receives encoded images via serial port from ESP32,
decodes them (JPEG or CS), and displays them.
"""
import argparse
import time
from typing import Optional

import cv2

from jpeg_decoder import JPEGDecoder
from cs_decoder import CSDecoder
from serial_comm import SerialComm
import sys
sys.path.insert(0, '..')
from common.protocol import decode_frame, get_frame_type_name, TYPE_JPEG, TYPE_CS
from common.config import WINDOW_TITLE_RECEIVER


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PC CCTV Receiver')
    parser.add_argument('--port', type=str, help='Serial port (auto-detect if not specified)')
    return parser.parse_args()


def main():
    """Main application loop."""
    args = parse_args()
    
    print("=" * 60)
    print("PC CCTV Receiver")
    print("=" * 60)
    
    # Initialize decoders
    jpeg_decoder = JPEGDecoder()
    cs_decoder = CSDecoder()
    
    # Initialize serial communication
    serial_comm = SerialComm(port=args.port)
    if not serial_comm.open():
        print("Failed to open serial port")
        return
    
    print("=" * 60)
    print("System initialized successfully")
    print("Waiting for frames...")
    print("Press 'q' in display window or Ctrl+C to quit")
    print("=" * 60)
    
    frame_count = 0
    error_count = 0
    jpeg_count = 0
    cs_count = 0
    start_time = time.time()
    last_display = None
    
    try:
        while True:
            # Receive frame
            frame_bytes = serial_comm.receive_frame()
            if frame_bytes is None:
                # No complete frame yet, check display
                if last_display is not None:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nQuit requested by user")
                        break
                time.sleep(0.001)
                continue
            
            # Decode protocol frame
            result = decode_frame(frame_bytes)
            if result is None:
                error_count += 1
                print("Warning: Invalid frame received")
                continue
            
            frame_type, data = result
            
            # Decode image based on type
            image = None
            if frame_type == TYPE_JPEG:
                image = jpeg_decoder.decode(data)
                jpeg_count += 1
            elif frame_type == TYPE_CS:
                image = cs_decoder.decode(data)
                cs_count += 1
            else:
                error_count += 1
                print(f"Warning: Unknown frame type: {frame_type}")
                continue
            
            if image is None:
                error_count += 1
                print(f"Warning: Failed to decode {get_frame_type_name(frame_type)} image")
                continue
            
            # Display image
            cv2.imshow(WINDOW_TITLE_RECEIVER, image)
            last_display = image
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuit requested by user")
                break
            
            # Update statistics
            frame_count += 1
            
            # Print status every 10 frames
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Frames: {frame_count} (JPEG: {jpeg_count}, CS: {cs_count}), "
                      f"Data size: {len(data)} bytes, "
                      f"FPS: {fps:.2f}, "
                      f"Errors: {error_count}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        serial_comm.close()
        cv2.destroyAllWindows()
        
        # Print final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print("\n" + "=" * 60)
        print("Session Statistics")
        print("=" * 60)
        print(f"Total frames: {frame_count}")
        print(f"JPEG frames: {jpeg_count}")
        print(f"CS frames: {cs_count}")
        print(f"Total errors: {error_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        print("=" * 60)


if __name__ == '__main__':
    main()
