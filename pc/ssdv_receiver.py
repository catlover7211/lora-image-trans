"""SSDV receiver application for PC.

This application receives SSDV packets via serial port, decodes them,
and displays/saves the reconstructed JPEG images.
"""
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from serial_comm import SerialComm
from jpeg_decoder import JPEGDecoder
from common.protocol import decode_frame, TYPE_SSDV, get_frame_type_name
from common.ssdv import SSDVDecoder


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SSDV Image Receiver')
    parser.add_argument('--port', type=str, help='Serial port (auto-detect if not specified)')
    parser.add_argument('--save-dir', type=str, default='ssdv_received',
                       help='Directory to save received images (default: ssdv_received)')
    parser.add_argument('--auto-save', action='store_true',
                       help='Automatically save complete images')
    parser.add_argument('--show-partial', action='store_true',
                       help='Display partial images as packets arrive')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed packet information')
    return parser.parse_args()


def main():
    """Main application loop."""
    args = parse_args()
    
    print("=" * 70)
    print("SSDV Image Receiver")
    print("=" * 70)
    
    # Create save directory if needed
    if args.auto_save:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Save directory: {save_dir.absolute()}")
    
    # Initialize serial communication
    serial_comm = SerialComm(port=args.port)
    if not serial_comm.open():
        print("Failed to open serial port")
        return
    
    # Initialize SSDV decoder
    ssdv_decoder = SSDVDecoder()
    jpeg_decoder = JPEGDecoder()
    
    print("=" * 70)
    print("System initialized successfully")
    print("Waiting for SSDV packets...")
    if args.show_partial:
        print("Partial image preview enabled")
    print("Press Ctrl+C to quit")
    print("=" * 70)
    
    # Statistics
    total_frames = 0
    ssdv_frames = 0
    other_frames = 0
    invalid_frames = 0
    packets_received = 0
    images_completed = 0
    
    last_image_id = None
    last_packet_count = 0
    
    try:
        while True:
            # Receive frame from serial
            frame_data = serial_comm.receive_frame()
            
            if frame_data is None:
                time.sleep(0.01)
                continue
            
            total_frames += 1
            
            # Decode protocol frame
            result = decode_frame(frame_data)
            if result is None:
                invalid_frames += 1
                if args.verbose:
                    print(f"Warning: Invalid frame (total invalid: {invalid_frames})")
                continue
            
            frame_type, payload = result
            
            # Check if it's an SSDV frame
            if frame_type != TYPE_SSDV:
                other_frames += 1
                if args.verbose:
                    print(f"Received {get_frame_type_name(frame_type)} frame (not SSDV)")
                continue
            
            ssdv_frames += 1
            
            # The payload should be a 256-byte SSDV packet
            if len(payload) != 256:
                if args.verbose:
                    print(f"Warning: SSDV packet has wrong size: {len(payload)} bytes (expected 256)")
                continue
            
            # Add packet to decoder
            if not ssdv_decoder.add_packet(payload):
                if args.verbose:
                    print(f"Warning: Failed to decode SSDV packet")
                continue
            
            packets_received += 1
            
            # Get image info
            image_info = ssdv_decoder.get_image_info()
            if image_info:
                current_image_id = image_info['image_id']
                current_packet_count = image_info['packet_count']
                
                # Print status when new image starts or every 10 packets
                if last_image_id != current_image_id or \
                   current_packet_count - last_packet_count >= 10:
                    print(f"\rImage {current_image_id} from {image_info['callsign']}: "
                          f"{current_packet_count} packets received "
                          f"({'COMPLETE' if image_info['is_complete'] else 'in progress'})...", 
                          end='', flush=True)
                    last_packet_count = current_packet_count
                
                last_image_id = current_image_id
                
                # Check if image is complete
                if image_info['is_complete']:
                    print(f"\n\n{'='*70}")
                    print(f"Image {current_image_id} complete!")
                    print(f"{'='*70}")
                    print(f"Callsign: {image_info['callsign']}")
                    print(f"Resolution: {image_info['width']}x{image_info['height']}")
                    print(f"Packets: {image_info['packet_count']}")
                    
                    # Reconstruct JPEG
                    jpeg_data = ssdv_decoder.get_jpeg(current_image_id)
                    if jpeg_data:
                        print(f"Reconstructed JPEG: {len(jpeg_data)} bytes")
                        
                        # Decode JPEG
                        image = jpeg_decoder.decode(jpeg_data)
                        if image is not None:
                            print(f"Image decoded successfully: {image.shape}")
                            
                            # Display image
                            window_name = f"SSDV Image {current_image_id}"
                            cv2.imshow(window_name, image)
                            print("Displaying image... Press any key to continue")
                            cv2.waitKey(0)
                            cv2.destroyWindow(window_name)
                            
                            # Save if requested
                            if args.auto_save:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"ssdv_{current_image_id}_{timestamp}.jpg"
                                filepath = Path(args.save_dir) / filename
                                cv2.imwrite(str(filepath), image)
                                print(f"Saved to: {filepath}")
                            
                            images_completed += 1
                        else:
                            print("Error: Failed to decode JPEG data")
                    else:
                        print("Error: Failed to reconstruct JPEG")
                    
                    print(f"{'='*70}\n")
                    
                    # Reset for next image
                    last_packet_count = 0
                
                # Show partial image if requested
                elif args.show_partial and current_packet_count > 0 and current_packet_count % 20 == 0:
                    jpeg_data = ssdv_decoder.get_jpeg(current_image_id)
                    if jpeg_data:
                        image = jpeg_decoder.decode(jpeg_data)
                        if image is not None:
                            window_name = f"SSDV Image {current_image_id} (Partial)"
                            cv2.imshow(window_name, image)
                            cv2.waitKey(1)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        serial_comm.close()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n" + "=" * 70)
        print("Session Statistics")
        print("=" * 70)
        print(f"Total frames received: {total_frames}")
        print(f"SSDV frames: {ssdv_frames}")
        print(f"Other frames: {other_frames}")
        print(f"Invalid frames: {invalid_frames}")
        print(f"SSDV packets decoded: {packets_received}")
        print(f"Images completed: {images_completed}")
        if ssdv_frames > 0:
            success_rate = (packets_received / ssdv_frames) * 100
            print(f"Packet decode success rate: {success_rate:.1f}%")
        print("=" * 70)


if __name__ == '__main__':
    main()
