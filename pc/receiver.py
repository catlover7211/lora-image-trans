"""Main receiver application for PC.

This application receives encoded images via serial port from ESP32,
decodes them (JPEG or CS), and displays them.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from jpeg_decoder import JPEGDecoder
from cs_decoder import CSDecoder
from serial_comm import SerialComm
from common.protocol import decode_frame, get_frame_type_name, TYPE_JPEG, TYPE_CS
from common.config import WINDOW_TITLE_RECEIVER, WINDOW_TITLE_PHOTO_RECEIVER, MODE_CCTV, MODE_PHOTO


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PC Image Receiver')
    parser.add_argument('--mode', type=str, choices=['cctv', 'photo'], default='cctv',
                        help='Operating mode: cctv (continuous video) or photo (single image) (default: cctv)')
    parser.add_argument('--port', type=str, help='Serial port (auto-detect if not specified)')
    parser.add_argument('--save', type=str, help='Save received photo to file (photo mode only)')
    parser.add_argument('--gap-iters', type=int, default=0,
                        help='Number of GAP reconstruction iterations for CS (default: 0)')
    parser.add_argument('--debug-buffer', action='store_true',
                        help='Print serial buffer usage when backlog grows (diagnostics)')
    return parser.parse_args()


def main():
    """Main application loop."""
    args = parse_args()
    
    window_title = WINDOW_TITLE_PHOTO_RECEIVER if args.mode == MODE_PHOTO else WINDOW_TITLE_RECEIVER
    
    print("=" * 60)
    print(f"PC Image Receiver - {args.mode.upper()} Mode")
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
    
    if args.mode == MODE_PHOTO:
        print("Waiting for photo...")
        if args.save:
            print(f"Will save to: {args.save}")
    else:
        print("Waiting for frames...")
    
    print("Press 'q' in display window or Ctrl+C to quit")
    print("=" * 60)
    
    # Photo mode: receive and display single image
    if args.mode == MODE_PHOTO:
        try:
            print("\nWaiting to receive photo...")
            
            # Receive frame
            frame_bytes = None
            while frame_bytes is None:
                frame_bytes = serial_comm.receive_frame()
                time.sleep(0.01)  # 10ms delay to reduce CPU usage
            
            print(f"Received {len(frame_bytes)} bytes")
            
            # Decode protocol frame
            result = decode_frame(frame_bytes)
            if result is None:
                print("Error: Invalid frame received")
                serial_comm.close()
                return
            
            frame_type, data = result
            print(f"Frame type: {get_frame_type_name(frame_type)}")
            print(f"Data size: {len(data)} bytes")
            
            # Decode image based on type
            image = None
            if frame_type == TYPE_JPEG:
                image = jpeg_decoder.decode(data)
            elif frame_type == TYPE_CS:
                image = cs_decoder.decode(data, iterations=args.gap_iters)
            else:
                print(f"Error: Unknown frame type: {frame_type}")
                serial_comm.close()
                return
            
            if image is None:
                print(f"Error: Failed to decode {get_frame_type_name(frame_type)} image")
                serial_comm.close()
                return
            
            print(f"Photo decoded successfully! Resolution: {image.shape[1]}x{image.shape[0]}")
            
            # Save if requested
            if args.save:
                cv2.imwrite(args.save, image)
                print(f"Photo saved to: {args.save}")
            
            # Display image
            print("\nDisplaying photo. Press any key to close...")
            cv2.imshow(window_title, image)
            cv2.waitKey(0)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            # Cleanup
            print("\nCleaning up...")
            serial_comm.close()
            cv2.destroyAllWindows()
        
        return
    
    # CCTV mode: continuous reception (original behavior)
    frame_count = 0
    error_count = 0
    jpeg_count = 0
    cs_count = 0
    crc_errors = 0
    invalid_frames = 0
    start_time = time.time()
    last_display = None
    last_buffer_warn = 0.0
    
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
                if args.debug_buffer:
                    buf_level = serial_comm.get_buffer_level()
                    usage = buf_level / max(1, serial_comm.get_buffer_capacity())
                    if usage > 0.7 and time.time() - last_buffer_warn > 0.5:
                        print(f"[SerialBuffer] usage={usage*100:.1f}% ({buf_level} bytes cached)")
                        last_buffer_warn = time.time()
                time.sleep(0.001)
                continue
            
            # Decode protocol frame
            result = decode_frame(frame_bytes)
            if result is None:
                error_count += 1
                invalid_frames += 1
                # More detailed error logging
                print(f"Warning: Invalid frame received (length: {len(frame_bytes)} bytes)")
                if len(frame_bytes) >= 9:
                    # Debug: Print header and footer
                    header = frame_bytes[:5]
                    footer = frame_bytes[-2:]
                    print(f"  Header: {header.hex().upper()}")
                    print(f"  Footer: {footer.hex().upper()}")
                    # Check CRC manually for debug
                    try:
                        from common.protocol import crc16
                        payload_with_crc = frame_bytes[2:-2]
                        if len(payload_with_crc) >= 2:
                            crc_received = (payload_with_crc[-2] << 8) | payload_with_crc[-1]
                            payload = payload_with_crc[:-2]
                            crc_calc = crc16(payload)
                            print(f"  CRC: Recv={crc_received:04X}, Calc={crc_calc:04X}")
                            if crc_received != crc_calc:
                                print("  -> CRC Mismatch")
                            
                            # Check length field
                            data_len = (frame_bytes[3] << 8) | frame_bytes[4]
                            real_data_len = len(payload) - 1 # minus TYPE
                            print(f"  Length field: {data_len}, Actual payload: {real_data_len}")
                    except Exception as e:
                        print(f"  Debug error: {e}")
                continue
            
            frame_type, data = result
            
            # Decode image based on type
            image = None
            if frame_type == TYPE_JPEG:
                image = jpeg_decoder.decode(data)
                jpeg_count += 1
            elif frame_type == TYPE_CS:
                if args.gap_iters > 0:
                    print(f"Decoding CS frame (GAP {args.gap_iters} iters)...", end='\r', flush=True)
                image = cs_decoder.decode(data, iterations=args.gap_iters)
                if args.gap_iters > 0:
                    print(" " * 40, end='\r', flush=True) # Clear line
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
            cv2.imshow(window_title, image)
            last_display = image
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuit requested by user")
                break
            
            # Update statistics
            frame_count += 1
            
            # Print status every 10 frames
            if frame_count % 10 == 0 or frame_count < 5:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                error_rate = (error_count / (frame_count + error_count) * 100) if (frame_count + error_count) > 0 else 0
                print(f"Frames: {frame_count} (JPEG: {jpeg_count}, CS: {cs_count}), "
                      f"Data size: {len(data)} bytes, "
                      f"FPS: {fps:.2f}, "
                      f"Errors: {error_count} ({error_rate:.1f}%)")
    
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
        total_received = frame_count + error_count
        success_rate = (frame_count / total_received * 100) if total_received > 0 else 0
        print("\n" + "=" * 60)
        print("Session Statistics")
        print("=" * 60)
        print(f"Total frames received: {total_received}")
        print(f"Successfully decoded: {frame_count}")
        print(f"JPEG frames: {jpeg_count}")
        print(f"CS frames: {cs_count}")
        print(f"Total errors: {error_count}")
        print(f"Invalid frames: {invalid_frames}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        print("=" * 60)


if __name__ == '__main__':
    main()
