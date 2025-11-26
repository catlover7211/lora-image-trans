"""SSDV sender application for Raspberry Pi with motion detection.

This application captures images when motion is detected (or on trigger),
encodes them using SSDV format, and transmits via serial port to ESP32.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from camera_capture import CameraCapture
from motion_detector import MotionDetector, ManualTrigger
from jpeg_encoder import JPEGEncoder
from serial_comm import SerialComm
from common.protocol import encode_frame, TYPE_SSDV
from common.ssdv import SSDVEncoder
from common.config import (
    SSDV_WIDTH, SSDV_HEIGHT, SSDV_QUALITY, SSDV_CALLSIGN, 
    SSDV_IMAGE_ID, SSDV_PACKET_DELAY, SSDV_MAX_IMAGE_ID, PHOTO_JPEG_QUALITY
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SSDV Image Sender with Motion Detection')
    parser.add_argument('--port', type=str, help='Serial port (auto-detect if not specified)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=SSDV_WIDTH, 
                       help=f'Image width (default: {SSDV_WIDTH})')
    parser.add_argument('--height', type=int, default=SSDV_HEIGHT,
                       help=f'Image height (default: {SSDV_HEIGHT})')
    parser.add_argument('--callsign', type=str, default=SSDV_CALLSIGN,
                       help=f'SSDV callsign up to 6 chars (default: {SSDV_CALLSIGN})')
    parser.add_argument('--quality', type=int, default=SSDV_QUALITY,
                       help=f'SSDV quality level 0-7 (default: {SSDV_QUALITY})')
    parser.add_argument('--packet-delay', type=float, default=SSDV_PACKET_DELAY,
                       help=f'Delay between SSDV packets in seconds (default: {SSDV_PACKET_DELAY})')
    
    # Motion detection options
    parser.add_argument('--motion-mode', type=str, choices=['auto', 'manual'], default='manual',
                       help='Motion detection mode: auto (detect motion) or manual (timed trigger) (default: manual)')
    parser.add_argument('--motion-threshold', type=int, default=25,
                       help='Motion detection threshold 0-255 (default: 25)')
    parser.add_argument('--motion-area', type=int, default=500,
                       help='Minimum motion area in pixels (default: 500)')
    parser.add_argument('--trigger-interval', type=float, default=10.0,
                       help='Trigger interval in seconds for manual mode (default: 10.0)')
    
    parser.add_argument('--preview', action='store_true', help='Show preview window')
    parser.add_argument('--continuous', action='store_true', 
                       help='Continuous mode: capture and send every trigger without waiting')
    
    return parser.parse_args()


def main():
    """Main application loop."""
    args = parse_args()
    
    print("=" * 70)
    print("SSDV Image Sender with Motion Detection")
    print("=" * 70)
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Callsign: {args.callsign}")
    print(f"SSDV Quality: {args.quality} (0=highest compression, 7=lowest)")
    print(f"Motion Mode: {args.motion_mode}")
    print(f"Packet Delay: {args.packet_delay}s")
    
    # Initialize camera
    camera = CameraCapture(camera_index=args.camera, width=args.width, height=args.height)
    if not camera.open():
        print("Failed to open camera")
        return
    
    # Initialize motion detector or manual trigger
    if args.motion_mode == 'auto':
        detector = MotionDetector(
            threshold=args.motion_threshold,
            min_area=args.motion_area
        )
        print(f"Motion Detection: threshold={args.motion_threshold}, min_area={args.motion_area}")
    else:
        detector = ManualTrigger(trigger_interval=args.trigger_interval)
        print(f"Manual Trigger: interval={args.trigger_interval}s")
    
    # Initialize JPEG encoder (for creating JPEG before SSDV encoding)
    jpeg_encoder = JPEGEncoder(quality=PHOTO_JPEG_QUALITY)
    
    # Initialize SSDV encoder
    ssdv_encoder = SSDVEncoder(
        callsign=args.callsign,
        image_id=SSDV_IMAGE_ID,
        use_fec=False,  # FEC not implemented in basic version
        quality=args.quality
    )
    
    # Initialize serial communication
    serial_comm = SerialComm(port=args.port, inter_frame_delay=args.packet_delay)
    if not serial_comm.open():
        print("Failed to open serial port")
        camera.close()
        return
    
    print("=" * 70)
    print("System initialized successfully")
    if args.continuous:
        print("Continuous mode: Will capture and transmit on every trigger")
    else:
        print("Single-shot mode: Will capture once per motion/trigger")
    print("Press Ctrl+C to quit")
    print("=" * 70)
    
    image_count = 0
    last_capture_time = 0
    cooldown_period = 2.0  # Cooldown between captures in non-continuous mode
    
    try:
        while True:
            # Capture frame
            frame = camera.capture()
            if frame is None:
                print("Warning: Failed to capture frame")
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            
            # Check for motion/trigger
            if args.motion_mode == 'auto':
                motion_detected, motion_score = detector.detect(frame)
                
                # Show preview with motion visualization if enabled
                if args.preview:
                    display_frame = detector.draw_motion(frame)
                    cv2.imshow('SSDV Sender - Motion Detection', display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nQuit requested by user")
                        break
                
                should_capture = motion_detected
            else:  # manual mode
                should_capture = detector.should_trigger(current_time)
                
                # Show preview if enabled
                if args.preview:
                    display_frame = frame.copy()
                    # Add countdown timer
                    next_trigger = args.trigger_interval - (current_time - detector.last_trigger_time)
                    status_text = f"Next trigger in: {next_trigger:.1f}s"
                    cv2.putText(display_frame, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('SSDV Sender - Manual Trigger', display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nQuit requested by user")
                        break
            
            # Apply cooldown in non-continuous mode
            if not args.continuous and (current_time - last_capture_time) < cooldown_period:
                should_capture = False
            
            if should_capture:
                print(f"\n{'='*70}")
                print(f"Capture triggered! (Image #{image_count + 1})")
                print(f"{'='*70}")
                
                # Encode frame as JPEG first
                print("Encoding JPEG...")
                jpeg_data = jpeg_encoder.encode(frame)
                if jpeg_data is None:
                    print("Error: Failed to encode JPEG")
                    continue
                
                print(f"JPEG size: {len(jpeg_data)} bytes")
                
                # Encode JPEG as SSDV packets
                print("Creating SSDV packets...")
                ssdv_packets = ssdv_encoder.encode(jpeg_data)
                print(f"Generated {len(ssdv_packets)} SSDV packets (256 bytes each)")
                
                est_time = len(ssdv_packets) * args.packet_delay * 2
                print(f"Estimated transmission time: {est_time:.1f}s (Delay: {args.packet_delay}s/packet)")

                # Send each SSDV packet via protocol frames
                print("Transmitting SSDV packets...")
                sent_packets = 0
                failed_packets = 0
                
                for i, ssdv_packet in enumerate(ssdv_packets):
                    # Wrap SSDV packet in protocol frame
                    protocol_frame = encode_frame(TYPE_SSDV, ssdv_packet)
                    
                    # Send via serial
                    if serial_comm.send(protocol_frame):
                        sent_packets += 1
                        if (i + 1) % 10 == 0 or (i + 1) == len(ssdv_packets):
                            print(f"  Sent packet {i+1}/{len(ssdv_packets)}")
                    else:
                        failed_packets += 1
                        print(f"  Warning: Failed to send packet {i+1}")
                    
                    # Delay between packets
                    if args.packet_delay > 0 and i < len(ssdv_packets) - 1:
                        time.sleep(args.packet_delay)
                
                print(f"{'='*70}")
                print(f"Transmission complete: {sent_packets} sent, {failed_packets} failed")
                print(f"Total data: {len(ssdv_packets) * 256} bytes")
                print(f"{'='*70}")
                
                # Update counters
                image_count += 1
                last_capture_time = current_time
                
                # Increment image ID for next capture (wrap around at max)
                ssdv_encoder.image_id = (ssdv_encoder.image_id + 1) % SSDV_MAX_IMAGE_ID
                
                # In non-continuous mode, reset detector
                if not args.continuous and args.motion_mode == 'auto':
                    detector.reset()
                
                # Flush camera buffer to ensure next frame is fresh
                # This prevents processing old frames that accumulated during transmission
                print("Flushing camera buffer...")
                camera.flush()
            
            # Small delay to prevent CPU spinning
            time.sleep(0.03)  # ~30 FPS check rate
    
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
        print("\n" + "=" * 70)
        print("Session Statistics")
        print("=" * 70)
        print(f"Total images captured and transmitted: {image_count}")
        print("=" * 70)


if __name__ == '__main__':
    main()
