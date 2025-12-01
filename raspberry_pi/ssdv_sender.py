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
from common.protocol import encode_frame, decode_frame, TYPE_SSDV, TYPE_STOP
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
    print("SSDV 影像發送端 (含移動偵測)")
    print("=" * 70)
    print(f"解析度: {args.width}x{args.height}")
    print(f"呼號: {args.callsign}")
    print(f"SSDV 品質: {args.quality} (0=最高壓縮, 7=最低)")
    print(f"移動偵測模式: {args.motion_mode}")
    print(f"封包延遲: {args.packet_delay}s")
    
    # Initialize camera
    camera = CameraCapture(camera_index=args.camera, width=args.width, height=args.height)
    if not camera.open():
        print("無法開啟攝影機")
        return
    
    # Initialize motion detector or manual trigger
    if args.motion_mode == 'auto':
        detector = MotionDetector(
            threshold=args.motion_threshold,
            min_area=args.motion_area
        )
        print(f"移動偵測: 閾值={args.motion_threshold}, 最小面積={args.motion_area}")
    else:
        detector = ManualTrigger(trigger_interval=args.trigger_interval)
        print(f"手動觸發: 間隔={args.trigger_interval}s")
    
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
        print("無法開啟 Serial Port")
        camera.close()
        return
    
    print("=" * 70)
    print("系統初始化成功")
    if args.continuous:
        print("連續模式: 每次觸發都會擷取並傳輸")
    else:
        print("單次模式: 每次移動/觸發僅擷取一次")
    print("按 Ctrl+C 退出")
    print("=" * 70)
    
    image_count = 0
    last_capture_time = 0
    cooldown_period = 2.0  # Cooldown between captures in non-continuous mode
    
    try:
        while True:
            # Check for incoming commands (e.g. STOP)
            stop_requested = False
            while True:
                line = serial_comm.read_line()
                if not line:
                    break
                if not line.startswith("[FC]"):
                    print(f"收到: {line}")
                    if "STOP" in line.upper():
                        print("\n收到停止指令，正在停止發送...")
                        stop_requested = True
                        break
            if stop_requested:
                break

            # Capture frame
            frame = camera.capture()
            if frame is None:
                print("警告: 無法擷取畫面")
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
                        print("\n使用者請求退出")
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
                        print("\n使用者請求退出")
                        break
            
            # Apply cooldown in non-continuous mode
            if not args.continuous and (current_time - last_capture_time) < cooldown_period:
                should_capture = False
            
            if should_capture:
                print(f"\n{'='*70}")
                print(f"觸發擷取! (影像 #{image_count + 1})")
                print(f"{'='*70}")
                
                # Encode frame as JPEG first
                print("正在編碼 JPEG...")
                jpeg_data = jpeg_encoder.encode(frame)
                if jpeg_data is None:
                    print("錯誤: 無法編碼 JPEG")
                    continue
                
                print(f"JPEG 大小: {len(jpeg_data)} bytes")
                
                # Encode JPEG as SSDV packets
                print("正在建立 SSDV 封包...")
                ssdv_packets = ssdv_encoder.encode(jpeg_data)
                print(f"產生了 {len(ssdv_packets)} 個 SSDV 封包 (每個 256 bytes)")
                
                est_time = len(ssdv_packets) * args.packet_delay * 2
                print(f"預估傳輸時間: {est_time:.1f}s (延遲: {args.packet_delay}s/封包)")

                # Send each SSDV packet via protocol frames
                print("正在傳輸 SSDV 封包...")
                sent_packets = 0
                failed_packets = 0
                aborted = False
                
                for i, ssdv_packet in enumerate(ssdv_packets):
                    # Check for stop command
                    while True:
                        line = serial_comm.read_line()
                        if not line:
                            break
                        if not line.startswith("[FC]"):
                            print(f"收到: {line}")
                            if "stop" in line.lower():
                                print("\n收到停止指令，正在停止發送...")
                                aborted = True
                                break
                    if aborted:
                        break

                    # Wrap SSDV packet in protocol frame
                    protocol_frame = encode_frame(TYPE_SSDV, ssdv_packet)
                    
                    # Send via serial
                    if serial_comm.send(protocol_frame):
                        sent_packets += 1
                        if (i + 1) % 10 == 0 or (i + 1) == len(ssdv_packets):
                            print(f"  已發送封包 {i+1}/{len(ssdv_packets)}")
                    else:
                        failed_packets += 1
                        print(f"  警告: 無法發送封包 {i+1}")
                    
                    # Delay between packets
                    if args.packet_delay > 0 and i < len(ssdv_packets) - 1:
                        start_sleep = time.time()
                        while time.time() - start_sleep < args.packet_delay:
                            # Check for stop command during delay
                            while True:
                                line = serial_comm.read_line()
                                if not line:
                                    break
                                if not line.startswith("[FC]"):
                                    print(f"收到: {line}")
                                    if "stop" in line.lower():
                                        print("\n收到停止指令，正在停止發送...")
                                        aborted = True
                                        break
                            if aborted:
                                break
                            time.sleep(0.01)
                        if aborted:
                            break
                
                if aborted:
                    print(f"{'='*70}")
                    print(f"傳輸已中止 (已發送 {sent_packets} 封包)")
                    print(f"{'='*70}")
                else:
                    print(f"{'='*70}")
                    print(f"傳輸完成: {sent_packets} 成功, {failed_packets} 失敗")
                    print(f"總資料量: {len(ssdv_packets) * 256} bytes")
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
                print("正在清除攝影機緩衝區...")
                camera.flush()
            
            # Small delay to prevent CPU spinning
            time.sleep(0.03)  # ~30 FPS check rate
    
    except KeyboardInterrupt:
        print("\n\n使用者中斷")
    
    finally:
        # Cleanup
        print("\n正在清理資源...")
        camera.close()
        serial_comm.close()
        if args.preview:
            cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n" + "=" * 70)
        print("工作階段統計")
        print("=" * 70)
        print(f"總共擷取並傳輸的影像數: {image_count}")
        print("=" * 70)


if __name__ == '__main__':
    main()
