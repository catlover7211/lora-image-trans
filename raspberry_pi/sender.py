"""Main sender application for Raspberry Pi.

This application captures images from camera, encodes them using JPEG or 
Compressed Sensing (CS), and transmits via serial port to ESP32.
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
from jpeg_encoder import JPEGEncoder
from cs_encoder import CSEncoder
from serial_comm import SerialComm
from common.protocol import encode_frame, decode_frame, TYPE_JPEG, TYPE_CS, TYPE_STOP
from common.config import (
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_JPEG_QUALITY,
    WINDOW_TITLE_SENDER, WINDOW_TITLE_PHOTO_SENDER,
    CS_MEASUREMENT_RATE, CS_BLOCK_SIZE,
    INTER_FRAME_DELAY, MAX_FRAME_SIZE,
    MODE_CCTV, MODE_PHOTO, PHOTO_WIDTH, PHOTO_HEIGHT, PHOTO_JPEG_QUALITY
)

def _encode_jpeg_with_limit(encoder: JPEGEncoder, image, max_size: int, min_quality: int = 30):
    """Encode JPEG ensuring size <= max_size by reducing quality and then downscaling if needed.

    Returns bytes on success or None on failure.
    """
    try:
        base_quality = encoder.quality
        working = image
        quality = base_quality
        last_data = None

        while True:
            q = quality
            while q >= min_quality:
                encoder.set_quality(q)
                data = encoder.encode(working)
                if data is None:
                    return None
                last_data = data
                if len(data) <= max_size:
                    encoder.set_quality(base_quality)
                    return data
                q -= 5

            # Downscale by 10% and retry
            h, w = working.shape[:2]
            new_w = int(w * 0.9)
            new_h = int(h * 0.9)
            if new_w < 80 or new_h < 60:
                # Too small, give up and return last attempt
                encoder.set_quality(base_quality)
                return last_data
            working = cv2.resize(working, (new_w, new_h), interpolation=cv2.INTER_AREA)
            quality = base_quality
    except Exception as e:
        print(f"JPEG size limiting error: {e}")
        return None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Raspberry Pi Image Sender')
    parser.add_argument('--mode', type=str, choices=['cctv', 'photo'], default='cctv',
                        help='Operating mode: cctv (continuous video) or photo (single high-quality image) (default: cctv)')
    parser.add_argument('--port', type=str, help='Serial port (auto-detect if not specified)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, help='Image width (auto-selected based on mode if not specified)')
    parser.add_argument('--height', type=int, help='Image height (auto-selected based on mode if not specified)')
    parser.add_argument('--codec', type=str, choices=['jpeg', 'cs'], default='cs',
                        help='Encoding method: jpeg or cs (Compressed Sensing) (default: jpeg)')
    parser.add_argument('--jpeg-quality', type=int, help='JPEG quality 1-100 (auto-selected based on mode if not specified)')
    parser.add_argument('--cs-rate', type=float, default=CS_MEASUREMENT_RATE,
                        help=f'CS measurement rate 0.0-1.0 (default: {CS_MEASUREMENT_RATE})')
    parser.add_argument('--cs-block', type=int, default=CS_BLOCK_SIZE,
                        help=f'CS block size (default: {CS_BLOCK_SIZE})')
    parser.add_argument('--fps', type=float, default=5.0, help='Target FPS for CCTV mode (default: 10.0)')
    parser.add_argument('--inter-frame-delay', type=float, default=INTER_FRAME_DELAY,
                        help=f'Delay between frames in seconds to prevent receiver overflow (default: {INTER_FRAME_DELAY})')
    parser.add_argument('--chunk-delay-ms', type=float, default=0.0,
                        help='Delay between chunks in milliseconds (default: 0.0)')
    parser.add_argument('--preview', action='store_true', help='Show preview window')
    return parser.parse_args()


def cleanup_resources(camera, serial_comm, preview_enabled):
    """Clean up camera, serial, and display resources."""
    print("\n正在清理資源...")
    camera.close()
    serial_comm.close()
    if preview_enabled:
        cv2.destroyAllWindows()


def main():
    """Main application loop."""
    args = parse_args()
    
    # Set mode-specific defaults
    if args.mode == MODE_PHOTO:
        width = args.width if args.width else PHOTO_WIDTH
        height = args.height if args.height else PHOTO_HEIGHT
        jpeg_quality = args.jpeg_quality if args.jpeg_quality else PHOTO_JPEG_QUALITY
        window_title = WINDOW_TITLE_PHOTO_SENDER
    else:  # MODE_CCTV
        width = args.width if args.width else DEFAULT_WIDTH
        height = args.height if args.height else DEFAULT_HEIGHT
        jpeg_quality = args.jpeg_quality if args.jpeg_quality else DEFAULT_JPEG_QUALITY
        window_title = WINDOW_TITLE_SENDER
    
    print("=" * 60)
    print(f"Raspberry Pi 影像發送端 - {args.mode.upper()} 模式")
    print("=" * 60)
    print(f"編碼: {args.codec.upper()}")
    print(f"解析度: {width}x{height}")
    if args.mode == MODE_CCTV:
        print(f"目標 FPS: {args.fps}")
    
    # Initialize camera
    camera = CameraCapture(camera_index=args.camera, width=width, height=height)
    if not camera.open():
        print("無法開啟攝影機")
        return
    
    # Initialize encoder
    if args.codec == 'jpeg':
        encoder = JPEGEncoder(quality=jpeg_quality)
        frame_type = TYPE_JPEG
        print(f"JPEG 品質: {jpeg_quality}")
    else:  # cs
        encoder = CSEncoder(measurement_rate=args.cs_rate, block_size=args.cs_block)
        frame_type = TYPE_CS
        print(f"CS 採樣率: {args.cs_rate}")
        print(f"CS 區塊大小: {args.cs_block}")
    
    # Initialize serial communication
    serial_comm = SerialComm(
        port=args.port,
        inter_frame_delay=args.inter_frame_delay,
        chunk_delay_s=max(0.0, args.chunk_delay_ms / 1000.0)
    )
    if not serial_comm.open():
        print("無法開啟 Serial Port")
        camera.close()
        return
    
    print("=" * 60)
    print("系統初始化成功")
    if args.inter_frame_delay > 0:
        print(f"幀間延遲: {args.inter_frame_delay:.3f}s")
    if args.chunk_delay_ms > 0:
        print(f"區塊延遲: {args.chunk_delay_ms:.3f}ms")
    
    if args.mode == MODE_PHOTO:
        print("在預覽視窗按 'q' 或按 Ctrl+C 拍攝照片")
    else:
        print("在預覽視窗按 'q' 或按 Ctrl+C 退出")
    print("=" * 60)
    
    # Photo mode: capture and send single image
    if args.mode == MODE_PHOTO:
        try:
            # Wait for user to be ready if preview is enabled
            if args.preview:
                print("\n顯示攝影機預覽...")
                print("按 'q' 拍攝照片或按 Ctrl+C 取消")
                while True:
                    frame = camera.capture()
                    if frame is None:
                        print("警告: 無法擷取預覽畫面")
                        time.sleep(0.1)
                        continue
                    
                    cv2.imshow(window_title, frame)
                    key = cv2.waitKey(30) & 0xFF  # 30ms delay = ~33 FPS
                    if key == ord('q'):
                        break
            else:
                # Small delay to let camera stabilize
                time.sleep(0.5)
            
            # Capture the photo
            print("\n正在拍攝照片...")
            frame = camera.capture()
            if frame is None:
                print("錯誤: 無法拍攝照片")
                cleanup_resources(camera, serial_comm, args.preview)
                return
            
            # Show captured image if preview enabled
            if args.preview:
                cv2.imshow(window_title, frame)
                print("照片已拍攝! 按任意鍵發送...")
                cv2.waitKey(0)
            
            # Encode frame
            print("正在編碼照片...")
            encoded_data = encoder.encode(frame)
            if encoded_data is None:
                print("錯誤: 無法編碼照片")
                cleanup_resources(camera, serial_comm, args.preview)
                return
            # Ensure JPEG fits protocol limit
            if frame_type == TYPE_JPEG and len(encoded_data) > MAX_FRAME_SIZE:
                print(f"編碼後的 JPEG 太大 ({len(encoded_data)} bytes)。正在降低品質/大小以符合 {MAX_FRAME_SIZE}...")
                adjusted = _encode_jpeg_with_limit(encoder, frame, MAX_FRAME_SIZE)
                if adjusted is None or len(adjusted) > MAX_FRAME_SIZE:
                    print("錯誤: 無法將 JPEG 縮小至協議限制以下")
                    cleanup_resources(camera, serial_comm, args.preview)
                    return
                encoded_data = adjusted
            
            # Build protocol frame
            try:
                protocol_frame = encode_frame(frame_type, encoded_data)
            except ValueError as e:
                print(f"錯誤: 無法建立幀: {e}")
                cleanup_resources(camera, serial_comm, args.preview)
                return
            
            # Send via serial
            print(f"正在發送照片 ({len(encoded_data)} bytes 編碼, {len(protocol_frame)} bytes 總計)...")
            if not serial_comm.send(protocol_frame):
                print("錯誤: 無法發送照片")
            else:
                print("照片發送成功!")
        
        except KeyboardInterrupt:
            print("\n\n使用者取消")
        
        finally:
            # Cleanup
            cleanup_resources(camera, serial_comm, args.preview)
        
        return
    
    # CCTV mode: continuous capture (original behavior)
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
            
            # Check for incoming commands (e.g. STOP)
            incoming_frame = serial_comm.receive_frame()
            if incoming_frame:
                result = decode_frame(incoming_frame)
                if result:
                    frame_type_in, _ = result
                    if frame_type_in == TYPE_STOP:
                        print("\n收到停止指令，正在停止發送...")
                        break

            # Capture frame
            frame = camera.capture()
            if frame is None:
                error_count += 1
                print("警告: 無法擷取畫面")
                time.sleep(0.1)
                continue
            
            # Show preview if requested
            if args.preview:
                cv2.imshow(window_title, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n使用者請求退出")
                    break
            
            # Encode frame
            encoded_data = encoder.encode(frame)
            if encoded_data is None:
                error_count += 1
                print("警告: 無法編碼畫面")
                continue
            if frame_type == TYPE_JPEG and len(encoded_data) > MAX_FRAME_SIZE:
                adjusted = _encode_jpeg_with_limit(encoder, frame, MAX_FRAME_SIZE)
                if adjusted is None or len(adjusted) > MAX_FRAME_SIZE:
                    error_count += 1
                    print("警告: JPEG 超過協議限制且無法縮小")
                    continue
                encoded_data = adjusted
            
            # Build protocol frame
            try:
                protocol_frame = encode_frame(frame_type, encoded_data)
            except ValueError as e:
                error_count += 1
                print(f"警告: 無法建立幀: {e}")
                continue
            
            # Send via serial
            if not serial_comm.send(protocol_frame):
                error_count += 1
                print("警告: 無法發送幀")
                continue
            
            # Update statistics
            frame_count += 1
            last_frame_time = current_time
            
            # Print status every 10 frames
            if frame_count % 10 == 0:
                elapsed_total = current_time - start_time
                actual_fps = frame_count / elapsed_total if elapsed_total > 0 else 0
                print(f"幀數: {frame_count}, "
                      f"編碼大小: {len(encoded_data)} bytes, "
                      f"幀大小: {len(protocol_frame)} bytes, "
                      f"FPS: {actual_fps:.2f}, "
                      f"錯誤: {error_count}")
    
    except KeyboardInterrupt:
        print("\n\n使用者中斷")
    
    finally:
        # Cleanup
        cleanup_resources(camera, serial_comm, args.preview)
        
        # Print final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print("\n" + "=" * 60)
        print("工作階段統計")
        print("=" * 60)
        print(f"總幀數: {frame_count}")
        print(f"總錯誤: {error_count}")
        print(f"總時間: {total_time:.2f} 秒")
        print(f"平均 FPS: {avg_fps:.2f}")
        print("=" * 60)


if __name__ == '__main__':
    main()
