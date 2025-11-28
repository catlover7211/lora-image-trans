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
    print("SSDV 影像接收端")
    print("=" * 70)
    
    # Create save directory if needed
    if args.auto_save:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"儲存目錄: {save_dir.absolute()}")
    
    # Initialize serial communication
    serial_comm = SerialComm(port=args.port)
    if not serial_comm.open():
        print("無法開啟 Serial Port")
        return
    
    # Autostart handshake
    print("\n" + "-" * 30)
    print("Raspberry Pi 自動啟動控制")
    print("-" * 30)
    print("1. 發送 'ssdv start' (啟動 SSDV 發送端)")
    print("2. 發送 'start' (啟動標準發送端)")
    print("3. 跳過 (不發送指令)")
    
    while True:
        choice = input("請選擇 (1-3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("無效的選擇，請重試。")
    
    if choice == '1':
        print("正在發送 'ssdv start'...")
        if serial_comm.ser:
            serial_comm.ser.write(b"ssdv start\n")
            serial_comm.ser.flush()
    elif choice == '2':
        print("正在發送 'start'...")
        if serial_comm.ser:
            serial_comm.ser.write(b"start\n")
            serial_comm.ser.flush()
    else:
        print("跳過發送啟動指令。")
    
    time.sleep(2) # Wait for Pi to start the script
    
    # Initialize SSDV decoder
    ssdv_decoder = SSDVDecoder()
    jpeg_decoder = JPEGDecoder()
    
    print("=" * 70)
    print("系統初始化成功")
    print("等待 SSDV 封包中...")
    if args.show_partial:
        print("已啟用部分影像預覽")
    print("按 Ctrl+C 退出")
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
                    print(f"警告: 無效的幀 (總無效數: {invalid_frames})")
                continue
            
            frame_type, payload = result
            
            # Check if it's an SSDV frame
            if frame_type != TYPE_SSDV:
                other_frames += 1
                if args.verbose:
                    print(f"接收到 {get_frame_type_name(frame_type)} 幀 (非 SSDV)")
                continue
            
            ssdv_frames += 1
            
            # The payload should be a 256-byte SSDV packet
            if len(payload) != 256:
                if args.verbose:
                    print(f"警告: SSDV 封包大小錯誤: {len(payload)} bytes (預期 256)")
                continue
            
            # Add packet to decoder
            if not ssdv_decoder.add_packet(payload):
                if args.verbose:
                    print(f"警告: 無法解碼 SSDV 封包")
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
                    print(f"\r影像 {current_image_id} 來自 {image_info['callsign']}: "
                          f"已接收 {current_packet_count} 個封包 "
                          f"({'完成' if image_info['is_complete'] else '進行中'})...", 
                          end='', flush=True)
                    last_packet_count = current_packet_count
                
                last_image_id = current_image_id
                
                # Check if image is complete
                if image_info['is_complete']:
                    print(f"\n\n{'='*70}")
                    print(f"影像 {current_image_id} 完成!")
                    print(f"{'='*70}")
                    print(f"呼號: {image_info['callsign']}")
                    print(f"解析度: {image_info['width']}x{image_info['height']}")
                    print(f"封包數: {image_info['packet_count']}")
                    
                    # Reconstruct JPEG
                    jpeg_data = ssdv_decoder.get_jpeg(current_image_id)
                    if jpeg_data:
                        print(f"重建 JPEG: {len(jpeg_data)} bytes")
                        
                        # Decode JPEG
                        image = jpeg_decoder.decode(jpeg_data)
                        if image is not None:
                            print(f"影像解碼成功: {image.shape}")
                            
                            # Display image
                            window_name = "SSDV Receiver"
                            cv2.imshow(window_name, image)
                            print("顯示影像中...")
                            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                                raise KeyboardInterrupt
                            
                            # Save if requested
                            if args.auto_save:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"ssdv_{current_image_id}_{timestamp}.jpg"
                                filepath = Path(args.save_dir) / filename
                                cv2.imwrite(str(filepath), image)
                                print(f"已儲存至: {filepath}")
                            
                            images_completed += 1
                        else:
                            print("錯誤: 無法解碼 JPEG 資料")
                    else:
                        print("錯誤: 無法重建 JPEG")
                    
                    print(f"{'='*70}\n")
                    
                    # Reset for next image
                    last_packet_count = 0
                
                # Show partial image if requested
                elif args.show_partial and current_packet_count > 0 and current_packet_count % 20 == 0:
                    jpeg_data = ssdv_decoder.get_jpeg(current_image_id)
                    if jpeg_data:
                        image = jpeg_decoder.decode(jpeg_data)
                        if image is not None:
                            window_name = "SSDV Receiver"
                            cv2.imshow(window_name, image)
                            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                                raise KeyboardInterrupt
    
    except KeyboardInterrupt:
        print("\n\n使用者中斷")
    
    finally:
        # Cleanup
        print("\n正在清理資源...")
        serial_comm.close()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n" + "=" * 70)
        print("工作階段統計")
        print("=" * 70)
        print(f"總接收幀數: {total_frames}")
        print(f"SSDV 幀數: {ssdv_frames}")
        print(f"其他幀數: {other_frames}")
        print(f"無效幀數: {invalid_frames}")
        print(f"已解碼 SSDV 封包: {packets_received}")
        print(f"已完成影像: {images_completed}")
        if ssdv_frames > 0:
            success_rate = (packets_received / ssdv_frames) * 100
            print(f"封包解碼成功率: {success_rate:.1f}%")
        print("=" * 70)


if __name__ == '__main__':
    main()
