"""Main receiver application for PC.

This application receives encoded images via serial port from ESP32,
decodes them (JPEG or CS), and displays them.
"""
import argparse
import sys
import time
import threading
from pathlib import Path
import msvcrt

import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from jpeg_decoder import JPEGDecoder
from cs_decoder import CSDecoder
from serial_comm import SerialComm
from common.protocol import decode_frame, encode_frame, get_frame_type_name, TYPE_JPEG, TYPE_CS, TYPE_STOP
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
    print(f"PC 影像接收端 - {args.mode.upper()} 模式")
    print("=" * 60)
    
    # Initialize decoders
    jpeg_decoder = JPEGDecoder()
    cs_decoder = CSDecoder()
    
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
    
    print("=" * 60)
    print("系統初始化成功")
    
    if args.mode == MODE_PHOTO:
        print("等待照片中...")
        if args.save:
            print(f"將儲存至: {args.save}")
    else:
        print("等待幀中...")
    
    print("在顯示視窗按 'q' 或按 Ctrl+C 退出")
    print("在終端機按 's' 發送停止指令")
    print("=" * 60)
    
    # Photo mode: receive and display single image
    if args.mode == MODE_PHOTO:
        try:
            print("\n等待接收照片...")
            
            # Receive frame
            frame_bytes = None
            while frame_bytes is None:
                frame_bytes = serial_comm.receive_frame()
                time.sleep(0.01)  # 10ms delay to reduce CPU usage
            
            print(f"已接收 {len(frame_bytes)} bytes")
            
            # Decode protocol frame
            result = decode_frame(frame_bytes)
            if result is None:
                print("錯誤: 接收到無效的幀")
                serial_comm.close()
                return
            
            frame_type, data = result
            print(f"幀類型: {get_frame_type_name(frame_type)}")
            print(f"資料大小: {len(data)} bytes")
            
            # Decode image based on type
            image = None
            if frame_type == TYPE_JPEG:
                image = jpeg_decoder.decode(data)
            elif frame_type == TYPE_CS:
                image = cs_decoder.decode(data, iterations=args.gap_iters)
            else:
                print(f"錯誤: 未知的幀類型: {frame_type}")
                serial_comm.close()
                return
            
            if image is None:
                print(f"錯誤: 無法解碼 {get_frame_type_name(frame_type)} 影像")
                serial_comm.close()
                return
            
            print(f"照片解碼成功! 解析度: {image.shape[1]}x{image.shape[0]}")
            
            # Save if requested
            if args.save:
                cv2.imwrite(args.save, image)
                print(f"照片已儲存至: {args.save}")
            
            # Display image
            print("\n顯示照片中。按任意鍵關閉...")
            cv2.imshow(window_title, image)
            cv2.waitKey(0)
        
        except KeyboardInterrupt:
            print("\n\n使用者中斷")
        
        finally:
            # Cleanup
            print("\n正在清理資源...")
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
    
    # High-performance frame poller thread
    # This thread continuously pulls frames from serial_comm into memory
    # so the serial buffer never overflows while the main thread is busy decoding.
    class FramePoller:
        def __init__(self, comm):
            self.comm = comm
            self.latest_frame = None
            self.lock = threading.Lock()
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            
        def _loop(self):
            while self.running:
                frame = self.comm.receive_frame()
                if frame:
                    with self.lock:
                        self.latest_frame = frame
                else:
                    time.sleep(0.001)
                    
        def get_latest(self):
            with self.lock:
                frame = self.latest_frame
                self.latest_frame = None # Consume it
                return frame
                
        def stop(self):
            self.running = False
            self.thread.join(timeout=1.0)

    poller = FramePoller(serial_comm)

    try:
        while True:
            # Check CLI input
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                try:
                    char = ch.decode('utf-8').lower()
                except:
                    char = None
                
                if char == 's':
                    print("\n發送停止指令...")
                    stop_frame = encode_frame(TYPE_STOP, b'')
                    if serial_comm.ser:
                        serial_comm.ser.write(stop_frame)
                        serial_comm.ser.flush()
                elif char == 'q':
                    print("\n使用者請求退出")
                    break

            # Receive frame from poller (memory) instead of serial (IO)
            frame_bytes = poller.get_latest()
            
            if frame_bytes is None:
                # No complete frame yet, check display
                if last_display is not None:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n使用者請求退出")
                        break
                if args.debug_buffer:
                    buf_level = serial_comm.get_buffer_level()
                    usage = buf_level / max(1, serial_comm.get_buffer_capacity())
                    if usage > 0.7 and time.time() - last_buffer_warn > 0.5:
                        print(f"[SerialBuffer] 使用率={usage*100:.1f}% ({buf_level} bytes 已快取)")
                        last_buffer_warn = time.time()
                time.sleep(0.001)
                continue
            
            # Decode protocol frame
            result = decode_frame(frame_bytes)
            if result is None:
                error_count += 1
                invalid_frames += 1
                # More detailed error logging
                print(f"警告: 接收到無效的幀 (長度: {len(frame_bytes)} bytes)")
                if len(frame_bytes) >= 9:
                    # Debug: Print header and footer
                    header = frame_bytes[:5]
                    footer = frame_bytes[-2:]
                    print(f"  標頭: {header.hex().upper()}")
                    print(f"  結尾: {footer.hex().upper()}")
                    # Check CRC manually for debug
                    try:
                        from common.protocol import crc16
                        payload_with_crc = frame_bytes[2:-2]
                        if len(payload_with_crc) >= 2:
                            crc_received = (payload_with_crc[-2] << 8) | payload_with_crc[-1]
                            payload = payload_with_crc[:-2]
                            crc_calc = crc16(payload)
                            print(f"  CRC: 接收={crc_received:04X}, 計算={crc_calc:04X}")
                            if crc_received != crc_calc:
                                print("  -> CRC 不符")
                            
                            # Check length field
                            data_len = (frame_bytes[3] << 8) | frame_bytes[4]
                            real_data_len = len(payload) - 1 # minus TYPE
                            print(f"  長度欄位: {data_len}, 實際 payload: {real_data_len}")
                    except Exception as e:
                        print(f"  除錯錯誤: {e}")
                continue
            
            frame_type, data = result
            
            # Decode image based on type
            image = None
            if frame_type == TYPE_JPEG:
                image = jpeg_decoder.decode(data)
                jpeg_count += 1
            elif frame_type == TYPE_CS:
                if args.gap_iters > 0:
                    print(f"正在解碼 CS 幀 (GAP {args.gap_iters} 次迭代)...", end='\r', flush=True)
                image = cs_decoder.decode(data, iterations=args.gap_iters)
                if args.gap_iters > 0:
                    print(" " * 40, end='\r', flush=True) # Clear line
                cs_count += 1
            else:
                error_count += 1
                print(f"警告: 未知的幀類型: {frame_type}")
                continue
            
            if image is None:
                error_count += 1
                print(f"警告: 無法解碼 {get_frame_type_name(frame_type)} 影像")
                continue
            
            # Display image
            cv2.imshow(window_title, image)
            last_display = image
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n使用者請求退出")
                break
            
            # Update statistics
            frame_count += 1
            
            # Print status every 10 frames
            if frame_count % 10 == 0 or frame_count < 5:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                error_rate = (error_count / (frame_count + error_count) * 100) if (frame_count + error_count) > 0 else 0
                print(f"幀數: {frame_count} (JPEG: {jpeg_count}, CS: {cs_count}), "
                      f"資料大小: {len(data)} bytes, "
                      f"FPS: {fps:.2f}, "
                      f"錯誤: {error_count} ({error_rate:.1f}%)")
    
    except KeyboardInterrupt:
        print("\n\n使用者中斷")
    
    finally:
        # Cleanup
        print("\n正在清理資源...")
        if 'poller' in locals():
            poller.stop()
        serial_comm.close()
        cv2.destroyAllWindows()
        
        # Print final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        total_received = frame_count + error_count
        success_rate = (frame_count / total_received * 100) if total_received > 0 else 0
        print("\n" + "=" * 60)
        print("工作階段統計")
        print("=" * 60)
        print(f"總接收幀數: {total_received}")
        print(f"成功解碼: {frame_count}")
        print(f"JPEG 幀數: {jpeg_count}")
        print(f"CS 幀數: {cs_count}")
        print(f"總錯誤: {error_count}")
        print(f"無效幀: {invalid_frames}")
        print(f"成功率: {success_rate:.1f}%")
        print(f"總時間: {total_time:.2f} 秒")
        print(f"平均 FPS: {avg_fps:.2f}")
        print("=" * 60)


if __name__ == '__main__':
    main()
