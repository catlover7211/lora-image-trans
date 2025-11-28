import sys
import os
import time
import subprocess
import serial
import serial.tools.list_ports

# Add parent directory to path so we can import common modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import BAUD_RATE

def find_port():
    """Auto-detect available serial port."""
    ports = serial.tools.list_ports.comports()
    # Filter out blocked ports if any (copying logic from serial_comm.py)
    BLOCKED_PORTS = {"/dev/cu.usbserial-10"}
    
    for port in ports:
        if port.device in BLOCKED_PORTS:
            continue
        if 'USB' in port.description or 'ACM' in port.device or 'USB' in port.device:
            return port.device
            
    if ports:
        for port in ports:
            if port.device in BLOCKED_PORTS:
                continue
            return port.device
    return None

def main():
    print("自動啟動腳本正在執行...")
    while True:
        port = find_port()
        if not port:
            print("未找到 Serial Port。5 秒後重試...")
            time.sleep(5)
            continue
            
        print(f"正在監聽 {port}...")
        cmd_to_run = None
        
        try:
            with serial.Serial(port, BAUD_RATE, timeout=1) as ser:
                while True:
                    try:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            print(f"收到: {line}")
                            
                        if line == "ssdv start":
                            print("收到指令: ssdv start")
                            cmd_to_run = [sys.executable, os.path.join(os.path.dirname(__file__), "ssdv_sender.py"), "--port", port]
                            break 
                            
                        elif line == "start":
                            print("收到指令: start")
                            cmd_to_run = [sys.executable, os.path.join(os.path.dirname(__file__), "sender.py"), "--port", port]
                            break
                            
                    except serial.SerialException as e:
                        print(f"Serial 錯誤: {e}")
                        break
        except Exception as e:
            print(f"開啟 Serial Port 錯誤: {e}")
            time.sleep(5)
            continue

        if cmd_to_run:
            print(f"正在執行: {' '.join(cmd_to_run)}")
            try:
                # Run the sender program. This will block until the sender program exits.
                subprocess.run(cmd_to_run)
            except Exception as e:
                print(f"執行子程序錯誤: {e}")
            
            print("子程序結束。重新啟動監聽器...")
            time.sleep(2) # Short delay before restarting listener

if __name__ == "__main__":
    main()