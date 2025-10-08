import cv2
import serial
import time
import struct
import platform
import glob
import numpy as np
import zlib  # --- 新增：用於CRC32 ---

# --- 設定 ---
BAUD_RATE = 115200
MAX_IMAGE_SIZE = 1920*1080 # 設定一個合理的影像大小上限 (例如 100KB)
# --- 結束設定 ---

# --- 新增：通訊協定標記 (更穩健的版本) ---
START_OF_FRAME = b'\x01'  # 使用 SOH (Start of Header) 控制字元
END_OF_FRAME   = b'\x04'  # 使用 EOT (End of Transmission) 控制字元
ESC            = b'\x1B'  # 使用 ESC (Escape) 控制字元
# --- 結束設定 ---


def unstuff_data(stuffed_data):
    """
    對資料進行反向位元組填充 (Unstuffing)。
    """
    unstuffed = bytearray()
    i = 0
    while i < len(stuffed_data):
        if stuffed_data[i] == ESC[0]:
            # 如果 ESC 是最後一個字元，這是一個錯誤，但我們還是處理一下
            if i + 1 < len(stuffed_data):
                unstuffed.append(stuffed_data[i+1])
                i += 2
            else:
                # 忽略結尾的單獨 ESC
                i += 1
        else:
            unstuffed.append(stuffed_data[i])
            i += 1
    return bytes(unstuffed)

def get_serial_port():
    """自動偵測並返回可用的序列埠"""
    # 在不同作業系統下序列埠名稱的常見模式
    if platform.system() == "Windows":
        # Windows 上的序列埠通常是 COMx
        ports = [f'COM{i}' for i in range(1, 256)]
    elif platform.system() == "Linux":
        # Linux 上的序列埠通常是 /dev/ttyUSBx 或 /dev/ttyACMx
        ports = [f'/dev/ttyUSB{i}' for i in range(8)] + [f'/dev/ttyACM{i}' for i in range(8)]
    elif platform.system() == "Darwin": # macOS
        # macOS 上的序列埠通常是 /dev/tty.usbserial-xxxx 或 /dev/tty.usbmodemxxxx
        # 使用 glob 來尋找可能的埠
        ports = glob.glob('/dev/tty.usbserial*') + glob.glob('/dev/tty.usbmodem*')
    else:
        return None

    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            print(f"找到可用的序列埠: {port}")
            return port
        except (OSError, serial.SerialException):
            pass
    return None

def main():
    """
    主函式，用於從序列埠接收、解碼並顯示影像。
    """
    serial_port = get_serial_port()
    if not serial_port:
        print("錯誤: 找不到任何可用的序列埠。")
        print("請確認您的微控制器(Arduino/ESP32)已連接。")
        return

    try:
        # 初始化序列埠
        # 將預設 timeout 設為一個較短的值，避免在沒有資料時卡住太久
        ser = serial.Serial(serial_port, BAUD_RATE, timeout=0.1)
        print(f"成功打開序列埠: {serial_port} @ {BAUD_RATE} bps")
    except serial.SerialException as e:
        print(f"錯誤: 無法打開序列埠 {serial_port}。")
        print(f"詳細資訊: {e}")
        return

    print("已連接序列埠，開始等待接收影像...")
    print("按下 'q' 鍵或 Ctrl+C 停止程式。")

    try:
        while True:
            # --- 修改：使用新的通訊協定 ---
            # 1. 尋找幀的開始標記 (SOF)
            ser.read_until(START_OF_FRAME)

            # 2. 讀取 8 位元組的標頭 (填充大小 + CRC)
            header = ser.read(8)
            if len(header) < 8:
                continue

            # 3. 解碼標頭以取得填充大小和CRC
            try:
                stuffed_size, received_crc = struct.unpack('>II', header)
                
                if stuffed_size > MAX_IMAGE_SIZE:
                    print(f"錯誤: 偵測到不合理的填充大小 ({stuffed_size} bytes)，重新尋找 SOF...")
                    continue

            except struct.error:
                print("錯誤: 標頭解碼失敗，重新尋找 SOF...")
                continue
            
            # --- 修改：分段讀取填充資料，直到滿stuffed_size ---
            # 暫時禁用超時，以確保讀取完整
            original_timeout = ser.timeout
            ser.timeout = None 
            stuffed_img_data = bytearray()
            bytes_read = 0
            CHUNK_SIZE = 128  # 與發送端一致
            while bytes_read < stuffed_size:
                remaining = stuffed_size - bytes_read
                chunk_size = min(CHUNK_SIZE, remaining)
                chunk = ser.read(chunk_size)
                if len(chunk) < chunk_size:
                    print(f"錯誤: 讀取不完整，預期 {chunk_size} bytes, 收到 {len(chunk)} bytes")
                    break
                stuffed_img_data.extend(chunk)
                bytes_read += len(chunk)
            ser.timeout = original_timeout  # 恢復超時

            if bytes_read != stuffed_size:
                print(f"錯誤: 填充資料不完整。預期 {stuffed_size} bytes, 收到 {bytes_read} bytes")
                continue

            # 讀取並確認EOF
            eof = ser.read(len(END_OF_FRAME))
            if eof != END_OF_FRAME:
                print("警告: 未找到 EOF 標記，收到的資料可能不完整。")
                continue
            # --- 結束修改 ---

            # --- 修改：對資料進行反向位元組填充 ---
            img_data = unstuff_data(stuffed_img_data)
            # --- 結束修改 ---

            # --- 新增：驗證CRC ---
            calculated_crc = zlib.crc32(img_data) & 0xffffffff
            if calculated_crc != received_crc:
                print(f"錯誤: CRC不匹配。計算: {calculated_crc:08x}, 接收: {received_crc:08x}")
                continue
            # --- 結束新增 ---

            # 5. 驗證資料完整性 (unstuffed大小應合理)
            if len(img_data) > MAX_IMAGE_SIZE or len(img_data) == 0:
                print(f"警告: 反填充後大小異常 ({len(img_data)} bytes)")
                continue

            # 6. 解碼 JPEG 影像
            try:
                # 將位元組流轉換為 numpy 陣列
                np_arr = np.frombuffer(img_data, np.uint8)
                # 從 numpy 陣列解碼影像
                frame = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

                if frame is None:
                    print("錯誤: 無法解碼影像。資料可能已損毀。")
                    continue
                
                print(f"成功接收並解碼一幀影像，反填充大小: {len(img_data)} bytes")

                # 7. 顯示影像
                cv2.imshow('Received CCTV (Press q to quit)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"影像處理時發生錯誤: {e}")
            # --- 結束修改 ---

    except KeyboardInterrupt:
        print("\n程式被使用者中斷。")
    finally:
        # 釋放資源
        print("正在關閉程式並釋放資源...")
        cv2.destroyAllWindows()
        ser.close()
        print("程式已關閉。")

if __name__ == '__main__':
    main()


# --- 新增：通訊協定標記 (更穩健的版本) ---
START_OF_FRAME = b'\x01'  # 使用 SOH (Start of Header) 控制字元
END_OF_FRAME   = b'\x04'  # 使用 EOT (End of Transmission) 控制字元
ESC            = b'\x1B'  # 使用 ESC (Escape) 控制字元
# --- 結束設定 ---

def unstuff_data(stuffed_data):
    """
    對資料進行反向位元組填充 (Unstuffing)。
    """
    unstuffed = bytearray()
    i = 0
    while i < len(stuffed_data):
        if stuffed_data[i] == ESC[0]:
            # 如果 ESC 是最後一個字元，這是一個錯誤，但我們還是處理一下
            if i + 1 < len(stuffed_data):
                unstuffed.append(stuffed_data[i+1])
                i += 2
            else:
                # 忽略結尾的單獨 ESC
                i += 1
        else:
            unstuffed.append(stuffed_data[i])
            i += 1
    return bytes(unstuffed)

def get_serial_port():
    """自動偵測並返回可用的序列埠"""
    # 在不同作業系統下序列埠名稱的常見模式
    if platform.system() == "Windows":
        # Windows 上的序列埠通常是 COMx
        ports = [f'COM{i}' for i in range(1, 256)]
    elif platform.system() == "Linux":
        # Linux 上的序列埠通常是 /dev/ttyUSBx 或 /dev/ttyACMx
        ports = [f'/dev/ttyUSB{i}' for i in range(8)] + [f'/dev/ttyACM{i}' for i in range(8)]
    elif platform.system() == "Darwin": # macOS
        # macOS 上的序列埠通常是 /dev/tty.usbserial-xxxx 或 /dev/tty.usbmodemxxxx
        # 使用 glob 來尋找可能的埠
        ports = glob.glob('/dev/tty.usbserial*') + glob.glob('/dev/tty.usbmodem*')
    else:
        return None

    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            print(f"找到可用的序列埠: {port}")
            return port
        except (OSError, serial.SerialException):
            pass
    return None

def main():
    """
    主函式，用於從序列埠接收、解碼並顯示影像。
    """
    serial_port = get_serial_port()
    if not serial_port:
        print("錯誤: 找不到任何可用的序列埠。")
        print("請確認您的微控制器(Arduino/ESP32)已連接。")
        return

    try:
        # 初始化序列埠
        # 將預設 timeout 設為一個較短的值，避免在沒有資料時卡住太久
        ser = serial.Serial(serial_port, BAUD_RATE, timeout=0.1)
        print(f"成功打開序列埠: {serial_port} @ {BAUD_RATE} bps")
    except serial.SerialException as e:
        print(f"錯誤: 無法打開序列埠 {serial_port}。")
        print(f"詳細資訊: {e}")
        return

    print("已連接序列埠，開始等待接收影像...")
    print("按下 'q' 鍵或 Ctrl+C 停止程式。")

    try:
        while True:
            # --- 修改：使用新的通訊協定 ---
            # 1. 尋找幀的開始標記 (SOF)
            ser.read_until(START_OF_FRAME)

            # 2. 讀取 8 位元組的標頭 (填充大小 + CRC)
            header = ser.read(8)
            if len(header) < 8:
                continue

            # 3. 解碼標頭以取得填充大小和CRC
            try:
                stuffed_size, received_crc = struct.unpack('>II', header)
                
                if stuffed_size > MAX_IMAGE_SIZE:
                    #print(f"錯誤: 偵測到不合理的填充大小 ({stuffed_size} bytes)，重新尋找 SOF...")
                    continue

            except struct.error:
                print("錯誤: 標頭解碼失敗，重新尋找 SOF...")
                continue
            
            # --- 修改：分段讀取填充資料，直到滿stuffed_size ---
            # 暫時禁用超時，以確保讀取完整
            original_timeout = ser.timeout
            ser.timeout = None 
            stuffed_img_data = bytearray()
            bytes_read = 0
            CHUNK_SIZE = 128  # 與發送端一致
            while bytes_read < stuffed_size:
                remaining = stuffed_size - bytes_read
                chunk_size = min(CHUNK_SIZE, remaining)
                chunk = ser.read(chunk_size)
                if len(chunk) < chunk_size:
                    print(f"錯誤: 讀取不完整，預期 {chunk_size} bytes, 收到 {len(chunk)} bytes")
                    break
                stuffed_img_data.extend(chunk)
                bytes_read += len(chunk)
            ser.timeout = original_timeout  # 恢復超時

            if bytes_read != stuffed_size:
                print(f"錯誤: 填充資料不完整。預期 {stuffed_size} bytes, 收到 {bytes_read} bytes")
                continue

            # 讀取並確認EOF
            eof = ser.read(len(END_OF_FRAME))
            if eof != END_OF_FRAME:
                print("警告: 未找到 EOF 標記，收到的資料可能不完整。")
                continue
            # --- 結束修改 ---

            # --- 修改：對資料進行反向位元組填充 ---
            img_data = unstuff_data(stuffed_img_data)
            # --- 結束修改 ---

            # --- 新增：驗證CRC ---
            calculated_crc = zlib.crc32(img_data) & 0xffffffff
            if calculated_crc != received_crc:
                print(f"錯誤: CRC不匹配。計算: {calculated_crc:08x}, 接收: {received_crc:08x}")
                continue
            # --- 結束新增 ---

            # 5. 驗證資料完整性 (unstuffed大小應合理)
            if len(img_data) > MAX_IMAGE_SIZE or len(img_data) == 0:
                print(f"警告: 反填充後大小異常 ({len(img_data)} bytes)")
                continue

            # 6. 解碼 JPEG 影像
            try:
                # 將位元組流轉換為 numpy 陣列
                np_arr = np.frombuffer(img_data, np.uint8)
                # 從 numpy 陣列解碼影像
                frame = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

                if frame is None:
                    print("錯誤: 無法解碼影像。資料可能已損毀。")
                    continue
                
                print(f"成功接收並解碼一幀影像，反填充大小: {len(img_data)} bytes")

                # 7. 顯示影像
                cv2.imshow('Received CCTV (Press q to quit)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"影像處理時發生錯誤: {e}")
            # --- 結束修改 ---

    except KeyboardInterrupt:
        print("\n程式被使用者中斷。")
    finally:
        # 釋放資源
        print("正在關閉程式並釋放資源...")
        cv2.destroyAllWindows()
        ser.close()
        print("程式已關閉。")

if __name__ == '__main__':
    main()