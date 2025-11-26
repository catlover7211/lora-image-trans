# SSDV 範例與測試

本目錄包含 SSDV (Slow Scan Digital Video) 的使用範例。

## 範例 1: 基本 SSDV 編碼/解碼

示範如何將 JPEG 影像轉換為 SSDV 封包並解碼回來。

```python
# 參見 ssdv_example.py
python ssdv_example.py
```

## 範例 2: 移動偵測測試

測試移動偵測功能，不需要實際的 LoRa 硬體。

```python
# 需要攝影機
python motion_detection_test.py
```

## 範例 3: SSDV 完整傳輸模擬

模擬完整的 SSDV 傳輸流程（發送端和接收端）。

```python
python ssdv_simulation.py
```

## 使用現有範例

### JPEG 範例

已有的 JPEG 編碼範例也適用於 SSDV 前置處理：

```bash
python jpeg_example.py
```

## 快速測試 SSDV

### 1. 使用虛擬串列埠測試（Linux）

```bash
# 建立虛擬串列埠對
socat -d -d pty,raw,echo=0 pty,raw,echo=0

# 在一個終端執行發送端
cd raspberry_pi
python ssdv_sender.py --port /dev/pts/X --motion-mode manual --trigger-interval 5

# 在另一個終端執行接收端
cd pc
python ssdv_receiver.py --port /dev/pts/Y --auto-save --verbose
```

### 2. 使用環回測試（無硬體）

修改程式碼使用檔案或記憶體緩衝區來模擬傳輸：

```python
# 參見 ssdv_simulation.py
python examples/ssdv_simulation.py
```

## 效能測試

測試不同設定下的傳輸速度和影像品質：

```bash
# 測試不同品質等級
for quality in 0 2 4 6; do
    echo "Testing quality $quality"
    python ssdv_sender.py --quality $quality --motion-mode manual --trigger-interval 5
done

# 測試不同解析度
for res in "160x120" "320x240" "640x480"; do
    echo "Testing resolution $res"
    python ssdv_sender.py --width ${res%x*} --height ${res#*x} --motion-mode manual
done
```

## 詳細文檔

更多資訊請參考：
- [SSDV 快速入門](../docs/SSDV_README.md)
- [SSDV 完整使用指南](../docs/SSDV_GUIDE.md)
