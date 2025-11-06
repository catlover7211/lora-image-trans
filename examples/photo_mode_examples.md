# 照片模式使用範例 (Photo Mode Usage Examples)

這個文件展示如何使用新的照片模式來拍攝和傳輸高清照片。
This document demonstrates how to use the new photo mode to capture and transmit high-definition photos.

## 基本使用 (Basic Usage)

### 1. 啟動接收端 (Start Receiver)

在 PC 端執行：
```bash
cd pc
python receiver.py --mode photo --save my_photo.jpg
```

這將等待接收一張照片並儲存為 `my_photo.jpg`。

### 2. 啟動發送端 (Start Sender)

在 Raspberry Pi 端執行：
```bash
cd raspberry_pi
python sender.py --mode photo --preview
```

這將：
1. 開啟攝影機預覽
2. 按 'q' 鍵拍照
3. 確認照片後按任意鍵發送
4. 傳輸完成後自動結束

## 進階範例 (Advanced Examples)

### 超高清照片 (Ultra HD Photo)

如果需要更高解析度的照片：
```bash
# 發送端
python sender.py --mode photo --width 1280 --height 720 --jpeg-quality 98 --preview

# 接收端
python receiver.py --mode photo --save ultra_hd_photo.jpg
```

### 快速照片模式 (Quick Photo Mode)

不顯示預覽，直接拍攝並發送：
```bash
# 發送端（不使用 --preview）
python sender.py --mode photo

# 接收端
python receiver.py --mode photo --save quick_photo.jpg
```

### 使用壓縮感知編碼 (Using CS Encoding)

雖然照片模式推薦使用 JPEG，但也可以使用 CS 編碼：
```bash
# 發送端
python sender.py --mode photo --codec cs --cs-rate 0.5 --preview

# 接收端
python receiver.py --mode photo --save cs_photo.jpg
```

## 照片模式 vs CCTV 模式 (Photo Mode vs CCTV Mode)

| 特性 | CCTV 模式 | 照片模式 |
|------|-----------|----------|
| 用途 | 連續監控 | 單張照片 |
| 預設解析度 | 128x72 | 640x480 |
| 預設 JPEG 品質 | 85 | 95 |
| FPS | 可調整 (預設 10) | N/A（單張） |
| 自動結束 | 否 | 是 |
| 適用場景 | 即時監控、錄影 | 文件拍攝、高品質記錄 |

## 注意事項 (Notes)

1. **傳輸時間**：高解析度照片需要較長的傳輸時間（數十秒到數分鐘）
2. **品質優先**：照片模式優先考慮品質而非速度
3. **預覽功能**：建議使用 `--preview` 來確保拍攝到理想的畫面
4. **儲存選項**：接收端可以選擇性儲存照片到檔案

## 故障排除 (Troubleshooting)

### 傳輸時間過長
降低解析度或品質：
```bash
python sender.py --mode photo --width 320 --height 240 --jpeg-quality 90
```

### 照片品質不夠高
提高解析度和品質：
```bash
python sender.py --mode photo --width 1280 --height 720 --jpeg-quality 98
```

### 需要多張照片
執行多次照片模式，或考慮使用 CCTV 模式配合較低的 FPS：
```bash
python sender.py --mode cctv --fps 0.2 --width 640 --height 480 --jpeg-quality 95
```
這將每 5 秒拍攝一張高品質照片。
