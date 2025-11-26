# SSDV (Slow Scan Digital Video) 使用指南

## 簡介

SSDV (Slow Scan Digital Video) 是一種專為低頻寬、高錯誤率的通道設計的影像傳輸協定。相比於直接傳輸 JPEG 影像，SSDV 提供了更好的錯誤恢復能力，特別適合 LoRa 這類的無線傳輸應用。

### SSDV 的優勢

1. **固定長度封包**：每個 SSDV 封包固定為 256 位元組，便於傳輸管理
2. **封包獨立性**：即使部分封包遺失，也能重建部分影像
3. **錯誤容忍**：比原始 JPEG 對傳輸錯誤更具容忍性
4. **元數據嵌入**：每個封包包含呼號、影像 ID、封包 ID 等資訊
5. **適合間歇傳輸**：支援移動偵測觸發，節省頻寬

## 系統架構

```
┌─────────────────────────────────────────────────────────────────┐
│                      SSDV 傳輸流程                               │
└─────────────────────────────────────────────────────────────────┘

發送端 (Raspberry Pi):
  1. 攝影機擷取 → 移動偵測/觸發
  2. 擷取靜態影像 (320x240)
  3. JPEG 編碼
  4. SSDV 封包化 (256 bytes/packet)
  5. 協定框架封裝 (TYPE_SSDV)
  6. 分包傳輸 (透過 LoRa)

接收端 (PC):
  1. 接收協定框架
  2. 解析 SSDV 封包
  3. 累積封包並重建 JPEG
  4. 解碼並顯示影像
  5. 可選：自動儲存完整影像
```

## 快速開始

### 1. 啟動接收端 (PC)

```bash
cd pc
python ssdv_receiver.py --auto-save
```

### 2. 啟動發送端 (Raspberry Pi)

```bash
cd raspberry_pi
# 手動定時觸發模式（每 10 秒拍一張）
python ssdv_sender.py --motion-mode manual --trigger-interval 10.0 --preview
```

## 使用方式

### 接收端選項

```bash
python ssdv_receiver.py [選項]
```

**選項：**
- `--port`: 指定串列埠（可選，會自動偵測）
- `--save-dir`: 儲存目錄（預設：ssdv_received）
- `--auto-save`: 自動儲存完整影像
- `--show-partial`: 顯示部分接收的影像
- `--verbose`: 顯示詳細封包資訊

### 發送端選項

```bash
python ssdv_sender.py [選項]
```

**基本選項：**
- `--width 320`: 影像寬度（預設：320）
- `--height 240`: 影像高度（預設：240）
- `--callsign LORA01`: SSDV 呼號（最多 6 字元）
- `--quality 4`: SSDV 品質等級 0-7（預設：4）
- `--packet-delay 0.01`: 封包間延遲秒數
- `--preview`: 顯示預覽視窗

**移動偵測選項：**
- `--motion-mode`: `auto`（自動偵測）或 `manual`（定時觸發）
- `--motion-threshold 25`: 移動偵測閾值
- `--motion-area 500`: 最小移動區域像素
- `--trigger-interval 10.0`: 手動模式觸發間隔秒數
- `--continuous`: 連續模式

## 使用範例

### 自動移動偵測

```bash
# 偵測到移動時拍照並傳輸
python ssdv_sender.py --motion-mode auto --preview

# 調整靈敏度（較低閾值 = 更敏感）
python ssdv_sender.py --motion-mode auto --motion-threshold 15 --motion-area 300
```

### 定時觸發

```bash
# 每 10 秒自動拍照
python ssdv_sender.py --motion-mode manual --trigger-interval 10.0

# 快速測試（每 5 秒）
python ssdv_sender.py --motion-mode manual --trigger-interval 5.0 --preview
```

### 高品質影像

```bash
python ssdv_sender.py --width 640 --height 480 --quality 6
```

## 效能調校

### 影像品質 vs 傳輸時間

| 品質 | 封包數 (320x240) | 傳輸時間 (10ms/封包) |
|------|------------------|---------------------|
| 0-2  | ~20-50 封包      | ~0.2-0.5 秒        |
| 4    | ~50-80 封包      | ~0.5-0.8 秒        |
| 6-7  | ~100-200 封包    | ~1.0-2.0 秒        |

### 解析度建議

| 解析度  | 用途           | 傳輸時間估計 |
|---------|----------------|-------------|
| 160x120 | 極快速監控     | ~0.2-0.3s   |
| 320x240 | 標準監控（建議）| ~0.5-0.8s   |
| 640x480 | 高品質影像     | ~2.0-3.0s   |

## 故障排除

### 封包遺失率高
```bash
# 增加封包延遲
python ssdv_sender.py --packet-delay 0.02

# 降低解析度
python ssdv_sender.py --width 320 --height 240

# 降低品質
python ssdv_sender.py --quality 2
```

### 移動偵測不靈敏
```bash
# 降低閾值和最小區域
python ssdv_sender.py --motion-mode auto --motion-threshold 15 --motion-area 300
```

### 移動偵測過於敏感
```bash
# 提高閾值和最小區域
python ssdv_sender.py --motion-mode auto --motion-threshold 35 --motion-area 1000
```

## 完整文檔

更詳細的說明請參考 `docs/SSDV_GUIDE.md`

## 參考資源

- **SSDV 原始專案**：https://github.com/fsphil/ssdv
- **LoRa Gateway 範例**：https://github.com/daveake/LoRa-Gateway
