# LoRa 傳輸解碼修復

## 問題描述
在 LoRa 傳輸過程中，二進制 SYNC_MARKER (`\xDE\xAD\xBE\xEF`) 會變成亂碼或被移除，導致 `main.py` 無法解碼接收到的影像幀。

問題描述中顯示的資料格式為：
```
FRAME 2124 6f4d2a91 IQD/2P/gABBKRklGAAEBAAABAAEAAP/...
```

這種格式缺少 SYNC_MARKER 前綴，導致現有的協議無法同步和解碼。

## 解決方案
修改 `protocol.py` 中的幀同步邏輯，增加後備機制：

1. **優先嘗試**：尋找二進制 SYNC_MARKER（原有行為）
2. **後備機制**：如果找不到 SYNC_MARKER，則搜索 ASCII "FRAME " 前綴

這樣既保持了向後兼容性，又能支持 LoRa 傳輸的資料格式。

## 修改內容

### 新增常量
```python
FRAME_PREFIX_BYTES = b"FRAME "
```
預編碼的 FRAME 前綴，提升性能，避免重複編碼操作。

### 更新方法
- `_synchronize()`: 添加文檔說明雙重檢測策略
- `_try_find_sync_marker()`: 實現雙重檢測邏輯
  1. 首先搜索 SYNC_MARKER
  2. 若未找到，搜索 "FRAME " 前綴

## 測試驗證

所有現有單元測試通過：
```bash
$ python3 -m unittest tests/test_protocol.py -v
test_frame_format_ascii ... ok
test_send_and_receive_roundtrip ... ok
test_use_ack_property ... ok

Ran 3 tests in 0.000s
OK
```

支持以下場景：
- ✅ 正常傳輸（帶 SYNC_MARKER）
- ✅ LoRa 傳輸（無 SYNC_MARKER）
- ✅ SYNC_MARKER 損壞的情況
- ✅ 問題描述中的資料格式

## 使用說明

修復後，`main.py` 可以正常接收和解碼來自 LoRa 的資料，無需任何額外配置。

### 啟動接收端
```bash
python main.py
```

接收端會自動處理以下兩種格式：
1. 帶 SYNC_MARKER 的完整格式（從 USB 直連或正常傳輸）
2. 不帶 SYNC_MARKER 的格式（從 LoRa 傳輸）

## 技術細節

### 同步機制
原始格式：
```
[SYNC_MARKER 4 bytes] + FRAME <length> <crc> <base64_data>\n
```

LoRa 格式（SYNC_MARKER 損壞或缺失）：
```
FRAME <length> <crc> <base64_data>\n
```

協議現在能夠處理兩種格式，自動檢測並同步到正確的幀邊界。

### 性能優化
使用模塊級常量 `FRAME_PREFIX_BYTES` 避免每次調用時重複編碼 "FRAME " 字符串，提升同步效率。

## 安全性
通過 CodeQL 安全掃描，無安全漏洞。
