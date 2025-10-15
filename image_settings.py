"""集中管理影像調校參數，方便在單一位置調整並加入註解說明。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import cv2

ColorMode = Literal["gray", "bgr"]
VideoCodec = Literal["h264", "h265", "av1", "wavelet", "jpeg", "contour", "yolo"]


@dataclass(frozen=True)
class ImageSettings:
    """影像擷取與編碼相關的預設參數。"""

    width: int = 1600
    """輸出影像寬度（像素）。維持 4:3 並控制輸出資料量。"""

    height: int = 800
    """輸出影像高度（像素）。配合 160×120 解析度可在 115200 bps 串列埠下達成 >10fps。"""

    target_bitrate: int = 120_000
    """H.264 目標位元率（每秒位元數）。經估算可讓單幀 payload 約 1.2 KB，足以在串口帶寬內維持 10 fps。"""

    keyframe_interval: int = 30
    """每隔多少幀強制產生一次關鍵幀（I-Frame）。30 對應約 2 秒更新一次，平衡壓縮效率與畫面恢復。"""

    motion_threshold: float = 12.0
    """平均灰階差異門檻（0-255）。設定在 12 可濾除感測器雜訊，同時保留真實位移。"""

    max_idle_seconds: float = 1.0
    """允許最長無傳輸的時間（秒）。最多 1 秒沒變化仍會送幀，避免畫面卡住。"""

    transmit_interval: float = 0.03
    """兩幀之間的最短間隔（秒）。0.03 秒相當於 ~33 fps，上限由串口吞吐決定。"""

    color_mode: ColorMode = "gray"
    """預設以灰階做運動檢測，再在編碼前轉回 BGR，兼顧壓縮與相容性。"""

    codec: VideoCodec = 'contour'
    """影像編碼器類型。預設改用 H.264，因為在 160×120@120kbps 下壓縮效率最佳。"""

    wavelet_levels: int = 1
    """Wavelet 轉換層數（僅 wavelet 編碼器使用）；在 160×120 下 1 層可兼顧速度與細節。"""

    wavelet_quant: int = 40
    """Wavelet 係數量化步階（僅 wavelet 編碼器使用），40 約可將單幀 payload 控制在 10 KB 內。"""

    use_chunk_ack: bool = True
    """是否在串流傳輸過程啟用每個 chunk 的 ACK。停用可降低延遲，但在雜訊環境下可靠度變差。"""

    tx_buffer_size: int = 8
    """傳送端待發緩衝區容量（以幀為單位）。預設 8 幀可吸收暫時的串列埠延遲。"""

    rx_buffer_size: int = 32
    """接收端待顯示緩衝區容量。32 幀可避免主程式顯示延遲阻塞解碼。"""

    jpeg_quality: int = 15
    """JPEG 壓縮品質 (1-100)。預設 85 在串口頻寬下兼顧畫質與大小。"""

    contour_samples: int = 128
    """Contour 模式採樣點數，用於建立 r(θ) 函數。"""

    contour_coefficients: int = 128
    """Contour 模式保留的低頻傅立葉係數數量。"""

    yolo_weights: str = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-seg.pt"
    """YOLOv5 權重檔路徑或模型名稱。預設使用官方 yolov5n 模型。"""

    yolo_confidence: float = 0.25
    """YOLOv5 信心門檻（0-1）。"""

    yolo_iou: float = 0.45
    """YOLOv5 NMS IoU 門檻。"""

    yolo_device: str = "cpu"
    """YOLOv5 推論使用的裝置，例如 'cpu' 或 'cuda:0'。"""

    yolo_max_detections: int = 10
    """YOLOv5 單張影像保留的最大框數。"""


DEFAULT_IMAGE_SETTINGS = ImageSettings()
"""提供給應用程式載入的預設參數實例。"""


def color_conversion(mode: ColorMode) -> Optional[int]:
    """將設定的顏色模式轉換為 OpenCV 所需的色彩轉換常數。"""
    if mode == "gray":
        return cv2.COLOR_BGR2GRAY
    return None
