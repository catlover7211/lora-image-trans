"""集中管理影像調校參數，方便在單一位置調整並加入註解說明。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import cv2

ColorMode = Literal["gray", "bgr"]
VideoCodec = Literal["h264", "h265", "av1", "wavelet"]


@dataclass(frozen=True)
class ImageSettings:
    """影像擷取與編碼相關的預設參數。"""

    width: int = 160
    """輸出影像寬度（像素）。維持 16:9 且兼顧畫質與編碼效率。"""

    height: int = 90
    """輸出影像高度（像素）。搭配 640x360 約等於 360p，便於在低頻寬下維持較高幀率。"""

    target_bitrate: int = 400_000
    """H.264 目標位元率（每秒位元數）。約 400 kbps 可支援 360p@20fps 且避免 libx264 初始化失敗。"""

    keyframe_interval: int = 30
    """每隔多少幀強制產生一次關鍵幀（I-Frame）。30 代表約 1 秒更新一次，有利串流穩定性。"""

    motion_threshold: float = 6.0
    """平均灰階差異門檻（0-255）。提高到 6 可減少雜訊導致的誤判，同時保留明顯變化。"""

    max_idle_seconds: float = 2.0
    """允許最長無傳輸的時間（秒）。最長 2 秒沒變化也會送幀，避免畫面停住。"""

    transmit_interval: float = 0.05
    """兩幀之間的最短間隔（秒）。0.05 約等於 20 fps，可在硬體允許下提升流暢度。"""

    color_mode: ColorMode = "gray"
    """預設採用灰階輸出，可改為 'bgr' 取得彩色畫面。"""

    codec: VideoCodec = 'wavelet'
    """影像編碼器類型。另支援 'av1' 與 'wavelet'，須視 FFmpeg/硬體或自訂解碼器支援而定。"""

    wavelet_levels: int = 2
    """Wavelet 轉換層數（僅 wavelet 編碼器使用），需小於等於影像尺寸的 log2。"""

    wavelet_quant: int = 12
    """Wavelet 係數量化步階（僅 wavelet 編碼器使用），越大壓縮越高但細節越少。"""


DEFAULT_IMAGE_SETTINGS = ImageSettings()
"""提供給應用程式載入的預設參數實例。"""


def color_conversion(mode: ColorMode) -> Optional[int]:
    """將設定的顏色模式轉換為 OpenCV 所需的色彩轉換常數。"""
    if mode == "gray":
        return cv2.COLOR_BGR2GRAY
    return None
