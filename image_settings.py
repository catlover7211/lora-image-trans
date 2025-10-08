"""集中管理影像調校參數，方便在單一位置調整並加入註解說明。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import cv2

ColorMode = Literal["gray", "bgr"]


@dataclass(frozen=True)
class ImageSettings:
    """影像擷取與編碼相關的預設參數。"""

    width: int = 160
    """輸出影像寬度（像素）。"""

    height: int = 90
    """輸出影像高度（像素）。"""

    target_bitrate: int = 150_000
    """H.264 目標位元率（每秒位元數）。實測低於 10 kbps 會導致編碼器初始化失敗。"""

    keyframe_interval: int = 30
    """每隔多少幀強制產生一次關鍵幀（I-Frame）。數值越大，差分壓縮越積極。"""

    motion_threshold: float = 2.0
    """平均灰階差異門檻（0-255）。低於該值視為無顯著變化，可跳過傳送。"""

    max_idle_seconds: float = 10.0
    """允許最長無傳輸的時間（秒）。超過後即使無變化也會強制送出一幀。"""

    transmit_interval: float = 10.0
    """兩幀之間的最短間隔（秒），用於節流避免過度佔用頻寬。"""

    color_mode: ColorMode = "gray"
    """預設採用灰階輸出，可改為 'bgr' 取得彩色畫面。"""


DEFAULT_IMAGE_SETTINGS = ImageSettings()
"""提供給應用程式載入的預設參數實例。"""


def color_conversion(mode: ColorMode) -> Optional[int]:
    """將設定的顏色模式轉換為 OpenCV 所需的色彩轉換常數。"""
    if mode == "gray":
        return cv2.COLOR_BGR2GRAY
    return None
