"""JPEG encoder for image transmission."""
import io
from typing import Optional

import cv2
import numpy as np

from common.config import DEFAULT_JPEG_QUALITY


class JPEGEncoder:
    """Encodes images to JPEG format."""
    
    def __init__(self, quality: int = DEFAULT_JPEG_QUALITY):
        """Initialize JPEG encoder.
        
        Args:
            quality: JPEG quality (1-100), default 85
        """
        if not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be between 1 and 100")
        self.quality = quality
    
    def encode(self, image: np.ndarray) -> Optional[bytes]:
        """Encode image to JPEG bytes.
        
        Args:
            image: BGR image array
            
        Returns:
            JPEG encoded bytes or None on failure
        """
        try:
            # Encode image to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            result, encoded_img = cv2.imencode('.jpg', image, encode_param)
            
            if not result:
                return None
            
            return encoded_img.tobytes()
        except Exception as e:
            print(f"JPEG encoding error: {e}")
            return None
    
    def set_quality(self, quality: int) -> None:
        """Update JPEG quality setting.
        
        Args:
            quality: New JPEG quality (1-100)
        """
        if not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be between 1 and 100")
        self.quality = quality
