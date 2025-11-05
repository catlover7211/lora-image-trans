"""JPEG decoder for received images."""
from typing import Optional

import cv2
import numpy as np


class JPEGDecoder:
    """Decodes JPEG encoded images."""
    
    def __init__(self):
        """Initialize JPEG decoder."""
        pass
    
    def decode(self, data: bytes) -> Optional[np.ndarray]:
        """Decode JPEG bytes to image array.
        
        Args:
            data: JPEG encoded bytes
            
        Returns:
            BGR image array or None on failure
        """
        try:
            # Decode JPEG data
            nparr = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return None
            
            return image
            
        except Exception as e:
            print(f"JPEG decoding error: {e}")
            return None
