"""Camera capture module for Raspberry Pi."""
from typing import Optional, Tuple

import cv2
import numpy as np

from common.config import DEFAULT_WIDTH, DEFAULT_HEIGHT


class CameraCapture:
    """Handles camera capture and preprocessing."""
    
    def __init__(self, camera_index: int = 0, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        """Initialize camera capture.
        
        Args:
            camera_index: Camera device index, default 0
            width: Target image width
            height: Target image height
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
    
    def open(self) -> bool:
        """Open camera device.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Error: Cannot open camera {self.camera_index}")
                return False
            
            # Set camera resolution (may not be supported by all cameras)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Camera opened: {actual_width}x{actual_height}")
            
            return True
            
        except Exception as e:
            print(f"Error opening camera: {e}")
            return False
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture a frame from camera.
        
        Returns:
            BGR image array or None on failure
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return None
            
            # Resize to target resolution
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            
            return frame
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def close(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
