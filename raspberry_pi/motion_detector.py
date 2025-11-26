"""Motion detection module for triggering image capture.

This module provides simple motion detection using frame differencing.
When motion is detected above a threshold, it triggers a capture event.
"""
from typing import Optional, Tuple

import cv2
import numpy as np


class MotionDetector:
    """Simple motion detector using frame differencing."""
    
    def __init__(self, threshold: int = 25, min_area: int = 500, 
                 blur_size: int = 21, history_frames: int = 2):
        """Initialize motion detector.
        
        Args:
            threshold: Pixel difference threshold (0-255)
            min_area: Minimum contour area to consider as motion
            blur_size: Gaussian blur kernel size (odd number)
            history_frames: Number of frames to keep in history
        """
        self.threshold = threshold
        self.min_area = min_area
        self.blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        self.history_frames = history_frames
        
        self.previous_frames = []
        self.motion_detected = False
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Detect motion in the current frame.
        
        Args:
            frame: Current BGR frame
            
        Returns:
            Tuple of (motion_detected, motion_score)
            motion_score is the percentage of changed pixels (0-100)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        
        # If this is the first frame, just store it
        if len(self.previous_frames) == 0:
            self.previous_frames.append(gray)
            return False, 0.0
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(self.previous_frames[-1], gray)
        
        # Threshold the difference
        _, thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Dilate to fill gaps
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate motion score
        motion_pixels = np.sum(thresh > 0)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        motion_score = (motion_pixels / total_pixels) * 100
        
        # Check if motion is significant
        has_motion = False
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_area:
                has_motion = True
                break
        
        # Update frame history
        self.previous_frames.append(gray)
        if len(self.previous_frames) > self.history_frames:
            self.previous_frames.pop(0)
        
        self.motion_detected = has_motion
        return has_motion, motion_score
    
    def reset(self):
        """Reset the motion detector state."""
        self.previous_frames.clear()
        self.motion_detected = False
    
    def get_motion_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Get the motion detection mask for visualization.
        
        Args:
            frame: Current BGR frame
            
        Returns:
            Binary mask showing detected motion areas
        """
        if len(self.previous_frames) == 0:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(self.previous_frames[-1], gray)
        
        # Threshold the difference
        _, thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Dilate to fill gaps
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        return thresh
    
    def draw_motion(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw motion detection visualization on frame.
        
        Args:
            frame: BGR frame to draw on
            color: BGR color for motion rectangles
            
        Returns:
            Frame with motion visualization
        """
        if len(self.previous_frames) == 0:
            return frame
        
        result = frame.copy()
        
        # Get motion mask
        mask = self.get_motion_mask(frame)
        if mask is None:
            return result
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw rectangles around significant motion
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        
        # Add motion status text
        status = "MOTION DETECTED" if self.motion_detected else "No Motion"
        cv2.putText(result, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, color if self.motion_detected else (128, 128, 128), 2)
        
        return result


class ManualTrigger:
    """Manual trigger for testing without actual motion detection."""
    
    def __init__(self, trigger_interval: float = 5.0):
        """Initialize manual trigger.
        
        Args:
            trigger_interval: Time interval between automatic triggers (seconds)
        """
        self.trigger_interval = trigger_interval
        self.last_trigger_time = 0
        self.manual_trigger_flag = False
    
    def should_trigger(self, current_time: float) -> bool:
        """Check if trigger should fire.
        
        Args:
            current_time: Current time in seconds
            
        Returns:
            True if trigger should fire
        """
        # Check manual flag
        if self.manual_trigger_flag:
            self.manual_trigger_flag = False
            self.last_trigger_time = current_time
            return True
        
        # Check automatic interval
        if current_time - self.last_trigger_time >= self.trigger_interval:
            self.last_trigger_time = current_time
            return True
        
        return False
    
    def trigger(self):
        """Manually trigger the capture."""
        self.manual_trigger_flag = True
    
    def reset(self):
        """Reset the trigger state."""
        self.last_trigger_time = 0
        self.manual_trigger_flag = False
