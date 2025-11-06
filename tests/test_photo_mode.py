"""Test photo mode functionality."""
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import (
    MODE_CCTV, MODE_PHOTO,
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_JPEG_QUALITY,
    PHOTO_WIDTH, PHOTO_HEIGHT, PHOTO_JPEG_QUALITY,
    WINDOW_TITLE_SENDER, WINDOW_TITLE_PHOTO_SENDER,
    WINDOW_TITLE_RECEIVER, WINDOW_TITLE_PHOTO_RECEIVER
)


class TestPhotoModeConfig(unittest.TestCase):
    """Test photo mode configuration constants."""
    
    def test_mode_constants_exist(self):
        """Test that mode constants are defined."""
        self.assertEqual(MODE_CCTV, 'cctv')
        self.assertEqual(MODE_PHOTO, 'photo')
    
    def test_photo_mode_defaults(self):
        """Test that photo mode has higher quality defaults than CCTV."""
        # Photo mode should have higher resolution
        self.assertGreater(PHOTO_WIDTH, DEFAULT_WIDTH)
        self.assertGreater(PHOTO_HEIGHT, DEFAULT_HEIGHT)
        
        # Photo mode should have higher JPEG quality
        self.assertGreater(PHOTO_JPEG_QUALITY, DEFAULT_JPEG_QUALITY)
    
    def test_window_titles_exist(self):
        """Test that window titles are defined for both modes."""
        self.assertIsNotNone(WINDOW_TITLE_SENDER)
        self.assertIsNotNone(WINDOW_TITLE_PHOTO_SENDER)
        self.assertIsNotNone(WINDOW_TITLE_RECEIVER)
        self.assertIsNotNone(WINDOW_TITLE_PHOTO_RECEIVER)
        
        # They should be different
        self.assertNotEqual(WINDOW_TITLE_SENDER, WINDOW_TITLE_PHOTO_SENDER)
        self.assertNotEqual(WINDOW_TITLE_RECEIVER, WINDOW_TITLE_PHOTO_RECEIVER)


class TestPhotoModeQuality(unittest.TestCase):
    """Test that photo mode provides high-definition quality."""
    
    def test_photo_resolution_is_high_definition(self):
        """Test that photo mode resolution is at least VGA quality."""
        # VGA is 640x480, considered minimum for "high-definition" in this context
        self.assertGreaterEqual(PHOTO_WIDTH, 640)
        self.assertGreaterEqual(PHOTO_HEIGHT, 480)
    
    def test_photo_jpeg_quality_is_high(self):
        """Test that photo mode JPEG quality is high."""
        # High quality should be at least 90
        self.assertGreaterEqual(PHOTO_JPEG_QUALITY, 90)
        # Should not exceed maximum
        self.assertLessEqual(PHOTO_JPEG_QUALITY, 100)
    
    def test_cctv_resolution_is_lower(self):
        """Test that CCTV mode uses lower resolution for streaming."""
        # CCTV mode should use lower resolution for better frame rate
        self.assertLess(DEFAULT_WIDTH, PHOTO_WIDTH)
        self.assertLess(DEFAULT_HEIGHT, PHOTO_HEIGHT)
    
    def test_quality_difference_is_significant(self):
        """Test that there's a significant quality difference between modes."""
        # Quality difference should be at least 10 points
        self.assertGreaterEqual(PHOTO_JPEG_QUALITY - DEFAULT_JPEG_QUALITY, 10)
        
        # Resolution difference should be at least 4x in terms of pixels
        cctv_pixels = DEFAULT_WIDTH * DEFAULT_HEIGHT
        photo_pixels = PHOTO_WIDTH * PHOTO_HEIGHT
        self.assertGreaterEqual(photo_pixels / cctv_pixels, 4)


if __name__ == '__main__':
    unittest.main()
