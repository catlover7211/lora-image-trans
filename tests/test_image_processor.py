"""Unit tests for the ImageProcessor and related classes in main.py."""
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path to import main module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from protocol import FrameStats


class TestFrameStatistics(unittest.TestCase):
    """Test the FrameStatistics class."""
    
    def test_initialization(self):
        """Test that FrameStatistics initializes correctly."""
        from main import FrameStatistics
        stats = FrameStatistics()
        self.assertEqual(stats.pending_fragments, 0)
        self.assertEqual(len(stats.pending_stats), 0)
    
    def test_add_fragment(self):
        """Test adding fragments to statistics."""
        from main import FrameStatistics
        stats = FrameStatistics()
        
        frame_stat = FrameStats(payload_size=100, stuffed_size=133, crc=12345)
        stats.add_fragment(frame_stat)
        
        self.assertEqual(stats.pending_fragments, 1)
        self.assertEqual(len(stats.pending_stats), 1)
    
    def test_reset(self):
        """Test resetting statistics."""
        from main import FrameStatistics
        stats = FrameStatistics()
        
        frame_stat = FrameStats(payload_size=100, stuffed_size=133, crc=12345)
        stats.add_fragment(frame_stat)
        stats.reset()
        
        self.assertEqual(stats.pending_fragments, 0)
        self.assertEqual(len(stats.pending_stats), 0)
    
    def test_compression_summary(self):
        """Test compression summary calculation."""
        from main import FrameStatistics
        stats = FrameStatistics()
        
        stats.add_fragment(FrameStats(payload_size=100, stuffed_size=133, crc=1))
        stats.add_fragment(FrameStats(payload_size=100, stuffed_size=133, crc=2))
        
        summary = stats.get_compression_summary()
        
        self.assertEqual(summary['total_payload'], 200)
        self.assertEqual(summary['total_encoded'], 266)
        self.assertEqual(summary['fragment_count'], 2)
        self.assertAlmostEqual(summary['compression_ratio'], 133.0, places=1)
    
    def test_compression_summary_empty(self):
        """Test compression summary with no data."""
        from main import FrameStatistics
        stats = FrameStatistics()
        
        summary = stats.get_compression_summary()
        
        self.assertEqual(summary['total_payload'], 0)
        self.assertEqual(summary['total_encoded'], 0)
        self.assertEqual(summary['compression_ratio'], 0)
        self.assertEqual(summary['fragment_count'], 0)


class TestImageProcessor(unittest.TestCase):
    """Test the ImageProcessor class."""
    
    @patch('main.H264Decoder')
    def test_initialization(self, mock_decoder_class):
        """Test that ImageProcessor initializes correctly."""
        from main import ImageProcessor
        
        decoder = mock_decoder_class()
        processor = ImageProcessor(decoder)
        
        self.assertEqual(processor.decoder, decoder)
        self.assertIsNone(processor.current_codec)
        self.assertEqual(processor.statistics.pending_fragments, 0)
    
    @patch('main.H264Decoder')
    @patch('main.EncodedChunk')
    def test_process_chunk_success(self, mock_chunk_class, mock_decoder_class):
        """Test successful chunk processing."""
        from main import ImageProcessor
        import numpy as np
        
        decoder = mock_decoder_class()
        processor = ImageProcessor(decoder)
        
        # Create mock chunk and stats
        chunk = MagicMock()
        chunk.is_config = False
        chunk.codec = 'h264'
        chunk.data = b'test_data'
        
        stats = FrameStats(payload_size=100, stuffed_size=133, crc=12345)
        
        # Mock decoder to return a frame
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        decoder.decode.return_value = [mock_frame]
        
        # Process chunk
        frames = processor.process_chunk(chunk, stats)
        
        self.assertEqual(len(frames), 1)
        self.assertEqual(processor.statistics.pending_fragments, 1)
        decoder.decode.assert_called_once_with(chunk)
    
    @patch('main.H264Decoder')
    def test_process_chunk_codec_change(self, mock_decoder_class):
        """Test codec change detection."""
        from main import ImageProcessor
        import numpy as np
        
        decoder = mock_decoder_class()
        processor = ImageProcessor(decoder)
        
        # Create config chunk with new codec
        chunk = MagicMock()
        chunk.is_config = True
        chunk.codec = 'h265'
        chunk.data = b'config_data'
        
        stats = FrameStats(payload_size=50, stuffed_size=67, crc=54321)
        
        # Mock decoder
        decoder.decode.return_value = []
        
        # Process chunk
        with patch('builtins.print') as mock_print:
            frames = processor.process_chunk(chunk, stats)
            mock_print.assert_called_once()
            self.assertIn('H265', mock_print.call_args[0][0])
        
        self.assertEqual(processor.current_codec, 'h265')
    
    @patch('main.H264Decoder')
    def test_process_chunk_decode_error(self, mock_decoder_class):
        """Test handling of decoding errors."""
        from main import ImageProcessor
        
        decoder = mock_decoder_class()
        processor = ImageProcessor(decoder)
        
        # Add some stats first
        chunk1 = MagicMock()
        chunk1.is_config = False
        chunk1.codec = 'h264'
        stats1 = FrameStats(payload_size=100, stuffed_size=133, crc=1)
        decoder.decode.return_value = []
        processor.process_chunk(chunk1, stats1)
        
        # Now cause a decoding error
        chunk2 = MagicMock()
        chunk2.is_config = False
        chunk2.codec = 'h264'
        stats2 = FrameStats(payload_size=100, stuffed_size=133, crc=2)
        decoder.decode.side_effect = RuntimeError("Decode failed")
        
        with patch('builtins.print'):
            frames = processor.process_chunk(chunk2, stats2)
        
        self.assertEqual(len(frames), 0)
        # Statistics should be reset on error
        self.assertEqual(processor.statistics.pending_fragments, 0)


class TestImageDisplay(unittest.TestCase):
    """Test the ImageDisplay class."""
    
    def test_initialization(self):
        """Test that ImageDisplay initializes correctly."""
        from main import ImageDisplay
        
        display = ImageDisplay("Test Window")
        self.assertEqual(display.window_title, "Test Window")
    
    @patch('main.cv2.imshow')
    @patch('builtins.print')
    def test_show_frame(self, mock_print, mock_imshow):
        """Test displaying a frame."""
        from main import ImageDisplay, FrameStatistics
        import numpy as np
        
        display = ImageDisplay("Test Window")
        stats = FrameStatistics()
        stats.add_fragment(FrameStats(payload_size=100, stuffed_size=133, crc=1))
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        display.show_frame(frame, stats)
        
        # Verify print was called with statistics
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        self.assertIn('100 bytes', call_args)
        
        # Verify cv2.imshow was called
        mock_imshow.assert_called_once_with("Test Window", frame)
        
        # Verify statistics were reset
        self.assertEqual(stats.pending_fragments, 0)


if __name__ == '__main__':
    unittest.main()
