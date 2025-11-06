"""Test inter-frame delay functionality to prevent receiver buffer overflow."""
import unittest
import sys
import os
import time
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# We need to mock serial before importing SerialComm
sys.modules['serial'] = MagicMock()
sys.modules['serial.tools'] = MagicMock()
sys.modules['serial.tools.list_ports'] = MagicMock()

from raspberry_pi.serial_comm import SerialComm
from common.config import INTER_FRAME_DELAY


class TestInterFrameDelay(unittest.TestCase):
    """Test inter-frame delay functionality."""
    
    def test_default_inter_frame_delay(self):
        """Test that default inter-frame delay is set correctly."""
        comm = SerialComm()
        self.assertEqual(comm.inter_frame_delay, INTER_FRAME_DELAY)
    
    def test_custom_inter_frame_delay(self):
        """Test that custom inter-frame delay can be set."""
        custom_delay = 0.1
        comm = SerialComm(inter_frame_delay=custom_delay)
        self.assertEqual(comm.inter_frame_delay, custom_delay)
    
    def test_zero_inter_frame_delay(self):
        """Test that zero inter-frame delay is allowed."""
        comm = SerialComm(inter_frame_delay=0.0)
        self.assertEqual(comm.inter_frame_delay, 0.0)
    
    @patch('raspberry_pi.serial_comm.serial.Serial')
    @patch('raspberry_pi.serial_comm.time.sleep')
    def test_send_with_inter_frame_delay(self, mock_sleep, mock_serial_class):
        """Test that send method applies inter-frame delay."""
        # Setup mock serial
        mock_serial = Mock()
        mock_serial.is_open = True
        mock_serial_class.return_value = mock_serial
        
        # Create SerialComm with custom delay
        custom_delay = 0.05
        comm = SerialComm(inter_frame_delay=custom_delay)
        comm.ser = mock_serial
        
        # Send small data (less than chunk size)
        test_data = b"test data"
        result = comm.send(test_data)
        
        # Verify send was successful
        self.assertTrue(result)
        
        # Verify inter-frame delay was applied
        # The last sleep call should be the inter-frame delay
        self.assertGreater(mock_sleep.call_count, 0)
        last_call_args = mock_sleep.call_args_list[-1][0]
        self.assertEqual(last_call_args[0], custom_delay)
    
    @patch('raspberry_pi.serial_comm.serial.Serial')
    @patch('raspberry_pi.serial_comm.time.sleep')
    def test_send_without_inter_frame_delay(self, mock_sleep, mock_serial_class):
        """Test that send method skips inter-frame delay when set to 0."""
        # Setup mock serial
        mock_serial = Mock()
        mock_serial.is_open = True
        mock_serial_class.return_value = mock_serial
        
        # Create SerialComm with zero delay
        comm = SerialComm(inter_frame_delay=0.0)
        comm.ser = mock_serial
        
        # Send small data
        test_data = b"test data"
        result = comm.send(test_data)
        
        # Verify send was successful
        self.assertTrue(result)
        
        # Verify sleep was not called (no inter-chunk delay for small data, no inter-frame delay)
        mock_sleep.assert_not_called()
    
    @patch('raspberry_pi.serial_comm.serial.Serial')
    @patch('raspberry_pi.serial_comm.time.sleep')
    def test_send_large_data_with_inter_frame_delay(self, mock_sleep, mock_serial_class):
        """Test inter-frame delay with large data that requires chunking."""
        # Setup mock serial
        mock_serial = Mock()
        mock_serial.is_open = True
        mock_serial_class.return_value = mock_serial
        
        # Create SerialComm with custom delay and small chunk size
        custom_delay = 0.05
        chunk_size = 10
        comm = SerialComm(inter_frame_delay=custom_delay, chunk_size=chunk_size)
        comm.ser = mock_serial
        
        # Send data larger than chunk size
        test_data = b"x" * 35  # Will need 4 chunks
        result = comm.send(test_data)
        
        # Verify send was successful
        self.assertTrue(result)
        
        # Verify sleep was called: 3 times for inter-chunk (0.003s) + 1 time for inter-frame (custom_delay)
        self.assertEqual(mock_sleep.call_count, 4)
        
        # Verify the last sleep call is the inter-frame delay
        last_call_args = mock_sleep.call_args_list[-1][0]
        self.assertEqual(last_call_args[0], custom_delay)
        
        # Verify the inter-chunk delays (first 3 calls)
        for i in range(3):
            call_args = mock_sleep.call_args_list[i][0]
            self.assertEqual(call_args[0], 0.003)


class TestInterFrameDelayIntegration(unittest.TestCase):
    """Integration test for inter-frame delay impact on throughput."""
    
    @patch('raspberry_pi.serial_comm.serial.Serial')
    def test_inter_frame_delay_reduces_throughput(self, mock_serial_class):
        """Test that inter-frame delay effectively reduces frame throughput."""
        # Setup mock serial
        mock_serial = Mock()
        mock_serial.is_open = True
        mock_serial_class.return_value = mock_serial
        
        # Test without delay
        comm_no_delay = SerialComm(inter_frame_delay=0.0)
        comm_no_delay.ser = mock_serial
        
        start_time = time.time()
        for _ in range(10):
            comm_no_delay.send(b"test data")
        time_no_delay = time.time() - start_time
        
        # Test with delay
        comm_with_delay = SerialComm(inter_frame_delay=0.05)
        comm_with_delay.ser = mock_serial
        
        start_time = time.time()
        for _ in range(10):
            comm_with_delay.send(b"test data")
        time_with_delay = time.time() - start_time
        
        # With 10 frames and 0.05s delay, should take at least 0.5s
        # Without delay, should be much faster
        self.assertGreater(time_with_delay, 0.4)  # Allow some margin
        self.assertLess(time_no_delay, time_with_delay)


if __name__ == '__main__':
    unittest.main()
