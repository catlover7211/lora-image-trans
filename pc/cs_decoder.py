"""Compressed Sensing decoder for received images."""
from typing import Optional

import cv2
import numpy as np


class CSDecoder:
    """Decodes Compressed Sensing encoded images."""
    
    def __init__(self):
        """Initialize CS decoder."""
        pass
    
    def decode(self, data: bytes) -> Optional[np.ndarray]:
        """Decode CS bytes to image array.
        
        Args:
            data: CS encoded bytes
            
        Returns:
            BGR grayscale image or None on failure
        """
        try:
            # Parse header
            data_arr = np.frombuffer(data, dtype=np.uint8)
            if len(data_arr) < 6:
                return None
            
            height = data_arr[0] | (data_arr[1] << 8)
            width = data_arr[2] | (data_arr[3] << 8)
            block_size = data_arr[4]
            measurements_per_block = data_arr[5]
            
            payload = data_arr[6:]
            
            # Calculate padded dimensions
            padded_h = ((height + block_size - 1) // block_size) * block_size
            padded_w = ((width + block_size - 1) // block_size) * block_size
            
            # Calculate expected data length
            num_blocks_h = padded_h // block_size
            num_blocks_w = padded_w // block_size
            total_blocks = num_blocks_h * num_blocks_w
            expected_length = total_blocks * measurements_per_block
            
            if len(payload) < expected_length:
                return None
            
            # Reconstruct image
            reconstructed = np.zeros((padded_h, padded_w), dtype=np.float32)
            
            block_idx = 0
            for i in range(0, padded_h, block_size):
                for j in range(0, padded_w, block_size):
                    # Extract measurements for this block
                    start_idx = block_idx * measurements_per_block
                    end_idx = start_idx + measurements_per_block
                    measurements = payload[start_idx:end_idx].astype(np.float32)
                    
                    # Dequantize
                    dequantized = (measurements - 128) * 16.0
                    
                    # Reconstruct DCT coefficients (fill with zeros for missing coefficients)
                    dct_block = np.zeros(block_size * block_size, dtype=np.float32)
                    dct_block[:len(dequantized)] = dequantized
                    
                    # Reverse zigzag scan
                    dct_2d = self._inverse_zigzag_scan(dct_block, block_size)
                    
                    # Apply inverse DCT
                    block = cv2.idct(dct_2d)
                    
                    reconstructed[i:i+block_size, j:j+block_size] = block
                    block_idx += 1
            
            # Crop to original size
            reconstructed = reconstructed[:height, :width]
            
            # Clip and convert to uint8
            reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
            
            # Convert grayscale to BGR for display
            bgr_image = cv2.cvtColor(reconstructed, cv2.COLOR_GRAY2BGR)
            
            return bgr_image
            
        except Exception as e:
            print(f"CS decoding error: {e}")
            return None
    
    def _inverse_zigzag_scan(self, flat: np.ndarray, size: int) -> np.ndarray:
        """Perform inverse zigzag scan to reconstruct 2D block."""
        block = np.zeros((size, size), dtype=np.float32)
        idx = 0
        
        for s in range(2 * size - 1):
            if s < size:
                if s % 2 == 0:
                    for i in range(s + 1):
                        if idx < len(flat):
                            block[s - i, i] = flat[idx]
                            idx += 1
                else:
                    for i in range(s + 1):
                        if idx < len(flat):
                            block[i, s - i] = flat[idx]
                            idx += 1
            else:
                if s % 2 == 0:
                    for i in range(s - size + 1, size):
                        if idx < len(flat):
                            block[s - i, i] = flat[idx]
                            idx += 1
                else:
                    for i in range(s - size + 1, size):
                        if idx < len(flat):
                            block[i, s - i] = flat[idx]
                            idx += 1
        
        return block
