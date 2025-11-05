"""Compressed Sensing encoder for image transmission."""
from typing import Optional

import cv2
import numpy as np

from common.config import CS_MEASUREMENT_RATE, CS_BLOCK_SIZE


class CSEncoder:
    """Encodes images using Compressed Sensing (CS) technique."""
    
    def __init__(self, measurement_rate: float = CS_MEASUREMENT_RATE, block_size: int = CS_BLOCK_SIZE):
        """Initialize CS encoder.
        
        Args:
            measurement_rate: Sampling rate (0.0-1.0), default 0.3
            block_size: Block size for block-based CS, default 8
        """
        if not 0.0 < measurement_rate <= 1.0:
            raise ValueError("Measurement rate must be between 0 and 1")
        if block_size < 4 or block_size > 64:
            raise ValueError("Block size must be between 4 and 64")
        
        self.measurement_rate = measurement_rate
        self.block_size = block_size
        
    def encode(self, image: np.ndarray) -> Optional[bytes]:
        """Encode image using Compressed Sensing.
        
        This implementation uses a simplified CS approach:
        1. Convert to grayscale
        2. Apply DCT (Discrete Cosine Transform) on blocks
        3. Sample coefficients based on measurement rate
        4. Quantize and pack
        
        Args:
            image: BGR image array
            
        Returns:
            CS encoded bytes or None on failure
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            height, width = gray.shape
            
            # Pad image to be divisible by block_size
            pad_h = (self.block_size - height % self.block_size) % self.block_size
            pad_w = (self.block_size - width % self.block_size) % self.block_size
            
            if pad_h > 0 or pad_w > 0:
                gray = np.pad(gray, ((0, pad_h), (0, pad_w)), mode='edge')
            
            padded_h, padded_w = gray.shape
            
            # Calculate number of measurements per block
            measurements_per_block = max(1, int(self.block_size * self.block_size * self.measurement_rate))
            
            # Process blocks
            encoded_blocks = []
            for i in range(0, padded_h, self.block_size):
                for j in range(0, padded_w, self.block_size):
                    block = gray[i:i+self.block_size, j:j+self.block_size].astype(np.float32)
                    
                    # Apply DCT
                    dct_block = cv2.dct(block)
                    
                    # Zigzag scan and sample top coefficients
                    flat_dct = self._zigzag_scan(dct_block)
                    sampled = flat_dct[:measurements_per_block]
                    
                    # Quantize to 8-bit
                    quantized = np.clip(sampled / 16.0 + 128, 0, 255).astype(np.uint8)
                    encoded_blocks.append(quantized)
            
            # Pack header and data
            header = np.array([
                height & 0xFF, (height >> 8) & 0xFF,
                width & 0xFF, (width >> 8) & 0xFF,
                self.block_size,
                measurements_per_block
            ], dtype=np.uint8)
            
            data = np.concatenate([header] + encoded_blocks)
            return data.tobytes()
            
        except Exception as e:
            print(f"CS encoding error: {e}")
            return None
    
    def _zigzag_scan(self, block: np.ndarray) -> np.ndarray:
        """Perform zigzag scan on a 2D block."""
        size = block.shape[0]
        result = []
        
        for s in range(2 * size - 1):
            if s < size:
                if s % 2 == 0:
                    for i in range(s + 1):
                        result.append(block[s - i, i])
                else:
                    for i in range(s + 1):
                        result.append(block[i, s - i])
            else:
                if s % 2 == 0:
                    for i in range(s - size + 1, size):
                        result.append(block[s - i, i])
                else:
                    for i in range(s - size + 1, size):
                        result.append(block[i, s - i])
        
        return np.array(result)
