"""Compressed Sensing encoder for image transmission."""
from typing import Optional
from functools import lru_cache

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
            n = self.block_size * self.block_size
            measurements_per_block = max(1, int(n * self.measurement_rate))

            num_blocks_h = padded_h // self.block_size
            num_blocks_w = padded_w // self.block_size
            total_blocks = num_blocks_h * num_blocks_w

            # Views of blocks: (nbh, bs, nbw, bs) -> (nb, bs, bs)
            bs = self.block_size
            blocks = (
                gray.reshape(num_blocks_h, bs, num_blocks_w, bs)
                .swapaxes(1, 2)
                .reshape(total_blocks, bs, bs)
                .astype(np.float32)
            )

            # Precompute DCT basis and zigzag indices
            C = _dct_basis(bs)  # orthonormal DCT-II
            zz_idx = _zigzag_indices(bs)

            # Batched DCT: C @ block @ C.T for all blocks
            dct_blocks = np.einsum('ij,bjk,kl->bil', C, blocks, C.T, optimize=True)

            # Flatten and sample zigzag K coefficients for all blocks
            flat = dct_blocks.reshape(total_blocks, bs * bs)
            sampled = flat[:, zz_idx[:measurements_per_block]]

            # Quantize to 8-bit
            encoded_blocks = np.clip(sampled / 16.0 + 128.0, 0, 255).astype(np.uint8)

            # Pack header and data
            header = np.array([
                height & 0xFF, (height >> 8) & 0xFF,
                width & 0xFF, (width >> 8) & 0xFF,
                self.block_size,
                measurements_per_block
            ], dtype=np.uint8)
            
            data = np.concatenate([header, encoded_blocks.reshape(-1)])
            return data.tobytes()
            
        except Exception as e:
            print(f"CS encoding error: {e}")
            return None
    
@lru_cache(maxsize=None)
def _zigzag_indices(size: int) -> np.ndarray:
    """Return raster indices for zigzag order as a 1D numpy array of length size*size."""
    idx = []
    for s in range(2 * size - 1):
        if s < size:
            rng = range(s + 1)
            if s % 2 == 0:
                for i in rng:
                    idx.append((s - i) * size + i)
            else:
                for i in rng:
                    idx.append(i * size + (s - i))
        else:
            rng = range(s - size + 1, size)
            if s % 2 == 0:
                for i in rng:
                    idx.append((s - i) * size + i)
            else:
                for i in rng:
                    idx.append(i * size + (s - i))
    return np.array(idx, dtype=np.int32)

@lru_cache(maxsize=None)
def _dct_basis(n: int) -> np.ndarray:
    """Create an orthonormal DCT-II basis matrix of size n x n."""
    C = np.zeros((n, n), dtype=np.float32)
    factor = np.pi / (2.0 * n)
    scale0 = np.sqrt(1.0 / n)
    scale = np.sqrt(2.0 / n)
    for k in range(n):
        s = scale0 if k == 0 else scale
        for i in range(n):
            C[k, i] = s * np.cos((2 * i + 1) * k * factor)
    return C
