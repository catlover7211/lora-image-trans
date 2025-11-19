"""Compressed Sensing decoder for received images."""
from typing import Optional
from functools import lru_cache

import cv2
import numpy as np


class CSDecoder:
    """Decodes Compressed Sensing encoded images."""
    
    def __init__(self):
        """Initialize CS decoder."""
        pass
    
    def decode(self, data: bytes, iterations: int = 3) -> Optional[np.ndarray]:
        """Decode CS bytes to image array.
        
        Args:
            data: CS encoded bytes
            iterations: Number of GAP reconstruction iterations (default: 3)
            
        Returns:
            BGR grayscale image or None on failure
        """
        try:
            # Parse header safely (avoid uint8 overflow by casting to Python ints)
            data_arr = np.frombuffer(data, dtype=np.uint8)
            if data_arr.size < 6:
                return None

            h_lo = int(data_arr[0])
            h_hi = int(data_arr[1])
            w_lo = int(data_arr[2])
            w_hi = int(data_arr[3])
            height = (h_hi << 8) | h_lo
            width = (w_hi << 8) | w_lo
            block_size = int(data_arr[4])
            measurements_per_block = int(data_arr[5])

            # Basic sanity checks
            if height <= 0 or width <= 0:
                return None
            if block_size < 4 or block_size > 64:
                return None
            max_coeffs = block_size * block_size
            if measurements_per_block <= 0 or measurements_per_block > max_coeffs:
                return None

            payload = data_arr[6:]
            
            # Calculate padded dimensions
            padded_h = ((height + block_size - 1) // block_size) * block_size
            padded_w = ((width + block_size - 1) // block_size) * block_size
            
            # Calculate expected data length
            num_blocks_h = padded_h // block_size
            num_blocks_w = padded_w // block_size
            total_blocks = int(num_blocks_h * num_blocks_w)
            expected_length = int(total_blocks * measurements_per_block)
            
            if payload.size < expected_length:
                return None
            
            # Prepare reshaped view: (total_blocks, K)
            K = measurements_per_block
            payload = payload[:expected_length]
            meas2d = payload.reshape((total_blocks, K)).astype(np.float32)

            # Dequantize all at once
            deq2d = (meas2d - 128.0) * 16.0

            # Precompute zigzag indices and DCT basis
            zz_idx = _zigzag_indices(block_size)
            C = _dct_basis(block_size)

            # Scatter dequantized coefficients into flat DCT arrays for all blocks
            n = block_size * block_size
            dct_flat = np.zeros((total_blocks, n), dtype=np.float32)
            dct_flat[np.arange(total_blocks)[:, None], zz_idx[:K]] = deq2d
            dct_blocks = dct_flat.reshape((total_blocks, block_size, block_size))

            # Initial reconstruction (Zero-filling IDCT)
            # Batched IDCT: C.T @ block @ C
            blocks = np.einsum('ij,bjk,kl->bil', C.T, dct_blocks, C, optimize=True)

            # Reassemble image from blocks
            reconstructed = (
                blocks.reshape(num_blocks_h, num_blocks_w, block_size, block_size)
                .swapaxes(1, 2)
                .reshape(padded_h, padded_w)
            )
            
            # GAP (Generalized Alternating Projection) Reconstruction
            # Inspired by DeSCI (https://github.com/liuyang12/DeSCI)
            # We use a simplified GAP-TV/Denoise approach for real-time performance.
            if iterations > 0:
                curr_im = reconstructed.copy()
                
                # Pre-calculate mask for data consistency
                # We need to enforce the measured coefficients
                # The mask is 1 for unmeasured, 0 for measured (because we replace measured)
                # Actually we just need to know which indices are measured: zz_idx[:K]
                
                for _ in range(iterations):
                    # 1. Denoise step (Projection onto signal manifold)
                    # Use Fast Non-Local Means as a proxy for WNNM (DeSCI) or TV
                    # For small images (128x72), this is reasonably fast
                    denoised = cv2.fastNlMeansDenoising(
                        np.clip(curr_im, 0, 255).astype(np.uint8), 
                        None, 
                        h=10, 
                        templateWindowSize=7, 
                        searchWindowSize=21
                    ).astype(np.float32)
                    
                    # 2. Data Consistency step (Projection onto measurement subspace)
                    # Transform denoised image to DCT domain
                    
                    # Split into blocks
                    d_blocks = (
                        denoised.reshape(num_blocks_h, block_size, num_blocks_w, block_size)
                        .swapaxes(1, 2)
                        .reshape(total_blocks, block_size, block_size)
                    )
                    
                    # Forward DCT: C @ block @ C.T
                    d_dct_blocks = np.einsum('ij,bjk,kl->bil', C, d_blocks, C.T, optimize=True)
                    
                    # Flatten
                    d_dct_flat = d_dct_blocks.reshape(total_blocks, n)
                    
                    # Enforce measurements: Replace the first K zigzag coefficients with measured ones
                    # We use the original dequantized values 'deq2d'
                    d_dct_flat[np.arange(total_blocks)[:, None], zz_idx[:K]] = deq2d
                    
                    # Reshape back to blocks
                    d_dct_blocks_enforced = d_dct_flat.reshape((total_blocks, block_size, block_size))
                    
                    # Inverse DCT
                    blocks_enforced = np.einsum('ij,bjk,kl->bil', C.T, d_dct_blocks_enforced, C, optimize=True)
                    
                    # Reassemble
                    curr_im = (
                        blocks_enforced.reshape(num_blocks_h, num_blocks_w, block_size, block_size)
                        .swapaxes(1, 2)
                        .reshape(padded_h, padded_w)
                    )
                
                reconstructed = curr_im

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
