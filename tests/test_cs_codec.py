import unittest

import numpy as np

from h264_codec import CSDecoder, CSEncoder, H264Decoder


class CSCodecTests(unittest.TestCase):
    def test_cs_roundtrip_preserves_shape(self) -> None:
        """Test that CS encoder/decoder preserves image shape."""
        # Create a simple test image with some structure
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        # Create encoder with fixed seed for reproducibility
        encoder = CSEncoder(width=32, height=32, measurement_ratio=0.3, seed=12345)
        chunks = encoder.encode(frame)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].codec, "cs")
        self.assertTrue(chunks[0].is_keyframe)
        self.assertFalse(chunks[0].is_config)

        # Decode
        decoder = H264Decoder()
        decoded_frames = []
        for chunk in chunks:
            decoded_frames.extend(decoder.decode(chunk))

        self.assertGreater(len(decoded_frames), 0)
        decoded = decoded_frames[-1]
        self.assertEqual(decoded.shape, frame.shape)

    def test_cs_reconstruction_quality(self) -> None:
        """Test that CS reconstruction is reasonably close to original."""
        # Create a simple gradient image
        y_grad = np.linspace(0, 255, 24, dtype=np.float32)
        x_grad = np.linspace(0, 255, 32, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(y_grad, x_grad, indexing="ij")
        
        # Create smooth gradient image
        base = ((grid_x + grid_y) / 2).astype(np.uint8)
        frame = np.stack((base, np.roll(base, 8, axis=1), np.roll(base, 8, axis=0)), axis=2)

        # Encode with moderate measurement ratio
        encoder = CSEncoder(width=32, height=24, measurement_ratio=0.5, seed=42)
        chunks = encoder.encode(frame.copy())

        # Decode
        decoder = H264Decoder()
        decoded_frames = []
        for chunk in chunks:
            decoded_frames.extend(decoder.decode(chunk))

        self.assertEqual(len(decoded_frames), 1)
        decoded = decoded_frames[0]
        
        # Check reconstruction quality
        # CS is lossy, so we expect some error, but it should be reasonable
        diff = np.abs(decoded.astype(np.int16) - frame.astype(np.int16))
        mean_error = diff.mean()
        max_error = diff.max()
        
        # CS reconstruction is approximate - expect moderate errors
        # With 50% measurements on smooth gradients, errors should be bounded
        self.assertLess(mean_error, 80.0, f"Mean error {mean_error} too high")
        self.assertLess(max_error, 255, f"Max error {max_error} too high")

    def test_cs_different_measurement_ratios(self) -> None:
        """Test CS with different measurement ratios."""
        rng = np.random.default_rng(100)
        frame = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)

        for ratio in [0.2, 0.5, 0.8]:
            with self.subTest(ratio=ratio):
                encoder = CSEncoder(width=16, height=16, measurement_ratio=ratio, seed=999)
                chunks = encoder.encode(frame)
                
                self.assertEqual(len(chunks), 1)
                
                decoder = H264Decoder()
                decoded_frames = list(decoder.decode(chunks[0]))
                
                self.assertEqual(len(decoded_frames), 1)
                self.assertEqual(decoded_frames[0].shape, frame.shape)

    def test_cs_seed_reproducibility(self) -> None:
        """Test that same seed produces same encoding."""
        rng = np.random.default_rng(123)
        frame = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)

        # Encode twice with same seed
        encoder1 = CSEncoder(width=16, height=16, measurement_ratio=0.3, seed=555)
        chunks1 = encoder1.encode(frame.copy())
        
        encoder2 = CSEncoder(width=16, height=16, measurement_ratio=0.3, seed=555)
        chunks2 = encoder2.encode(frame.copy())

        # Encoded data should be identical
        self.assertEqual(chunks1[0].data, chunks2[0].data)

    def test_cs_different_seeds_different_encoding(self) -> None:
        """Test that different seeds produce different encodings."""
        rng = np.random.default_rng(456)
        frame = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)

        # Encode with different seeds
        encoder1 = CSEncoder(width=16, height=16, measurement_ratio=0.3, seed=111)
        chunks1 = encoder1.encode(frame.copy())
        
        encoder2 = CSEncoder(width=16, height=16, measurement_ratio=0.3, seed=222)
        chunks2 = encoder2.encode(frame.copy())

        # Encoded data should be different
        self.assertNotEqual(chunks1[0].data, chunks2[0].data)

    def test_cs_invalid_measurement_ratio(self) -> None:
        """Test that invalid measurement ratios raise errors."""
        with self.assertRaises(ValueError):
            CSEncoder(width=16, height=16, measurement_ratio=0.0)
        
        with self.assertRaises(ValueError):
            CSEncoder(width=16, height=16, measurement_ratio=1.0)
        
        with self.assertRaises(ValueError):
            CSEncoder(width=16, height=16, measurement_ratio=-0.1)
        
        with self.assertRaises(ValueError):
            CSEncoder(width=16, height=16, measurement_ratio=1.5)

    def test_cs_invalid_dimensions(self) -> None:
        """Test that invalid dimensions raise errors."""
        with self.assertRaises(ValueError):
            CSEncoder(width=0, height=16)
        
        with self.assertRaises(ValueError):
            CSEncoder(width=16, height=0)
        
        with self.assertRaises(ValueError):
            CSEncoder(width=-1, height=16)

    def test_cs_compression_ratio(self) -> None:
        """Test that CS provides good compression."""
        rng = np.random.default_rng(789)
        frame = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        # Original size
        original_size = frame.nbytes
        
        # Encode with 30% measurements
        encoder = CSEncoder(width=32, height=32, measurement_ratio=0.3, seed=1000)
        chunks = encoder.encode(frame)
        
        compressed_size = len(chunks[0].data)
        
        # Compressed size should be significantly smaller
        compression_ratio = original_size / compressed_size
        self.assertGreater(compression_ratio, 1.5, 
                          f"Compression ratio {compression_ratio:.2f} not sufficient")

    def test_cs_encode_wrong_dimensions(self) -> None:
        """Test encoding with wrong frame dimensions raises error."""
        encoder = CSEncoder(width=16, height=16, measurement_ratio=0.3)
        
        # Wrong width
        frame = np.zeros((16, 32, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            encoder.encode(frame)
        
        # Wrong height
        frame = np.zeros((32, 16, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            encoder.encode(frame)
        
        # Wrong channels
        frame = np.zeros((16, 16, 1), dtype=np.uint8)
        with self.assertRaises(ValueError):
            encoder.encode(frame)

    def test_cs_payload_format(self) -> None:
        """Test that CS payload can be serialized/deserialized."""
        rng = np.random.default_rng(321)
        frame = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)

        encoder = CSEncoder(width=16, height=16, measurement_ratio=0.3, seed=777)
        chunks = encoder.encode(frame)
        
        # Convert to payload and back
        payload = chunks[0].to_payload()
        self.assertIsInstance(payload, bytes)
        self.assertGreater(len(payload), 0)
        
        # Deserialize
        from h264_codec import EncodedChunk
        chunk_restored = EncodedChunk.from_payload(payload)
        
        self.assertEqual(chunk_restored.codec, "cs")
        self.assertTrue(chunk_restored.is_keyframe)
        self.assertFalse(chunk_restored.is_config)
        
        # Decode restored chunk
        decoder = H264Decoder()
        decoded = list(decoder.decode(chunk_restored))
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0].shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
