import unittest

import numpy as np

from h264_codec import H264Decoder, WaveletEncoder


class WaveletCodecTests(unittest.TestCase):
    def test_wavelet_roundtrip_preserves_shape(self) -> None:
        rng = np.random.default_rng(1234)
        frame = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)

        encoder = WaveletEncoder(width=16, height=16, levels=2, quant_step=1)
        chunks = encoder.encode(frame)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].codec, "wavelet")
        self.assertTrue(chunks[0].is_keyframe)
        self.assertFalse(chunks[0].is_config)

        decoder = H264Decoder()
        decoded_frames = []
        for chunk in chunks:
            decoded_frames.extend(decoder.decode(chunk))

        self.assertGreater(len(decoded_frames), 0)
        decoded = decoded_frames[-1]
        self.assertEqual(decoded.shape, frame.shape)

        diff = np.abs(decoded.astype(np.int16) - frame.astype(np.int16))
        self.assertLess(diff.mean(), 2.0)
        self.assertLess(diff.max(), 16)


if __name__ == "__main__":
    unittest.main()
