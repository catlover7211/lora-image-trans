import unittest

import numpy as np

from h264_codec import H264Decoder, JPEGEncoder


class JPEGCodecTests(unittest.TestCase):
    def test_jpeg_roundtrip_preserves_dimensions(self) -> None:
        grid_y, grid_x = np.meshgrid(
            np.linspace(0, 255, 24, dtype=np.float32),
            np.linspace(0, 255, 32, dtype=np.float32),
            indexing="ij",
        )
        base = (0.7 * grid_x + 0.3 * grid_y).astype(np.uint8)
        frame = np.stack((base, np.roll(base, 5, axis=1), np.roll(base, 3, axis=0)), axis=2)

        encoder = JPEGEncoder(width=32, height=24, quality=90)
        chunks = encoder.encode(frame.copy())

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.codec, "jpeg")
        self.assertTrue(chunk.is_keyframe)
        self.assertFalse(chunk.is_config)

        decoder = H264Decoder()
        decoded_frames = []
        for chunk in chunks:
            decoded_frames.extend(decoder.decode(chunk))

        self.assertGreater(len(decoded_frames), 0)
        decoded = decoded_frames[-1]
        self.assertEqual(decoded.shape, frame.shape)

        diff = np.abs(decoded.astype(np.int16) - frame.astype(np.int16))
        self.assertLess(diff.mean(), 6.5)
        self.assertLess(diff.max(), 70)


if __name__ == "__main__":
    unittest.main()
