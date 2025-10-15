import unittest

import cv2
import numpy as np

from h264_codec import ContourEncoder, H264Decoder


class ContourCodecTests(unittest.TestCase):
    def test_contour_roundtrip_generates_outline(self) -> None:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.circle(frame, (32, 32), 18, (255, 255, 255), thickness=2)

        encoder = ContourEncoder(width=64, height=64, samples=64, coefficients=12)
        chunks = encoder.encode(frame)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.codec, "contour")
        self.assertTrue(chunk.is_keyframe)

        decoder = H264Decoder()
        decoded_frames = []
        for chunk in chunks:
            decoded_frames.extend(decoder.decode(chunk))

        self.assertEqual(len(decoded_frames), 1)
        decoded = decoded_frames[0]
        self.assertEqual(decoded.shape, frame.shape)
        self.assertGreater(int(decoded.sum()), 0)


if __name__ == "__main__":
    unittest.main()
