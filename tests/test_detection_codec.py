import unittest

import numpy as np

from h264_codec import DetectionBox, H264Decoder, YOLODetectionEncoder


class DummyDetector:
    def detect(self, frame_bgr: np.ndarray) -> list[DetectionBox]:
        return [(0.5, 0.5, 0.3, 0.4, 0.9)]


class YOLOCodecTests(unittest.TestCase):
    def test_yolo_roundtrip_produces_overlay(self) -> None:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        encoder = YOLODetectionEncoder(
            width=64,
            height=48,
            weights_path="unused",
            confidence=0.5,
            iou=0.4,
            device="cpu",
            max_detections=5,
            detector=DummyDetector(),
        )
        chunks = encoder.encode(frame)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.codec, "yolo")
        self.assertTrue(chunk.is_keyframe)

        decoder = H264Decoder()
        decoded_frames = []
        for chunk in chunks:
            decoded_frames.extend(decoder.decode(chunk))

        self.assertEqual(len(decoded_frames), 1)
        decoded = decoded_frames[0]
        self.assertEqual(decoded.shape, frame.shape)
        self.assertGreater(int(decoded.sum()), 0)
 
        preview = encoder.preview_frame
        self.assertEqual(preview.shape, frame.shape)
        self.assertGreater(int(preview.sum()), 0)


if __name__ == "__main__":
    unittest.main()
