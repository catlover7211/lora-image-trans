from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np


class YOLOv5Detector:
    """Wrapper for running YOLOv5 inference via torch.hub."""

    def __init__(
        self,
        *,
        weights_path: str,
        confidence: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
        max_detections: int = 10,
    ) -> None:
        try:
            import torch  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("YOLOv5 偵測需要安裝 torch 套件。") from exc

        weights = Path(weights_path)
        if weights.exists():
            self.model = torch.hub.load("ultralytics/yolov5n-seg.pt", "custom", path=str(weights), source="local")
        else:
            # 若提供的是官方模型名稱 (例如 yolov5s)，讓 torch.hub 自行下載
            self.model = torch.hub.load("ultralytics/yolov5n-seg.pt", weights_path, pretrained=True)
        self.model.to(device)
        self.model.conf = float(confidence)
        self.model.iou = float(iou)
        self.model.max_det = int(max_detections)
        self.device = device

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("YOLOv5 偵測僅支援 BGR 三通道影像")

        # YOLOv5 期望 RGB 格式
        frame_rgb = frame_bgr[:, :, ::-1]
        results = self.model(frame_rgb, size=640)
        tensor = results.xyxy[0]
        if tensor.is_cuda:
            tensor = tensor.cpu()
        detections: List[Tuple[float, float, float, float, float]] = []
        height, width = frame_bgr.shape[0], frame_bgr.shape[1]
        for row in tensor.numpy():
            x1, y1, x2, y2, conf, _cls = row.tolist()
            w = max(x2 - x1, 0.0)
            h = max(y2 - y1, 0.0)
            if width <= 0 or height <= 0:
                continue
            cx = (x1 + x2) / 2.0 / width
            cy = (y1 + y2) / 2.0 / height
            detections.append(
                (
                    float(np.clip(cx, 0.0, 1.0)),
                    float(np.clip(cy, 0.0, 1.0)),
                    float(np.clip(w / width, 0.0, 1.0)),
                    float(np.clip(h / height, 0.0, 1.0)),
                    float(np.clip(conf, 0.0, 1.0)),
                )
            )
        return detections
