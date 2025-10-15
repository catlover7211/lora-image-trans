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
        try:
            if weights.exists():
                self.model = torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path=str(weights),
                    source="local",
                    trust_repo=True,
                )
            elif weights_path.endswith(".pt"):
                # 依官方文件，若提供 .pt 檔名會視為自訂權重並透過第三個參數傳入
                if weights_path.startswith("http://") or weights_path.startswith("https://"):
                    self.model = torch.hub.load(
                        "ultralytics/yolov5",
                        "custom",
                        weights_path,
                        trust_repo=True,
                        source="github",
                    )
                else:
                    raise RuntimeError(
                        "找不到指定的 YOLOv5 權重檔 '"
                        + weights_path
                        + "'。請提供有效的本機路徑、HTTP(S) 連結，或改用官方模型名稱 (例如 yolov5n)。"
                    )
            else:
                # 若提供的是官方模型名稱 (例如 yolov5s) 則透過 pretrained=True 載入
                self.model = torch.hub.load(
                    "ultralytics/yolov5",
                    weights_path,
                    pretrained=True,
                    trust_repo=True,
                    source="github",
                )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "載入 YOLOv5 模型時找不到 'ultralytics' 相依套件，請先執行 pip install ultralytics"
            ) from exc
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
