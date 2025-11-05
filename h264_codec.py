"""Utilities for video encoding and decoding over the serial protocol."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, List, Literal, Optional, Tuple
import struct
import zlib

import av  # type: ignore
import cv2
import numpy as np
from av.packet import Packet
from av.video.frame import PictureType

VideoCodec = Literal["h264", "h265", "av1", "wavelet", "jpeg", "contour", "yolo", "cs"]

DetectionBox = Tuple[float, float, float, float, float]


@dataclass(frozen=True)
class EncodedChunk:
    """Represents a single video payload fragment."""

    data: bytes
    codec: VideoCodec = "h264"
    is_keyframe: bool = False
    is_config: bool = False

    FLAG_KEYFRAME = 0x01
    FLAG_CONFIG = 0x02
    FLAG_CODEC_HEVC = 0x04
    FLAG_CODEC_AV1 = 0x08
    FLAG_CODEC_WAVELET = 0x10
    FLAG_CODEC_JPEG = 0x20
    FLAG_CODEC_CONTOUR = 0x40
    FLAG_CODEC_YOLO = 0x80
    FLAG_CODEC_CS = 0x100

    def to_payload(self) -> bytes:
        """Convert the chunk into a protocol payload with a flag prefix."""
        flags = 0
        if self.is_keyframe:
            flags |= self.FLAG_KEYFRAME
        if self.is_config:
            flags |= self.FLAG_CONFIG
        if self.codec == "h265":
            flags |= self.FLAG_CODEC_HEVC
        elif self.codec == "av1":
            flags |= self.FLAG_CODEC_AV1
        elif self.codec == "wavelet":
            flags |= self.FLAG_CODEC_WAVELET
        elif self.codec == "jpeg":
            flags |= self.FLAG_CODEC_JPEG
        elif self.codec == "contour":
            flags |= self.FLAG_CODEC_CONTOUR
        elif self.codec == "yolo":
            flags |= self.FLAG_CODEC_YOLO
        elif self.codec == "cs":
            flags |= self.FLAG_CODEC_CS
        # Use 2 bytes for flags to support extended codec types
        return struct.pack("<H", flags) + self.data

    @classmethod
    def from_payload(cls, payload: bytes) -> "EncodedChunk":
        if not payload:
            raise ValueError("payload 不可為空")
        
        # Try to detect format by checking if we can decode as 2-byte flags
        # Legacy format uses single byte (0x00-0xFF) followed by data
        # New format uses 2 bytes in little-endian
        if len(payload) >= 2:
            # Try reading as 2-byte flags first
            flags_2byte = struct.unpack("<H", payload[:2])[0]
            
            # Check if this could be a valid CS codec (FLAG_CODEC_CS = 0x100)
            # or if high byte is set for any new codec
            if flags_2byte & 0xFF00:
                # This is definitely new format (high byte is set)
                flags = flags_2byte
                data = payload[2:]
            else:
                # High byte is 0, could be either format
                # Check if low byte matches known legacy flags
                flags_1byte = payload[0]
                if flags_1byte & (cls.FLAG_CODEC_HEVC | cls.FLAG_CODEC_AV1 | 
                                 cls.FLAG_CODEC_WAVELET | cls.FLAG_CODEC_JPEG |
                                 cls.FLAG_CODEC_CONTOUR | cls.FLAG_CODEC_YOLO):
                    # Looks like legacy format
                    flags = flags_1byte
                    data = payload[1:]
                else:
                    # Use new format
                    flags = flags_2byte
                    data = payload[2:]
        else:
            # Single byte payload, must be legacy
            flags = payload[0]
            data = payload[1:]
        
        if flags & cls.FLAG_CODEC_CS:
            codec: VideoCodec = "cs"
        elif flags & cls.FLAG_CODEC_WAVELET:
            codec = "wavelet"
        elif flags & cls.FLAG_CODEC_CONTOUR:
            codec = "contour"
        elif flags & cls.FLAG_CODEC_JPEG:
            codec = "jpeg"
        elif flags & cls.FLAG_CODEC_YOLO:
            codec = "yolo"
        elif flags & cls.FLAG_CODEC_AV1:
            codec = "av1"
        elif flags & cls.FLAG_CODEC_HEVC:
            codec = "h265"
        else:
            codec = "h264"
        return cls(
            data=data,
            codec=codec,
            is_keyframe=bool(flags & cls.FLAG_KEYFRAME),
            is_config=bool(flags & cls.FLAG_CONFIG),
        )


MIN_ENCODER_BITRATE = 10_000


class H264Encoder:
    """Encodes numpy image frames into H.264/H.265/AV1 packets."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        fps: float = 10.0,
        bitrate: int = 400_000,
        keyframe_interval: int = 30,
        codec: VideoCodec = "h264",
    ) -> None:
        if codec == "wavelet":
            raise ValueError("wavelet 編碼請使用 WaveletEncoder")
        if width <= 0 or height <= 0:
            raise ValueError("影像尺寸必須為正整數")
        if fps <= 0:
            raise ValueError("fps 必須為正數")
        if bitrate < MIN_ENCODER_BITRATE:
            raise ValueError(
                f"位元率過低 (須 >= {MIN_ENCODER_BITRATE} bps) 可能導致編碼器無法啟動，"
                "請提高 bitrate 或改用較低解析度/幀率。"
            )

        self.width = width
        self.height = height
        self.fps = fps
        self.codec_name: VideoCodec = codec

        if codec == "h265":
            encoder_name = "libx265"
            fallback_name = "hevc"
        elif codec == "av1":
            encoder_name = "libaom-av1"
            fallback_name = "av1"
        else:
            encoder_name = "libx264"
            fallback_name = "h264"
        try:
            codec_ctx: Any = av.CodecContext.create(encoder_name, "w")
        except Exception:
            codec_ctx = av.CodecContext.create(fallback_name, "w")

        self.codec: Any = codec_ctx

        frame_rate = max(int(round(fps)), 1)
        codec_ctx.width = width
        codec_ctx.height = height
        codec_ctx.time_base = Fraction(1, frame_rate)
        codec_ctx.framerate = Fraction(frame_rate, 1)
        codec_ctx.pix_fmt = "yuv420p"
        codec_ctx.bit_rate = bitrate

        gop_size = max(keyframe_interval, 1)
        if codec == "h265":
            x265_params = ":".join(
                [
                    f"keyint={gop_size}",
                    f"min-keyint={gop_size}",
                    "scenecut=0",
                    "bframes=0",
                    "repeat-headers=1",
                    "rc-lookahead=0",
                    "frame-threads=1",
                ]
            )
            codec_ctx.options = {
                "preset": "veryfast",
                "tune": "zerolatency",
                "x265-params": x265_params,
            }
        elif codec == "av1":
            codec_ctx.options = {
                "cpu-used": "8",
                "end-usage": "cbr",
                "lag-in-frames": "0",
                "row-mt": "1",
                "tile-columns": "0",
                "tile-rows": "0",
                "enable-cdef": "1",
                "usage": "realtime",
                "kf-max-dist": str(gop_size),
                "kf-min-dist": str(gop_size),
            }
        else:
            codec_ctx.options = {
                "preset": "veryfast",
                "tune": "zerolatency",
                "profile": "baseline",
                "x264opts": "no-scenecut",
            }
        codec_ctx.gop_size = gop_size
        codec_ctx.open()

        self.keyframe_interval = gop_size
        self._frame_index = 0
        self._config_burst = 3  # how many upcoming frames should resend config
        self._force_config = True

    def encode(self, frame_bgr: np.ndarray) -> List[EncodedChunk]:
        """Encode a BGR frame and return zero or more codec chunks."""
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            raise ValueError("輸入影像尺寸與編碼器不符")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("輸入影像需為 BGR 三通道格式")

        chunks: List[EncodedChunk] = []

        is_keyframe_due = (self._frame_index % self.keyframe_interval == 0)
        if is_keyframe_due:
            self._force_config = True

        if self.codec.extradata and (self._force_config or self._config_burst > 0):
            chunks.append(
                EncodedChunk(
                    data=self.codec.extradata,
                    codec=self.codec_name,
                    is_config=True,
                    is_keyframe=True,
                )
            )
            if self._config_burst > 0:
                self._config_burst -= 1
            self._force_config = False

        video_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        if is_keyframe_due:
            try:
                video_frame.pict_type = PictureType.I
            except (AttributeError, TypeError):
                # 部分後端僅接受整數或不支援強制設定，忽略錯誤
                pass
        self._frame_index += 1

        for packet in self.codec.encode(video_frame):
            chunks.append(
                EncodedChunk(
                    data=bytes(packet),
                    codec=self.codec_name,
                    is_keyframe=packet.is_keyframe,
                    is_config=False,
                )
            )
        return chunks

    def flush(self) -> List[EncodedChunk]:
        """Flush any delayed packets from the encoder."""
        chunks: List[EncodedChunk] = []
        for packet in self.codec.encode(None):
            chunks.append(
                EncodedChunk(
                    data=bytes(packet),
                    codec=self.codec_name,
                    is_keyframe=packet.is_keyframe,
                    is_config=False,
                )
            )
        return chunks

    def force_keyframe(self) -> None:
        """Request the next frame to be encoded as a keyframe."""
        self._frame_index = 0
        self._force_config = True
        self._config_burst = max(self._config_burst, 1)

    def force_config_repeat(self, count: int = 3) -> None:
        """Schedule codec configuration to be resent for upcoming frames."""
        self._force_config = True
        self._config_burst = max(self._config_burst, count)


class JPEGEncoder:
    """Encodes numpy image frames into baseline JPEG payloads."""

    def __init__(self, width: int, height: int, *, quality: int = 85) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("影像尺寸必須為正整數")
        if not (1 <= quality <= 100):
            raise ValueError("JPEG 品質需介於 1 到 100")
        self.width = width
        self.height = height
        self.quality = int(quality)
        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]

    def encode(self, frame_bgr: np.ndarray) -> List[EncodedChunk]:
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            raise ValueError("輸入影像尺寸與編碼器不符")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("輸入影像需為 BGR 三通道格式")

        success, buffer = cv2.imencode(".jpg", frame_bgr, self._encode_params)
        if not success:
            raise RuntimeError("JPEG 編碼失敗")

        chunk = EncodedChunk(
            data=buffer.tobytes(),
            codec="jpeg",
            is_keyframe=True,
            is_config=False,
        )
        return [chunk]

    def force_keyframe(self) -> None:
        return

    def force_config_repeat(self, count: int = 3) -> None:
        return


class ContourEncoder:
    """Approximates frame contours with a truncated Fourier series."""

    VERSION = 1
    HEADER_STRUCT = struct.Struct("<BHHHffH")
    COEFF_STRUCT = struct.Struct("<ff")

    def __init__(self, width: int, height: int, *, samples: int = 128, coefficients: int = 16) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("影像尺寸必須為正整數")
        if samples <= 0:
            raise ValueError("samples 必須為正整數")
        if coefficients <= 0:
            raise ValueError("coefficients 必須為正整數")
        self.width = width
        self.height = height
        self.samples = samples
        max_coeffs = samples // 2 + 1
        self.coefficients = min(coefficients, max(1, max_coeffs))

    def encode(self, frame_bgr: np.ndarray) -> List[EncodedChunk]:
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            raise ValueError("輸入影像尺寸與編碼器不符")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("輸入影像需為 BGR 三通道格式")

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 使用自適應二值化，將影像轉為黑白，白色為前景
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # 尋找二值化影像中的輪廓
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            center = np.array([self.width / 2.0, self.height / 2.0], dtype=np.float32)
            radii_samples = np.zeros(self.samples, dtype=np.float32)
        else:
            largest = max(contours, key=cv2.contourArea)
            points = largest.reshape(-1, 2).astype(np.float32)
            center = points.mean(axis=0)
            vectors = points - center
            norms = np.linalg.norm(vectors, axis=1)
            if np.all(norms == 0):
                radii_samples = np.zeros(self.samples, dtype=np.float32)
            else:
                angles = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]) + 2 * np.pi, 2 * np.pi)
                order = np.argsort(angles)
                angles_sorted = angles[order]
                radii_sorted = norms[order]
                # 若角度重複，np.interp 仍可處理；擴展 2π 週期確保補間
                angles_ext = np.concatenate([angles_sorted, angles_sorted + 2 * np.pi])
                radii_ext = np.concatenate([radii_sorted, radii_sorted])
                sample_angles = np.linspace(0.0, 2 * np.pi, self.samples, endpoint=False, dtype=np.float32)
                radii_samples = np.interp(sample_angles, angles_ext, radii_ext).astype(np.float32)

        spectrum = np.fft.rfft(radii_samples)
        keep = min(self.coefficients, spectrum.shape[0])
        kept_coeffs = spectrum[:keep]

        header = self.HEADER_STRUCT.pack(
            self.VERSION,
            self.width,
            self.height,
            self.samples,
            float(center[0]),
            float(center[1]),
            keep,
        )

        body = bytearray(self.COEFF_STRUCT.size * keep)
        offset = 0
        for coeff in kept_coeffs:
            struct.pack_into("<ff", body, offset, float(np.real(coeff)), float(np.imag(coeff)))
            offset += self.COEFF_STRUCT.size

        chunk = EncodedChunk(
            data=header + bytes(body),
            codec="contour",
            is_keyframe=True,
            is_config=False,
        )
        return [chunk]

    def force_keyframe(self) -> None:
        return

    def force_config_repeat(self, count: int = 3) -> None:
        return


class DetectionEncoder:
    """Serialize normalized bounding boxes into protocol payloads."""

    VERSION = 1
    HEADER_STRUCT = struct.Struct("<BHHB")
    BOX_STRUCT = struct.Struct("<fffff")

    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("影像尺寸必須為正整數")
        self.width = width
        self.height = height

    def encode(self, boxes: List[DetectionBox]) -> List[EncodedChunk]:
        count = min(len(boxes), 255)
        header = self.HEADER_STRUCT.pack(
            self.VERSION,
            self.width,
            self.height,
            count,
        )
        body = bytearray(self.BOX_STRUCT.size * count)
        for idx, (cx, cy, w, h, conf) in enumerate(boxes[:count]):
            cx_c = float(np.clip(cx, 0.0, 1.0))
            cy_c = float(np.clip(cy, 0.0, 1.0))
            w_c = float(np.clip(w, 0.0, 1.0))
            h_c = float(np.clip(h, 0.0, 1.0))
            conf_c = float(np.clip(conf, 0.0, 1.0))
            offset = idx * self.BOX_STRUCT.size
            struct.pack_into("<fffff", body, offset, cx_c, cy_c, w_c, h_c, conf_c)

        chunk = EncodedChunk(
            data=header + bytes(body),
            codec="yolo",
            is_keyframe=True,
            is_config=False,
        )
        return [chunk]

    def force_keyframe(self) -> None:
        return

    def force_config_repeat(self, count: int = 3) -> None:
        return


class YOLODetectionEncoder:
    """Run YOLOv5 on frames to produce bounding boxes."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        weights_path: str,
        confidence: float,
        iou: float,
        device: str,
        max_detections: int,
        detector: Optional[Any] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.packetizer = DetectionEncoder(width, height)
        self.preview_frame = np.zeros((height, width, 3), dtype=np.uint8)
        if detector is not None:
            self.detector = detector
        else:
            from yolo_detector import YOLOv5Detector

            self.detector = YOLOv5Detector(
                weights_path=weights_path,
                confidence=confidence,
                iou=iou,
                device=device,
                max_detections=max_detections,
            )

    def encode(self, frame_bgr: np.ndarray) -> List[EncodedChunk]:
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            raise ValueError("輸入影像尺寸與編碼器不符")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("輸入影像需為 BGR 三通道格式")

        detections = self.detector.detect(frame_bgr)
        overlay = frame_bgr.copy()
        for cx, cy, w, h, conf in detections:
            abs_w = max(int(round(w * self.width)), 1)
            abs_h = max(int(round(h * self.height)), 1)
            x1 = int(round(cx * self.width - abs_w / 2))
            y1 = int(round(cy * self.height - abs_h / 2))
            x2 = x1 + abs_w
            y2 = y1 + abs_h
            x1 = int(np.clip(x1, 0, self.width - 1))
            y1 = int(np.clip(y1, 0, self.height - 1))
            x2 = int(np.clip(x2, 0, self.width - 1))
            y2 = int(np.clip(y2, 0, self.height - 1))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            label = f"{conf:.2f}"
            cv2.putText(overlay, label, (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        self.preview_frame = overlay
        return self.packetizer.encode(detections)

    def force_keyframe(self) -> None:
        return

    def force_config_repeat(self, count: int = 3) -> None:
        return


def _bgr_to_ycocg(frame: np.ndarray) -> np.ndarray:
    b = frame[..., 0].astype(np.int32)
    g = frame[..., 1].astype(np.int32)
    r = frame[..., 2].astype(np.int32)
    co = r - b
    t = b + (co >> 1)
    cg = g - t
    y = t + (cg >> 1)
    return np.stack((y, co, cg), axis=2)


def _ycocg_to_bgr(data: np.ndarray) -> np.ndarray:
    y = data[..., 0]
    co = data[..., 1]
    cg = data[..., 2]
    t = y - (cg >> 1)
    g = cg + t
    b = t - (co >> 1)
    r = co + b
    stacked = np.stack((b, g, r), axis=2)
    return np.clip(stacked, 0, 255).astype(np.uint8)


def _max_wavelet_levels(width: int, height: int) -> int:
    levels = 0
    while width % 2 == 0 and height % 2 == 0 and width > 1 and height > 1:
        levels += 1
        width //= 2
        height //= 2
    return levels


def _haar_forward_block(block: np.ndarray) -> None:
    rows, cols = block.shape
    half_cols = cols // 2
    temp = block.copy()
    block[:, :half_cols] = (temp[:, 0::2] + temp[:, 1::2]) / 2.0
    block[:, half_cols:half_cols * 2] = temp[:, 0::2] - temp[:, 1::2]

    temp = block.copy()
    half_rows = rows // 2
    block[:half_rows, :] = (temp[0::2, :] + temp[1::2, :]) / 2.0
    block[half_rows:half_rows * 2, :] = temp[0::2, :] - temp[1::2, :]


def _haar_inverse_block(block: np.ndarray) -> None:
    rows, cols = block.shape
    half_rows = rows // 2
    temp = block.copy()
    avg = temp[:half_rows, :]
    diff = temp[half_rows:half_rows * 2, :]
    recon = np.empty_like(block)
    recon[0::2, :] = avg + diff / 2.0
    recon[1::2, :] = avg - diff / 2.0

    temp = recon.copy()
    half_cols = cols // 2
    avg = temp[:, :half_cols]
    diff = temp[:, half_cols:half_cols * 2]
    block[:, 0::2] = avg + diff / 2.0
    block[:, 1::2] = avg - diff / 2.0


def _haar_forward(channel: np.ndarray, levels: int) -> np.ndarray:
    if levels <= 0:
        return channel.astype(np.float32)
    result = channel.astype(np.float32, copy=True)
    height, width = result.shape
    for level in range(levels):
        rows = height >> level
        cols = width >> level
        if rows < 2 or cols < 2:
            break
        _haar_forward_block(result[:rows, :cols])
    return result


def _haar_inverse(coeffs: np.ndarray, levels: int) -> np.ndarray:
    if levels <= 0:
        return coeffs.astype(np.float32)
    result = coeffs.astype(np.float32, copy=True)
    height, width = result.shape
    for level in reversed(range(levels)):
        rows = height >> level
        cols = width >> level
        if rows < 2 or cols < 2:
            continue
        _haar_inverse_block(result[:rows, :cols])
    return result


class WaveletEncoder:
    """Simple YCoCg + Haar wavelet encoder producing intra-only frames."""

    VERSION = 1
    HEADER_STRUCT = struct.Struct("<BHHHBB")  # version, width, height, quant, levels, channels

    def __init__(self, width: int, height: int, *, levels: int = 2, quant_step: int = 12) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("影像尺寸必須為正整數")
        if quant_step <= 0:
            raise ValueError("量化步階必須為正整數")
        self.width = width
        self.height = height
        self.quant_step = quant_step
        self.levels = min(max(0, levels), _max_wavelet_levels(width, height))

    def encode(self, frame_bgr: np.ndarray) -> List[EncodedChunk]:
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width or frame_bgr.shape[2] != 3:
            raise ValueError("輸入影像尺寸與編碼器不符")

        ycocg = _bgr_to_ycocg(frame_bgr)
        coeffs = []
        for c in range(3):
            forward = _haar_forward(ycocg[..., c], self.levels)
            quantized = np.rint(forward / self.quant_step).astype(np.int16)
            coeffs.append(quantized)
        quantized_stack = np.stack(coeffs, axis=2)
        payload_body = zlib.compress(quantized_stack.tobytes(), level=6)
        header = self.HEADER_STRUCT.pack(
            self.VERSION,
            self.width,
            self.height,
            self.quant_step,
            self.levels,
            3,
        )
        chunk = EncodedChunk(
            data=header + payload_body,
            codec="wavelet",
            is_keyframe=True,
            is_config=False,
        )
        return [chunk]

    def force_keyframe(self) -> None:
        # All frames are intra-only; nothing to do.
        return

    def force_config_repeat(self, count: int = 3) -> None:
        # Wavelet 編碼為每幀獨立，不需要額外的設定封包。
        return


class WaveletDecoder:
    """Inverse transform for the custom YCoCg + wavelet encoder."""

    HEADER_STRUCT = WaveletEncoder.HEADER_STRUCT

    def decode_chunk(self, chunk: EncodedChunk) -> np.ndarray:
        data = chunk.data
        if len(data) <= self.HEADER_STRUCT.size:
            raise RuntimeError("Wavelet 資料長度不正確")
        header = data[:self.HEADER_STRUCT.size]
        version, width, height, quant_step, levels, channels = self.HEADER_STRUCT.unpack(header)
        if version != WaveletEncoder.VERSION:
            raise RuntimeError(f"Wavelet 版本不支援: {version}")
        if channels != 3:
            raise RuntimeError("僅支援 3 通道波形資料")
        compressed = data[self.HEADER_STRUCT.size:]
        try:
            decompressed = zlib.decompress(compressed)
        except zlib.error as exc:
            raise RuntimeError(f"Wavelet 解壓失敗: {exc}") from exc
        expected_bytes = width * height * channels * np.dtype(np.int16).itemsize
        if len(decompressed) != expected_bytes:
            raise RuntimeError("Wavelet 解壓後長度不符")
        coeffs = np.frombuffer(decompressed, dtype=np.int16).reshape((height, width, channels)).astype(np.float32)
        coeffs *= quant_step
        channels_rec = []
        for c in range(channels):
            restored = _haar_inverse(coeffs[..., c], levels)
            channels_rec.append(restored)
        ycocg = np.stack(channels_rec, axis=2).astype(np.int32)
        return _ycocg_to_bgr(ycocg)

    def decode(self, chunk: EncodedChunk) -> Iterable[np.ndarray]:
        if chunk.is_config:
            return []
        return [self.decode_chunk(chunk)]

    def flush(self) -> Iterable[np.ndarray]:
        return []


class CSEncoder:
    """Compressed Sensing encoder using random Gaussian measurement matrix with DCT sparsifying basis."""

    VERSION = 1
    HEADER_STRUCT = struct.Struct("<BHHHHH")  # version, width, height, measurements, seed_high, seed_low

    def __init__(
        self, 
        width: int, 
        height: int, 
        *, 
        measurement_ratio: float = 0.3,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize CS encoder.
        
        Args:
            width: Image width
            height: Image height
            measurement_ratio: Ratio of measurements to original signal size (0 < ratio < 1)
            seed: Random seed for measurement matrix generation (for reproducibility)
        """
        if width <= 0 or height <= 0:
            raise ValueError("影像尺寸必須為正整數")
        if not (0 < measurement_ratio < 1):
            raise ValueError("測量比率必須介於 0 和 1 之間")
        
        self.width = width
        self.height = height
        self.measurement_ratio = measurement_ratio
        
        # Use provided seed or generate a random one
        if seed is None:
            self.seed = np.random.randint(0, 2**32 - 1)
        else:
            self.seed = seed % (2**32)
        
        # Calculate number of measurements per channel
        signal_length = width * height
        self.num_measurements = max(1, int(signal_length * measurement_ratio))
        
        # Generate measurement matrix (will be regenerated with same seed for decoding)
        rng = np.random.RandomState(self.seed)
        self.measurement_matrix = rng.randn(self.num_measurements, signal_length).astype(np.float32)
        # Normalize rows
        row_norms = np.linalg.norm(self.measurement_matrix, axis=1, keepdims=True)
        self.measurement_matrix /= (row_norms + 1e-8)

    def encode(self, frame_bgr: np.ndarray) -> List[EncodedChunk]:
        """Encode a BGR frame using compressed sensing."""
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            raise ValueError("輸入影像尺寸與編碼器不符")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("輸入影像需為 BGR 三通道格式")

        # Process each channel separately
        measurements_list = []
        for c in range(3):
            channel = frame_bgr[:, :, c].astype(np.float32)
            # Flatten channel
            signal = channel.flatten()
            # Apply measurement matrix: y = Φx
            measurements = self.measurement_matrix @ signal
            measurements_list.append(measurements)

        # Stack measurements for all channels
        all_measurements = np.stack(measurements_list, axis=0)  # Shape: (3, num_measurements)
        
        # Quantize to int16 for compression
        quantized = np.clip(all_measurements, -32768, 32767).astype(np.int16)
        
        # Compress the measurements
        payload_body = zlib.compress(quantized.tobytes(), level=6)
        
        # Pack header: version, width, height, num_measurements, seed (split into 2 uint16)
        seed_high = (self.seed >> 16) & 0xFFFF
        seed_low = self.seed & 0xFFFF
        header = self.HEADER_STRUCT.pack(
            self.VERSION,
            self.width,
            self.height,
            self.num_measurements,
            seed_high,
            seed_low,
        )
        
        chunk = EncodedChunk(
            data=header + payload_body,
            codec="cs",
            is_keyframe=True,
            is_config=False,
        )
        return [chunk]

    def force_keyframe(self) -> None:
        # All frames are intra-only; nothing to do.
        return

    def force_config_repeat(self, count: int = 3) -> None:
        # CS 編碼為每幀獨立，不需要額外的設定封包。
        return


class CSDecoder:
    """Compressed Sensing decoder using regularized least-squares reconstruction."""

    HEADER_STRUCT = CSEncoder.HEADER_STRUCT

    def __init__(self, regularization: float = 0.01):
        """
        Initialize CS decoder.
        
        Args:
            regularization: Regularization parameter for least-squares (ridge regression)
        """
        self.regularization = regularization

    def _reconstruct(
        self,
        measurements: np.ndarray,
        measurement_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Reconstruct signal using regularized least-squares (ridge regression).
        
        Args:
            measurements: Measurement vector y
            measurement_matrix: Measurement matrix Φ
            
        Returns:
            Reconstructed signal
        """
        m, n = measurement_matrix.shape
        
        # Use ridge regression: x = (Φ^T Φ + λI)^(-1) Φ^T y
        # This is more stable and faster than OMP for this application
        try:
            phi_t = measurement_matrix.T
            phi_t_phi = phi_t @ measurement_matrix
            
            # Add regularization to diagonal
            reg_matrix = phi_t_phi + self.regularization * np.eye(n, dtype=np.float32)
            
            # Solve: reg_matrix @ x = Φ^T y
            phi_t_y = phi_t @ measurements
            signal = np.linalg.solve(reg_matrix, phi_t_y)
            
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if solve fails
            try:
                signal = np.linalg.lstsq(measurement_matrix, measurements, rcond=None)[0]
            except np.linalg.LinAlgError:
                # Last resort: return zeros
                signal = np.zeros(n, dtype=np.float32)
        
        return signal

    def decode_chunk(self, chunk: EncodedChunk) -> np.ndarray:
        """Decode a CS chunk back to image."""
        data = chunk.data
        if len(data) <= self.HEADER_STRUCT.size:
            raise RuntimeError("CS 資料長度不正確")
        
        # Unpack header
        header = data[:self.HEADER_STRUCT.size]
        version, width, height, num_measurements, seed_high, seed_low = self.HEADER_STRUCT.unpack(header)
        
        if version != CSEncoder.VERSION and version != 0:
            raise RuntimeError(f"CS 版本不支援: {version}")
        
        # Reconstruct seed
        seed = (seed_high << 16) | seed_low
        
        # Decompress measurements
        compressed = data[self.HEADER_STRUCT.size:]
        try:
            decompressed = zlib.decompress(compressed)
        except zlib.error as exc:
            raise RuntimeError(f"CS 解壓失敗: {exc}") from exc
        
        # Verify size
        expected_bytes = 3 * num_measurements * np.dtype(np.int16).itemsize
        if len(decompressed) != expected_bytes:
            raise RuntimeError("CS 解壓後長度不符")
        
        # Reshape measurements
        measurements = np.frombuffer(decompressed, dtype=np.int16).reshape((3, num_measurements)).astype(np.float32)
        
        # Regenerate measurement matrix with same seed
        signal_length = width * height
        rng = np.random.RandomState(seed)
        measurement_matrix = rng.randn(num_measurements, signal_length).astype(np.float32)
        row_norms = np.linalg.norm(measurement_matrix, axis=1, keepdims=True)
        measurement_matrix /= (row_norms + 1e-8)
        
        # Reconstruct each channel
        reconstructed_channels = []
        for c in range(3):
            signal = self._reconstruct(
                measurements[c],
                measurement_matrix
            )
            channel = signal.reshape((height, width))
            # Clip to valid range
            channel = np.clip(channel, 0, 255)
            reconstructed_channels.append(channel)
        
        # Stack channels and convert to uint8
        frame_bgr = np.stack(reconstructed_channels, axis=2).astype(np.uint8)
        return frame_bgr

    def decode(self, chunk: EncodedChunk) -> Iterable[np.ndarray]:
        if chunk.is_config:
            return []
        return [self.decode_chunk(chunk)]

    def flush(self) -> Iterable[np.ndarray]:
        return []


class ContourDecoder:
    """Reconstructs contour renderings from Fourier coefficients."""

    HEADER_STRUCT = ContourEncoder.HEADER_STRUCT
    COEFF_STRUCT = ContourEncoder.COEFF_STRUCT

    def decode_chunk(self, chunk: EncodedChunk) -> np.ndarray:
        data = chunk.data
        if len(data) < self.HEADER_STRUCT.size:
            raise RuntimeError("Contour 資料過短")
        version, width, height, samples, center_x, center_y, keep = self.HEADER_STRUCT.unpack_from(data, 0)
        if version != ContourEncoder.VERSION:
            raise RuntimeError(f"Contour 版本不支援: {version}")
        if samples <= 0:
            raise RuntimeError("Contour 樣本數無效")
        expected_coeff_bytes = keep * self.COEFF_STRUCT.size
        payload = data[self.HEADER_STRUCT.size:]
        if len(payload) != expected_coeff_bytes:
            raise RuntimeError("Contour 係數長度不符")

        spectrum = np.zeros(samples // 2 + 1, dtype=np.complex64)
        for i in range(keep):
            offset = i * self.COEFF_STRUCT.size
            real, imag = self.COEFF_STRUCT.unpack_from(payload, offset)
            spectrum[i] = complex(real, imag)

        radii = np.fft.irfft(spectrum, n=samples).astype(np.float32)
        radii = np.maximum(radii, 0.0)

        width_i = int(width)
        height_i = int(height)
        if width_i <= 0 or height_i <= 0:
            raise RuntimeError("Contour 影像尺寸無效")
        canvas = np.zeros((height_i, width_i, 3), dtype=np.uint8)

        # 如果半徑都很小，代表輪廓不存在或非常微弱，顯示提示方塊
        if np.all(radii < 0.1):
            box_size = 20
            x1 = width_i // 2 - box_size // 2
            y1 = height_i // 2 - box_size // 2
            x2 = x1 + box_size
            y2 = y1 + box_size
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (64, 64, 64), 1)
            cv2.line(canvas, (x1, y1), (x2, y2), (64, 64, 64), 1)
            cv2.line(canvas, (x1, y2), (x2, y1), (64, 64, 64), 1)
        else:
            angles = np.linspace(0.0, 2 * np.pi, samples, endpoint=False, dtype=np.float32)
            xs = center_x + radii * np.cos(angles)
            ys = center_y + radii * np.sin(angles)
            points = np.stack((xs, ys), axis=1)
            points = np.round(points).astype(np.int32)

            if points.size > 0:
                points[:, 0] = np.clip(points[:, 0], 0, max(0, width_i - 1))
                points[:, 1] = np.clip(points[:, 1], 0, max(0, height_i - 1))
            poly = points.reshape(-1, 1, 2)

            if poly.size > 0:
                cv2.polylines(canvas, [poly], True, (0, 255, 0), thickness=2)

        center_point = (int(round(center_x)), int(round(center_y)))
        if 0 <= center_point[0] < width_i and 0 <= center_point[1] < height_i:
            cv2.circle(canvas, center_point, 2, (0, 0, 255), thickness=-1)
        return canvas


class DetectionDecoder:
    """Reconstruct detection overlays from serialized bounding boxes."""

    HEADER_STRUCT = DetectionEncoder.HEADER_STRUCT
    BOX_STRUCT = DetectionEncoder.BOX_STRUCT

    def decode_chunk(self, chunk: EncodedChunk) -> np.ndarray:
        data = chunk.data
        if len(data) < self.HEADER_STRUCT.size:
            raise RuntimeError("Detection 資料過短")
        version, width, height, count = self.HEADER_STRUCT.unpack_from(data, 0)
        if version != DetectionEncoder.VERSION:
            raise RuntimeError(f"Detection 版本不支援: {version}")
        expected_bytes = count * self.BOX_STRUCT.size
        payload = data[self.HEADER_STRUCT.size:]
        if len(payload) != expected_bytes:
            raise RuntimeError("Detection 係數長度不符")

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for idx in range(count):
            offset = idx * self.BOX_STRUCT.size
            cx, cy, w, h, conf = self.BOX_STRUCT.unpack_from(payload, offset)
            cx = float(np.clip(cx, 0.0, 1.0))
            cy = float(np.clip(cy, 0.0, 1.0))
            w = float(np.clip(w, 0.0, 1.0))
            h = float(np.clip(h, 0.0, 1.0))
            abs_w = max(int(round(w * width)), 1)
            abs_h = max(int(round(h * height)), 1)
            x1 = int(round(cx * width - abs_w / 2))
            y1 = int(round(cy * height - abs_h / 2))
            x2 = x1 + abs_w
            y2 = y1 + abs_h
            x1 = int(np.clip(x1, 0, max(0, width - 1)))
            y1 = int(np.clip(y1, 0, max(0, height - 1)))
            x2 = int(np.clip(x2, 0, max(0, width - 1)))
            y2 = int(np.clip(y2, 0, max(0, height - 1)))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
            label = f"{conf:.2f}"
            cv2.putText(canvas, label, (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return canvas


class H264Decoder:
    """Decodes H.264/H.265/AV1 payloads back into numpy image frames."""

    def __init__(self) -> None:
        self.codec: Optional[Any] = None
        self._codec_name: Optional[VideoCodec] = None
        self._wavelet_decoder = WaveletDecoder()
        self._contour_decoder = ContourDecoder()
        self._detection_decoder = DetectionDecoder()
        self._cs_decoder = CSDecoder()

    def _open_codec(self, codec_name: VideoCodec) -> Any:
        if codec_name in {"wavelet", "contour", "yolo", "cs"}:
            raise RuntimeError("該編碼器應透過專用路徑建立")
        if self.codec is not None and self._codec_name == codec_name:
            return self.codec

        if codec_name == "h265":
            decode_candidates = ("hevc", "h265")
        elif codec_name == "av1":
            decode_candidates = ("av1", "libaom-av1")
        else:
            decode_candidates = ("h264",)

        context: Optional[Any] = None
        for name in decode_candidates:
            try:
                context = av.CodecContext.create(name, "r")
                break
            except Exception:
                continue
        if context is None:
            raise RuntimeError(f"無法建立 {codec_name} 解碼器，請確認 FFmpeg 支援 {codec_name}。")
        context.options = {"refcounted_frames": "0"}
        context.open()

        self.codec = context
        self._codec_name = codec_name
        return context

    def decode(self, chunk: EncodedChunk) -> Iterable[np.ndarray]:
        codec_name: VideoCodec = chunk.codec
        if codec_name == "wavelet":
            self._codec_name = "wavelet"
            if chunk.is_config:
                return []
            return [self._wavelet_decoder.decode_chunk(chunk)]
        if codec_name == "contour":
            self._codec_name = "contour"
            if chunk.is_config:
                return []
            return [self._contour_decoder.decode_chunk(chunk)]
        if codec_name == "yolo":
            self._codec_name = "yolo"
            if chunk.is_config:
                return []
            return [self._detection_decoder.decode_chunk(chunk)]
        if codec_name == "cs":
            self._codec_name = "cs"
            if chunk.is_config:
                return []
            return [self._cs_decoder.decode_chunk(chunk)]
        if codec_name == "jpeg":
            self._codec_name = "jpeg"
            if chunk.is_config:
                return []
            buffer = np.frombuffer(chunk.data, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("JPEG 解碼失敗")
            return [image]

        context = self._open_codec(codec_name)

        if chunk.is_config:
            context.extradata = chunk.data
            return []

        packet = Packet(chunk.data)
        frames = []
        try:
            decoded_iter = context.decode(packet)
        except Exception as exc:
            raise RuntimeError(f"解碼 {codec_name} 影像失敗: {exc}") from exc
        for frame in decoded_iter:
            frames.append(frame.to_ndarray(format="bgr24"))
        return frames

    def flush(self) -> Iterable[np.ndarray]:
        if self._codec_name == "wavelet":
            return list(self._wavelet_decoder.flush())
        if self._codec_name == "jpeg":
            return []
        if self._codec_name == "contour":
            return []
        if self._codec_name == "yolo":
            return []
        if self._codec_name == "cs":
            return []
        if self.codec is None:
            return []

        frames = []
        try:
            decoded_iter = self.codec.decode(None)
        except Exception as exc:
            raise RuntimeError(f"解碼 {self._codec_name or 'unknown'} 影像失敗: {exc}") from exc
        for frame in decoded_iter:
            frames.append(frame.to_ndarray(format="bgr24"))
        return frames
