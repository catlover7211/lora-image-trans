"""Utilities for video encoding and decoding over the serial protocol.

This module provides comprehensive video codec support including:
- Standard codecs: H.264, H.265, AV1, JPEG
- Custom codecs: Wavelet, Contour, Compressed Sensing (CS)
- Object detection: YOLO-based detection encoding

All encoders produce EncodedChunk objects that can be serialized for transmission.
All decoders accept EncodedChunk objects and return numpy image arrays.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, List, Literal, Optional, Tuple
import struct
import zlib
from scipy.fftpack import dct, idct
from sklearn.linear_model import OrthogonalMatchingPursuit

import av  # type: ignore
import cv2
import numpy as np
from av.packet import Packet
from av.video.frame import PictureType

VideoCodec = Literal["h264", "h265", "av1", "wavelet", "jpeg", "contour", "yolo", "cs"]

DetectionBox = Tuple[float, float, float, float, float]


class BaseEncoder(ABC):
    """Base class for all encoders with common validation logic.
    
    Provides standard frame validation and defines the interface that all
    encoders must implement. Encoders convert numpy BGR images into
    EncodedChunk objects suitable for transmission.
    """
    
    def __init__(self, width: int, height: int) -> None:
        """Initialize encoder with frame dimensions.
        
        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            
        Raises:
            ValueError: If width or height is not positive.
        """
        if width <= 0 or height <= 0:
            raise ValueError("影像尺寸必須為正整數")
        self.width = width
        self.height = height
    
    def validate_frame(self, frame_bgr: np.ndarray) -> None:
        """Validate input frame dimensions and format."""
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            raise ValueError("輸入影像尺寸與編碼器不符")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("輸入影像需為 BGR 三通道格式")
    
    @abstractmethod
    def encode(self, frame_bgr: np.ndarray) -> List[EncodedChunk]:
        """Encode a frame and return encoded chunks."""
        pass
    
    def force_keyframe(self) -> None:
        """Request the next frame to be encoded as a keyframe (optional)."""
        pass
    
    def force_config_repeat(self, count: int = 3) -> None:
        """Schedule codec configuration to be resent (optional)."""
        pass


@dataclass(frozen=True)
class EncodedChunk:
    """Represents a single video payload fragment.
    
    This is the fundamental unit of encoded video data. Each chunk contains:
    - data: The actual encoded bytes
    - codec: Which codec was used to encode this chunk
    - is_keyframe: Whether this is a keyframe (I-frame)
    - is_config: Whether this contains codec configuration data (SPS/PPS)
    
    Chunks are serialized to/from protocol payloads with a 2-byte flag prefix.
    """

    data: bytes
    codec: VideoCodec = "h264"
    is_keyframe: bool = False
    is_config: bool = False

    # Flag bits for serialization
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
        """Deserialize chunk from protocol payload.
        
        Supports both legacy (1-byte flags) and new (2-byte flags) formats for
        backward compatibility. The new format is required for codecs with flag
        values >= 0x100 (e.g., CS codec with FLAG_CODEC_CS = 0x100).
        
        Format detection:
        - If payload length < 2: treat as legacy format (1-byte flags)
        - If 2nd byte (high byte of 2-byte flags) is non-zero: new format
        - Otherwise: legacy format (1-byte flags)
        
        Args:
            payload: Raw bytes from protocol frame
            
        Returns:
            EncodedChunk instance
            
        Raises:
            ValueError: If payload is empty
        """
        if not payload:
            raise ValueError("payload 不可為空")
        
        # Detect format: new format if length >= 2 AND high byte is set
        if len(payload) >= 2:
            # Read as 2-byte little-endian
            flags_2byte = struct.unpack("<H", payload[:2])[0]
            # Check if high byte is set (indicates new format or CS codec)
            if flags_2byte & 0xFF00:
                # New format with 2-byte flags
                flags = flags_2byte
                data = payload[2:]
            else:
                # Legacy format with 1-byte flags
                flags = payload[0]
                data = payload[1:]
        else:
            # Single byte payload - legacy format
            flags = payload[0]
            data = payload[1:]
        
        # Determine codec from flags
        # Priority order: CS > YOLO > CONTOUR > JPEG > WAVELET > AV1 > HEVC > H264
        if flags & cls.FLAG_CODEC_CS:
            codec: VideoCodec = "cs"
        elif flags & cls.FLAG_CODEC_YOLO:
            codec = "yolo"
        elif flags & cls.FLAG_CODEC_CONTOUR:
            codec = "contour"
        elif flags & cls.FLAG_CODEC_JPEG:
            codec = "jpeg"
        elif flags & cls.FLAG_CODEC_WAVELET:
            codec = "wavelet"
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
MAX_SEED_VALUE = 2**32  # Maximum value for random seed in compressed sensing


class H264Encoder(BaseEncoder):
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
        super().__init__(width, height)
        
        if codec == "wavelet":
            raise ValueError("wavelet 編碼請使用 WaveletEncoder")
        if fps <= 0:
            raise ValueError("fps 必須為正數")
        if bitrate < MIN_ENCODER_BITRATE:
            raise ValueError(
                f"位元率過低 (須 >= {MIN_ENCODER_BITRATE} bps) 可能導致編碼器無法啟動，"
                "請提高 bitrate 或改用較低解析度/幀率。"
            )

        self.fps = fps
        self.codec_name: VideoCodec = codec
        self.codec: Any = self._create_codec_context(codec, width, height, fps, bitrate)
        self.keyframe_interval = max(keyframe_interval, 1)
        
        self._configure_codec(codec, self.keyframe_interval)
        self.codec.open()
        
        self._frame_index = 0
        self._config_burst = 3
        self._force_config = True

    def _create_codec_context(
        self,
        codec: VideoCodec,
        width: int,
        height: int,
        fps: float,
        bitrate: int
    ) -> Any:
        """Create and configure codec context."""
        encoder_name, fallback_name = self._get_encoder_names(codec)
        
        try:
            codec_ctx: Any = av.CodecContext.create(encoder_name, "w")
        except Exception as e:
            # Fallback to alternative encoder if preferred one fails
            import warnings
            warnings.warn(
                f"Failed to create {encoder_name} encoder: {e}. Falling back to {fallback_name}.",
                RuntimeWarning
            )
            codec_ctx = av.CodecContext.create(fallback_name, "w")

        frame_rate = max(int(round(fps)), 1)
        codec_ctx.width = width
        codec_ctx.height = height
        codec_ctx.time_base = Fraction(1, frame_rate)
        codec_ctx.framerate = Fraction(frame_rate, 1)
        codec_ctx.pix_fmt = "yuv420p"
        codec_ctx.bit_rate = bitrate
        
        return codec_ctx

    @staticmethod
    def _get_encoder_names(codec: VideoCodec) -> tuple[str, str]:
        """Get encoder name and fallback for the given codec."""
        if codec == "h265":
            return "libx265", "hevc"
        elif codec == "av1":
            return "libaom-av1", "av1"
        else:
            return "libx264", "h264"

    def _configure_codec(self, codec: VideoCodec, gop_size: int) -> None:
        """Configure codec-specific options."""
        if codec == "h265":
            self._configure_h265(gop_size)
        elif codec == "av1":
            self._configure_av1(gop_size)
        else:
            self._configure_h264(gop_size)
        
        self.codec.gop_size = gop_size

    def _configure_h265(self, gop_size: int) -> None:
        """Configure H.265 specific options."""
        x265_params = ":".join([
            f"keyint={gop_size}",
            f"min-keyint={gop_size}",
            "scenecut=0",
            "bframes=0",
            "repeat-headers=1",
            "rc-lookahead=0",
            "frame-threads=1",
        ])
        self.codec.options = {
            "preset": "veryfast",
            "tune": "zerolatency",
            "x265-params": x265_params,
        }

    def _configure_av1(self, gop_size: int) -> None:
        """Configure AV1 specific options."""
        self.codec.options = {
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

    def _configure_h264(self, gop_size: int) -> None:
        """Configure H.264 specific options."""
        self.codec.options = {
            "preset": "veryfast",
            "tune": "zerolatency",
            "profile": "baseline",
            "x264opts": "no-scenecut",
        }

    def encode(self, frame_bgr: np.ndarray) -> List[EncodedChunk]:
        """Encode a BGR frame and return zero or more codec chunks."""
        self.validate_frame(frame_bgr)

        chunks: List[EncodedChunk] = []
        is_keyframe_due = (self._frame_index % self.keyframe_interval == 0)
        
        if is_keyframe_due:
            self._force_config = True

        # Add config chunk if needed
        if self.codec.extradata and (self._force_config or self._config_burst > 0):
            chunks.append(self._create_config_chunk())
            if self._config_burst > 0:
                self._config_burst -= 1
            self._force_config = False

        # Encode video frame
        video_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        if is_keyframe_due:
            self._set_keyframe_flag(video_frame)
        
        self._frame_index += 1

        # Get encoded packets
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

    def _create_config_chunk(self) -> EncodedChunk:
        """Create configuration chunk."""
        return EncodedChunk(
            data=self.codec.extradata,
            codec=self.codec_name,
            is_config=True,
            is_keyframe=True,
        )

    @staticmethod
    def _set_keyframe_flag(video_frame: Any) -> None:
        """Set keyframe flag on video frame (with error handling)."""
        try:
            video_frame.pict_type = PictureType.I
        except (AttributeError, TypeError):
            # Some backends only accept integers or don't support forced setting
            pass

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


class JPEGEncoder(BaseEncoder):
    """Encodes numpy image frames into baseline JPEG payloads."""

    def __init__(self, width: int, height: int, *, quality: int = 85) -> None:
        super().__init__(width, height)
        
        if not (1 <= quality <= 100):
            raise ValueError("JPEG 品質需介於 1 到 100")
        
        self.quality = int(quality)
        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]

    def encode(self, frame_bgr: np.ndarray) -> List[EncodedChunk]:
        self.validate_frame(frame_bgr)

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


class ContourEncoder(BaseEncoder):
    """Approximates frame contours with a truncated Fourier series."""

    VERSION = 1
    HEADER_STRUCT = struct.Struct("<BHHHffH")
    COEFF_STRUCT = struct.Struct("<ff")

    def __init__(self, width: int, height: int, *, samples: int = 128, coefficients: int = 16) -> None:
        super().__init__(width, height)
        
        if samples <= 0:
            raise ValueError("samples 必須為正整數")
        if coefficients <= 0:
            raise ValueError("coefficients 必須為正整數")
        
        self.samples = samples
        max_coeffs = samples // 2 + 1
        self.coefficients = min(coefficients, max(1, max_coeffs))

    def encode(self, frame_bgr: np.ndarray) -> List[EncodedChunk]:
        self.validate_frame(frame_bgr)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        center, radii_samples = self._extract_contour_features(contours)
        spectrum = np.fft.rfft(radii_samples)
        kept_coeffs = spectrum[:self.coefficients]

        header = self._pack_header(center, len(kept_coeffs))
        body = self._pack_coefficients(kept_coeffs)

        chunk = EncodedChunk(
            data=header + bytes(body),
            codec="contour",
            is_keyframe=True,
            is_config=False,
        )
        return [chunk]

    def _extract_contour_features(self, contours) -> tuple[np.ndarray, np.ndarray]:
        """Extract center and radii samples from contours."""
        if not contours:
            center = np.array([self.width / 2.0, self.height / 2.0], dtype=np.float32)
            radii_samples = np.zeros(self.samples, dtype=np.float32)
            return center, radii_samples

        largest = max(contours, key=cv2.contourArea)
        points = largest.reshape(-1, 2).astype(np.float32)
        center = points.mean(axis=0)
        
        vectors = points - center
        norms = np.linalg.norm(vectors, axis=1)
        
        if np.all(norms == 0):
            radii_samples = np.zeros(self.samples, dtype=np.float32)
        else:
            radii_samples = self._interpolate_radii(vectors, norms)
        
        return center, radii_samples

    def _interpolate_radii(self, vectors: np.ndarray, norms: np.ndarray) -> np.ndarray:
        """Interpolate radii values for uniform angular sampling."""
        angles = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]) + 2 * np.pi, 2 * np.pi)
        order = np.argsort(angles)
        angles_sorted = angles[order]
        radii_sorted = norms[order]
        
        # Extend 2π period for interpolation
        angles_ext = np.concatenate([angles_sorted, angles_sorted + 2 * np.pi])
        radii_ext = np.concatenate([radii_sorted, radii_sorted])
        
        sample_angles = np.linspace(0.0, 2 * np.pi, self.samples, endpoint=False, dtype=np.float32)
        return np.interp(sample_angles, angles_ext, radii_ext).astype(np.float32)

    def _pack_header(self, center: np.ndarray, num_coeffs: int) -> bytes:
        """Pack header information."""
        return self.HEADER_STRUCT.pack(
            self.VERSION,
            self.width,
            self.height,
            self.samples,
            float(center[0]),
            float(center[1]),
            num_coeffs,
        )

    def _pack_coefficients(self, coeffs: np.ndarray) -> bytearray:
        """Pack Fourier coefficients."""
        body = bytearray(self.COEFF_STRUCT.size * len(coeffs))
        offset = 0
        for coeff in coeffs:
            struct.pack_into("<ff", body, offset, float(np.real(coeff)), float(np.imag(coeff)))
            offset += self.COEFF_STRUCT.size
        return body


class DetectionEncoder(BaseEncoder):
    """Serialize normalized bounding boxes into protocol payloads."""

    VERSION = 1
    HEADER_STRUCT = struct.Struct("<BHHB")
    BOX_STRUCT = struct.Struct("<fffff")

    def encode(self, boxes: List[DetectionBox]) -> List[EncodedChunk]:
        count = min(len(boxes), 255)
        header = self.HEADER_STRUCT.pack(self.VERSION, self.width, self.height, count)
        
        body = bytearray(self.BOX_STRUCT.size * count)
        for idx, (cx, cy, w, h, conf) in enumerate(boxes[:count]):
            offset = idx * self.BOX_STRUCT.size
            struct.pack_into(
                "<fffff", body, offset,
                float(np.clip(cx, 0.0, 1.0)),
                float(np.clip(cy, 0.0, 1.0)),
                float(np.clip(w, 0.0, 1.0)),
                float(np.clip(h, 0.0, 1.0)),
                float(np.clip(conf, 0.0, 1.0))
            )

        chunk = EncodedChunk(
            data=header + bytes(body),
            codec="yolo",
            is_keyframe=True,
            is_config=False,
        )
        return [chunk]


class YOLODetectionEncoder(BaseEncoder):
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
        super().__init__(width, height)
        
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
        self.validate_frame(frame_bgr)

        detections = self.detector.detect(frame_bgr)
        overlay = self._create_detection_overlay(frame_bgr, detections)
        self.preview_frame = overlay
        
        return self.packetizer.encode(detections)

    def _create_detection_overlay(self, frame_bgr: np.ndarray, detections: List[DetectionBox]) -> np.ndarray:
        """Create an overlay image with detection boxes."""
        overlay = frame_bgr.copy()
        
        for cx, cy, w, h, conf in detections:
            abs_w = max(int(round(w * self.width)), 1)
            abs_h = max(int(round(h * self.height)), 1)
            x1 = int(round(cx * self.width - abs_w / 2))
            y1 = int(round(cy * self.height - abs_h / 2))
            x2 = x1 + abs_w
            y2 = y1 + abs_h
            
            # Clip coordinates
            x1 = int(np.clip(x1, 0, self.width - 1))
            y1 = int(np.clip(y1, 0, self.height - 1))
            x2 = int(np.clip(x2, 0, self.width - 1))
            y2 = int(np.clip(y2, 0, self.height - 1))
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            label = f"{conf:.2f}"
            cv2.putText(overlay, label, (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        return overlay


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


class WaveletEncoder(BaseEncoder):
    """Simple YCoCg + Haar wavelet encoder producing intra-only frames."""

    VERSION = 1
    HEADER_STRUCT = struct.Struct("<BHHHBB")  # version, width, height, quant, levels, channels

    def __init__(self, width: int, height: int, *, levels: int = 2, quant_step: int = 12) -> None:
        super().__init__(width, height)
        
        if quant_step <= 0:
            raise ValueError("量化步階必須為正整數")
        
        self.quant_step = quant_step
        self.levels = min(max(0, levels), _max_wavelet_levels(width, height))

    def encode(self, frame_bgr: np.ndarray) -> List[EncodedChunk]:
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width or frame_bgr.shape[2] != 3:
            raise ValueError("輸入影像尺寸與編碼器不符")

        ycocg = _bgr_to_ycocg(frame_bgr)
        
        # Process each channel
        coeffs = [
            np.rint(_haar_forward(ycocg[..., c], self.levels) / self.quant_step).astype(np.int16)
            for c in range(3)
        ]
        
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


class CSEncoder(BaseEncoder):
    """
    Compressed Sensing encoder using a random Gaussian measurement matrix
    on the DCT coefficients of the image.
    """

    VERSION = 2  # Bump version due to fundamental change in algorithm
    HEADER_STRUCT = struct.Struct("<BHHHHH")  # version, width, height, measurements, seed_high, seed_low

    def __init__(
        self,
        width: int,
        height: int,
        *,
        measurement_ratio: float = 0.3,
        seed: Optional[int] = None
    ) -> None:
        super().__init__(width, height)
        
        if not (0 < measurement_ratio < 1):
            raise ValueError("測量比率必須介於 0 和 1 之間")

        self.measurement_ratio = measurement_ratio
        self.seed = seed % MAX_SEED_VALUE if seed is not None else np.random.randint(0, MAX_SEED_VALUE - 1)

        signal_length = width * height
        self.num_measurements = max(1, int(signal_length * measurement_ratio))

        # Create measurement matrix
        rng = np.random.RandomState(self.seed)
        self.measurement_matrix = rng.randn(self.num_measurements, signal_length).astype(np.float32)
        # Normalize rows, adding epsilon to prevent division by zero
        row_norms = np.linalg.norm(self.measurement_matrix, axis=1, keepdims=True)
        self.measurement_matrix /= (row_norms + 1e-8)

    @staticmethod
    def _dct2(block: np.ndarray) -> np.ndarray:
        """2D DCT transform."""
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def encode(self, frame_bgr: np.ndarray) -> List[EncodedChunk]:
        self.validate_frame(frame_bgr)

        measurements_list = []
        for c in range(3):
            channel = frame_bgr[:, :, c].astype(np.float32)
            dct_coeffs = self._dct2(channel)
            dct_vector = dct_coeffs.flatten()
            measurements = self.measurement_matrix @ dct_vector
            measurements_list.append(measurements)

        all_measurements = np.stack(measurements_list, axis=0)
        quantized = np.clip(all_measurements, -32767, 32767).astype(np.int16)
        payload_body = zlib.compress(quantized.tobytes(), level=6)

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


class CSDecoder:
    """
    Compressed Sensing decoder using Orthogonal Matching Pursuit (OMP)
    to reconstruct the sparse DCT coefficients.
    """

    HEADER_STRUCT = CSEncoder.HEADER_STRUCT

    def __init__(self, n_nonzero_coeffs: Optional[int] = None, regularization: float = 0.01):
        """
        Args:
            n_nonzero_coeffs: For V2, the presumed number of non-zero DCT coefficients.
            regularization: For V0/V1, the regularization parameter for least-squares.
        """
        self.n_nonzero_coeffs = n_nonzero_coeffs
        self.regularization = regularization

    @staticmethod
    def _idct2(coeffs: np.ndarray) -> np.ndarray:
        """Inverse 2D DCT transform."""
        return idct(idct(coeffs.T, norm='ortho').T, norm='ortho')

    def decode_chunk(self, chunk: EncodedChunk) -> np.ndarray:
        data = chunk.data
        if len(data) <= self.HEADER_STRUCT.size:
            raise RuntimeError("CS 資料長度不正確")

        header = data[:self.HEADER_STRUCT.size]
        version, width, height, num_measurements, seed_high, seed_low = self.HEADER_STRUCT.unpack(header)

        # Version dispatch
        if version == 2:
            return self._decode_v2(data)
        elif version in (0, 1):
            return self._decode_legacy(data)
        
        raise RuntimeError(f"CS 版本不支援: {version}")

    def _decode_legacy(self, data: bytes) -> np.ndarray:
        """Decoder for legacy versions 0 and 1."""
        _ver, width, height, num_measurements, seed_high, seed_low = self.HEADER_STRUCT.unpack_from(data)
        seed = (seed_high << 16) | seed_low

        measurements = self._decompress_measurements(data, num_measurements, "legacy")
        measurement_matrix = self._create_measurement_matrix(seed, num_measurements, width * height)

        reconstructed_channels = [
            self._reconstruct_legacy(measurements[c], measurement_matrix).reshape((height, width))
            for c in range(3)
        ]
        
        frame_bgr = np.stack(reconstructed_channels, axis=2)
        return np.clip(frame_bgr, 0, 255).astype(np.uint8)

    def _decode_v2(self, data: bytes) -> np.ndarray:
        """Decoder for new version 2."""
        _ver, width, height, num_measurements, seed_high, seed_low = self.HEADER_STRUCT.unpack_from(data)
        seed = (seed_high << 16) | seed_low

        measurements = self._decompress_measurements(data, num_measurements, "v2")
        measurement_matrix = self._create_measurement_matrix(seed, num_measurements, width * height)

        n_nonzero = self.n_nonzero_coeffs or max(1, int(num_measurements * 0.1))
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        
        reconstructed_channels = []
        for c in range(3):
            omp.fit(measurement_matrix, measurements[c])
            dct_coeffs = omp.coef_.reshape((height, width))
            channel = self._idct2(dct_coeffs)
            reconstructed_channels.append(channel)

        frame_bgr = np.stack(reconstructed_channels, axis=2)
        return np.clip(frame_bgr, 0, 255).astype(np.uint8)

    def _decompress_measurements(self, data: bytes, num_measurements: int, version: str) -> np.ndarray:
        """Decompress and validate measurements."""
        compressed = data[self.HEADER_STRUCT.size:]
        try:
            decompressed = zlib.decompress(compressed)
        except zlib.error as exc:
            raise RuntimeError(f"CS ({version}) 解壓失敗: {exc}") from exc

        expected_bytes = 3 * num_measurements * np.dtype(np.int16).itemsize
        if len(decompressed) != expected_bytes:
            raise RuntimeError(f"CS ({version}) 解壓後長度不符: 應為 {expected_bytes}, 實際 {len(decompressed)}")

        return np.frombuffer(decompressed, dtype=np.int16).reshape((3, num_measurements)).astype(np.float32)

    @staticmethod
    def _create_measurement_matrix(seed: int, num_measurements: int, signal_length: int) -> np.ndarray:
        """Create measurement matrix from seed."""
        rng = np.random.RandomState(seed)
        measurement_matrix = rng.randn(num_measurements, signal_length).astype(np.float32)
        # Normalize rows, adding epsilon to prevent division by zero
        row_norms = np.linalg.norm(measurement_matrix, axis=1, keepdims=True)
        measurement_matrix /= (row_norms + 1e-8)
        return measurement_matrix

    def _reconstruct_legacy(
        self,
        measurements: np.ndarray,
        measurement_matrix: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct signal using regularized least-squares (for V0/V1)."""
        m, n = measurement_matrix.shape
        try:
            phi_t = measurement_matrix.T
            phi_t_phi = phi_t @ measurement_matrix
            reg_matrix = phi_t_phi + self.regularization * np.eye(n, dtype=np.float32)
            phi_t_y = phi_t @ measurements
            signal = np.linalg.solve(reg_matrix, phi_t_y)
        except np.linalg.LinAlgError:
            try:
                signal = np.linalg.lstsq(measurement_matrix, measurements, rcond=None)[0]
            except np.linalg.LinAlgError:
                signal = np.zeros(n, dtype=np.float32)
        return signal


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
