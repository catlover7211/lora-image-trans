"""Utilities for H.264 encoding and decoding over the serial protocol."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, List

import av  # type: ignore
import numpy as np


@dataclass(frozen=True)
class EncodedChunk:
    """Represents a single H.264 payload fragment."""

    data: bytes
    is_keyframe: bool = False
    is_config: bool = False

    FLAG_KEYFRAME = 0x01
    FLAG_CONFIG = 0x02

    def to_payload(self) -> bytes:
        """Convert the chunk into a protocol payload with a flag prefix."""
        flags = 0
        if self.is_keyframe:
            flags |= self.FLAG_KEYFRAME
        if self.is_config:
            flags |= self.FLAG_CONFIG
        return bytes((flags,)) + self.data

    @classmethod
    def from_payload(cls, payload: bytes) -> "EncodedChunk":
        if not payload:
            raise ValueError("payload 不可為空")
        flags = payload[0]
        data = payload[1:]
        return cls(
            data=data,
            is_keyframe=bool(flags & cls.FLAG_KEYFRAME),
            is_config=bool(flags & cls.FLAG_CONFIG),
        )


class H264Encoder:
    """Encodes numpy image frames into H.264 packets."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        fps: float = 10.0,
        bitrate: int = 400_000,
        keyframe_interval: int = 30,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("影像尺寸必須為正整數")
        if fps <= 0:
            raise ValueError("fps 必須為正數")
        self.width = width
        self.height = height
        self.fps = fps
        try:
            self.codec = av.CodecContext.create("libx264", "w")
        except av.AVError:
            self.codec = av.CodecContext.create("h264", "w")
        self.codec.width = width
        self.codec.height = height
        self.codec.time_base = Fraction(1, max(int(round(fps)), 1))
        self.codec.framerate = Fraction(max(int(round(fps)), 1), 1)
        self.codec.pix_fmt = "yuv420p"
        self.codec.bit_rate = bitrate
        self.codec.options = {
            "preset": "veryfast",
            "tune": "zerolatency",
            "profile": "baseline",
            "x264opts": "no-scenecut",
        }
        self.codec.gop_size = max(keyframe_interval, 1)
        self.codec.open()
        self.keyframe_interval = max(keyframe_interval, 1)
        self._frame_index = 0
        self._config_sent = False

    def encode(self, frame_bgr: np.ndarray) -> List[EncodedChunk]:
        """Encode a BGR frame and return zero or more H.264 chunks."""
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            raise ValueError("輸入影像尺寸與編碼器不符")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("輸入影像需為 BGR 三通道格式")

        chunks: List[EncodedChunk] = []

        if not self._config_sent and self.codec.extradata:
            chunks.append(EncodedChunk(data=self.codec.extradata, is_config=True, is_keyframe=True))
            self._config_sent = True

        video_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        if self._frame_index % self.keyframe_interval == 0:
            video_frame.pict_type = "I"
        self._frame_index += 1

        for packet in self.codec.encode(video_frame):
            chunks.append(EncodedChunk(
                data=packet.to_bytes(),
                is_keyframe=packet.is_keyframe,
                is_config=False,
            ))
        return chunks

    def flush(self) -> List[EncodedChunk]:
        """Flush any delayed packets from the encoder."""
        chunks: List[EncodedChunk] = []
        for packet in self.codec.encode(None):
            chunks.append(EncodedChunk(
                data=packet.to_bytes(),
                is_keyframe=packet.is_keyframe,
                is_config=False,
            ))
        return chunks


class H264Decoder:
    """Decodes H.264 payloads back into numpy image frames."""

    def __init__(self) -> None:
        self.codec = av.CodecContext.create("h264", "r")
        self.codec.options = {"refcounted_frames": "0"}
        self._opened = False

    def _ensure_open(self) -> None:
        if not self._opened:
            self.codec.open()
            self._opened = True

    def decode(self, chunk: EncodedChunk) -> Iterable[np.ndarray]:
        if chunk.is_config:
            self.codec.extradata = chunk.data
            self._ensure_open()
            return []

        self._ensure_open()
        packet = av.packet.Packet(chunk.data)
        frames = []
        for frame in self.codec.decode(packet):
            frames.append(frame.to_ndarray(format="bgr24"))
        return frames

    def flush(self) -> Iterable[np.ndarray]:
        if not self._opened:
            return []
        frames = []
        for frame in self.codec.decode(None):
            frames.append(frame.to_ndarray(format="bgr24"))
        return frames
