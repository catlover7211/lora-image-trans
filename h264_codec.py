"""Utilities for video encoding and decoding over the serial protocol."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, List, Literal, Optional

import av  # type: ignore
import numpy as np
from av.packet import Packet
from av.video.frame import PictureType

VideoCodec = Literal["h264", "h265"]


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

    def to_payload(self) -> bytes:
        """Convert the chunk into a protocol payload with a flag prefix."""
        flags = 0
        if self.is_keyframe:
            flags |= self.FLAG_KEYFRAME
        if self.is_config:
            flags |= self.FLAG_CONFIG
        if self.codec == "h265":
            flags |= self.FLAG_CODEC_HEVC
        return bytes((flags,)) + self.data

    @classmethod
    def from_payload(cls, payload: bytes) -> "EncodedChunk":
        if not payload:
            raise ValueError("payload 不可為空")
        flags = payload[0]
        data = payload[1:]
        codec: VideoCodec = "h265" if flags & cls.FLAG_CODEC_HEVC else "h264"
        return cls(
            data=data,
            codec=codec,
            is_keyframe=bool(flags & cls.FLAG_KEYFRAME),
            is_config=bool(flags & cls.FLAG_CONFIG),
        )


MIN_ENCODER_BITRATE = 10_000


class H264Encoder:
    """Encodes numpy image frames into H.264 or H.265 packets."""

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

        encoder_name = "libx265" if codec == "h265" else "libx264"
        fallback_name = "hevc" if codec == "h265" else "h264"
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
        self._config_sent = False

    def encode(self, frame_bgr: np.ndarray) -> List[EncodedChunk]:
        """Encode a BGR frame and return zero or more codec chunks."""
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            raise ValueError("輸入影像尺寸與編碼器不符")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("輸入影像需為 BGR 三通道格式")

        chunks: List[EncodedChunk] = []

        if not self._config_sent and self.codec.extradata:
            chunks.append(
                EncodedChunk(
                    data=self.codec.extradata,
                    codec=self.codec_name,
                    is_config=True,
                    is_keyframe=True,
                )
            )
            self._config_sent = True

        video_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        if self._frame_index % self.keyframe_interval == 0:
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


class H264Decoder:
    """Decodes H.264 or H.265 payloads back into numpy image frames."""

    def __init__(self) -> None:
        self.codec: Optional[Any] = None
        self._codec_name: Optional[VideoCodec] = None

    def _open_codec(self, codec_name: VideoCodec) -> Any:
        if self.codec is not None and self._codec_name == codec_name:
            return self.codec

        decode_name = "hevc" if codec_name == "h265" else "h264"
        context: Any = av.CodecContext.create(decode_name, "r")
        context.options = {"refcounted_frames": "0"}
        context.open()

        self.codec = context
        self._codec_name = codec_name
        return context

    def decode(self, chunk: EncodedChunk) -> Iterable[np.ndarray]:
        codec_name: VideoCodec = "h265" if chunk.codec == "h265" else "h264"
        context = self._open_codec(codec_name)

        if chunk.is_config:
            context.extradata = chunk.data
            return []

        packet = Packet(chunk.data)
        frames = []
        for frame in context.decode(packet):
            frames.append(frame.to_ndarray(format="bgr24"))
        return frames

    def flush(self) -> Iterable[np.ndarray]:
        if self.codec is None:
            return []

        frames = []
        for frame in self.codec.decode(None):
            frames.append(frame.to_ndarray(format="bgr24"))
        return frames
