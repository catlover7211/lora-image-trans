"""Utilities for video encoding and decoding over the serial protocol."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, List, Literal, Optional

import av  # type: ignore
import numpy as np
from av.packet import Packet
from av.video.frame import PictureType

VideoCodec = Literal["h264", "h265", "av1"]


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
        return bytes((flags,)) + self.data

    @classmethod
    def from_payload(cls, payload: bytes) -> "EncodedChunk":
        if not payload:
            raise ValueError("payload 不可為空")
        flags = payload[0]
        data = payload[1:]
        if flags & cls.FLAG_CODEC_AV1:
            codec: VideoCodec = "av1"
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


class H264Decoder:
    """Decodes H.264/H.265/AV1 payloads back into numpy image frames."""

    def __init__(self) -> None:
        self.codec: Optional[Any] = None
        self._codec_name: Optional[VideoCodec] = None

    def _open_codec(self, codec_name: VideoCodec) -> Any:
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
