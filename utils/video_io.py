"""
Video I/O via PyAV (FFmpeg bindings).

Functions
---------
read_frames(path, max_frames)
    Yield (index, frame_rgb) tuples from an MP4 file.

write_video(frames, output_path, fps, width, height, codec, crf)
    Encode a list of uint8 RGB frames to an MP4 file.

read_video_metadata(path)
    Return dict with codec, fps, width, height, duration, n_frames.

apply_watermark_to_video(input_path, output_path, watermarker, payload_bits,
                         frame_selector, max_frames, codec, crf)
    Full encode pipeline: read → watermark selected frames → write.

decode_watermark_from_video(input_path, watermarker, frame_selector, max_frames)
    Full decode pipeline: read → decode → aggregate bits.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, Optional

import av
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_video_metadata(path: str | Path) -> dict[str, Any]:
    """Return codec, fps, resolution, duration information."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 25)
        n_frames = stream.frames or 0
        duration = float(stream.duration or 0) * float(stream.time_base or 1)
        codec_name = stream.codec_context.name
        return {
            "codec": codec_name,
            "fps": fps,
            "width": stream.width,
            "height": stream.height,
            "duration_s": duration,
            "n_frames": n_frames,
            "pixel_format": str(stream.codec_context.pix_fmt),
        }


def read_frames(
    path: str | Path,
    max_frames: Optional[int] = None,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """
    Yield (frame_index, frame_rgb_uint8) for every frame in the video.

    Parameters
    ----------
    path : str | Path
    max_frames : int, optional  — stop after this many frames

    Yields
    ------
    (int, np.ndarray)  index and (H, W, 3) uint8 RGB frame
    """
    with av.open(str(path)) as container:
        for idx, frame in enumerate(container.decode(video=0)):
            if max_frames is not None and idx >= max_frames:
                break
            arr = frame.to_ndarray(format="rgb24")   # (H, W, 3) uint8
            yield idx, arr


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_video(
    frames: list[np.ndarray],
    output_path: str | Path,
    fps: float,
    codec: str = "libx264",
    crf: int = 23,
    pixel_format: str = "yuv420p",
) -> None:
    """
    Write a list of (H, W, 3) uint8 RGB frames to an MP4 file.

    Parameters
    ----------
    frames : list of np.ndarray  (H, W, 3) uint8 RGB
    output_path : str | Path
    fps : float
    codec : str  'libx264' or 'libx265'
    crf : int    quality factor — 0 (lossless) to 51 (worst). Default 23.
    pixel_format : str  default 'yuv420p' (required by most players)
    """
    if not frames:
        raise ValueError("No frames to write")

    H, W = frames[0].shape[:2]
    out_path = str(output_path)

    with av.open(out_path, "w") as container:
        from fractions import Fraction
        stream = container.add_stream(codec, rate=Fraction(fps).limit_denominator(1000))
        stream.width = W
        stream.height = H
        stream.pix_fmt = pixel_format
        stream.options = {"crf": str(crf)}

        for frame_arr in frames:
            av_frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
            av_frame = av_frame.reformat(format=pixel_format)
            for packet in stream.encode(av_frame):
                container.mux(packet)
        # Flush
        for packet in stream.encode():
            container.mux(packet)

    log.info("Wrote %d frames to %s", len(frames), out_path)


# ---------------------------------------------------------------------------
# High-level pipelines
# ---------------------------------------------------------------------------

FrameSelector = Callable[[int], bool]


def every_frame(_: int) -> bool:
    return True


def every_nth(n: int) -> FrameSelector:
    """Select every n-th frame (0-indexed)."""
    return lambda i: i % n == 0


def apply_watermark_to_video(
    input_path: str | Path,
    output_path: str | Path,
    watermarker,
    payload_bits: np.ndarray,
    frame_selector: FrameSelector = every_frame,
    max_frames: Optional[int] = None,
    codec: str = "libx264",
    crf: int = 23,
) -> dict[str, Any]:
    """
    Read input MP4, watermark selected frames, write output MP4.

    Non-selected frames are passed through unchanged (for speed/comparison).

    Returns
    -------
    dict with keys: n_frames_total, n_frames_watermarked, fps, codec_out
    """
    meta = read_video_metadata(input_path)
    fps = meta["fps"]

    frames_out: list[np.ndarray] = []
    n_wm = 0

    frame_iter = read_frames(input_path, max_frames=max_frames)
    total = min(meta["n_frames"] or 9999, max_frames or 9999)

    for idx, frame in tqdm(frame_iter, total=total, desc="Encoding", unit="fr"):
        if frame_selector(idx):
            frame = watermarker.encode(frame, payload_bits)
            n_wm += 1
        frames_out.append(frame)

    write_video(frames_out, output_path, fps=fps, codec=codec, crf=crf)

    return {
        "n_frames_total": len(frames_out),
        "n_frames_watermarked": n_wm,
        "fps": fps,
        "codec_out": codec,
    }


def decode_watermark_from_video(
    input_path: str | Path,
    watermarker,
    frame_selector: FrameSelector = every_frame,
    max_frames: Optional[int] = None,
    majority_vote: bool = True,
) -> dict[str, Any]:
    """
    Read watermarked MP4, run decode on selected frames, aggregate bits.

    Aggregation: soft majority vote across frames (mean ≥ 0.5 → bit=1).

    Returns
    -------
    dict with keys:
        payload_bits  np.ndarray (64,)
        frame_bits    list of (frame_idx, bits) for every decoded frame
        confidence    np.ndarray (64,) float — per-bit vote fraction
        n_decoded     int
    """
    meta = read_video_metadata(input_path)
    total = min(meta["n_frames"] or 9999, max_frames or 9999)

    soft_accum = np.zeros(64, dtype=np.float64)
    frame_bits: list[tuple[int, np.ndarray]] = []
    n_decoded = 0

    frame_iter = read_frames(input_path, max_frames=max_frames)

    for idx, frame in tqdm(frame_iter, total=total, desc="Decoding", unit="fr"):
        if not frame_selector(idx):
            continue

        # Support both DCT (returns array) and TrustMark (returns (array, bool))
        result = watermarker.decode(frame)
        if isinstance(result, tuple):
            bits, _ = result
        else:
            bits = result

        soft_accum += bits.astype(np.float64)
        frame_bits.append((idx, bits.copy()))
        n_decoded += 1

    if n_decoded == 0:
        return {
            "payload_bits": np.zeros(64, dtype=np.uint8),
            "frame_bits": [],
            "confidence": np.zeros(64),
            "n_decoded": 0,
        }

    confidence = soft_accum / n_decoded
    payload = (confidence >= 0.5).astype(np.uint8)

    return {
        "payload_bits": payload,
        "frame_bits": frame_bits,
        "confidence": confidence,
        "n_decoded": n_decoded,
    }
