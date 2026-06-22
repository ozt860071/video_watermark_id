"""
Robustness attacks for watermarked video.

Each `attack_*` function reads an MP4, transforms it, and writes a new MP4 to
`output_path`. Spatial attacks (resize, crop) restore the original resolution
so the watermarker's coordinate system still lines up on decode.

Spatial / temporal / noise attacks use a near-lossless re-encode (CRF 18) by
default so the attack itself dominates the result rather than codec
compression. The `recompress` and `transcode` attacks let the user pick the
CRF/codec explicitly — for them the codec IS the attack.

Public surface
--------------
ATTACKS              dict[str, callable]  — short name → attack function
DEFAULT_PRESETS      dict[str, dict]      — preset name → {attack, params}
run_preset(...)      apply a preset by name
evaluate_decoded(...)decode an attacked file and score BER vs. expected bits
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from video_watermark.utils.video_io import (
    read_frames,
    read_video_metadata,
    write_video,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-frame transforms
# ---------------------------------------------------------------------------

def _resize_roundtrip(frame: np.ndarray, scale: float) -> np.ndarray:
    """Downscale by `scale`, then upscale back to the original resolution."""
    H, W = frame.shape[:2]
    nW = max(2, int(round(W * scale)))
    nH = max(2, int(round(H * scale)))
    img = Image.fromarray(frame)
    small = img.resize((nW, nH), Image.BILINEAR)
    back = small.resize((W, H), Image.BILINEAR)
    return np.asarray(back, dtype=np.uint8)


def _center_crop_and_restore(frame: np.ndarray, keep_fraction: float) -> np.ndarray:
    """Center-crop to `keep_fraction` of each dim, then upscale back to size."""
    H, W = frame.shape[:2]
    kH = max(2, int(round(H * keep_fraction)))
    kW = max(2, int(round(W * keep_fraction)))
    y0 = (H - kH) // 2
    x0 = (W - kW) // 2
    cropped = frame[y0:y0 + kH, x0:x0 + kW]
    img = Image.fromarray(cropped).resize((W, H), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _add_gaussian_noise(frame: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise (sigma on the 0-255 scale)."""
    noise = rng.normal(0.0, sigma, size=frame.shape)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# File-level attacks
# ---------------------------------------------------------------------------

def _write_attacked(
    frames: list[np.ndarray],
    output_path: Path,
    fps: float,
    codec: str,
    crf: int,
) -> Path:
    write_video(frames, output_path, fps=fps, codec=codec, crf=crf)
    return Path(output_path)


def attack_recompress(
    input_path: Path,
    output_path: Path,
    crf: int = 32,
    codec: str = "libx264",
    max_frames: Optional[int] = None,
) -> Path:
    """Re-encode the video at a (typically harsher) CRF."""
    meta = read_video_metadata(input_path)
    frames = [f for _, f in read_frames(input_path, max_frames=max_frames)]
    return _write_attacked(frames, Path(output_path), meta["fps"], codec, crf)


def attack_transcode(
    input_path: Path,
    output_path: Path,
    codec: str = "libx265",
    crf: int = 28,
    max_frames: Optional[int] = None,
) -> Path:
    """Re-encode with a different codec (default x264 → x265)."""
    return attack_recompress(input_path, output_path, crf=crf, codec=codec, max_frames=max_frames)


def attack_resize(
    input_path: Path,
    output_path: Path,
    scale: float = 0.5,
    crf: int = 18,
    codec: str = "libx264",
    max_frames: Optional[int] = None,
) -> Path:
    """Downscale by `scale`, upscale back, then re-encode near-losslessly."""
    meta = read_video_metadata(input_path)
    frames = [
        _resize_roundtrip(f, scale)
        for _, f in read_frames(input_path, max_frames=max_frames)
    ]
    return _write_attacked(frames, Path(output_path), meta["fps"], codec, crf)


def attack_crop(
    input_path: Path,
    output_path: Path,
    keep_fraction: float = 0.8,
    crf: int = 18,
    codec: str = "libx264",
    max_frames: Optional[int] = None,
) -> Path:
    """Center-crop keeping `keep_fraction` of each dim, then restore size."""
    meta = read_video_metadata(input_path)
    frames = [
        _center_crop_and_restore(f, keep_fraction)
        for _, f in read_frames(input_path, max_frames=max_frames)
    ]
    return _write_attacked(frames, Path(output_path), meta["fps"], codec, crf)


def attack_noise(
    input_path: Path,
    output_path: Path,
    sigma: float = 5.0,
    crf: int = 18,
    codec: str = "libx264",
    seed: int = 0,
    max_frames: Optional[int] = None,
) -> Path:
    """Add zero-mean Gaussian noise (sigma on 0-255), then re-encode."""
    meta = read_video_metadata(input_path)
    rng = np.random.default_rng(seed)
    frames = [
        _add_gaussian_noise(f, sigma, rng)
        for _, f in read_frames(input_path, max_frames=max_frames)
    ]
    return _write_attacked(frames, Path(output_path), meta["fps"], codec, crf)


def attack_frame_drop(
    input_path: Path,
    output_path: Path,
    drop_ratio: float = 0.5,
    crf: int = 18,
    codec: str = "libx264",
    max_frames: Optional[int] = None,
) -> Path:
    """Uniformly drop `drop_ratio` of frames (keeps every k-th)."""
    if not 0 <= drop_ratio < 1:
        raise ValueError("drop_ratio must be in [0, 1)")
    meta = read_video_metadata(input_path)
    keep_every = max(1, int(round(1.0 / (1.0 - drop_ratio))))
    frames = [
        f for i, f in read_frames(input_path, max_frames=max_frames)
        if i % keep_every == 0
    ]
    if not frames:
        raise ValueError("frame_drop produced 0 frames")
    return _write_attacked(frames, Path(output_path), meta["fps"], codec, crf)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ATTACKS: dict[str, Callable[..., Path]] = {
    "recompress": attack_recompress,
    "transcode":  attack_transcode,
    "resize":     attack_resize,
    "crop":       attack_crop,
    "noise":      attack_noise,
    "frame_drop": attack_frame_drop,
}

# Pre-baked parameter sweeps. Each preset = {"attack": <key in ATTACKS>, "params": {...}}.
DEFAULT_PRESETS: dict[str, dict[str, Any]] = {
    "recompress_crf28":   {"attack": "recompress", "params": {"crf": 28}},
    "recompress_crf32":   {"attack": "recompress", "params": {"crf": 32}},
    "recompress_crf36":   {"attack": "recompress", "params": {"crf": 36}},
    "transcode_x265":     {"attack": "transcode",  "params": {"codec": "libx265", "crf": 28}},
    "resize_0.75":        {"attack": "resize",     "params": {"scale": 0.75}},
    "resize_0.5":         {"attack": "resize",     "params": {"scale": 0.5}},
    "crop_0.8":           {"attack": "crop",       "params": {"keep_fraction": 0.8}},
    "crop_0.6":           {"attack": "crop",       "params": {"keep_fraction": 0.6}},
    "noise_sigma2":       {"attack": "noise",      "params": {"sigma": 2.0}},
    "noise_sigma5":       {"attack": "noise",      "params": {"sigma": 5.0}},
    "noise_sigma10":      {"attack": "noise",      "params": {"sigma": 10.0}},
    "frame_drop_0.5":     {"attack": "frame_drop", "params": {"drop_ratio": 0.5}},
}


def run_preset(
    preset_name: str,
    input_path: str | Path,
    output_path: str | Path,
    max_frames: Optional[int] = None,
) -> Path:
    """Apply a named preset from DEFAULT_PRESETS to `input_path`."""
    if preset_name not in DEFAULT_PRESETS:
        raise KeyError(f"Unknown preset {preset_name!r}. "
                       f"Known: {sorted(DEFAULT_PRESETS)}")
    spec = DEFAULT_PRESETS[preset_name]
    fn = ATTACKS[spec["attack"]]
    return fn(Path(input_path), Path(output_path), max_frames=max_frames, **spec["params"])


def resolve_preset_names(names: list[str] | str) -> list[str]:
    """Expand 'all' or a comma string into a concrete preset list."""
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    if not names or names == ["all"]:
        return list(DEFAULT_PRESETS.keys())
    unknown = [n for n in names if n not in DEFAULT_PRESETS]
    if unknown:
        raise KeyError(f"Unknown preset(s): {unknown}. "
                       f"Known: {sorted(DEFAULT_PRESETS)}")
    return list(names)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_decoded(
    attacked_path: str | Path,
    watermarker,
    expected_bits: np.ndarray,
    max_frames: Optional[int] = None,
    eval_every_nth: int = 1,
) -> dict[str, Any]:
    """
    Decode every selected frame of `attacked_path` with `watermarker`,
    score against `expected_bits`, and return aggregate BER stats.

    Returns
    -------
    dict with keys:
        n_frames_evaluated, n_frames_exact_match, exact_match_rate,
        ber_mean, ber_std, ber_min, ber_max,
        bit_accuracy_mean, ber_per_bit (list[float]), majority_vote_ber
    """
    n_bits = len(expected_bits)
    per_frame_ber: list[float] = []
    decoded_stack: list[np.ndarray] = []
    n_exact = 0

    for idx, frame in read_frames(attacked_path, max_frames=max_frames):
        if idx % eval_every_nth != 0:
            continue
        result = watermarker.decode(frame)
        bits = result[0] if isinstance(result, tuple) else result
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        if bits.shape[0] != n_bits:
            raise ValueError(
                f"Decoded {bits.shape[0]} bits but expected {n_bits}"
            )
        decoded_stack.append(bits)
        ber = float(np.mean(bits != expected_bits))
        per_frame_ber.append(ber)
        if ber == 0.0:
            n_exact += 1

    n = len(per_frame_ber)
    if n == 0:
        return {
            "n_frames_evaluated": 0,
            "n_frames_exact_match": 0,
            "exact_match_rate": 0.0,
            "ber_mean": float("nan"),
            "ber_std":  float("nan"),
            "ber_min":  float("nan"),
            "ber_max":  float("nan"),
            "bit_accuracy_mean": float("nan"),
            "ber_per_bit": [float("nan")] * n_bits,
            "majority_vote_ber": float("nan"),
        }

    bers = np.asarray(per_frame_ber)
    stacked = np.stack(decoded_stack, axis=0)
    ber_per_bit = np.mean(stacked != expected_bits[None, :], axis=0)
    # Soft majority vote over frames, then BER against expected.
    voted = (stacked.mean(axis=0) >= 0.5).astype(np.uint8)
    maj_ber = float(np.mean(voted != expected_bits))

    return {
        "n_frames_evaluated":  int(n),
        "n_frames_exact_match": int(n_exact),
        "exact_match_rate":    float(n_exact / n),
        "ber_mean":            float(bers.mean()),
        "ber_std":             float(bers.std()),
        "ber_min":             float(bers.min()),
        "ber_max":             float(bers.max()),
        "bit_accuracy_mean":   float(1.0 - bers.mean()),
        "ber_per_bit":         ber_per_bit.tolist(),
        "majority_vote_ber":   maj_ber,
    }
