"""
Quality and robustness metrics for watermarking comparison.

Metrics
-------
psnr(original, watermarked)
    Peak Signal-to-Noise Ratio in dB.  Target ≥ 42 dB for invisible WM.

ssim_frame(original, watermarked)
    Structural Similarity Index (SSIM).  Target ≥ 0.98.

bit_error_rate(original_bits, decoded_bits)
    Fraction of bits that differ.  Lower = more robust.

bit_accuracy(original_bits, decoded_bits)
    1 - BER.

payload_match(original_bits, decoded_bits)
    True if all 56 bits match exactly.

per_frame_metrics(original_frames, watermarked_frames, original_bits,
                  decoded_bits_per_frame)
    Compute all metrics per frame and return a summary dict.

aggregate_video_metrics(frame_metrics_list)
    Aggregate per-frame metrics into overall video statistics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr_sk
from skimage.metrics import structural_similarity as _ssim_sk


# ---------------------------------------------------------------------------
# Per-frame quality metrics
# ---------------------------------------------------------------------------

def psnr(original: np.ndarray, watermarked: np.ndarray) -> float:
    """PSNR in dB between two uint8 RGB images."""
    return float(_psnr_sk(original, watermarked, data_range=255))


def ssim_frame(original: np.ndarray, watermarked: np.ndarray) -> float:
    """SSIM between two uint8 RGB images."""
    return float(
        _ssim_sk(original, watermarked, channel_axis=-1, data_range=255)
    )


def pixel_diff_stats(original: np.ndarray, watermarked: np.ndarray) -> dict:
    """
    Return statistics on per-pixel absolute difference.

    Keys: mean_abs_diff, max_abs_diff, std_abs_diff
    """
    diff = np.abs(original.astype(np.int32) - watermarked.astype(np.int32))
    return {
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff":  int(diff.max()),
        "std_abs_diff":  float(diff.std()),
    }


# ---------------------------------------------------------------------------
# Per-frame robustness metrics
# ---------------------------------------------------------------------------

def bit_error_rate(original_bits: np.ndarray, decoded_bits: np.ndarray) -> float:
    """Fraction of bits that differ (0.0 = perfect, 0.5 = random noise)."""
    assert len(original_bits) == len(decoded_bits)
    return float(np.mean(original_bits != decoded_bits))


def bit_accuracy(original_bits: np.ndarray, decoded_bits: np.ndarray) -> float:
    """1 - BER."""
    return 1.0 - bit_error_rate(original_bits, decoded_bits)


def payload_match(original_bits: np.ndarray, decoded_bits: np.ndarray) -> bool:
    """True if all bits match exactly."""
    return bool(np.array_equal(original_bits, decoded_bits))


def normalized_correlation(original_bits: np.ndarray, decoded_bits: np.ndarray) -> float:
    """
    Normalised correlation between ±1-coded bit arrays.
    +1 = perfect match, 0 = random, -1 = inverted.
    """
    a = original_bits.astype(np.float32) * 2 - 1
    b = decoded_bits.astype(np.float32) * 2 - 1
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Structured result types
# ---------------------------------------------------------------------------

@dataclass
class FrameMetrics:
    frame_idx: int
    psnr_db: float
    ssim: float
    mean_abs_diff: float
    max_abs_diff: int
    ber: float
    bit_accuracy: float
    payload_exact_match: bool
    norm_correlation: float


@dataclass
class VideoMetrics:
    """Aggregated metrics for a complete watermarked video."""
    method: str                     # 'dct' or 'trustmark'
    n_frames_evaluated: int
    n_frames_exact_match: int

    # Quality (mean ± std across frames)
    psnr_mean: float
    psnr_std: float
    psnr_min: float

    ssim_mean: float
    ssim_std: float
    ssim_min: float

    mean_abs_diff_mean: float

    # Robustness
    ber_mean: float                 # mean over all frames
    ber_std: float
    ber_per_bit: np.ndarray         # (56,) per-bit error rate

    bit_accuracy_mean: float
    exact_match_rate: float         # fraction of frames with all-correct bits
    norm_correlation_mean: float

    # Raw per-frame records for plotting
    per_frame: list[FrameMetrics] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Method: {self.method}",
            f"  Frames evaluated : {self.n_frames_evaluated}",
            f"  Exact match rate : {self.exact_match_rate:.1%}",
            f"  PSNR             : {self.psnr_mean:.2f} ± {self.psnr_std:.2f} dB  (min {self.psnr_min:.2f})",
            f"  SSIM             : {self.ssim_mean:.4f} ± {self.ssim_std:.4f}  (min {self.ssim_min:.4f})",
            f"  Mean |Δpx|       : {self.mean_abs_diff_mean:.3f} / 255",
            f"  BER              : {self.ber_mean:.4f} ± {self.ber_std:.4f}  (bit accuracy {self.bit_accuracy_mean:.2%})",
            f"  Norm correlation : {self.norm_correlation_mean:.4f}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ber_per_bit"] = self.ber_per_bit.tolist()
        d["per_frame"] = [asdict(f) for f in self.per_frame]
        return d


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def compute_frame_metrics(
    frame_idx: int,
    original: np.ndarray,
    watermarked: np.ndarray,
    original_bits: np.ndarray,
    decoded_bits: np.ndarray,
) -> FrameMetrics:
    diff = pixel_diff_stats(original, watermarked)
    return FrameMetrics(
        frame_idx=frame_idx,
        psnr_db=psnr(original, watermarked),
        ssim=ssim_frame(original, watermarked),
        mean_abs_diff=diff["mean_abs_diff"],
        max_abs_diff=diff["max_abs_diff"],
        ber=bit_error_rate(original_bits, decoded_bits),
        bit_accuracy=bit_accuracy(original_bits, decoded_bits),
        payload_exact_match=payload_match(original_bits, decoded_bits),
        norm_correlation=normalized_correlation(original_bits, decoded_bits),
    )


def aggregate_frame_metrics(
    method: str,
    frame_metrics: list[FrameMetrics],
    original_bits: np.ndarray,
    all_decoded_bits: list[np.ndarray],
) -> VideoMetrics:
    """
    Aggregate a list of FrameMetrics into a VideoMetrics summary.

    Parameters
    ----------
    method : str
    frame_metrics : list[FrameMetrics]
    original_bits : np.ndarray (56,)
    all_decoded_bits : list of np.ndarray (56,)  — one per frame
    """
    if not frame_metrics:
        raise ValueError("No frame metrics to aggregate")

    psnrs = np.array([f.psnr_db for f in frame_metrics])
    ssims = np.array([f.ssim for f in frame_metrics])
    diffs = np.array([f.mean_abs_diff for f in frame_metrics])
    bers  = np.array([f.ber for f in frame_metrics])
    accs  = np.array([f.bit_accuracy for f in frame_metrics])
    corrs = np.array([f.norm_correlation for f in frame_metrics])
    n_exact = sum(1 for f in frame_metrics if f.payload_exact_match)

    # Per-bit error rate across all frames
    if all_decoded_bits:
        stacked = np.stack(all_decoded_bits, axis=0)   # (N_frames, 56)
        ber_per_bit = np.mean(stacked != original_bits[None, :], axis=0)
    else:
        ber_per_bit = np.zeros(56)

    return VideoMetrics(
        method=method,
        n_frames_evaluated=len(frame_metrics),
        n_frames_exact_match=n_exact,
        psnr_mean=float(psnrs.mean()),
        psnr_std=float(psnrs.std()),
        psnr_min=float(psnrs.min()),
        ssim_mean=float(ssims.mean()),
        ssim_std=float(ssims.std()),
        ssim_min=float(ssims.min()),
        mean_abs_diff_mean=float(diffs.mean()),
        ber_mean=float(bers.mean()),
        ber_std=float(bers.std()),
        ber_per_bit=ber_per_bit,
        bit_accuracy_mean=float(accs.mean()),
        exact_match_rate=float(n_exact / len(frame_metrics)),
        norm_correlation_mean=float(corrs.mean()),
        per_frame=frame_metrics,
    )
