"""
Results reporter: load benchmark_results.json and generate comparison plots.

Usage
-----
    python -m video_watermark.benchmark.report results/benchmark_results.json

Produces in the same directory:
    comparison_quality.png   — PSNR / SSIM per frame
    comparison_ber.png       — BER per frame
    comparison_per_bit.png   — per-bit error rate heatmap
    comparison_summary.png   — bar chart of headline metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS = {"dct": "#1f77b4", "trustmark": "#d62728"}
LABELS = {"dct": "DCT (frequency domain)", "trustmark": "TrustMark (ML)"}


def _load(results_path: str | Path) -> dict:
    with open(results_path) as f:
        return json.load(f)


def _per_frame_array(method_data: dict, key: str) -> np.ndarray:
    return np.array([f[key] for f in method_data.get("per_frame", [])])


def plot_quality(results: dict, out_dir: Path) -> None:
    """PSNR and SSIM per frame, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Per-frame visual quality", fontsize=13, y=1.02)

    for method in ("dct", "trustmark"):
        data = results.get(method, {})
        if "error" in data or "per_frame" not in data:
            continue
        frames = _per_frame_array(data, "frame_idx")
        color  = COLORS[method]
        label  = LABELS[method]

        psnr = _per_frame_array(data, "psnr_db")
        ssim = _per_frame_array(data, "ssim")

        axes[0].plot(frames, psnr, color=color, label=label, linewidth=1.5, alpha=0.85)
        axes[1].plot(frames, ssim, color=color, label=label, linewidth=1.5, alpha=0.85)

    axes[0].set_xlabel("Frame index"); axes[0].set_ylabel("PSNR (dB)")
    axes[0].axhline(42, color="grey", linewidth=0.8, linestyle="--", label="42 dB target")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Frame index"); axes[1].set_ylabel("SSIM")
    axes[1].axhline(0.98, color="grey", linewidth=0.8, linestyle="--", label="0.98 target")
    axes[1].set_ylim(0.9, 1.0)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path = out_dir / "comparison_quality.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_ber(results: dict, out_dir: Path) -> None:
    """BER per frame."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Per-frame bit error rate (BER)")

    for method in ("dct", "trustmark"):
        data = results.get(method, {})
        if "error" in data or "per_frame" not in data:
            continue
        frames = _per_frame_array(data, "frame_idx")
        ber    = _per_frame_array(data, "ber")
        ax.plot(frames, ber, color=COLORS[method], label=LABELS[method],
                linewidth=1.5, alpha=0.85)

    ax.axhline(0.0,  color="green",  linewidth=0.8, linestyle="--", label="Perfect (0)")
    ax.axhline(0.5,  color="red",    linewidth=0.8, linestyle="--", label="Random (0.5)")
    ax.set_xlabel("Frame index"); ax.set_ylabel("BER")
    ax.set_ylim(-0.02, 0.55)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    path = out_dir / "comparison_ber.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_per_bit(results: dict, out_dir: Path) -> None:
    """Per-bit error rate heatmap (64 bits)."""
    methods = [m for m in ("dct", "trustmark") if "ber_per_bit" in results.get(m, {})]
    if not methods:
        return

    n = len(methods)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), squeeze=False)
    fig.suptitle("Per-bit error rate across all evaluated frames", fontsize=12)

    for i, method in enumerate(methods):
        bpb = np.array(results[method]["ber_per_bit"])   # (56,)
        heat = bpb.reshape(7, 8)
        im = axes[i][0].imshow(heat, vmin=0, vmax=0.5, cmap="RdYlGn_r", aspect="auto")
        axes[i][0].set_title(LABELS[method], fontsize=10)
        axes[i][0].set_xlabel("Bit position within byte (0=MSB)")
        axes[i][0].set_yticks(range(7))
        axes[i][0].set_yticklabels([f"byte {b}" for b in range(7)])
        fig.colorbar(im, ax=axes[i][0], fraction=0.02, pad=0.02, label="BER")

    fig.tight_layout()
    path = out_dir / "comparison_per_bit.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_summary(results: dict, out_dir: Path) -> None:
    """Bar chart of headline metrics."""
    metrics = {
        "PSNR (dB)":           ("psnr_mean",           None,  None),
        "SSIM":                 ("ssim_mean",            None,  None),
        "Bit accuracy (%)":     ("bit_accuracy_mean",    None,  100),
        "Exact match rate (%)": ("exact_match_rate",     None,  100),
    }

    methods = [m for m in ("dct", "trustmark")
               if results.get(m) and "error" not in results[m]]
    if not methods:
        return

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4))
    fig.suptitle("Headline metrics comparison", fontsize=13)

    for ax, (label, (key, lo, hi)) in zip(axes, metrics.items()):
        vals = []
        cols = []
        lbls = []
        for method in methods:
            data = results[method]
            raw = data.get(key, 0.0)
            if hi == 100:
                raw = float(raw) * 100
            vals.append(float(raw))
            cols.append(COLORS[method])
            lbls.append(LABELS[method])

        bars = ax.bar(lbls, vals, color=cols, alpha=0.85, width=0.5)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(range(len(lbls)))
        ax.set_xticklabels([l.replace(" ", "\n") for l in lbls], fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        if lo is not None:
            ax.set_ylim(bottom=lo)

    fig.tight_layout()
    path = out_dir / "comparison_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Generate watermark benchmark plots")
    p.add_argument("results_json", help="Path to benchmark_results.json")
    args = p.parse_args()

    results_path = Path(args.results_json)
    out_dir = results_path.parent
    results = _load(results_path)

    plot_quality(results, out_dir)
    plot_ber(results, out_dir)
    plot_per_bit(results, out_dir)
    plot_summary(results, out_dir)

    print("\nAll plots written to", out_dir)


if __name__ == "__main__":
    main()
