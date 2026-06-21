"""
Benchmark harness: DCT vs TrustMark video watermarking comparison.

Usage (CLI)
-----------
    python -m video_watermark.benchmark.run \
        --input  clip.mp4 \
        --outdir results/ \
        --payload-int 0xDEADBEEFCAFE1234 \
        --max-frames 60 \
        --crf 23 \
        --codec libx264 \
        --dct-delta 10 \
        --trustmark-model Q \
        --trustmark-encoding BCH_5

Usage (Python API)
------------------
    from video_watermark.benchmark.run import BenchmarkRunner
    runner = BenchmarkRunner(config)
    results = runner.run("clip.mp4")
    runner.print_comparison(results)
    runner.save_results(results, "results/")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from video_watermark.benchmark.metrics import (
    VideoMetrics,
    aggregate_frame_metrics,
    compute_frame_metrics,
)
from video_watermark.dct.watermarker import DCTWatermarker
from video_watermark.utils.video_io import (
    every_frame,
    every_nth,
    read_frames,
    read_video_metadata,
    write_video,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    payload_int: int = 0xDEADBEEFCAFEBA            # 56-bit payload (7 bytes)
    max_frames: Optional[int] = 60                 # None = all frames
    eval_every_nth: int = 1                        # evaluate every N-th frame
    crf: int = 23                                  # re-encode quality
    codec: str = "libx264"                         # or libx265

    # DCT settings
    dct_delta: float = 30.0
    dct_num_blocks: int = 4
    dct_secret_key: str = "video_watermark_key"

    # TrustMark settings
    trustmark_model: str = "Q"
    trustmark_strength: float = 1.0

    # Whether to run each method
    run_dct: bool = True
    run_trustmark: bool = True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        self.cfg = config or BenchmarkConfig()
        self._payload_bits = self._int_to_bits(self.cfg.payload_int)
        log.info("Payload (first 16 bits): %s ...", "".join(map(str, self._payload_bits[:16])))

    @staticmethod
    def _int_to_bits(value: int, n: int = 56) -> np.ndarray:
        return np.array([(value >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)

    # ------------------------------------------------------------------
    # Single-method run
    # ------------------------------------------------------------------

    def _run_dct(
        self,
        original_frames: list[np.ndarray],
        output_dir: Path,
        input_path: Path,
    ) -> tuple[VideoMetrics, float]:
        """Run DCT encode → re-encode → decode → measure. Returns (metrics, wall_s)."""
        from video_watermark.utils.video_io import write_video

        cfg = self.cfg
        wm = DCTWatermarker(
            secret_key=cfg.dct_secret_key,
            delta=cfg.dct_delta,
            num_blocks=cfg.dct_num_blocks,
        )

        log.info("[DCT] Encoding %d frames …", len(original_frames))
        t0 = time.perf_counter()

        wm_frames: list[np.ndarray] = []
        for frame in original_frames:
            wm_frames.append(wm.encode(frame, self._payload_bits))

        encode_wall = time.perf_counter() - t0

        # Write + re-read to simulate real codec round-trip
        wm_path = output_dir / "dct_watermarked.mp4"
        meta = read_video_metadata(input_path)
        write_video(wm_frames, wm_path, fps=meta["fps"], codec=cfg.codec, crf=cfg.crf)

        log.info("[DCT] Decoding from re-encoded file …")
        decoded_list: list[np.ndarray] = []
        frame_metrics = []

        for idx, frame in read_frames(wm_path, max_frames=cfg.max_frames):
            if idx % cfg.eval_every_nth != 0:
                continue
            bits = wm.decode(frame)
            decoded_list.append(bits)
            fm = compute_frame_metrics(
                frame_idx=idx,
                original=original_frames[min(idx, len(original_frames) - 1)],
                watermarked=wm_frames[min(idx, len(wm_frames) - 1)],
                original_bits=self._payload_bits,
                decoded_bits=bits,
            )
            frame_metrics.append(fm)

        wall = time.perf_counter() - t0
        metrics = aggregate_frame_metrics("dct", frame_metrics, self._payload_bits, decoded_list)
        log.info("[DCT] Done in %.1f s", wall)
        return metrics, encode_wall

    def _run_trustmark(
        self,
        original_frames: list[np.ndarray],
        output_dir: Path,
        input_path: Path,
    ) -> tuple[VideoMetrics, float]:
        """Run TrustMark encode → re-encode → decode → measure."""
        try:
            from video_watermark.trustmark_video.adapter import TrustMarkVideoWatermarker
        except ImportError as e:
            log.error("TrustMark not available: %s", e)
            raise

        cfg = self.cfg
        wm = TrustMarkVideoWatermarker(
            model_type=cfg.trustmark_model,
            strength=cfg.trustmark_strength,
            verbose=False,
        )

        log.info("[TrustMark] Encoding %d frames (model=%s) …", len(original_frames), cfg.trustmark_model)
        t0 = time.perf_counter()

        wm_frames: list[np.ndarray] = []
        for frame in original_frames:
            wm_frames.append(wm.encode_frame(frame, self._payload_bits))

        encode_wall = time.perf_counter() - t0

        wm_path = output_dir / "trustmark_watermarked.mp4"
        meta = read_video_metadata(input_path)
        write_video(wm_frames, wm_path, fps=meta["fps"], codec=cfg.codec, crf=cfg.crf)

        log.info("[TrustMark] Decoding from re-encoded file …")
        decoded_list: list[np.ndarray] = []
        frame_metrics = []

        for idx, frame in read_frames(wm_path, max_frames=cfg.max_frames):
            if idx % cfg.eval_every_nth != 0:
                continue
            bits, _ = wm.decode_frame(frame)
            decoded_list.append(bits)
            fm = compute_frame_metrics(
                frame_idx=idx,
                original=original_frames[min(idx, len(original_frames) - 1)],
                watermarked=wm_frames[min(idx, len(wm_frames) - 1)],
                original_bits=self._payload_bits,
                decoded_bits=bits,
            )
            frame_metrics.append(fm)

        wall = time.perf_counter() - t0
        metrics = aggregate_frame_metrics("trustmark", frame_metrics, self._payload_bits, decoded_list)
        log.info("[TrustMark] Done in %.1f s", wall)
        return metrics, encode_wall

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, input_path: str | Path, output_dir: str | Path = "results") -> dict[str, Any]:
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        meta = read_video_metadata(input_path)
        log.info("Input: %s  %dx%d  %.1f fps  codec=%s",
                 input_path.name, meta["width"], meta["height"], meta["fps"], meta["codec"])

        # Load original frames into memory once
        log.info("Loading up to %s frames …", self.cfg.max_frames or "all")
        original_frames = [
            frame for _, frame in read_frames(input_path, max_frames=self.cfg.max_frames)
        ]
        log.info("Loaded %d frames", len(original_frames))

        results: dict[str, Any] = {
            "input": str(input_path),
            "meta": meta,
            "payload_bits": self._payload_bits.tolist(),
            "payload_hex": hex(self.cfg.payload_int),
            "config": {k: str(v) for k, v in self.cfg.__dict__.items()},
        }

        if self.cfg.run_dct:
            dct_metrics, dct_enc_time = self._run_dct(original_frames, output_dir, input_path)
            results["dct"] = dct_metrics.to_dict()
            results["dct"]["encode_wall_s"] = dct_enc_time
            log.info("\n%s", dct_metrics.summary())

        if self.cfg.run_trustmark:
            try:
                tm_metrics, tm_enc_time = self._run_trustmark(original_frames, output_dir, input_path)
                results["trustmark"] = tm_metrics.to_dict()
                results["trustmark"]["encode_wall_s"] = tm_enc_time
                log.info("\n%s", tm_metrics.summary())
            except ImportError:
                log.warning("Skipping TrustMark (not installed)")
                results["trustmark"] = {"error": "trustmark not installed"}

        return results

    def print_comparison(self, results: dict[str, Any]) -> None:
        """Print a side-by-side comparison table to stdout."""
        dct = results.get("dct", {})
        tm  = results.get("trustmark", {})

        def v(d: dict, key: str, fmt: str = ".4f") -> str:
            val = d.get(key, "N/A")
            if val == "N/A":
                return val
            try:
                return format(float(val), fmt)
            except (TypeError, ValueError):
                return str(val)

        print()
        print("=" * 62)
        print(f"{'Metric':<30} {'DCT':>14}  {'TrustMark':>14}")
        print("-" * 62)
        rows = [
            ("PSNR mean (dB)",         "psnr_mean",              ".2f"),
            ("PSNR min (dB)",          "psnr_min",               ".2f"),
            ("SSIM mean",              "ssim_mean",              ".4f"),
            ("SSIM min",               "ssim_min",               ".4f"),
            ("Mean |Δpx|",             "mean_abs_diff_mean",     ".3f"),
            ("BER mean",               "ber_mean",               ".4f"),
            ("Bit accuracy mean",      "bit_accuracy_mean",      ".2%"),
            ("Exact match rate",       "exact_match_rate",       ".2%"),
            ("Norm correlation mean",  "norm_correlation_mean",  ".4f"),
            ("Encode wall (s)",        "encode_wall_s",          ".1f"),
            ("Frames evaluated",       "n_frames_evaluated",     "d"),
        ]
        for label, key, fmt in rows:
            dv = v(dct, key, fmt)
            tv = v(tm, key, fmt)
            print(f"{label:<30} {dv:>14}  {tv:>14}")
        print("=" * 62)

    def save_results(self, results: dict[str, Any], output_dir: str | Path) -> Path:
        out = Path(output_dir) / "benchmark_results.json"

        # Make numpy arrays JSON-serialisable
        def _default(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return str(obj)

        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=_default)
        log.info("Results saved to %s", out)
        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_PAYLOAD_BYTES = 7  # 56 bits


def _payload_text_to_int(text: str) -> int:
    """Pack up to 7 ASCII chars into a 56-bit int (big-endian, NUL-padded)."""
    try:
        data = text.encode("ascii")
    except UnicodeEncodeError as e:
        raise argparse.ArgumentTypeError(
            f"--payload-text must be ASCII-only: {e}"
        ) from e
    if len(data) > _PAYLOAD_BYTES:
        raise argparse.ArgumentTypeError(
            f"--payload-text fits at most {_PAYLOAD_BYTES} ASCII chars "
            f"(got {len(data)})"
        )
    return int.from_bytes(data.ljust(_PAYLOAD_BYTES, b"\x00"), "big")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare DCT vs TrustMark video watermarking"
    )
    p.add_argument("--input",           required=True, help="Input MP4 path")
    p.add_argument("--outdir",          default="results", help="Output directory")
    payload = p.add_mutually_exclusive_group()
    payload.add_argument("--payload-int",  type=lambda x: int(x, 0),
                   default=None, help="56-bit payload as hex (0x…)")
    payload.add_argument("--payload-text", type=str, default=None,
                   help=f"ASCII payload, up to {_PAYLOAD_BYTES} chars "
                        "(NUL-padded, big-endian)")
    p.add_argument("--max-frames",      type=int, default=60)
    p.add_argument("--eval-every-nth",  type=int, default=1)
    p.add_argument("--crf",             type=int, default=23)
    p.add_argument("--codec",           default="libx264", choices=["libx264", "libx265"])
    p.add_argument("--dct-delta",       type=float, default=30.0)
    p.add_argument("--dct-num-blocks",  type=int, default=4)
    p.add_argument("--dct-key",         default="video_watermark_key")
    p.add_argument("--trustmark-model", default="Q", choices=["B","C","P","Q"])
    p.add_argument("--trustmark-strength", type=float, default=1.0)
    p.add_argument("--no-dct",          action="store_true")
    p.add_argument("--no-trustmark",    action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.payload_text is not None:
        payload_int = _payload_text_to_int(args.payload_text)
    elif args.payload_int is not None:
        payload_int = args.payload_int
    else:
        payload_int = 0xDEADBEEFCAFEBA
    cfg = BenchmarkConfig(
        payload_int=payload_int,
        max_frames=args.max_frames,
        eval_every_nth=args.eval_every_nth,
        crf=args.crf,
        codec=args.codec,
        dct_delta=args.dct_delta,
        dct_num_blocks=args.dct_num_blocks,
        dct_secret_key=args.dct_key,
        trustmark_model=args.trustmark_model,
        trustmark_strength=args.trustmark_strength,
        run_dct=not args.no_dct,
        run_trustmark=not args.no_trustmark,
    )
    runner = BenchmarkRunner(cfg)
    results = runner.run(args.input, args.outdir)
    runner.print_comparison(results)
    runner.save_results(results, args.outdir)


if __name__ == "__main__":
    main()
