"""Decode the screen-recorded watermark and report per-frame success.

The clip ToS-1920_60s_camera.mp4 is a phone recording of the watermarked
output played back on a laptop screen. Compared to the lab attack suite,
it stacks several real-world distortions in one shot: display→camera
optical path, frame-rate change (24→30 fps), spatial rescale,
perspective skew, screen reflection / moire, and a fresh H.264 encode.

We expect both methods to perform poorly on BER. The interesting signal
is success rate over time: how often, and where in the clip, can each
method still recover the 56-bit payload?

CLI:
    python -m video_watermark.benchmark.decode_camera \\
        --input content/ToS-1920_60s_camera.mp4 \\
        --payload-text "ABCDEFG" \\
        --outdir results_camera/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from video_watermark.dct.watermarker import DCTWatermarker, PAYLOAD_BITS
from video_watermark.utils.video_io import read_frames, read_video_metadata

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# Original watermarked clip dimensions — DCT and the letterbox-detection
# heuristic both need this.
ORIG_W, ORIG_H = 1920, 800


def _detect_screen_crop(
    frame: np.ndarray,
    luma_thresh: float = 25.0,
    row_frac: float = 0.05,
) -> tuple[int, int, int, int]:
    """Find the bright sub-rectangle of the recording (the laptop screen).

    Returns (y0, y1, x0, x1) — the bounding box of pixels whose row/col
    mean luma exceeds `luma_thresh`. `row_frac` requires that at least
    that fraction of pixels in the row/col exceed the threshold, to
    reject sparse reflections.
    """
    g = frame.mean(axis=2)   # crude luma; sufficient for letterbox detection
    row_active = (g > luma_thresh).mean(axis=1) > row_frac
    col_active = (g > luma_thresh).mean(axis=0) > row_frac
    if not row_active.any() or not col_active.any():
        return 0, frame.shape[0], 0, frame.shape[1]
    y0 = int(np.argmax(row_active))
    y1 = int(len(row_active) - np.argmax(row_active[::-1]))
    x0 = int(np.argmax(col_active))
    x1 = int(len(col_active) - np.argmax(col_active[::-1]))
    return y0, y1, x0, x1


def _crop_and_resize(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    y0, y1, x0, x1 = _detect_screen_crop(frame)
    sub = frame[y0:y1, x0:x1]
    if sub.size == 0:
        sub = frame
    img = Image.fromarray(sub).resize((target_w, target_h), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _payload_text_to_bits(text: str) -> np.ndarray:
    data = text.encode("ascii")
    if len(data) > 7:
        raise ValueError(f"--payload-text accepts ≤ 7 ASCII chars, got {len(data)}")
    padded = data.ljust(7, b"\x00")
    return np.unpackbits(np.frombuffer(padded, dtype=np.uint8))[:PAYLOAD_BITS]


def main() -> None:
    p = argparse.ArgumentParser(description="Decode screen-recorded watermark.")
    p.add_argument("--input", required=True, help="Recording MP4")
    p.add_argument("--payload-text", default="ABCDEFG",
                   help="Expected ASCII payload (≤7 chars).")
    p.add_argument("--outdir", default="results_camera")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--dct-key", default="video_watermark_key")
    p.add_argument("--dct-delta", type=float, default=30.0)
    p.add_argument("--dct-num-blocks", type=int, default=4)
    p.add_argument("--trustmark-model", default="Q")
    p.add_argument("--trustmark-device", default=None,
                   help="mps/cuda/cpu (auto-detect by default)")
    p.add_argument("--skip-dct", action="store_true",
                   help="DCT requires exact pixel alignment; skip for camera data.")
    p.add_argument("--skip-trustmark", action="store_true")
    p.add_argument("--rectify", action="store_true",
                   help="Auto-detect screen region & resize to 1920x800 before "
                        "decode. Off by default — recording fills the frame and "
                        "rectification hurts more than it helps.")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    expected_bits = _payload_text_to_bits(args.payload_text)
    expected_str  = "".join(str(int(b)) for b in expected_bits)
    log.info("Expected payload (%s): %s", args.payload_text, expected_str)

    meta = read_video_metadata(args.input)
    log.info("Recording: %dx%d  %.2f fps  %d frames  %.1f s",
             meta["width"], meta["height"], meta["fps"],
             meta["n_frames"], meta["duration_s"])

    # Init watermarkers
    dct_wm = None
    tm_wm  = None
    if not args.skip_dct:
        dct_wm = DCTWatermarker(
            secret_key=args.dct_key,
            delta=args.dct_delta,
            num_blocks=args.dct_num_blocks,
        )
    if not args.skip_trustmark:
        from video_watermark.trustmark_video.adapter import TrustMarkVideoWatermarker
        tm_wm = TrustMarkVideoWatermarker(
            model_type=args.trustmark_model,
            verbose=False,
            device=args.trustmark_device,
        )
        log.info("TrustMark device: %s", tm_wm.device)

    per_frame: list[dict[str, Any]] = []

    total = args.max_frames or meta["n_frames"] or None

    for idx, frame in tqdm(
        read_frames(args.input, max_frames=args.max_frames),
        total=total, desc="Decoding", unit="fr",
    ):
        if args.rectify:
            rectified = _crop_and_resize(frame, ORIG_W, ORIG_H)
        else:
            rectified = frame

        row: dict[str, Any] = {"frame_idx": int(idx)}

        if dct_wm is not None:
            try:
                stats = dct_wm.decode_with_stats(rectified)
                bits  = stats["payload_bits"]
                ber   = float(np.mean(bits != expected_bits))
                row["dct_ber"]     = ber
                row["dct_nerr"]    = int(stats["n_errors_corrected"])
                row["dct_success"] = bool(ber == 0.0)
            except Exception as e:
                row["dct_error"] = str(e)

        if tm_wm is not None:
            try:
                bits, present = tm_wm.decode_frame(rectified)
                ber = float(np.mean(bits != expected_bits))
                row["tm_ber"]     = ber
                row["tm_present"] = bool(present)
                row["tm_success"] = bool(ber == 0.0)
            except Exception as e:
                row["tm_error"] = str(e)

        per_frame.append(row)

    # Aggregate
    def _agg(prefix: str) -> dict[str, Any]:
        bers   = [r[f"{prefix}_ber"]     for r in per_frame if f"{prefix}_ber"     in r]
        succ   = [r[f"{prefix}_success"] for r in per_frame if f"{prefix}_success" in r]
        if not bers:
            return {}
        bers_a = np.asarray(bers)
        return {
            "n_evaluated":      len(bers),
            "n_success":        int(sum(succ)),
            "success_rate":     float(sum(succ) / len(bers)),
            "ber_mean":         float(bers_a.mean()),
            "ber_min":          float(bers_a.min()),
            "ber_max":          float(bers_a.max()),
            "ber_p50":          float(np.median(bers_a)),
            "ber_p10":          float(np.percentile(bers_a, 10)),
            "frac_ber_below_0.05": float(np.mean(bers_a < 0.05)),
            "frac_ber_below_0.10": float(np.mean(bers_a < 0.10)),
            "frac_ber_below_0.20": float(np.mean(bers_a < 0.20)),
        }

    summary = {
        "input":          str(args.input),
        "meta":           meta,
        "payload_text":   args.payload_text,
        "expected_bits":  expected_bits.tolist(),
    }
    if dct_wm is not None:
        summary["dct"] = _agg("dct")
    if tm_wm is not None:
        summary["trustmark"] = _agg("tm")

    summary["per_frame"] = per_frame
    out_json = outdir / "camera_decode.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote %s", out_json)

    # Print headline
    print()
    print("=" * 60)
    print(f"Source: {args.input}")
    print(f"Frames evaluated: {len(per_frame)}")
    if "dct" in summary:
        d = summary["dct"]
        print(f"\nDCT   : success {d['n_success']}/{d['n_evaluated']} "
              f"({d['success_rate']:.2%})  BER median {d['ber_p50']:.3f}  "
              f"BER<0.10: {d['frac_ber_below_0.10']:.2%}")
    if "trustmark" in summary:
        t = summary["trustmark"]
        print(f"TM    : success {t['n_success']}/{t['n_evaluated']} "
              f"({t['success_rate']:.2%})  BER median {t['ber_p50']:.3f}  "
              f"BER<0.10: {t['frac_ber_below_0.10']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
