"""Generate docs/report.docx — project report on the video watermarking benchmark.

Run:
    python docs/build_report.py
"""

from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Inches


OUT = Path(__file__).resolve().parent / "report.docx"
IMG_DIR = Path(__file__).resolve().parent

# Attack presets ordered for the snapshot grid (matches §8 results table)
_PRESETS = [
    "recompress_crf28", "recompress_crf32", "recompress_crf36",
    "transcode_x265",
    "resize_0.75", "resize_0.5",
    "crop_0.8", "crop_0.6",
    "noise_sigma2", "noise_sigma5", "noise_sigma10",
    "frame_drop_0.5",
]


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = "Helvetica"
    return h


def add_para(doc, text, bold=False, italic=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Helvetica"
    run.font.size = Pt(11)
    run.bold = bold
    run.italic = italic
    return p


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(item)
        run.font.name = "Helvetica"
        run.font.size = Pt(11)


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for p in hdr_cells[i].paragraphs:
            for run in p.runs:
                run.bold = True
                run.font.name = "Helvetica"
                run.font.size = Pt(10)
    for r, row in enumerate(rows, start=1):
        for i, val in enumerate(row):
            cell = table.rows[r].cells[i]
            cell.text = str(val)
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.name = "Helvetica"
                    run.font.size = Pt(10)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    if col_widths:
        for r in table.rows:
            for i, w in enumerate(col_widths):
                r.cells[i].width = w
    return table


def _add_snapshot_table(doc, img_dir: Path):
    """3-column grid: preset | DCT thumb | TrustMark thumb.

    Row 1 is the baseline (un-watermarked original) merged across the
    two image columns; rows 2-13 are the 12 attack presets.
    """
    cell_img_w = Inches(2.7)
    baseline_w = Inches(5.5)

    table = doc.add_table(rows=1 + 1 + len(_PRESETS), cols=3)
    table.style = "Light Grid Accent 1"

    # Header row
    for i, txt in enumerate(["Preset", "DCT", "TrustMark"]):
        cell = table.rows[0].cells[i]
        cell.text = txt
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.name = "Helvetica"
                r.font.size = Pt(10)

    # Baseline row (merged image columns)
    base_cells = table.rows[1].cells
    base_cells[0].text = "baseline\n(original)"
    for p in base_cells[0].paragraphs:
        for r in p.runs:
            r.font.name = "Helvetica"
            r.font.size = Pt(10)
    merged = base_cells[1].merge(base_cells[2])
    merged.paragraphs[0].add_run().add_picture(
        str(img_dir / "frame12_baseline.jpg"), width=baseline_w
    )

    # Attack rows
    for ri, preset in enumerate(_PRESETS, start=2):
        cells = table.rows[ri].cells
        cells[0].text = preset
        for p in cells[0].paragraphs:
            for r in p.runs:
                r.font.name = "Helvetica"
                r.font.size = Pt(10)
        cells[1].paragraphs[0].add_run().add_picture(
            str(img_dir / f"frame12_dct_{preset}.jpg"), width=cell_img_w
        )
        cells[2].paragraphs[0].add_run().add_picture(
            str(img_dir / f"frame12_trustmark_{preset}.jpg"), width=cell_img_w
        )
    return table


def main():
    doc = Document()

    # Default body font
    style = doc.styles["Normal"]
    style.font.name = "Helvetica"
    style.font.size = Pt(11)

    # ----- Title -----
    title = doc.add_heading("Video Watermarking Benchmark", level=0)
    for run in title.runs:
        run.font.name = "Helvetica"
    sub = doc.add_paragraph()
    sub_run = sub.add_run(
        "DCT-QIM vs. Adobe TrustMark — quality, robustness, and "
        "failure-mode characterisation under realistic codec and "
        "geometric attacks."
    )
    sub_run.italic = True
    sub_run.font.size = Pt(11)
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ----- 1. Problem framing -----
    add_heading(doc, "1. Problem framing", level=1)
    add_para(
        doc,
        "Embedding a recoverable identifier into video frames is the basis "
        "for provenance, leak tracing, and tamper detection. Two families "
        "of techniques dominate: classical signal-processing methods that "
        "modulate transform-domain coefficients (DCT-QIM, spread-spectrum), "
        "and learned methods that train an encoder/decoder pair to embed "
        "a payload as an imperceptible residual. This project benchmarks "
        "one representative from each family — a custom DCT-QIM scheme "
        "and Adobe TrustMark — on a common 56-bit payload, common video "
        "codec round-trip, and a shared attack suite, so the trade-offs "
        "can be compared on equal terms."
    )
    add_para(
        doc,
        "The benchmark answers three questions:"
    )
    add_bullets(doc, [
        "How invisible is each watermark on real content (PSNR, SSIM)?",
        "How robust is each watermark after a realistic H.264 / H.265 "
        "round-trip and downstream attacks (per-frame BER, exact-match rate)?",
        "Which frames are easier to recover from than others, and why?",
    ])

    # ----- 2. Assumptions and constraints -----
    add_heading(doc, "2. Assumptions and constraints", level=1)
    add_bullets(doc, [
        "Payload size is fixed at 56 bits (7 bytes). This matches both "
        "the DCT scheme's BCH(127, t=11) data field and TrustMark's BCH_5 "
        "61-bit capacity (with 5 unused bits of headroom).",
        "Watermarking is per-frame and independent. No temporal "
        "exploitation across frames during embedding; the decoder is free "
        "to majority-vote across frames after the fact.",
        "Encoder and decoder share a secret key and operate on a fixed "
        "frame geometry. The DCT scheme's PN block selection depends on "
        "frame dimensions, so geometric distortions break it unless the "
        "dimensions are restored before decode.",
        "The reference codec is H.264 (libx264) at CRF 23 — a common "
        "streaming quality. H.265 (libx265) is used to test cross-codec "
        "robustness.",
        "Evaluation is run on a 1920×800 cinematic clip excerpted from "
        "Tears of Steel, sampled at 24 frames. Frames with degenerate "
        "luma (e.g. opening black title cards) are excluded because the "
        "QIM scheme is mathematically unable to embed in a flat Y=0 block.",
    ])

    # ----- 3. Embedding strategy -----
    add_heading(doc, "3. Embedding strategy", level=1)
    add_heading(doc, "3.1 DCT-QIM", level=2)
    add_para(
        doc,
        "The frame is converted to BT.601 full-range YCbCr; embedding "
        "happens only in the Y plane. The Y plane is tiled into 8×8 "
        "blocks (the same grid H.264 uses for transform coding) and a "
        "2-D orthonormal DCT-II is applied to each block."
    )
    add_para(
        doc,
        "The 56-bit payload is expanded by BCH(n=127, k=57, t=11) into "
        "a 127-bit codeword. Each of the 127 codeword chips is embedded "
        "in a pseudo-randomly chosen block via Quantisation Index "
        "Modulation (QIM) of a mid-band AC coefficient (zigzag indices "
        "5–14). The block-selection sequence is keyed by SHA-256 of the "
        "user-supplied secret. With num_blocks=4, each chip is written "
        "to four distinct blocks at four different mid-band coefficients, "
        "trading payload density for an inner majority-vote on decode "
        "in addition to the outer BCH correction."
    )
    add_para(
        doc,
        "After modification, the inverse DCT reconstructs the luma plane; "
        "YCbCr is converted back to RGB, rounded, and cast to uint8 for "
        "storage. QIM step δ = 30 is chosen empirically to survive H.264 "
        "down to CRF 28 while preserving PSNR ≈ 60 dB on smooth content."
    )

    add_heading(doc, "3.2 TrustMark", level=2)
    add_para(
        doc,
        "TrustMark is a learned encoder/decoder pair (Adobe, 2023). The "
        "encoder operates internally at 256×256 and re-projects a learned "
        "residual onto the input frame. We wrap it in a per-frame adapter "
        "that exposes the same encode/decode interface as the DCT scheme."
    )
    add_para(
        doc,
        "The 56-bit payload is serialised as a 56-character binary "
        "string ('01001...') and passed to TrustMark.encode() with "
        "MODE='binary' and BCH_5 encoding (35 ECC bits, corrects up to "
        "5 bit-flips, 61-bit capacity). MODE='binary' is critical: the "
        "default MODE='text' packs the string as ASCII text, which then "
        "does not round-trip when decoded with MODE='binary' — see §9 "
        "failure analysis."
    )

    # ----- 4. Detection strategy -----
    add_heading(doc, "4. Detection strategy", level=1)
    add_heading(doc, "4.1 DCT-QIM", level=2)
    add_para(
        doc,
        "Decode mirrors encode: RGB → BT.601 YCbCr → 8×8 block tiling → "
        "DCT. For each of the 127 chips the decoder reads num_blocks "
        "(default 4) QIM coefficients and takes the soft mean as the "
        "chip's value (≥ 0.5 → bit = 1). The 127 hard chips are then "
        "BCH-decoded; up to 11 chip errors are corrected and the 56-bit "
        "payload is recovered. The decoder also returns "
        "n_errors_corrected (or −1 for an uncorrectable codeword), which "
        "is useful for confidence reporting."
    )
    add_heading(doc, "4.2 TrustMark", level=2)
    add_para(
        doc,
        "Each frame is passed through TrustMark.decode(MODE='binary'), "
        "which returns the recovered binary string, a watermark-present "
        "flag, and a schema id. The first 56 characters of the string "
        "are converted back to a (56,) uint8 array. The model's own "
        "presence flag is reported alongside per-frame BER."
    )
    add_para(
        doc,
        "At the video level, both methods support a soft majority vote "
        "across all decoded frames as an additional aggregation step "
        "(reported as 'majority-vote BER' in the attack results)."
    )

    # ----- 5. Attack model -----
    add_heading(doc, "5. Attack model", level=1)
    add_para(
        doc,
        "Attacks are applied as a separate stage after the main encode + "
        "codec round-trip. Each attack reads the watermarked MP4, "
        "transforms it, and writes a new MP4 — either with the user's "
        "chosen CRF (compression attacks) or near-lossless CRF 18 "
        "(non-codec attacks, to isolate the attack from extra codec noise). "
        "Twelve presets across six attack families ship by default:"
    )
    add_table(doc,
        headers=["Family", "Presets", "What it stresses"],
        rows=[
            ["Re-encode (compression)",
             "recompress_crf28, crf32, crf36",
             "Codec quantisation increasing in severity"],
            ["Transcode (codec switch)",
             "transcode_x265 (CRF 28)",
             "Different codec, different quantisation matrix"],
            ["Spatial rescale",
             "resize_0.75, resize_0.5",
             "Downscale → upscale (player resizing, sharing pipelines)"],
            ["Spatial crop",
             "crop_0.8, crop_0.6",
             "Center-crop fraction → rescale back (camera-zoom mimic)"],
            ["Additive Gaussian noise",
             "noise_sigma2, sigma5, sigma10",
             "Sensor noise, generation loss, transcoding artefacts"],
            ["Frame drop",
             "frame_drop_0.5",
             "Frame-rate reduction; the surviving frames must still decode"],
        ],
        col_widths=[Inches(2.0), Inches(2.4), Inches(2.6)],
    )
    add_para(
        doc,
        "The attack model is intentionally non-adversarial. There is no "
        "watermark-aware adversary trying to remove the signal; the "
        "presets approximate distortions that occur naturally in real "
        "distribution pipelines (re-streaming, mobile playback, format "
        "conversion). Stronger attacks (collusion, oracle attacks, "
        "deep-learning removal) are out of scope for this benchmark."
    )

    # ----- 6. Evaluation methodology -----
    add_heading(doc, "6. Evaluation methodology", level=1)
    add_para(
        doc,
        "The pipeline is one CLI invocation per run "
        "(python -m video_watermark.benchmark.run). For each frame in "
        "the input clip:"
    )
    add_bullets(doc, [
        "Load the original frame as (H, W, 3) uint8 RGB via PyAV.",
        "Run the chosen watermarker's encode(frame, payload_bits) — "
        "watermarked frames stay in memory.",
        "Write all watermarked frames out as an MP4 at the user-chosen "
        "codec/CRF — this is the main codec round-trip.",
        "Re-read the watermarked MP4 frame-by-frame, call decode(frame) "
        "per the per-method API, and compute per-frame quality and "
        "robustness metrics against the original frame and the known "
        "payload bits.",
        "If --attacks is given, for each attack preset: re-encode the "
        "watermarked MP4 into a per-attack file, decode each frame, "
        "score per-frame BER and success against the expected payload, "
        "and emit the full per-frame trace (frame_idx, ber, success) "
        "alongside the aggregate stats.",
    ])
    add_para(
        doc,
        "All numbers are saved to benchmark_results.json. The per-frame "
        "trace is the entry point for the 'which frames are easier?' "
        "analysis."
    )

    # ----- 7. Metrics -----
    add_heading(doc, "7. Metrics used", level=1)
    add_para(doc, "Quality (watermarked vs original frame):", bold=True)
    add_bullets(doc, [
        "PSNR (dB) — peak signal-to-noise ratio. Target ≥ 42 dB for "
        "invisible watermarking; ≥ 38 dB is the perceptibility floor.",
        "SSIM — structural similarity index, channel-averaged. Target "
        "≥ 0.98.",
        "Mean |Δpx| — average per-pixel absolute difference, on the "
        "0–255 scale.",
    ])
    add_para(doc, "Robustness (decoded bits vs. expected payload):", bold=True)
    add_bullets(doc, [
        "Per-frame BER — fraction of the 56 bits that differ. 0.0 is "
        "perfect, 0.5 is random output (watermark destroyed).",
        "Exact-match rate (recovered / total) — fraction of frames "
        "where all 56 bits decode correctly. Reported as the primary "
        "headline number in the attack table.",
        "Per-bit BER — error rate per individual payload bit across "
        "frames. Surfaces whether errors are uniform or concentrated.",
        "Majority-vote BER — soft mean of all per-frame decoded bits, "
        "thresholded to 0/1 and compared to expected. Models a real "
        "video-level decoder that aggregates across frames.",
        "BCH n_errors_corrected (DCT only) — number of chip errors the "
        "BCH decoder fixed per frame (or −1 = uncorrectable).",
    ])

    # ----- 8. Experimental results -----
    add_heading(doc, "8. Experimental results", level=1)
    add_para(
        doc,
        "Test setup: ToS-1920_60s.mp4 (1920×800), 24 frames, payload "
        "\"ABCDEFG\" (0x41424344454647), main pipeline at H.264 CRF 23. "
        "Both methods recover 100% of payload bits on the baseline "
        "(post-codec, no attack) after the fixes documented in §9. "
        "Quality numbers per method:"
    )
    add_table(doc,
        headers=["Metric", "DCT", "TrustMark"],
        rows=[
            ["PSNR mean (dB)",   "56.46", "48.07"],
            ["SSIM mean",        "0.9987", "0.9983"],
            ["Mean |Δpx|",       "0.035",  "0.426"],
            ["Encode wall (s)",  "0.5",    "0.6"],
        ],
        col_widths=[Inches(2.2), Inches(1.5), Inches(1.5)],
    )
    add_para(
        doc,
        "Attack-suite results — frames perfectly recovered out of 24 "
        "(or 12 for the frame-drop preset). Bold numbers indicate the "
        "stronger method for that attack."
    )
    add_table(doc,
        headers=["Attack preset", "DCT (ok/n)", "TrustMark (ok/n)"],
        rows=[
            ["baseline (CRF 23, no attack)", "24/24", "24/24"],
            ["recompress_crf28", "1/24",   "24/24"],
            ["recompress_crf32", "0/24",   "16/24"],
            ["recompress_crf36", "0/24",   "1/24"],
            ["transcode_x265",   "1/24",   "20/24"],
            ["resize_0.75",      "1/24",   "24/24"],
            ["resize_0.5",       "0/24",   "24/24"],
            ["crop_0.8",         "0/24",   "23/24"],
            ["crop_0.6",         "0/24",   "0/24"],
            ["noise_sigma2",     "24/24",  "24/24"],
            ["noise_sigma5",     "24/24",  "24/24"],
            ["noise_sigma10",    "24/24",  "24/24"],
            ["frame_drop_0.5",   "12/12",  "12/12"],
        ],
        col_widths=[Inches(2.2), Inches(1.5), Inches(1.5)],
    )
    add_para(
        doc,
        "Per-frame observation: under recompress_crf28, transcode_x265 "
        "and resize_0.75, the DCT decoder recovers frame 0 only — every "
        "other frame fails with BER ≥ 0.11. Frame 0 is the first decoded "
        "frame and almost certainly an I-frame, which is quantised more "
        "leniently than inter-coded frames. This confirms a key benefit "
        "of recording per-frame traces: aggregate BER alone would have "
        "hidden the I-frame-vs-P-frame robustness gap."
    )

    # ----- Snapshot grid -----
    add_heading(doc, "8.1 Visual reference (frame 12)", level=2)
    add_para(
        doc,
        "Frame 12 from each attacked clip, both methods side by side. The "
        "baseline image is the original un-watermarked frame for "
        "reference. For frame_drop_0.5 the corresponding kept frame "
        "(original index 12 → output index 6) is shown so the visual "
        "content lines up."
    )
    _add_snapshot_table(doc, IMG_DIR)

    # ----- 9. Failure analysis -----
    add_heading(doc, "9. Failure analysis", level=1)
    add_para(
        doc,
        "Two algorithmic failure modes account for the observed errors "
        "after the validated baselines were established."
    )

    add_heading(doc, "9.1 DCT — degenerate input frames", level=2)
    add_para(
        doc,
        "QIM on flat luma is mathematically impossible. The opening "
        "title sequence of the original test clip had Y = 0 for ~195 "
        "frames. Modifying an AC coefficient produces a spatial "
        "perturbation with peak amplitude ≈ δ × 0.25 ≈ 7.5; the "
        "negative half clips at the floor (Y ≥ 0), and the surviving "
        "coefficient comes back at ≈ δ/2 — exactly the QIM decision "
        "boundary. Any downstream perturbation then flips the bit and "
        "BCH cannot recover. Resolution: exclude content-poor frames "
        "(handled at the dataset level for this benchmark; could be "
        "automated with a luma-variance gate in the runner). The same "
        "mechanism makes any flat-luma block — solid backgrounds, "
        "letterbox bars, blown-out highlights — locally un-embeddable; "
        "the impact is bounded as long as the watermark is spread "
        "across enough content-bearing blocks for BCH to correct the "
        "lost chips."
    )

    add_heading(doc, "9.2 Genuine robustness limits", level=2)
    add_para(
        doc,
        "The remaining failures are genuine algorithmic limits, not "
        "implementation artefacts. Heavy crop (0.6×) breaks both "
        "methods: DCT loses its coordinate system as the block grid "
        "no longer aligns with the embedded positions, and TrustMark's "
        "learned receptive field saturates once 40% of the frame is "
        "discarded. Aggressive recompression (CRF 36) erases mid-band "
        "DCT coefficients faster than BCH(127, t=11) can recover, and "
        "is visible even in TrustMark's learned residual (1/24 frames "
        "survive). Codec quantisation is the dominant attacker for DCT "
        "across the moderate-CRF range (28–32); geometric distortion "
        "is the dominant attacker for TrustMark only at the extreme "
        "(crop 0.6×)."
    )

    # ----- 10. Tradeoff discussion -----
    add_heading(doc, "10. Tradeoff discussion", level=1)
    add_table(doc,
        headers=["Dimension", "DCT-QIM", "TrustMark"],
        rows=[
            ["Visual fidelity (PSNR)", "Higher (~56 dB)", "Lower (~48 dB)"],
            ["Visual fidelity (SSIM)", "Equivalent",      "Equivalent"],
            ["Codec robustness (CRF 28)", "Weak",            "Strong"],
            ["Spatial resize robustness", "Brittle",         "Strong (scale-invariant)"],
            ["Crop robustness",       "Brittle (coord loss)", "Robust to mild crop"],
            ["Gaussian noise robustness", "Strong",          "Strong"],
            ["Compute / latency",      "CPU, ms/frame",     "GPU preferred, 100s of ms/frame"],
            ["External dependencies",  "bchlib only",       "PyTorch + ~200 MB weights"],
            ["Determinism",            "Fully deterministic, single secret key", "Stochastic (model precision), single weights file"],
            ["Payload headroom",       "56 / 127 bits used", "56 / 61 bits used"],
        ],
        col_widths=[Inches(2.2), Inches(2.0), Inches(2.0)],
    )
    add_para(
        doc,
        "Headline reading: DCT is a clean choice when invisibility and "
        "deployment simplicity dominate and the only threat is codec "
        "compression at moderate CRF or additive noise. TrustMark is "
        "the clear winner under any pipeline that may resize, crop, or "
        "transcode the video before decode — paid for with model weights "
        "and inference cost. For high-stakes provenance the two are "
        "complementary, not substitutable."
    )

    # ----- 11. Future improvements -----
    add_heading(doc, "11. Future improvements", level=1)
    add_bullets(doc, [
        "Content-aware DCT embedding: skip blocks where the mean luma "
        "is within δ × peak-amplitude of 0 or 255 (where embedding "
        "would clip), or scale δ down on those blocks. Eliminates "
        "the failure mode in §9.2 without dataset curation.",
        "I-frame-aware embedding: PyAV exposes packet flags; raise δ "
        "selectively on I-frames (more aggressively quantised by the "
        "codec) and on cut-adjacent frames, lower it on stable runs.",
        "Hybrid pipeline: run DCT + TrustMark in parallel, decode both, "
        "and treat the two as independent channels (cross-check, "
        "fall back, or weighted vote). Coverage of each method's "
        "robustness gap is essentially disjoint (see §10).",
        "Temporal aggregation analytics: the runner already does "
        "soft majority voting across frames; export the per-frame "
        "confidence trajectory and plot how exact-match rate converges "
        "with number of frames aggregated.",
        "Scale-equivariant DCT embedding: estimate any scale factor "
        "between watermarked and attacked frame (via cross-correlation "
        "of a known marker) and rescale before decode. Would close the "
        "DCT–TrustMark gap on resize attacks.",
        "Adversarial attack additions: oracle attack (gradient-free "
        "search over the decoder), watermark-aware denoising, "
        "averaging attack across multiple watermarked copies.",
        "GPU-batched TrustMark adapter: today the per-frame loop "
        "incurs Python overhead per call; batching N frames at once "
        "would change the throughput characteristic from 'experimental' "
        "to 'production'.",
        "Quality of life: add a luma-variance gate to the runner so "
        "frames that cannot mathematically be watermarked (e.g. solid "
        "colour intros) are skipped automatically and reported in the "
        "summary, rather than silently inflating BER.",
    ])

    # ----- Save -----
    doc.save(str(OUT))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
