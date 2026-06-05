# CLAUDE.md — Video Watermark Benchmark: Handoff for Claude Code

This file gives Claude Code full context to continue development without
re-reading every source file.  Read it top-to-bottom before touching any code.

---

## Project in one sentence

A Python benchmark that watermarks video frames using two independent
methods — classical DCT-QIM and Adobe TrustMark (ML) — then compares
robustness and quality after a real H.264/H.265 codec round-trip.

---

## Repository layout

```
video_watermark/
├── dct/
│   └── watermarker.py          DCT encoder/decoder  (PAYLOAD_BITS=56)
├── trustmark_video/
│   └── adapter.py              TrustMark per-frame wrapper (BCH_5, 56 bits)
├── utils/
│   └── video_io.py             PyAV frame I/O helpers
├── benchmark/
│   ├── metrics.py              PSNR, SSIM, BER, per-bit error rate
│   ├── run.py                  CLI + BenchmarkRunner class
│   └── report.py               Matplotlib plot generator
├── docs/
│   └── architecture.svg        Pipeline diagram (used in README)
├── README.md                   GitHub README with full CLI reference
└── setup.py                    Package metadata
```

---

## Key design constants (must stay in sync across all files)

| Constant | Value | Location |
|---|---|---|
| `PAYLOAD_BITS` | 56 | `dct/watermarker.py`, `trustmark_video/adapter.py` |
| `N_CHIPS` | 127 | `dct/watermarker.py` |
| `_BCH_T` | 11 | `dct/watermarker.py` — BCH error-correction capability |
| `_BCH_M` | 7 | `dct/watermarker.py` — GF(2^7), codeword length = 2^7-1 = 127 |
| `DELTA` | 30.0 | `dct/watermarker.py` — QIM quantisation step |
| TrustMark mode | `BCH_5` | `trustmark_video/adapter.py` — hardcoded, 61-bit capacity |

If you change `PAYLOAD_BITS`, update it in both watermarker files, the
benchmark config defaults, the README table, and the report heatmap shape
(currently `7×8` for 56 bits).

---

## Known bugs to fix

### 1. CLI `--dct-delta` default is wrong  ← fix first

**File:** `benchmark/run.py`, line ~326

```python
# WRONG — CLI default is 10.0 but BenchmarkConfig.dct_delta is 30.0
p.add_argument("--dct-delta", type=float, default=10.0)
```

**Fix:** Change `default=10.0` → `default=30.0` to match the empirically
validated value in `BenchmarkConfig`.  The value 30.0 is what was tested and
proven to survive H.264 CRF 23.

### 2. `bchlib` missing from `setup.py` install_requires

**File:** `setup.py`

`bchlib` is a required dependency (`dct/watermarker.py` imports it at the top
level), but it is not listed in `install_requires`.

**Fix:** Add `"bchlib"` to the `install_requires` list alongside `"av"`, `"numpy"`, etc.

---

## How the DCT watermarker works (implementation detail)

```
encode(frame_rgb, payload_bits[56])
  1. RGB → YCbCr  (ITU-R BT.601 full-range)
  2. Tile luma plane into 8×8 blocks  →  (N_blocks, 8, 8)
  3. bch_encode(payload_bits)  →  chips[127]    ← BCH(127, t=11) codeword
  4. For each chip_i in 0..126:
       block_idx = _select_blocks(rng)[chip_i]   ← keyed PN selection
       coeff (r,c) = MID_BAND[slot % 10]         ← zigzag indices 5-14
       dct_block[r,c] = _qim_write(value, bit)   ← QIM embed
  5. IDCT all blocks → YCbCr → RGB

decode(frame_rgb)
  1–3. Same as encode up through block DCT
  4. Read QIM parity of each coefficient → soft_chips[127]
  5. bch_decode(hard_chips)  →  payload_bits[56], n_errors_corrected
```

The `_rng()` method is seeded deterministically from `sha256(secret_key)[:8]`,
so encode and decode always pick the same blocks.

**`decode_with_stats()`** returns a dict with keys:
- `payload_bits` — `(56,)` uint8, BCH-corrected
- `raw_chips` — `(127,)` uint8, before BCH correction
- `n_errors_corrected` — int (-1 = uncorrectable)

---

## How the TrustMark adapter works

TrustMark's Python API:
```python
tm.encode(pil_image, secret_string)  # -> watermarked PIL Image
tm.decode(pil_image, MODE='binary')  # -> (wm_secret, wm_present, wm_schema)
```

The adapter converts 56 payload bits → a 56-character binary string
`"01101..."` for `tm.encode()`, and reverses it on decode.  The string
fits within TrustMark's BCH_5 capacity (61 bits).

`encode_frame` / `decode_frame` are the primary methods.
`encode` / `decode` are aliases for compatibility with `video_io.py`.

---

## bchlib API (version 2.1.3) — non-obvious behaviour

```python
bch = bchlib.BCH(t=11, m=7)
# bch.n         = 127  (codeword bits)
# bch.ecc_bits  = 70   (parity bits)
# bch.ecc_bytes = 10   (bytes bchlib allocates for ECC)

# Encode: pass exactly 7 bytes of data
data  = bytearray(payload_7bytes)   # 7 bytes = 56 bits
ecc   = bytearray(bch.encode(data)) # returns 10 bytes

# Decode: decode() finds errors, correct() fixes them in-place
nerr = bch.decode(recv_data, recv_ecc)   # nerr=-1 means uncorrectable
if nerr >= 0:
    bch.correct(recv_data, recv_ecc)     # modifies recv_data in-place
```

**Gotcha:** `bch.decode()` only accepts `recv_data` of length ≤ 7 bytes for
this parameter set.  Passing 8 bytes raises `ValueError: invalid parameters`.
Our 56-bit payload packs to exactly 7 bytes, which is why the math works.

---

## video_io.py API summary

```python
read_video_metadata(path)         # -> dict: codec, fps, width, height, ...
read_frames(path, max_frames)     # -> Generator[(int, ndarray)]  (idx, RGB)
write_video(frames, path, fps, codec='libx264', crf=23)

# High-level pipelines:
apply_watermark_to_video(input, output, watermarker, payload_bits,
                          frame_selector, max_frames, codec, crf)
decode_watermark_from_video(input, watermarker, frame_selector, max_frames)

# Frame selectors:
every_frame          # lambda i: True
every_nth(n)         # lambda i: i % n == 0
```

`decode_watermark_from_video` supports both watermarker types: it checks
whether `watermarker.decode()` returns a plain array (DCT) or a tuple of
`(array, bool)` (TrustMark adapter), and handles both.

---

## BenchmarkConfig fields

```python
@dataclass
class BenchmarkConfig:
    payload_int: int = 0xDEADBEEFCAFEBA   # 56-bit payload
    max_frames: int | None = 60
    eval_every_nth: int = 1
    crf: int = 23
    codec: str = "libx264"

    # DCT
    dct_delta: float = 30.0              # ← BUG: CLI default is 10.0, fix it
    dct_num_blocks: int = 1
    dct_secret_key: str = "video_watermark_key"

    # TrustMark
    trustmark_model: str = "Q"
    trustmark_strength: float = 1.0

    run_dct: bool = True
    run_trustmark: bool = True
```

---

## Validated performance (synthetic smooth frames, 480p, H.264 CRF 23)

| Metric | DCT result | TrustMark result |
|---|---|---|
| PSNR | ~49 dB | 43–50 dB (model-dependent) |
| SSIM | ~0.995 | depends on model |
| BER after codec | 0.000 | untested (needs GPU env) |
| Exact match rate | 100 % (20 frames) | untested |
| BCH errors/frame | 0–10 corrected | N/A (internal) |

TrustMark has not been end-to-end tested in this repo because the model
weights require a download and GPU environment.  The adapter code is
complete; it just needs live testing.

---

## Suggested next tasks (priority order)

### Must-fix before any real testing
1. **Fix `--dct-delta` CLI default** (bug #1 above, one-liner)
2. **Add `bchlib` to `setup.py`** (bug #2 above, one-liner)

### Testing
3. **Run against a real MP4** — the full pipeline has only been tested on
   synthetic gradient frames.  Real video (natural scene content, motion,
   grain) will expose any remaining DCT robustness issues.
4. **TrustMark end-to-end test** — install `trustmark`, run with
   `--no-dct --trustmark-model Q` on a real clip, verify the adapter
   round-trips correctly.
5. **H.265 / high-CRF stress test** — `--codec libx265 --crf 32 --dct-delta 40`
   to find the DCT robustness floor under stronger compression.

### Enhancements
6. **Robustness attack suite** — add a `benchmark/attacks.py` module with
   common video processing attacks to test watermark survival:
   - Re-encode at different CRF values (18, 23, 28, 32, 36)
   - Spatial rescale (720p → 480p → 720p)
   - Gaussian blur (σ = 0.5, 1.0, 2.0)
   - Frame-rate conversion (drop every other frame)
   - Colour jitter (±10 brightness/contrast)
   Run each attack, then decode, and report BER per attack type.

7. **Temporal aggregation analysis** — `decode_watermark_from_video` already
   does majority voting across frames but doesn't report per-frame confidence
   variance.  Add a plot showing how confidence converges as more frames are
   aggregated (useful for showing how many frames you need for reliable decode).

8. **Scene-cut awareness** — currently every frame is watermarked
   independently.  For real video, scene cuts cause the codec to use I-frames,
   which are more aggressively quantised.  Detecting I-frame positions (via
   PyAV's `pkt_flags`) and adjusting `delta` for those frames could improve
   robustness on cut-heavy content.

9. **Payload capacity experiment** — the benchmark is hard-wired to 56 bits
   but the ECC parameters are easy to change.  A sweep over payload sizes
   (40, 56, 61, 68, 75 bits) vs BER would be a useful addition to the paper.

---

## Dependencies summary

```
av            PyAV FFmpeg bindings  (frame decode/encode)
numpy         array operations
scipy         DCT (scipy.fft.dctn / idctn)
scikit-image  PSNR, SSIM
Pillow        PIL Image for TrustMark
tqdm          progress bars
matplotlib    plots
bchlib==2.1.3 BCH ECC  ← MISSING FROM setup.py, add it
trustmark     optional ML model (pip install trustmark)
```

Install command for a fresh environment:
```bash
pip install av numpy scipy scikit-image Pillow tqdm matplotlib bchlib
pip install trustmark   # optional
pip install -e .
```

---

## Running the benchmark

```bash
# DCT only (no trustmark needed)
python -m video_watermark.benchmark.run \
    --input clip.mp4 \
    --no-trustmark \
    --outdir results/

# Both methods
python -m video_watermark.benchmark.run \
    --input clip.mp4 \
    --outdir results/

# Generate plots
python -m video_watermark.benchmark.report results/benchmark_results.json
```
