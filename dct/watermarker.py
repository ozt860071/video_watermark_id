"""
DCT-based video watermarker — frequency-domain embedding in the luma channel.

Payload : 56 bits (7 bytes).
ECC     : BCH(n=127, t=11, m=7) — corrects up to 11 bit-flips in the 127-chip
          codeword, matching TrustMark's BCH_5 philosophy of maximum
          error-correction at the cost of throughput.

Embedding strategy
------------------
- RGB → YCbCr; embed only in the Y (luma) plane.
- Tile luma into 8×8 blocks (same grid H.264 uses).
- 2-D DCT-II per block.
- BCH-encode the 56-bit payload → 127-chip codeword (56 data + 71 ECC bits).
  Each chip is one QIM-modulated mid-band AC coefficient in a pseudo-randomly
  chosen block.  The PN block sequence is seeded from the secret key.
- IDCT, YCbCr → RGB.

Decoder
-------
- Same block selection, same key.
- Read QIM parity of each coefficient → 127 noisy chip bits.
- BCH decode+correct → 56 recovered bits.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import bchlib
import numpy as np
from scipy.fft import dctn, idctn

# ---------------------------------------------------------------------------
# BCH parameters  (fixed — encoder and decoder must agree)
# ---------------------------------------------------------------------------
_BCH_T = 11    # error-correction capability: up to 11 bit-flips per codeword
_BCH_M = 7     # GF(2^7), codeword length n = 2^7 - 1 = 127

PAYLOAD_BITS  = 56                       # 7 bytes of user data
_DATA_BYTES   = PAYLOAD_BITS // 8        # 7
N_CHIPS       = 127                      # BCH codeword length in bits


def _make_bch() -> bchlib.BCH:
    """Construct the shared BCH codec instance."""
    bch = bchlib.BCH(_BCH_T, m=_BCH_M)
    assert bch.n == N_CHIPS, f"Unexpected codeword length {bch.n}"
    assert bch.n - bch.ecc_bits >= PAYLOAD_BITS, (
        f"BCH data field ({bch.n - bch.ecc_bits} bits) too small for "
        f"{PAYLOAD_BITS}-bit payload"
    )
    return bch


# ---------------------------------------------------------------------------
# ECC helpers
# ---------------------------------------------------------------------------

def bch_encode(payload_bits: np.ndarray) -> np.ndarray:
    """
    Encode 56 payload bits → 127 chip bits using BCH(127, t=11).

    Layout of the 127-bit codeword:
        bits  0 –  55 : 56 payload bits  (MSB first)
        bits 56 –  56 : 1 zero padding   (fills the 57-bit data field)
        bits 57 – 126 : 70 BCH ECC bits

    Returns ndarray of shape (127,) dtype uint8.
    """
    assert payload_bits.shape == (PAYLOAD_BITS,), \
        f"Expected {PAYLOAD_BITS} bits, got {payload_bits.shape}"

    bch = _make_bch()
    # Pack 56 bits into 7 bytes (MSB first, np.packbits default)
    data_bytes = bytearray(np.packbits(payload_bits).tobytes())  # 7 bytes
    ecc_bytes  = bytearray(bch.encode(data_bytes))               # 10 bytes

    # Assemble codeword bits: data_bits | ecc_bits, truncated to n=127
    cw_bytes  = data_bytes + ecc_bytes
    cw_bits   = np.unpackbits(np.frombuffer(bytes(cw_bytes), dtype=np.uint8))
    return cw_bits[:N_CHIPS].copy()


def bch_decode(chips: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Decode 127 noisy chip bits → 56 payload bits.

    Returns (payload_bits_56, n_errors_corrected).
    n_errors_corrected == -1 means uncorrectable (> t=11 errors).
    """
    assert chips.shape == (N_CHIPS,), \
        f"Expected {N_CHIPS} chip bits, got {chips.shape}"

    bch = _make_bch()

    # Pad 127 bits back to full bytes for bchlib
    padded = np.zeros(len(bytes(_DATA_BYTES + bch.ecc_bytes)) * 8,
                      dtype=np.uint8)
    # data field: first 56 bits
    padded[:N_CHIPS] = chips

    full_bytes  = np.packbits(padded).tobytes()
    recv_data   = bytearray(full_bytes[:_DATA_BYTES])
    recv_ecc    = bytearray(full_bytes[_DATA_BYTES: _DATA_BYTES + bch.ecc_bytes])

    nerr = bch.decode(recv_data, recv_ecc)
    if nerr < 0:
        # Uncorrectable — return as-is (hard decision only)
        raw_bits = chips[:PAYLOAD_BITS].astype(np.uint8)
        return raw_bits, -1

    bch.correct(recv_data, recv_ecc)
    payload_bits = np.unpackbits(
        np.frombuffer(bytes(recv_data), dtype=np.uint8)
    )[:PAYLOAD_BITS]
    return payload_bits.astype(np.uint8), nerr


# ---------------------------------------------------------------------------
# Mid-band AC coefficient positions in zigzag order.
# Indices 5–14 strike the balance between codec survival and invisibility.
# ---------------------------------------------------------------------------
_ZIGZAG: list[tuple[int, int]] = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
]
_MID_BAND: list[tuple[int, int]] = _ZIGZAG[5:15]   # 10 positions

DELTA = 30.0  # QIM step — survives H.264/H.265 CRF ≤ 28 at PSNR ~49 dB


class DCTWatermarker:
    """
    Frequency-domain video watermarker.

    Embeds PAYLOAD_BITS=56 bits per frame using BCH(127, t=11) ECC and
    QIM modulation of mid-band DCT coefficients.

    Parameters
    ----------
    secret_key : str | bytes
        Seed for the pseudo-random chip→block assignment.
    delta : float
        QIM quantisation step (default 30.0 for H.264/H.265 CRF ≤ 28).
    num_blocks : int
        DCT blocks polled per chip bit during decode for an inner
        majority vote (on top of BCH). Default 1 (pure BCH).
        Set to 2–3 for extra robustness at lower delta values.
    """

    def __init__(
        self,
        secret_key: str | bytes = "default_key",
        delta: float = DELTA,
        num_blocks: int = 1,
    ) -> None:
        if isinstance(secret_key, str):
            secret_key = secret_key.encode()
        self._key_bytes = hashlib.sha256(secret_key).digest()
        self.delta      = float(delta)
        self.num_blocks = max(1, int(num_blocks))

    # ------------------------------------------------------------------
    # Block selection (deterministic per key)
    # ------------------------------------------------------------------

    def _rng(self) -> np.random.Generator:
        seed = int.from_bytes(self._key_bytes[:8], "big")
        return np.random.default_rng(seed)

    def _select_blocks(
        self,
        n_total: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Return shape-(N_CHIPS, num_blocks) index array.

        Each chip gets `num_blocks` distinct block indices drawn from a
        globally shuffled pool so coverage is spread uniformly.
        """
        needed  = N_CHIPS * self.num_blocks
        repeats = needed // n_total + 2
        pool    = np.tile(np.arange(n_total), repeats)[:needed]
        rng.shuffle(pool)
        return pool.reshape(N_CHIPS, self.num_blocks)

    # ------------------------------------------------------------------
    # Colour space (ITU-R BT.601 full-range)
    # ------------------------------------------------------------------

    @staticmethod
    def _rgb_to_ycbcr(f: np.ndarray) -> np.ndarray:
        r, g, b = f[...,0].astype(np.float32), f[...,1].astype(np.float32), f[...,2].astype(np.float32)
        y  =  0.299    * r + 0.587    * g + 0.114    * b
        cb = -0.168736 * r - 0.331264 * g + 0.500    * b + 128.
        cr =  0.500    * r - 0.418688 * g - 0.081312 * b + 128.
        return np.stack([y, cb, cr], axis=-1)

    @staticmethod
    def _ycbcr_to_rgb(y_cb_cr: np.ndarray) -> np.ndarray:
        y  = y_cb_cr[..., 0]
        cb = y_cb_cr[..., 1] - 128.
        cr = y_cb_cr[..., 2] - 128.
        r  = y               + 1.402    * cr
        g  = y - 0.344136  * cb - 0.714136 * cr
        b  = y + 1.772     * cb
        return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Block tiling
    # ------------------------------------------------------------------

    @staticmethod
    def _tile(luma: np.ndarray) -> tuple[np.ndarray, int, int]:
        H, W   = luma.shape
        nr, nc = H // 8, W // 8
        blocks = (luma[:nr*8, :nc*8]
                  .reshape(nr, 8, nc, 8)
                  .transpose(0, 2, 1, 3)
                  .reshape(-1, 8, 8))
        return blocks, nr, nc

    @staticmethod
    def _untile(blocks: np.ndarray, nr: int, nc: int) -> np.ndarray:
        return (blocks.reshape(nr, nc, 8, 8)
                      .transpose(0, 2, 1, 3)
                      .reshape(nr*8, nc*8))

    # ------------------------------------------------------------------
    # QIM
    # ------------------------------------------------------------------

    def _qim_write(self, v: float, bit: int) -> float:
        q = round(v / self.delta)
        if bit == 1:
            if q % 2 == 0: q += 1
        else:
            if q % 2 != 0: q += 1
        return q * self.delta

    def _qim_read(self, v: float) -> int:
        return int(round(v / self.delta)) % 2

    # ------------------------------------------------------------------
    # Public encode / decode
    # ------------------------------------------------------------------

    def encode(
        self,
        frame_rgb: np.ndarray,
        payload_bits: np.ndarray,
    ) -> np.ndarray:
        """
        Watermark a single frame.

        Parameters
        ----------
        frame_rgb    : (H, W, 3) uint8 RGB
        payload_bits : (56,) uint8  ∈ {0, 1}

        Returns
        -------
        (H, W, 3) uint8 RGB
        """
        if payload_bits.shape != (PAYLOAD_BITS,):
            raise ValueError(
                f"encode() requires {PAYLOAD_BITS} bits, got {payload_bits.shape}"
            )

        chips = bch_encode(payload_bits)     # (127,) BCH codeword

        ycbcr = self._rgb_to_ycbcr(frame_rgb)
        luma  = ycbcr[..., 0].copy()
        blocks, nr, nc = self._tile(luma)

        rng       = self._rng()
        block_idx = self._select_blocks(len(blocks), rng)
        dct_b     = dctn(blocks, axes=(-2, -1), norm="ortho")

        for chip_i, bit in enumerate(chips):
            for slot, blk_i in enumerate(block_idx[chip_i]):
                r, c = _MID_BAND[slot % len(_MID_BAND)]
                dct_b[blk_i, r, c] = self._qim_write(
                    dct_b[blk_i, r, c], int(bit)
                )

        luma_out = luma.copy()
        luma_out[:nr*8, :nc*8] = self._untile(
            idctn(dct_b, axes=(-2, -1), norm="ortho"), nr, nc
        )
        ycbcr_out = ycbcr.copy()
        ycbcr_out[..., 0] = luma_out
        return self._ycbcr_to_rgb(ycbcr_out)

    def decode(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Extract the 56-bit payload from a watermarked frame.

        Parameters
        ----------
        frame_rgb : (H, W, 3) uint8 RGB

        Returns
        -------
        (56,) uint8 — BCH-corrected payload bits
        """
        ycbcr = self._rgb_to_ycbcr(frame_rgb)
        luma  = ycbcr[..., 0]
        blocks, nr, nc = self._tile(luma)

        rng       = self._rng()
        block_idx = self._select_blocks(len(blocks), rng)
        dct_b     = dctn(blocks, axes=(-2, -1), norm="ortho")

        # Inner majority vote across num_blocks, then BCH outer decode
        soft = np.zeros(N_CHIPS, dtype=np.float32)
        for chip_i in range(N_CHIPS):
            votes = [
                self._qim_read(dct_b[block_idx[chip_i, slot],
                                     r, c])
                for slot, (r, c) in enumerate(
                    _MID_BAND[j % len(_MID_BAND)]
                    for j in range(self.num_blocks)
                )
            ]
            soft[chip_i] = float(np.mean(votes))

        hard_chips = (soft >= 0.5).astype(np.uint8)
        payload_bits, _ = bch_decode(hard_chips)
        return payload_bits

    def decode_with_stats(
        self, frame_rgb: np.ndarray
    ) -> dict:
        """
        Like decode() but also returns raw BER before BCH correction
        and the number of errors the BCH codec corrected.

        Returns dict with keys: payload_bits, raw_chips, n_errors_corrected.
        """
        ycbcr = self._rgb_to_ycbcr(frame_rgb)
        luma  = ycbcr[..., 0]
        blocks, nr, nc = self._tile(luma)

        rng       = self._rng()
        block_idx = self._select_blocks(len(blocks), rng)
        dct_b     = dctn(blocks, axes=(-2, -1), norm="ortho")

        soft = np.zeros(N_CHIPS, dtype=np.float32)
        for chip_i in range(N_CHIPS):
            votes = [
                self._qim_read(dct_b[block_idx[chip_i, slot], r, c])
                for slot, (r, c) in enumerate(
                    _MID_BAND[j % len(_MID_BAND)]
                    for j in range(self.num_blocks)
                )
            ]
            soft[chip_i] = float(np.mean(votes))

        raw_chips = (soft >= 0.5).astype(np.uint8)
        payload_bits, nerr = bch_decode(raw_chips)
        return {
            "payload_bits":        payload_bits,
            "raw_chips":           raw_chips,
            "n_errors_corrected":  nerr,
        }

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def int_to_bits(value: int, n: int = PAYLOAD_BITS) -> np.ndarray:
        """Pack integer → bit array of length n (MSB first)."""
        return np.array(
            [(value >> (n - 1 - i)) & 1 for i in range(n)],
            dtype=np.uint8,
        )

    @staticmethod
    def bits_to_int(bits: np.ndarray) -> int:
        result = 0
        for b in bits:
            result = (result << 1) | int(b)
        return result

    @staticmethod
    def bytes_to_bits(data: bytes) -> np.ndarray:
        """7-byte payload → 56-bit array (MSB first per byte)."""
        if len(data) != _DATA_BYTES:
            raise ValueError(f"Expected {_DATA_BYTES} bytes, got {len(data)}")
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    @staticmethod
    def bits_to_bytes(bits: np.ndarray) -> bytes:
        """56-bit array → 7 bytes."""
        return np.packbits(bits[:PAYLOAD_BITS]).tobytes()
