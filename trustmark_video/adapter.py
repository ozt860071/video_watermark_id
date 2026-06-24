"""
TrustMark video adapter — 56-bit payload, BCH_5 encoding (best robustness).

TrustMark's BCH_5 mode:
  - 100-bit internal codeword
  - 61 protected payload bits  (+35 BCH ECC bits, corrects 5 bit-flips)
  - We use 56 of those 61 bits, leaving 5 spare for future use.

This gives the best robustness TrustMark offers while carrying a full
56-bit (7-byte) payload, matching the DCT watermarker's BCH(127, t=11)
configuration.

Payload wire format
-------------------
TrustMark.encode() accepts a plain string secret.  We pass a 56-character
binary string ("01101..." etc.) which is stored as 56 UTF-8 bytes in
TrustMark's 61-bit data field.  Decoding reverses this: we read the
first 56 characters and convert back to a bit array.

Interface
---------
    from trustmark_video.adapter import TrustMarkVideoWatermarker
    wm = TrustMarkVideoWatermarker()
    wm_frame = wm.encode_frame(frame_rgb, payload_bits_56)
    bits, ok  = wm.decode_frame(wm_frame)
"""

from __future__ import annotations

import numpy as np
from PIL import Image

try:
    from trustmark import TrustMark
    _TRUSTMARK_AVAILABLE = True
except ImportError:
    _TRUSTMARK_AVAILABLE = False
    TrustMark = None  # type: ignore

_INSTALL_MSG = (
    "TrustMark is not installed.\n"
    "    pip install trustmark\n"
    "Model weights (~200 MB) download automatically on first use."
)

PAYLOAD_BITS = 56   # must match dct/watermarker.py


class TrustMarkVideoWatermarker:
    """
    Per-frame TrustMark watermarker carrying a 56-bit payload.

    Uses BCH_5 encoding (61-bit capacity, 5-flip ECC) — the most robust
    TrustMark mode that fits 56 bits.

    Parameters
    ----------
    model_type : str
        'Q' (default — best robustness/quality balance)
        'P' (highest visual quality, similar robustness)
        'B' or 'C' (smaller/faster, slightly less robust)
    strength : float
        Watermark strength multiplier passed to encode().
        1.0 = default.  Raise to 1.5 for print/severe-distortion
        robustness; lower to 0.8 for minimal visibility.
    verbose : bool
        Print TrustMark model-loading messages.
    """

    # Official capacity table from Adobe docs (bits of user payload)
    # Mode       payload  ECC_bits  max_bit_flips_corrected
    # BCH_SUPER    40       56             8   ← NOT enough for 56 bits
    # BCH_5        61       35             5   ← best fit for 56 bits ✓
    # BCH_4        68       28             4
    # BCH_3        75       21             3
    _ENCODING = "BCH_5"
    _CAPACITY  = 61   # bits available under BCH_5

    def __init__(
        self,
        model_type: str = "Q",
        strength: float = 1.0,
        verbose: bool = False,
        device: str | None = None,
    ) -> None:
        if not _TRUSTMARK_AVAILABLE:
            raise ImportError(_INSTALL_MSG)
        if PAYLOAD_BITS > self._CAPACITY:
            raise AssertionError(
                f"PAYLOAD_BITS={PAYLOAD_BITS} exceeds BCH_5 capacity "
                f"{self._CAPACITY}"
            )

        self.model_type = model_type
        self.strength   = strength

        # TrustMark's own auto-detect is broken (self.device = device
        # immediately overwrites the MPS/CUDA detection). Pick a device
        # explicitly: caller override > MPS on Apple Silicon > CPU.
        if device is None:
            try:
                import torch
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"
        self.device = device

        enc_enum = getattr(TrustMark.Encoding, self._ENCODING)
        self._tm = TrustMark(
            verbose=verbose,
            model_type=model_type,
            encoding_type=enc_enum,
            device=device,
        )

    # ------------------------------------------------------------------
    # Payload ↔ TrustMark secret string
    # ------------------------------------------------------------------

    @staticmethod
    def _bits_to_secret(bits56: np.ndarray) -> str:
        """
        Convert 56-bit array to a 56-character binary string.

        TrustMark stores the string as UTF-8 in its 61-bit data field.
        '0'/'1' characters are single-byte ASCII — no encoding surprises.
        """
        assert bits56.shape == (PAYLOAD_BITS,)
        return "".join(str(int(b)) for b in bits56)

    @staticmethod
    def _secret_to_bits(secret: str) -> tuple[np.ndarray, bool]:
        """
        Parse TrustMark's decoded string back to a 56-bit array.

        Returns (bits56, parse_ok).  parse_ok is False if the string
        was shorter than expected or contained non-binary characters.
        """
        bits = np.zeros(PAYLOAD_BITS, dtype=np.uint8)
        clean = secret.strip()
        if len(clean) < PAYLOAD_BITS:
            return bits, False
        for i in range(PAYLOAD_BITS):
            ch = clean[i]
            if ch == "1":
                bits[i] = 1
            elif ch != "0":
                return bits, False   # unexpected character
        return bits, True

    # ------------------------------------------------------------------
    # Per-frame encode / decode
    # ------------------------------------------------------------------

    def encode_frame(
        self,
        frame_rgb: np.ndarray,
        payload_bits: np.ndarray,
    ) -> np.ndarray:
        """
        Embed a 56-bit payload into a single RGB frame via TrustMark.

        Parameters
        ----------
        frame_rgb    : (H, W, 3) uint8 RGB
        payload_bits : (56,) uint8

        Returns
        -------
        (H, W, 3) uint8 RGB — watermarked frame
        """
        if payload_bits.shape != (PAYLOAD_BITS,):
            raise ValueError(
                f"encode_frame() requires {PAYLOAD_BITS} bits, "
                f"got {payload_bits.shape}"
            )
        secret  = self._bits_to_secret(payload_bits)
        pil_in  = Image.fromarray(frame_rgb, mode="RGB")
        # MODE='binary' on both encode and decode — otherwise encode defaults
        # to MODE='text' which packs the secret as ASCII text in the available
        # bits, while decode(MODE='binary') reads them as raw bits, giving a
        # silent layout mismatch (~23/56 systematic errors).
        pil_out = self._tm.encode(
            pil_in, secret, MODE="binary", WM_STRENGTH=self.strength
        )
        return np.array(pil_out, dtype=np.uint8)

    def decode_frame(
        self,
        frame_rgb: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """
        Extract the 56-bit payload from a watermarked frame.

        Parameters
        ----------
        frame_rgb : (H, W, 3) uint8 RGB

        Returns
        -------
        (bits56, wm_detected)
            bits56      : (56,) uint8
            wm_detected : bool — TrustMark's own detection confidence flag
        """
        pil_in = Image.fromarray(frame_rgb, mode="RGB")
        wm_secret, wm_present, _ = self._tm.decode(pil_in, MODE="binary")

        if not wm_present or not wm_secret:
            return np.zeros(PAYLOAD_BITS, dtype=np.uint8), False

        bits, parse_ok = self._secret_to_bits(wm_secret)
        return bits, (wm_present and parse_ok)

    # ------------------------------------------------------------------
    # Utility helpers (same interface as DCTWatermarker)
    # ------------------------------------------------------------------

    @staticmethod
    def int_to_bits(value: int, n: int = PAYLOAD_BITS) -> np.ndarray:
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
        if len(data) != PAYLOAD_BITS // 8:
            raise ValueError(
                f"Expected {PAYLOAD_BITS // 8} bytes, got {len(data)}"
            )
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    @staticmethod
    def bits_to_bytes(bits: np.ndarray) -> bytes:
        return np.packbits(bits[:PAYLOAD_BITS]).tobytes()

    # For API compatibility with video_io.py which calls wm.decode() on DCT frames:
    def encode(self, frame_rgb: np.ndarray, payload_bits: np.ndarray) -> np.ndarray:
        return self.encode_frame(frame_rgb, payload_bits)

    def decode(self, frame_rgb: np.ndarray):
        return self.decode_frame(frame_rgb)
