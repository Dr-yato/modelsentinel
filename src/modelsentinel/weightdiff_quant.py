"""Vectorized numpy dequantization for the GGML block formats weight-diff reads.

Each function takes the raw bytes of whole blocks as a 1-D ``uint8`` array and
returns ``float32`` values. The layouts follow ggml's reference
``dequantize_row_*`` functions (ggml-quants.c); tests/test_weightdiff.py checks
each one against a literal scalar port of those loops and against hand-computed
blocks. Formats without a port here (Q2_K, Q3_K, the IQ family, ...) go through
the optional ``gguf`` package when it is importable, and are otherwise skipped
with an "unsupported quantization" note.

numpy is imported lazily by callers: this module is only imported once numpy is
known to be available.
"""

from __future__ import annotations

import numpy as np

# type id -> (name, elements per block, bytes per block)
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
}
TYPE_IDS = {name: type_id for type_id, (name, _e, _b) in GGML_TYPES.items()}


def _f16(blocks: np.ndarray, start: int) -> np.ndarray:
    return np.ascontiguousarray(blocks[:, start:start + 2]).view("<f2").astype(np.float32)


def _blocks(raw: np.ndarray, size: int) -> np.ndarray:
    if raw.size % size:
        raise ValueError(f"{raw.size} bytes is not a whole number of {size}-byte blocks")
    return raw.reshape(-1, size)


def bf16_to_f32(raw_u16: np.ndarray) -> np.ndarray:
    return (raw_u16.astype(np.uint32) << 16).view(np.float32)


def dequant_q4_0(raw: np.ndarray) -> np.ndarray:
    b = _blocks(raw, 18)
    qs = b[:, 2:]
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.int16) - 8
    return (q * _f16(b, 0)).astype(np.float32).reshape(-1)


def dequant_q4_1(raw: np.ndarray) -> np.ndarray:
    b = _blocks(raw, 20)
    qs = b[:, 4:]
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.float32)
    return (q * _f16(b, 0) + _f16(b, 2)).reshape(-1)


def _q5_high_bits(qh_bytes: np.ndarray) -> np.ndarray:
    qh = np.ascontiguousarray(qh_bytes).view("<u4")  # (n, 1)
    return ((qh >> np.arange(32, dtype=np.uint32)) & 1).astype(np.uint8)  # (n, 32)


def dequant_q5_0(raw: np.ndarray) -> np.ndarray:
    b = _blocks(raw, 22)
    bits = _q5_high_bits(b[:, 2:6])
    qs = b[:, 6:]
    lo = (qs & 0x0F) | (bits[:, :16] << 4)
    hi = (qs >> 4) | (bits[:, 16:] << 4)
    q = np.concatenate([lo, hi], axis=1).astype(np.int16) - 16
    return (q * _f16(b, 0)).astype(np.float32).reshape(-1)


def dequant_q5_1(raw: np.ndarray) -> np.ndarray:
    b = _blocks(raw, 24)
    bits = _q5_high_bits(b[:, 4:8])
    qs = b[:, 8:]
    lo = (qs & 0x0F) | (bits[:, :16] << 4)
    hi = (qs >> 4) | (bits[:, 16:] << 4)
    q = np.concatenate([lo, hi], axis=1).astype(np.float32)
    return (q * _f16(b, 0) + _f16(b, 2)).reshape(-1)


def dequant_q8_0(raw: np.ndarray) -> np.ndarray:
    b = _blocks(raw, 34)
    q = np.ascontiguousarray(b[:, 2:]).view(np.int8).astype(np.float32)
    return (q * _f16(b, 0)).reshape(-1)


def dequant_q8_k(raw: np.ndarray) -> np.ndarray:
    b = _blocks(raw, 292)
    d = np.ascontiguousarray(b[:, 0:4]).view("<f4")
    q = np.ascontiguousarray(b[:, 4:260]).view(np.int8).astype(np.float32)
    return (q * d).reshape(-1)


def _k4_scales(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ggml get_scale_min_k4 for j = 0..7, vectorized over blocks."""
    sc = scales.astype(np.uint8)
    d = np.empty((sc.shape[0], 8), dtype=np.float32)
    m = np.empty((sc.shape[0], 8), dtype=np.float32)
    d[:, :4] = sc[:, 0:4] & 63
    m[:, :4] = sc[:, 4:8] & 63
    d[:, 4:] = (sc[:, 8:12] & 0x0F) | ((sc[:, 0:4] >> 6) << 4)
    m[:, 4:] = (sc[:, 8:12] >> 4) | ((sc[:, 4:8] >> 6) << 4)
    return d, m


def _k_nibbles(qs: np.ndarray) -> np.ndarray:
    """128 bytes -> (n, 8, 32): sub-block 2c is the low nibbles of chunk c, 2c+1 the high."""
    q = qs.reshape(qs.shape[0], 4, 32)
    return np.stack([q & 0x0F, q >> 4], axis=2).reshape(qs.shape[0], 8, 32)


def dequant_q4_k(raw: np.ndarray) -> np.ndarray:
    b = _blocks(raw, 144)
    d, dmin = _f16(b, 0), _f16(b, 2)
    sc, mn = _k4_scales(b[:, 4:16])
    vals = _k_nibbles(b[:, 16:144]).astype(np.float32)
    y = (d * sc)[:, :, None] * vals - (dmin * mn)[:, :, None]
    return y.astype(np.float32).reshape(-1)


def dequant_q5_k(raw: np.ndarray) -> np.ndarray:
    b = _blocks(raw, 176)
    d, dmin = _f16(b, 0), _f16(b, 2)
    sc, mn = _k4_scales(b[:, 4:16])
    qh = b[:, 16:48]
    high = (qh[:, None, :] >> np.arange(8, dtype=np.uint8)[None, :, None]) & 1  # (n, 8, 32)
    vals = (_k_nibbles(b[:, 48:176]) + (high << 4)).astype(np.float32)
    y = (d * sc)[:, :, None] * vals - (dmin * mn)[:, :, None]
    return y.astype(np.float32).reshape(-1)


def dequant_q6_k(raw: np.ndarray) -> np.ndarray:
    b = _blocks(raw, 210)
    n = b.shape[0]
    ql = b[:, :128].reshape(n, 2, 64)
    qh = b[:, 128:192].reshape(n, 2, 32)
    sc = np.ascontiguousarray(b[:, 192:208]).view(np.int8).astype(np.float32)  # (n, 16)
    d = _f16(b, 208)
    q1 = (ql[:, :, :32] & 0x0F) | ((qh & 3) << 4)
    q2 = (ql[:, :, 32:] & 0x0F) | (((qh >> 2) & 3) << 4)
    q3 = (ql[:, :, :32] >> 4) | (((qh >> 4) & 3) << 4)
    q4 = (ql[:, :, 32:] >> 4) | (((qh >> 6) & 3) << 4)
    q = np.stack([q1, q2, q3, q4], axis=2).reshape(n, 256).astype(np.int16) - 32
    return (d * np.repeat(sc, 16, axis=1) * q).astype(np.float32).reshape(-1)


DEQUANT = {
    "Q4_0": dequant_q4_0, "Q4_1": dequant_q4_1, "Q5_0": dequant_q5_0, "Q5_1": dequant_q5_1,
    "Q8_0": dequant_q8_0, "Q4_K": dequant_q4_k, "Q5_K": dequant_q5_k, "Q6_K": dequant_q6_k,
    "Q8_K": dequant_q8_k,
}
PLAIN = {"F32": "<f4", "F16": "<f2", "F64": "<f8", "I8": "i1", "I16": "<i2", "I32": "<i4"}


def native_support(name: str) -> bool:
    return name in DEQUANT or name in PLAIN or name == "BF16"


def gguf_package_dequant(type_id: int):
    """Return a raw-bytes -> float32 function backed by the optional gguf package, or None."""
    try:
        from gguf.constants import GGMLQuantizationType  # type: ignore
        from gguf.quants import dequantize  # type: ignore
    except Exception:  # noqa: BLE001 - any import problem means "not available"
        return None
    try:
        qtype = GGMLQuantizationType(type_id)
    except ValueError:
        return None

    def run(raw: np.ndarray) -> np.ndarray:
        return np.asarray(dequantize(raw, qtype), dtype=np.float32).reshape(-1)

    return run


def decode(name: str, raw: np.ndarray, type_id: int | None = None) -> np.ndarray:
    """Decode raw bytes of type ``name`` to a flat float32 array."""
    if name in PLAIN:
        return raw.view(PLAIN[name]).astype(np.float32)
    if name == "BF16":
        return bf16_to_f32(raw.view("<u2"))
    if name in DEQUANT:
        return DEQUANT[name](raw)
    fallback = gguf_package_dequant(type_id) if type_id is not None else None
    if fallback is None:
        raise NotImplementedError(f"unsupported quantization {name}")
    return fallback(raw)


# Rough relative Frobenius error of each storage type against the float it came
# from. Used only to recognise "this is quantization noise", never as a verdict.
QUANT_NOISE = {
    "F64": 0.0, "F32": 0.0, "F16": 1e-3, "BF16": 5e-3, "Q8_0": 1.5e-2, "Q8_K": 1.5e-2, "Q8_1": 1.5e-2,
    "I8": 2e-2, "U8": 2e-2, "Q6_K": 3e-2, "Q5_K": 5e-2, "Q5_0": 6e-2, "Q5_1": 6e-2,
    "Q4_K": 1e-1, "Q4_0": 1.3e-1, "Q4_1": 1.2e-1, "IQ4_NL": 1.2e-1, "IQ4_XS": 1.2e-1,
}
DEFAULT_LOW_BIT_NOISE = 0.3  # Q3/Q2/IQ2/IQ1 and anything unknown


def expected_noise(*names: str) -> float:
    return max(QUANT_NOISE.get(name, DEFAULT_LOW_BIT_NOISE) for name in names)
