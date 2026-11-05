"""sentinel weight-diff: synthetic safetensors/GGUF models, no downloads, seconds to run."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from modelsentinel import weightdiff_quant as wq  # noqa: E402
from modelsentinel.cli import main  # noqa: E402
from modelsentinel.db import connect, get_scan  # noqa: E402
from modelsentinel.report import render  # noqa: E402
from modelsentinel.weightdiff import Options, weight_diff  # noqa: E402
from modelsentinel.weightdiff_io import PickleRefused, hf_to_gguf_name, open_model, read_rows  # noqa: E402

HIDDEN, FFN, VOCAB, LAYERS = 64, 128, 96, 6


# ------------------------------------------------------------------ writers

def write_safetensors(path: Path, tensors: dict, dtype: str = "F32") -> Path:
    header, blobs, offset = {}, [], 0
    for name, array in tensors.items():
        arr = np.asarray(array, dtype=np.float32)
        if dtype == "F16":
            raw = arr.astype("<f2").tobytes()
        elif dtype == "BF16":
            raw = (arr.view(np.uint32) >> 16).astype("<u2").tobytes()
        else:
            raw = arr.astype("<f4").tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    head = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))
    return path


def quant_q8_0(x: np.ndarray) -> bytes:
    blocks = x.astype(np.float32).reshape(-1, 32)
    d = np.abs(blocks).max(axis=1) / 127.0
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d > 0)
    q = np.round(blocks * inv[:, None]).astype(np.int8)
    return b"".join(np.float16(di).tobytes() + qi.tobytes() for di, qi in zip(d, q))


def quant_q4_0(x: np.ndarray) -> bytes:
    """ggml quantize_row_q4_0_ref."""
    out = []
    for block in x.astype(np.float32).reshape(-1, 32):
        peak = block[np.argmax(np.abs(block))]
        d = peak / -8.0
        inv = 1.0 / d if d else 0.0
        q = np.minimum(15, np.trunc(block * inv + 8.5)).astype(np.uint8)
        out.append(np.float16(d).tobytes() + (q[:16] | (q[16:] << 4)).astype(np.uint8).tobytes())
    return b"".join(out)


def encode(array: np.ndarray, kind: str) -> tuple[bytes, int]:
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    if kind == "F32":
        return flat.astype("<f4").tobytes(), 0
    if kind == "F16":
        return flat.astype("<f2").tobytes(), 1
    if kind == "Q8_0":
        return quant_q8_0(flat), 8
    if kind == "Q4_0":
        return quant_q4_0(flat), 2
    raise ValueError(kind)


def write_gguf(path: Path, tensors: dict, quant: str = "F32", kvs: dict | None = None,
               raw_tensors: list | None = None) -> Path:
    """Tiny GGUF v3 writer. 2-D tensors use ``quant``; 1-D tensors stay F32 (as llama.cpp does)."""
    kvs = {"general.architecture": "qwen2", "general.alignment": 32, **(kvs or {})}
    entries = []
    for name, array in tensors.items():
        kind = quant if np.ndim(array) >= 2 else "F32"
        data, type_id = encode(array, kind)
        entries.append((name, tuple(reversed(np.shape(array))), type_id, data))
    for name, dims, type_id, data in raw_tensors or []:
        entries.append((name, dims, type_id, data))
    out = b"GGUF" + struct.pack("<IQQ", 3, len(entries), len(kvs))
    for key, value in kvs.items():
        kb = key.encode()
        out += struct.pack("<Q", len(kb)) + kb
        if isinstance(value, str):
            vb = value.encode()
            out += struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", value)
    offset, datas = 0, []
    for name, dims, type_id, data in entries:
        nb = name.encode()
        out += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims))
        out += struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", type_id, offset)
        pad = (-len(data)) % 32
        datas.append(data + b"\x00" * pad)
        offset += len(data) + pad
    out += b"\x00" * ((-len(out)) % 32)
    path.write_bytes(out + b"".join(datas))
    return path


# ------------------------------------------------------------------- models

def base_gguf_names(rng: np.random.Generator) -> dict:
    tensors = {"token_embd.weight": rng.normal(0, 0.02, (VOCAB, HIDDEN))}
    for i in range(LAYERS):
        tensors[f"blk.{i}.attn_norm.weight"] = np.ones(HIDDEN) + rng.normal(0, 0.01, HIDDEN)
        tensors[f"blk.{i}.attn_q.weight"] = rng.normal(0, 0.05, (HIDDEN, HIDDEN))
        tensors[f"blk.{i}.attn_v.weight"] = rng.normal(0, 0.05, (HIDDEN, HIDDEN))
        tensors[f"blk.{i}.ffn_up.weight"] = rng.normal(0, 0.05, (FFN, HIDDEN))
    tensors["output_norm.weight"] = np.ones(HIDDEN)
    return {k: np.asarray(v, dtype=np.float32) for k, v in tensors.items()}


HF_NAMES = {"attn_q": "self_attn.q_proj", "attn_v": "self_attn.v_proj", "ffn_up": "mlp.up_proj",
            "attn_norm": "input_layernorm"}


def to_hf(tensors: dict) -> dict:
    out = {}
    for name, value in tensors.items():
        if name == "token_embd.weight":
            out["model.embed_tokens.weight"] = value
        elif name == "output_norm.weight":
            out["model.norm.weight"] = value
        else:
            _blk, layer, kind, suffix = name.split(".")
            out[f"model.layers.{layer}.{HF_NAMES[kind]}.{suffix}"] = value
    return out


def with_noise(tensors: dict, rng: np.random.Generator, rel: float) -> dict:
    return {k: (v + rng.normal(0, 1, v.shape) * rel * np.sqrt(np.mean(v ** 2))).astype(np.float32)
            for k, v in tensors.items()}


def rank2(rows: int, cols: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    u, v = rng.normal(0, 1, (rows, 2)), rng.normal(0, 1, (2, cols))
    return (scale * (u @ v) / np.sqrt(rows * cols)).astype(np.float32)


def run(target: Path, base: Path, **kwargs) -> dict:
    return weight_diff(str(target), str(base), persist=False, **kwargs)


def by_category(record: dict, category: str) -> list[dict]:
    return [f for f in record["findings"] if f["category"] == category]


def severities(record: dict) -> set[str]:
    return {f["severity"] for f in record["findings"]}


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


# -------------------------------------------------------- dequantization

def _f16(raw: bytes, at: int) -> float:
    return float(np.frombuffer(raw[at:at + 2], dtype="<f2")[0])


def ref_q4_0(b: bytes) -> list[float]:
    d, qs = _f16(b, 0), b[2:18]
    y = [0.0] * 32
    for j in range(16):
        y[j] = ((qs[j] & 0x0F) - 8) * d
        y[j + 16] = ((qs[j] >> 4) - 8) * d
    return y


def ref_q4_1(b: bytes) -> list[float]:
    d, m, qs = _f16(b, 0), _f16(b, 2), b[4:20]
    y = [0.0] * 32
    for j in range(16):
        y[j] = (qs[j] & 0x0F) * d + m
        y[j + 16] = (qs[j] >> 4) * d + m
    return y


def ref_q5(b: bytes, with_min: bool) -> list[float]:
    d = _f16(b, 0)
    m = _f16(b, 2) if with_min else 0.0
    start = 4 if with_min else 2
    qh = struct.unpack("<I", b[start:start + 4])[0]
    qs = b[start + 4:start + 20]
    y = [0.0] * 32
    for j in range(16):
        xh0 = ((qh >> j) << 4) & 0x10
        xh1 = (qh >> (j + 12)) & 0x10
        x0 = ((qs[j] & 0x0F) | xh0) - (0 if with_min else 16)
        x1 = ((qs[j] >> 4) | xh1) - (0 if with_min else 16)
        y[j] = x0 * d + m
        y[j + 16] = x1 * d + m
    return y


def ref_q8_0(b: bytes) -> list[float]:
    d = _f16(b, 0)
    return [q * d for q in np.frombuffer(b[2:34], dtype=np.int8).tolist()]


def ref_q8_k(b: bytes) -> list[float]:
    d = float(np.frombuffer(b[0:4], dtype="<f4")[0])
    return [q * d for q in np.frombuffer(b[4:260], dtype=np.int8).tolist()]


def _scale_min_k4(j: int, q: bytes) -> tuple[int, int]:
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def ref_q4_k(b: bytes) -> list[float]:
    d, dmin, scales, q = _f16(b, 0), _f16(b, 2), b[4:16], b[16:144]
    y, is_, qi = [], 0, 0
    for _j in range(0, 256, 64):
        sc, m = _scale_min_k4(is_, scales)
        d1, m1 = d * sc, dmin * m
        sc, m = _scale_min_k4(is_ + 1, scales)
        d2, m2 = d * sc, dmin * m
        y += [d1 * (q[qi + l] & 0x0F) - m1 for l in range(32)]
        y += [d2 * (q[qi + l] >> 4) - m2 for l in range(32)]
        qi += 32
        is_ += 2
    return y


def ref_q5_k(b: bytes) -> list[float]:
    d, dmin, scales, qh, ql = _f16(b, 0), _f16(b, 2), b[4:16], b[16:48], b[48:176]
    y, is_, qi, u1, u2 = [], 0, 0, 1, 2
    for _j in range(0, 256, 64):
        sc, m = _scale_min_k4(is_, scales)
        d1, m1 = d * sc, dmin * m
        sc, m = _scale_min_k4(is_ + 1, scales)
        d2, m2 = d * sc, dmin * m
        y += [d1 * ((ql[qi + l] & 0x0F) + (16 if qh[l] & u1 else 0)) - m1 for l in range(32)]
        y += [d2 * ((ql[qi + l] >> 4) + (16 if qh[l] & u2 else 0)) - m2 for l in range(32)]
        qi += 32
        is_ += 2
        u1 <<= 2
        u2 <<= 2
    return y


def ref_q6_k(b: bytes) -> list[float]:
    ql, qh = b[0:128], b[128:192]
    sc = np.frombuffer(b[192:208], dtype=np.int8).tolist()
    d = _f16(b, 208)
    y = [0.0] * 256
    for n in range(2):
        lo, ho, so, yo = 64 * n, 32 * n, 8 * n, 128 * n
        for l in range(32):
            is_ = l // 16
            q1 = ((ql[lo + l] & 0xF) | (((qh[ho + l] >> 0) & 3) << 4)) - 32
            q2 = ((ql[lo + l + 32] & 0xF) | (((qh[ho + l] >> 2) & 3) << 4)) - 32
            q3 = ((ql[lo + l] >> 4) | (((qh[ho + l] >> 4) & 3) << 4)) - 32
            q4 = ((ql[lo + l + 32] >> 4) | (((qh[ho + l] >> 6) & 3) << 4)) - 32
            y[yo + l] = d * sc[so + is_] * q1
            y[yo + l + 32] = d * sc[so + is_ + 2] * q2
            y[yo + l + 64] = d * sc[so + is_ + 4] * q3
            y[yo + l + 96] = d * sc[so + is_ + 6] * q4
    return y


REFERENCES = {
    "Q4_0": (18, [0], ref_q4_0), "Q4_1": (20, [0, 2], ref_q4_1),
    "Q5_0": (22, [0], lambda b: ref_q5(b, False)), "Q5_1": (24, [0, 2], lambda b: ref_q5(b, True)),
    "Q8_0": (34, [0], ref_q8_0), "Q4_K": (144, [0, 2], ref_q4_k), "Q5_K": (176, [0, 2], ref_q5_k),
    "Q6_K": (210, [208], ref_q6_k), "Q8_K": (292, [], ref_q8_k),
}


@pytest.mark.parametrize("kind", sorted(REFERENCES))
def test_vectorized_dequant_matches_scalar_ggml_port(kind):
    size, f16_fields, ref = REFERENCES[kind]
    gen = np.random.default_rng(7)
    blocks = []
    for _ in range(5):
        raw = bytearray(gen.integers(0, 256, size, dtype=np.uint8).tobytes())
        for at in f16_fields:
            raw[at:at + 2] = np.float16(gen.uniform(-0.5, 0.5)).tobytes()
        if kind == "Q8_K":
            raw[0:4] = np.float32(gen.uniform(-0.5, 0.5)).tobytes()
        blocks.append(bytes(raw))
    got = wq.DEQUANT[kind](np.frombuffer(b"".join(blocks), dtype=np.uint8))
    want = np.array([v for block in blocks for v in ref(block)], dtype=np.float32)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


def test_hand_computed_blocks():
    # Q8_0: d = 0.5, q = -16..15 -> q / 2
    q = np.arange(-16, 16, dtype=np.int8)
    block = np.float16(0.5).tobytes() + q.tobytes()
    np.testing.assert_array_equal(wq.dequant_q8_0(np.frombuffer(block, np.uint8)), q.astype(np.float32) * 0.5)
    # Q4_0: d = 2, low nibble j, high nibble 15 - j -> x[j] = (j - 8) * 2, x[j + 16] = (7 - j) * 2
    qs = bytes((j & 0xF) | ((15 - j) << 4) for j in range(16))
    got = wq.dequant_q4_0(np.frombuffer(np.float16(2.0).tobytes() + qs, np.uint8))
    want = [(j - 8) * 2.0 for j in range(16)] + [(7 - j) * 2.0 for j in range(16)]
    assert got.tolist() == want
    # Q4_1: d = 1, m = -3, all nibbles 5 -> 2
    got = wq.dequant_q4_1(np.frombuffer(np.float16(1).tobytes() + np.float16(-3).tobytes() + b"\x55" * 16, np.uint8))
    assert got.tolist() == [2.0] * 32
    # Q5_0: d = 1, qh bit 0 and bit 16 set, nibbles 0 -> x[0] = 16 - 16 = 0, x[16] = 0, the rest -16
    got = wq.dequant_q5_0(np.frombuffer(np.float16(1).tobytes() + struct.pack("<I", 1 | (1 << 16)) + b"\x00" * 16,
                                        np.uint8))
    assert got[0] == 0 and got[16] == 0 and got[1] == -16 and got[31] == -16
    # BF16: 1.0 is 0x3F80
    assert wq.bf16_to_f32(np.array([0x3F80, 0xC000], dtype=np.uint16)).tolist() == [1.0, -2.0]


def test_reference_quantizers_round_trip(rng):
    x = rng.normal(0, 1, 32 * 64).astype(np.float32)
    for kind, tol in (("Q8_0", 0.01), ("Q4_0", 0.2)):
        data, _ = encode(x, kind)
        y = wq.decode(kind, np.frombuffer(data, np.uint8))
        assert np.linalg.norm(y - x) / np.linalg.norm(x) < tol


# ---------------------------------------------------------- end to end

def test_identical_safetensors_no_warning(tmp_path, rng):
    base = to_hf(base_gguf_names(rng))
    b = write_safetensors(tmp_path / "base.safetensors", base)
    t = write_safetensors(tmp_path / "target.safetensors", base)
    record = run(t, b)
    assert record["verdict"] == "PASS"
    assert by_category(record, "weight_identical")
    assert severities(record) <= {"info"}
    assert not re.search(r"\bsafe\b", record["summary"].lower())
    assert "no evidence of tampering" in record["summary"]


def test_small_full_rank_noise_is_not_high(tmp_path, rng):
    base = to_hf(base_gguf_names(rng))
    b = write_safetensors(tmp_path / "base.safetensors", base)
    t = write_safetensors(tmp_path / "target.safetensors", with_noise(base, rng, 1e-2))
    record = run(t, b)
    assert "high" not in severities(record) and "critical" not in severities(record)
    assert not by_category(record, "weight_low_rank_update")
    assert not by_category(record, "weight_outlier_layer")
    assert by_category(record, "weight_uniform_diff")
    assert record["verdict"] == "PASS"


def test_rank2_update_flags_that_tensor_and_layer(tmp_path, rng):
    base = to_hf(base_gguf_names(rng))
    target = with_noise(base, rng, 1e-4)
    name = "model.layers.3.self_attn.q_proj.weight"
    target[name] = target[name] + rank2(HIDDEN, HIDDEN, rng, 0.5)
    b = write_safetensors(tmp_path / "base.safetensors", base)
    t = write_safetensors(tmp_path / "target.safetensors", target)
    record = run(t, b)
    low_rank = by_category(record, "weight_low_rank_update")
    assert [f["location"] for f in low_rank] == [name]
    assert low_rank[0]["severity"] == "high"
    outliers = by_category(record, "weight_outlier_layer")
    assert [f["location"] for f in outliers] == ["layer 3"]
    assert record["verdict"] == "WARN"
    payload = record["weightdiff"]
    flagged = [f for f in payload["flagged"] if f["low_rank"]]
    assert flagged[0]["name"] == name and len(flagged[0]["svd"]["spectrum"]) == 16
    assert flagged[0]["svd"]["spectrum"][0] == 1.0 and flagged[0]["svd"]["spectrum"][2] < 0.05
    assert payload["top_tensors"][0]["name"] == name
    assert [row["layer"] for row in payload["layers"]] == list(range(LAYERS))


def test_rank2_update_in_gguf(tmp_path, rng):
    base = base_gguf_names(rng)
    target = dict(base)
    target["blk.1.ffn_up.weight"] = base["blk.1.ffn_up.weight"] + rank2(FFN, HIDDEN, rng, 0.3)
    b = write_gguf(tmp_path / "base.gguf", base, "F16")
    t = write_gguf(tmp_path / "target.gguf", target, "F16")
    record = run(t, b)
    assert [f["location"] for f in by_category(record, "weight_low_rank_update")] == ["blk.1.ffn_up.weight"]
    assert [f["location"] for f in by_category(record, "weight_outlier_layer")] == ["layer 1"]


def test_shape_change_and_extra_tensor_are_high(tmp_path, rng):
    base = base_gguf_names(rng)
    target = dict(base)
    target["token_embd.weight"] = np.vstack([base["token_embd.weight"], rng.normal(0, 0.02, (4, HIDDEN))]).astype(np.float32)
    target["blk.2.attn_backdoor.weight"] = rng.normal(0, 0.05, (HIDDEN, HIDDEN)).astype(np.float32)
    record = run(write_gguf(tmp_path / "t.gguf", target), write_gguf(tmp_path / "b.gguf", base))
    shape = by_category(record, "weight_shape_mismatch")
    assert shape and shape[0]["severity"] == "high" and shape[0]["location"] == "token_embd.weight"
    added = by_category(record, "weight_tensor_added")
    assert added and added[0]["severity"] == "high" and "blk.2.attn_backdoor.weight" in added[0]["evidence"]


@pytest.mark.parametrize("quant", ["Q8_0", "Q4_0", "F16"])
def test_quant_mismatch_no_false_alarm(tmp_path, rng, quant):
    base = base_gguf_names(rng)
    record = run(write_gguf(tmp_path / "t.gguf", base, quant), write_gguf(tmp_path / "b.gguf", base, "F32"))
    assert record["weightdiff"]["quant_mismatch"]
    assert by_category(record, "weight_quant_mismatch")
    assert not severities(record) & {"critical", "high", "medium"}
    assert record["verdict"] == "PASS"


def test_quantized_target_with_injected_edit_still_flagged(tmp_path, rng):
    base = base_gguf_names(rng)
    target = dict(base)
    target["blk.4.attn_v.weight"] = base["blk.4.attn_v.weight"] + rank2(HIDDEN, HIDDEN, rng, 1.0)
    record = run(write_gguf(tmp_path / "t.gguf", target, "Q8_0"), write_gguf(tmp_path / "b.gguf", base, "F32"))
    assert [f["location"] for f in by_category(record, "weight_low_rank_update")] == ["blk.4.attn_v.weight"]
    assert "critical" not in severities(record)


def test_cross_format_hf_vs_gguf_identical(tmp_path, rng):
    gguf_names = base_gguf_names(rng)
    b = write_safetensors(tmp_path / "base.safetensors", to_hf(gguf_names))
    t = write_gguf(tmp_path / "target.gguf", gguf_names)
    record = run(t, b)
    assert record["weightdiff"]["cross_format"]
    assert by_category(record, "weight_identical")
    assert record["verdict"] == "PASS"
    assert hf_to_gguf_name("model.layers.7.mlp.down_proj.weight") == "blk.7.ffn_down.weight"
    assert hf_to_gguf_name("lm_head.weight") == "output.weight"


def test_cross_format_llama_qk_permutation_undone(tmp_path, rng):
    gguf_names = base_gguf_names(rng)
    heads = 4

    def permute(w):
        return w.reshape(heads, 2, w.shape[0] // heads // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)

    converted = {k: (permute(v) if ".attn_q." in k else v) for k, v in gguf_names.items()}
    b = write_safetensors(tmp_path / "base.safetensors", to_hf(gguf_names))
    t = write_gguf(tmp_path / "target.gguf", converted,
                   kvs={"general.architecture": "llama", "llama.attention.head_count": heads,
                        "llama.attention.head_count_kv": heads})
    record = run(t, b)
    assert by_category(record, "weight_identical"), record["findings"]


def test_bf16_and_f16_safetensors_read(tmp_path, rng):
    x = rng.normal(0, 1, (8, 32)).astype(np.float32)
    for dtype, tol in (("F16", 1e-3), ("BF16", 1e-2)):
        tset = open_model(write_safetensors(tmp_path / f"{dtype}.safetensors", {"w": x}, dtype))
        y = read_rows(tset.tensors["w"], 0, 8)
        assert np.abs(y - x).max() / np.abs(x).max() < tol


def test_pickle_inputs_refused(tmp_path, rng, capsys):
    bin_file = tmp_path / "pytorch_model.bin"
    bin_file.write_bytes(b"\x80\x02}q\x00.")
    ok = write_safetensors(tmp_path / "ok.safetensors", {"w": np.zeros((2, 2), np.float32)})
    with pytest.raises(PickleRefused, match="sentinel scan"):
        run(bin_file, ok)
    only_pickle = tmp_path / "hf"
    only_pickle.mkdir()
    (only_pickle / "pytorch_model.bin").write_bytes(b"\x80\x02}q\x00.")
    with pytest.raises(PickleRefused):
        run(only_pickle, ok)
    assert main(["--db", str(tmp_path / "x.db"), "weight-diff", str(bin_file), "--base", str(ok)]) == 2
    assert "never loads pickle" in capsys.readouterr().err


def test_sharded_hf_directory(tmp_path, rng):
    hf = to_hf(base_gguf_names(rng))
    names = list(hf)
    directory = tmp_path / "model"
    directory.mkdir()
    write_safetensors(directory / "model-00001-of-00002.safetensors", {n: hf[n] for n in names[:10]})
    write_safetensors(directory / "model-00002-of-00002.safetensors", {n: hf[n] for n in names[10:]})
    weight_map = {n: ("model-00001-of-00002.safetensors" if i < 10 else "model-00002-of-00002.safetensors")
                  for i, n in enumerate(names)}
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    single = write_safetensors(tmp_path / "base.safetensors", hf)
    record = run(directory, single)
    assert record["weightdiff"]["compared"] == len(names)
    assert by_category(record, "weight_identical")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ollama_store(root: Path, tag: str, gguf: Path, digest: str | None = None) -> None:
    digest = digest or _sha(gguf)
    blobs = root / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    (blobs / f"sha256-{digest}").write_bytes(gguf.read_bytes())
    manifest = root / "manifests" / "registry.ollama.ai" / "library" / tag
    manifest.mkdir(parents=True, exist_ok=True)
    (manifest / "latest").write_text(json.dumps({"layers": [
        {"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{digest}"}]}), encoding="utf-8")


def test_ollama_blob_digest_verified(tmp_path, rng, monkeypatch):
    base = base_gguf_names(rng)
    gguf = write_gguf(tmp_path / "src.gguf", base, "Q8_0")
    root = tmp_path / "ollama"
    _ollama_store(root, "tiny", gguf)
    tampered = dict(base)
    tampered["blk.0.attn_q.weight"] = base["blk.0.attn_q.weight"] * 1.01
    bad = write_gguf(tmp_path / "bad.gguf", tampered, "Q8_0")
    evil_root = tmp_path / "ollama-evil"
    _ollama_store(evil_root, "evil", bad, digest=_sha(gguf))  # blob name claims the clean digest
    monkeypatch.setenv("OLLAMA_MODELS", str(root))
    b = write_gguf(tmp_path / "base.gguf", base, "F32")
    clean = weight_diff("ollama:tiny", str(b), persist=False)
    checks = clean["weightdiff"]["hash_checks"]
    assert checks and all(c["ok"] for c in checks) and checks[0]["source"] == "ollama content address"
    assert not by_category(clean, "hash_mismatch")
    monkeypatch.setenv("OLLAMA_MODELS", str(evil_root))
    evil = weight_diff("ollama:evil", str(b), persist=False)
    mismatch = by_category(evil, "hash_mismatch")
    assert mismatch and mismatch[0]["severity"] == "high"


def test_manifest_and_sidecar_hashes(tmp_path, rng):
    base = to_hf(base_gguf_names(rng))
    model = tmp_path / "model"
    model.mkdir()
    t = write_safetensors(model / "model.safetensors", base)
    b = write_safetensors(tmp_path / "base.safetensors", base)
    good = tmp_path / "good.yaml"
    good.write_text(f"model.safetensors: {_sha(t)}\n", encoding="utf-8")
    record = run(model, b, manifest=str(good))
    assert not by_category(record, "hash_mismatch")
    assert any(c["ok"] and c["source"].startswith("manifest") for c in record["weightdiff"]["hash_checks"])
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"files": [{"path": "model.safetensors", "sha256": "0" * 64}]}), encoding="utf-8")
    record = run(model, b, manifest=str(bad))
    assert by_category(record, "hash_mismatch")[0]["severity"] == "high"
    (model / "model.safetensors.sha256").write_text("f" * 64 + "  model.safetensors\n", encoding="utf-8")
    record = run(model, b)
    assert by_category(record, "hash_mismatch")


def test_time_budget_and_svd_caps(tmp_path, rng):
    base = {"w": rng.normal(0, 0.05, (200, 96)).astype(np.float32)}
    target = {"w": base["w"] + rank2(200, 96, rng, 0.5)}
    b = write_safetensors(tmp_path / "b.safetensors", base)
    t = write_safetensors(tmp_path / "t.safetensors", target)
    record = run(t, b, options=Options(max_seconds=0))
    assert record["weightdiff"]["compared"] == 0 and record["weightdiff"]["skipped"]["budget"] == ["w"]
    assert by_category(record, "weight_diff_incomplete")[0]["severity"] == "medium"
    capped = run(t, b, options=Options(svd_rows=40, svd_cols=48))
    svd = capped["weightdiff"]["top_tensors"][0]["svd"]
    assert svd["sampled_shape"] == [40, 48]
    assert capped["weightdiff"]["top_tensors"][0]["low_rank"]


def test_unsupported_quant_is_skipped_not_fatal(tmp_path, rng):
    base = base_gguf_names(rng)
    raw = [("blk.0.odd.weight", (256, 2), 10, b"\x00" * 84 * 2)]  # 2 rows of one Q2_K block each
    t = write_gguf(tmp_path / "t.gguf", base, raw_tensors=raw)
    b = write_gguf(tmp_path / "b.gguf", base, raw_tensors=raw)
    record = run(t, b)
    coverage = [f for f in by_category(record, "weight_diff_coverage") if f["code"] == "unsupported_quantization"]
    if wq.gguf_package_dequant(10) is None:
        assert coverage and "Q2_K" in coverage[0]["message"]
    assert record["verdict"] == "PASS"


def test_cli_persists_and_renders(tmp_path, rng):
    base = base_gguf_names(rng)
    target = dict(base)
    target["blk.2.attn_q.weight"] = base["blk.2.attn_q.weight"] + rank2(HIDDEN, HIDDEN, rng, 0.5)
    b = write_gguf(tmp_path / "b.gguf", base)
    t = write_gguf(tmp_path / "t.gguf", target)
    db = tmp_path / "sentinel.db"
    out = tmp_path / "wd.json"
    code = main(["--db", str(db), "weight-diff", str(t), "--base", str(b), "--format", "json", "-o", str(out),
                 "--top", "5", "--ci"])
    assert code == 0  # WARN does not block CI by default
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["kind"] == "weightdiff" and len(record["weightdiff"]["top_tensors"]) == 5
    conn = connect(db)
    try:
        stored = get_scan(conn, record["id"])
    finally:
        conn.close()
    assert stored is not None
    assert "<html" in render(record, "html").lower()
    sarif = json.loads(render(record, "sarif"))
    assert any(r["ruleId"] == "low_rank_update" for r in sarif["runs"][0]["results"])
    assert main(["--db", str(db), "weight-diff", str(t), "--base", str(b), "--no-save"]) == 0


def test_without_numpy_prints_install_hint(monkeypatch, capsys):
    import sys

    monkeypatch.setitem(sys.modules, "numpy", None)
    assert main(["weight-diff", "a.gguf", "--base", "b.gguf"]) == 2
    assert "modelsentinel[advanced]" in capsys.readouterr().err
