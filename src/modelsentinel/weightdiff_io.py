"""Tensor readers for weight-diff: GGUF and safetensors, without torch or pickle.

A model is opened into a ``TensorSet``: a name -> ``TensorRef`` map plus notes.
Nothing is decoded at open time; ``read_rows`` decodes one row range of one
tensor at a time, so memory stays bounded by the chunk size, not the model.
Every tensor is viewed as 2-D (rows = product of the leading dims, cols = last
dim); a 1-D tensor is one row.

Pickle-based weights (.bin/.pt/.pth/.ckpt/.pkl) are refused: decoding them
means running the pickle VM. ``sentinel scan`` is the tool for those files.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from modelsentinel.backdoor import GGUFError, read_header
from modelsentinel.weightdiff_quant import GGML_TYPES, decode, gguf_package_dequant, native_support

PICKLE_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"
ST_DTYPES = {
    "F64": ("<f8", 8), "F32": ("<f4", 4), "F16": ("<f2", 2), "BF16": ("<u2", 2),
    "I8": ("i1", 1), "U8": ("u1", 1), "I16": ("<i2", 2), "I32": ("<i4", 4),
}
ST_UNSUPPORTED = {"BOOL", "I64", "U16", "U32", "U64", "F8_E4M3", "F8_E5M2", "F8_E8M0", "F4"}


class PickleRefused(ValueError):
    """Raised for weight files that can only be decoded by running pickle."""


@dataclass(frozen=True)
class TensorRef:
    name: str
    shape: tuple[int, ...]
    dtype: str
    path: str
    offset: int  # absolute byte offset of the first element
    nbytes: int
    fmt: str  # "gguf" | "safetensors"
    type_id: int | None = None
    supported: bool = True

    @property
    def rows(self) -> int:
        if len(self.shape) <= 1:
            return 1
        out = 1
        for dim in self.shape[:-1]:
            out *= dim
        return out

    @property
    def cols(self) -> int:
        return self.shape[-1] if self.shape else 1

    @property
    def elements(self) -> int:
        return self.rows * self.cols

    @property
    def row_bytes(self) -> int:
        return self.nbytes // max(1, self.rows)


@dataclass
class TensorSet:
    fmt: str
    tensors: dict[str, TensorRef] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    weight_files: list[str] = field(default_factory=list)


def refuse_pickle(path: Path) -> None:
    raise PickleRefused(
        f"{path} is a pickle-based weight file. weight-diff never loads pickle weights because decoding them "
        f"executes code. Run `sentinel scan {path}` for the static pickle scan, or compare a .safetensors/.gguf copy."
    )


def _check_lfs_pointer(path: Path) -> None:
    if path.stat().st_size < 1024:
        with path.open("rb") as handle:
            if handle.read(len(LFS_POINTER)) == LFS_POINTER:
                raise ValueError(f"{path} is a Git LFS pointer, not the weights. Run `git lfs pull` first.")


# ---------------------------------------------------------------- safetensors

def open_safetensors(path: Path, tensors: dict[str, TensorRef], notes: list[str]) -> dict:
    _check_lfs_pointer(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        head = handle.read(8)
        if len(head) != 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        header_len = struct.unpack("<Q", head)[0]
        if header_len > MAX_SAFETENSORS_HEADER or 8 + header_len > size:
            raise ValueError(f"{path} has an implausible safetensors header length {header_len}")
        try:
            header = json.loads(handle.read(header_len).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path} has a malformed safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{path} safetensors header is not an object")
    base = 8 + header_len
    metadata = header.get("__metadata__") if isinstance(header.get("__metadata__"), dict) else {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        tensors[name] = _st_ref(path, name, info, base, size, notes)
    return metadata


def _st_ref(path: Path, name: str, info: object, base: int, size: int, notes: list[str]) -> TensorRef:
    if not isinstance(info, dict):
        raise ValueError(f"{path}: tensor {name} has a malformed header entry")
    dtype = str(info.get("dtype", ""))
    shape = tuple(int(d) for d in info.get("shape") or [])
    offsets = info.get("data_offsets") or [0, 0]
    begin, end = int(offsets[0]), int(offsets[1])
    if any(d < 0 for d in shape) or begin < 0 or end < begin or base + end > size:
        raise ValueError(f"{path}: tensor {name} points outside the file")
    supported = dtype in ST_DTYPES
    if supported:
        count = 1
        for dim in shape:
            count *= dim
        if count * ST_DTYPES[dtype][1] != end - begin:
            raise ValueError(f"{path}: tensor {name} byte length does not match its shape")
    else:
        notes.append(f"unsupported dtype {dtype} for {name}")
    return TensorRef(name, shape, dtype, str(path), base + begin, end - begin, "safetensors", None, supported)


def _safetensors_files(directory: Path, notes: list[str]) -> list[Path]:
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text(encoding="utf-8"))
        names = sorted({str(v) for v in (data.get("weight_map") or {}).values()})
        files = []
        root = directory.resolve()
        for name in names:
            candidate = (directory / name).resolve()
            if root not in candidate.parents and candidate != root:
                raise ValueError(f"index entry {name} points outside {directory}")
            if not candidate.is_file():
                raise FileNotFoundError(f"shard {name} listed in {index.name} is missing")
            files.append(directory / name)
        notes.append(f"sharded safetensors: {len(files)} shard(s) from {index.name}")
        return files
    return sorted(directory.glob("*.safetensors"))


# ----------------------------------------------------------------------- GGUF

def _parse_tensor_infos(blob: bytes, count: int) -> list[tuple[str, tuple[int, ...], int, int]]:
    pos, out = 0, []
    for _ in range(count):
        (name_len,) = struct.unpack_from("<Q", blob, pos)
        pos += 8
        name = blob[pos:pos + name_len].decode("utf-8", "replace")
        pos += name_len
        (n_dims,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        dims = struct.unpack_from(f"<{n_dims}Q", blob, pos)
        pos += 8 * n_dims
        type_id, offset = struct.unpack_from("<IQ", blob, pos)
        pos += 12
        out.append((name, tuple(int(d) for d in dims), int(type_id), int(offset)))
    return out


def open_gguf(path: Path, tensors: dict[str, TensorRef], notes: list[str]) -> dict:
    _check_lfs_pointer(path)
    try:
        _version, count, kvs, info, data_offset = read_header(path)
    except GGUFError as exc:
        raise ValueError(str(exc)) from exc
    size = path.stat().st_size
    metadata = {kv.key: kv.value for kv in kvs if kv.value is not None and not kv.key.startswith("tokenizer.")}
    for name, dims, type_id, offset in _parse_tensor_infos(info, count):
        type_name, block_elems, block_bytes = GGML_TYPES.get(type_id, (f"type{type_id}", 0, 0))
        shape = tuple(reversed(dims))  # ggml ne[0] is the fastest-varying dim
        elements = 1
        for dim in dims:
            elements *= dim
        cols = dims[0] if dims else 1
        known = block_elems > 0 and cols % block_elems == 0
        nbytes = elements // block_elems * block_bytes if known else 0
        supported = known and (native_support(type_name) or gguf_package_dequant(type_id) is not None)
        if known and data_offset + offset + nbytes > size:
            raise ValueError(f"{path}: tensor {name} points outside the file")
        if not supported:
            notes.append(f"unsupported quantization {type_name} for {name}")
        tensors[name] = TensorRef(name, shape, type_name, str(path), data_offset + offset, nbytes, "gguf",
                                  type_id, supported)
    return metadata


# ------------------------------------------------------------------- dispatch

def open_model(path: Path) -> TensorSet:
    """Open a .gguf file, a .safetensors file, or a directory of safetensors shards."""
    path = Path(path)
    if path.is_file():
        return _open_file(path)
    if not path.is_dir():
        raise FileNotFoundError(f"model source not found: {path}")
    notes: list[str] = []
    shards = _safetensors_files(path, notes)
    if shards:
        pickles = sorted(p.name for p in path.iterdir() if p.suffix.lower() in PICKLE_SUFFIXES)
        if pickles:
            notes.append(f"ignored pickle weight files (never loaded): {', '.join(pickles[:5])}")
        result = TensorSet("safetensors", notes=notes)
        for shard in shards:
            meta = open_safetensors(shard, result.tensors, result.notes)
            result.metadata.update(meta)
            result.weight_files.append(str(shard))
        result.metadata.update(_hf_config(path))
        return result
    ggufs = sorted(path.glob("*.gguf"))
    if len(ggufs) == 1:
        return _open_file(ggufs[0])
    if len(ggufs) > 1:
        raise ValueError(f"{path} holds several .gguf files; pass the one to compare")
    pickles = sorted(p for p in path.iterdir() if p.suffix.lower() in PICKLE_SUFFIXES)
    if pickles:
        refuse_pickle(pickles[0])
    raise ValueError(f"{path} has no .safetensors or .gguf weights")


def _open_file(path: Path) -> TensorSet:
    suffix = path.suffix.lower()
    if suffix in PICKLE_SUFFIXES:
        refuse_pickle(path)
    with path.open("rb") as handle:
        magic = handle.read(4)
    result = TensorSet("gguf" if magic == b"GGUF" else "safetensors", weight_files=[str(path)])
    if magic == b"GGUF":
        result.metadata = open_gguf(path, result.tensors, result.notes)
    elif suffix == ".safetensors":
        result.metadata = open_safetensors(path, result.tensors, result.notes)
        result.metadata.update(_hf_config(path.parent))
    elif magic[:2] == b"\x80\x02" or magic[:1] == b"\x80" or magic[:2] == b"PK":
        refuse_pickle(path)
    else:
        raise ValueError(f"{path} is neither GGUF nor safetensors")
    return result


def _hf_config(directory: Path) -> dict:
    config = directory / "config.json"
    if not config.is_file():
        return {}
    try:
        data = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    keep = ("model_type", "num_attention_heads", "num_key_value_heads", "architectures")
    return {f"hf.{key}": data[key] for key in keep if key in data}


# ------------------------------------------------------------------- decoding

def read_rows(ref: TensorRef, start: int, stop: int) -> np.ndarray:
    """Decode rows [start, stop) of the 2-D view of ``ref`` as float32."""
    count = stop - start
    if count <= 0:
        return np.zeros((0, ref.cols), dtype=np.float32)
    row_bytes = ref.row_bytes
    with open(ref.path, "rb") as handle:
        handle.seek(ref.offset + start * row_bytes)
        raw = np.frombuffer(handle.read(count * row_bytes), dtype=np.uint8)
    if raw.size != count * row_bytes:
        raise ValueError(f"truncated tensor data for {ref.name}")
    if ref.fmt == "safetensors":
        np_dtype = ST_DTYPES[ref.dtype][0]
        values = raw.view(np_dtype)
        values = (values.astype(np.uint32) << 16).view(np.float32) if ref.dtype == "BF16" else values.astype(np.float32)
    else:
        values = decode(ref.dtype, raw, ref.type_id)
    return values.reshape(count, ref.cols)


def read_row_set(ref: TensorRef, rows: np.ndarray, perm: np.ndarray | None = None) -> np.ndarray:
    """Decode an arbitrary sorted set of rows, reading contiguous runs together."""
    wanted = perm[rows] if perm is not None else rows
    order = np.argsort(wanted, kind="stable")
    sorted_rows = wanted[order]
    out = np.empty((len(rows), ref.cols), dtype=np.float32)
    i = 0
    while i < len(sorted_rows):
        j = i
        while j + 1 < len(sorted_rows) and sorted_rows[j + 1] == sorted_rows[j] + 1:
            j += 1
        out[order[i:j + 1]] = read_rows(ref, int(sorted_rows[i]), int(sorted_rows[j]) + 1)
        i = j + 1
    return out


def read_rows_permuted(ref: TensorRef, start: int, stop: int, perm: np.ndarray | None) -> np.ndarray:
    if perm is None:
        return read_rows(ref, start, stop)
    wanted = perm[start:stop]
    lo, hi = int(wanted.min()), int(wanted.max()) + 1
    return read_rows(ref, lo, hi)[wanted - lo]


# ---------------------------------------------------------------- name mapping

HF_TO_GGUF_LAYER = {
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output", "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm",
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
    "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm",
}
HF_TO_GGUF_GLOBAL = {"model.embed_tokens": "token_embd", "model.norm": "output_norm", "lm_head": "output"}
_HF_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)\.(weight|bias)$")
_HF_GLOBAL = re.compile(r"^(.+)\.(weight|bias)$")
LAYER_RE = re.compile(r"(?:^|\.)(?:blk|layers|h|block|blocks)\.(\d+)\.")


def hf_to_gguf_name(name: str) -> str:
    match = _HF_LAYER.match(name)
    if match and match.group(2) in HF_TO_GGUF_LAYER:
        return f"blk.{match.group(1)}.{HF_TO_GGUF_LAYER[match.group(2)]}.{match.group(3)}"
    match = _HF_GLOBAL.match(name)
    if match and match.group(1) in HF_TO_GGUF_GLOBAL:
        return f"{HF_TO_GGUF_GLOBAL[match.group(1)]}.{match.group(2)}"
    return name


def layer_of(name: str) -> int | None:
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def canonical_map(tset: TensorSet, other_fmt: str) -> dict[str, TensorRef]:
    """Name -> ref; HF names are rewritten to GGUF names only for cross-format comparisons."""
    if tset.fmt == "safetensors" and other_fmt == "gguf":
        return {hf_to_gguf_name(name): ref for name, ref in tset.tensors.items()}
    return dict(tset.tensors)


def llama_qk_permutation(gguf_meta: dict, name: str, rows: int) -> np.ndarray | None:
    """Row order llama.cpp's HF->GGUF converter applies to llama q/k weights (qwen2 is not permuted)."""
    if gguf_meta.get("general.architecture") != "llama":
        return None
    kind = name.split(".")[2] if name.startswith("blk.") and name.count(".") >= 3 else ""
    if kind not in {"attn_q", "attn_k"} or not name.endswith(".weight"):
        return None
    heads = gguf_meta.get("llama.attention.head_count")
    kv_heads = gguf_meta.get("llama.attention.head_count_kv") or heads
    n_head = kv_heads if kind == "attn_k" else heads
    if not isinstance(n_head, int) or n_head <= 0 or rows % (2 * n_head):
        return None
    return np.arange(rows).reshape(n_head, 2, rows // n_head // 2).swapaxes(1, 2).reshape(-1)
