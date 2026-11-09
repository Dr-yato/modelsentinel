"""Per-tensor diff metrics, diff spectra, and robust per-layer outlier scores.

``compare_pair`` streams both tensors in row chunks and accumulates
float64 sums, so a 150k x 1.5k embedding never sits in memory twice. The SVD
runs on a bounded sub-matrix of the diff: half the rows/columns with the largest
diff energy (so a sparse row edit is not sampled away) and half chosen at random.
Row/column subsampling cannot raise the rank of a low-rank update, and keeps a
full-rank (noise-like) diff full-rank, which is what the low-rank test needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from modelsentinel.weightdiff_io import TensorRef, read_row_set, read_rows_permuted

CHUNK_ELEMENTS = 4 * 1024 * 1024
TOP_K = 8
SPECTRUM_LEN = 16
LOW_RANK_ENERGY = 0.90
LOW_RANK_MIN_DIM = 32
LOW_RANK_MIN_REL = 1e-3
SVD_MIN_REL = 1e-4
SVD_MIN_DIM = 8
OUTLIER_Z = 3.5
OUTLIER_RATIO = 1.5
OUTLIER_RATIO_QUANT = 3.0
OUTLIER_HIGH_Z = 10.0
OUTLIER_MIN_LAYERS = 4
NEGLIGIBLE_REL = 1e-6


@dataclass
class SvdOptions:
    max_rows: int = 512
    max_cols: int = 1024
    seed: int = 0


@dataclass
class TensorDiff:
    name: str
    layer: int | None
    shape: tuple[int, ...]
    target_dtype: str
    base_dtype: str
    rel_diff: float = 0.0
    max_abs_diff: float = 0.0
    frac_changed: float = 0.0
    rows_changed: float = 0.0
    cosine: float = 1.0
    diff_sq: float = 0.0
    base_sq: float = 0.0
    non_finite: int = 0
    svd: dict | None = None
    low_rank: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "layer": self.layer, "shape": list(self.shape),
            "target_dtype": self.target_dtype, "base_dtype": self.base_dtype,
            "rel_diff": _clean(self.rel_diff), "max_abs_diff": _clean(self.max_abs_diff),
            "frac_changed": _clean(self.frac_changed), "rows_changed": _clean(self.rows_changed),
            "cosine": _clean(self.cosine), "non_finite": self.non_finite, "svd": self.svd,
            "low_rank": self.low_rank, "notes": self.notes,
        }


def _clean(value: float) -> float | None:
    """JSON has no inf/nan; keep the numbers readable."""
    if value is None or math.isnan(value):
        return None
    if math.isinf(value):
        return 1e30 if value > 0 else -1e30
    return float(f"{value:.6g}")


class _Accumulator:
    def __init__(self, rows: int, cols: int) -> None:
        self.diff_sq = self.base_sq = self.target_sq = self.cross = 0.0
        self.max_abs = 0.0
        self.changed = 0
        self.non_finite = 0
        self.row_energy = np.zeros(rows, dtype=np.float64)
        self.col_energy = np.zeros(cols, dtype=np.float64)

    def add(self, start: int, target: np.ndarray, base: np.ndarray) -> None:
        bad = ~(np.isfinite(target) & np.isfinite(base))
        if bad.any():
            self.non_finite += int(np.count_nonzero(bad))
            target = np.where(bad, 0.0, target).astype(np.float32)
            base = np.where(bad, 0.0, base).astype(np.float32)
        diff = target - base
        sq = np.square(diff, dtype=np.float64)
        self.diff_sq += float(sq.sum())
        self.base_sq += float(np.square(base, dtype=np.float64).sum())
        self.target_sq += float(np.square(target, dtype=np.float64).sum())
        self.cross += float(np.multiply(target, base, dtype=np.float64).sum())
        if diff.size:
            self.max_abs = max(self.max_abs, float(np.abs(diff).max()))
        self.changed += int(np.count_nonzero(diff))
        self.row_energy[start:start + diff.shape[0]] = sq.sum(axis=1)
        self.col_energy += sq.sum(axis=0)


def compare_pair(name: str, layer: int | None, target: TensorRef, base: TensorRef, *,
                 target_perm: np.ndarray | None = None, base_perm: np.ndarray | None = None,
                 svd: SvdOptions | None = None, allow_svd: bool = True) -> TensorDiff:
    rows, cols = target.rows, target.cols
    result = TensorDiff(name, layer, target.shape, target.dtype, base.dtype)
    acc = _Accumulator(rows, cols)
    chunk = max(1, CHUNK_ELEMENTS // max(1, cols))
    for start in range(0, rows, chunk):
        stop = min(rows, start + chunk)
        acc.add(start, read_rows_permuted(target, start, stop, target_perm),
                read_rows_permuted(base, start, stop, base_perm))
    total = max(1, rows * cols)
    result.diff_sq, result.base_sq = acc.diff_sq, acc.base_sq
    result.max_abs_diff = acc.max_abs
    result.frac_changed = acc.changed / total
    result.rows_changed = float(np.count_nonzero(acc.row_energy)) / max(1, rows)
    result.non_finite = acc.non_finite
    if acc.base_sq > 0:
        result.rel_diff = math.sqrt(acc.diff_sq / acc.base_sq)
    else:
        result.rel_diff = 0.0 if acc.diff_sq == 0 else math.inf
    denom = math.sqrt(acc.target_sq * acc.base_sq)
    result.cosine = acc.cross / denom if denom > 0 else (1.0 if acc.diff_sq == 0 else 0.0)
    two_d = len(target.shape) >= 2 and min(rows, cols) >= SVD_MIN_DIM
    if allow_svd and two_d and result.rel_diff >= SVD_MIN_REL:
        result.svd = _diff_spectrum(target, base, acc, svd or SvdOptions(), target_perm, base_perm)
        result.low_rank = _is_low_rank(result)
    return result


def _pick(energy: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    count = energy.shape[0]
    if count <= limit:
        return np.arange(count)
    top = np.argsort(energy)[::-1][: limit // 2]
    rest = np.setdiff1d(np.arange(count), top, assume_unique=False)
    extra = rng.choice(rest, size=limit - top.shape[0], replace=False)
    return np.sort(np.concatenate([top, extra]))


def _diff_spectrum(target: TensorRef, base: TensorRef, acc: _Accumulator, options: SvdOptions,
                   target_perm: np.ndarray | None, base_perm: np.ndarray | None) -> dict:
    rng = np.random.default_rng(options.seed)
    rows = _pick(acc.row_energy, max(1, options.max_rows), rng)
    cols = _pick(acc.col_energy, max(1, options.max_cols), rng)
    sub_t = read_row_set(target, rows, target_perm)[:, cols]
    sub_b = read_row_set(base, rows, base_perm)[:, cols]
    diff = np.nan_to_num(sub_t.astype(np.float64) - sub_b.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.linalg.svd(diff, compute_uv=False)
    energy = np.square(values)
    total = float(energy.sum())
    if total <= 0:
        return {"sampled_shape": list(diff.shape), "top_k_energy": 0.0, "effective_rank": 0.0, "spectrum": []}
    share = energy / total
    nonzero = share[share > 0]
    eff_rank = float(math.exp(-float((nonzero * np.log(nonzero)).sum())))
    spectrum = (values[:SPECTRUM_LEN] / values[0]).tolist() if values[0] > 0 else []
    return {
        "sampled_shape": list(diff.shape),
        "top_k": TOP_K,
        "top_k_energy": float(f"{float(share[:TOP_K].sum()):.6g}"),
        "effective_rank": float(f"{eff_rank:.4g}"),
        "spectrum": [float(f"{v:.5g}") for v in spectrum],
    }


def _is_low_rank(diff: TensorDiff) -> bool:
    info = diff.svd or {}
    shape = info.get("sampled_shape") or [0, 0]
    return (min(shape) >= LOW_RANK_MIN_DIM and diff.rel_diff >= LOW_RANK_MIN_REL
            and float(info.get("top_k_energy") or 0.0) >= LOW_RANK_ENERGY)


def layer_series(diffs: list[TensorDiff]) -> list[dict]:
    grouped: dict[int, list[TensorDiff]] = {}
    for item in diffs:
        if item.layer is not None:
            grouped.setdefault(item.layer, []).append(item)
    series = []
    for layer in sorted(grouped):
        items = grouped[layer]
        diff_sq = sum(i.diff_sq for i in items)
        base_sq = sum(i.base_sq for i in items)
        rel = math.sqrt(diff_sq / base_sq) if base_sq > 0 else (0.0 if diff_sq == 0 else math.inf)
        series.append({"layer": layer, "rel_diff": rel, "tensors": len(items),
                       "low_rank_tensors": sum(1 for i in items if i.low_rank),
                       "max_tensor": max(items, key=lambda i: i.rel_diff).name})
    return robust_z(series)


def robust_z(series: list[dict]) -> list[dict]:
    """Add a median/MAD z-score of rel_diff to each layer entry (returns new dicts)."""
    values = np.array([min(item["rel_diff"], 1e30) for item in series], dtype=np.float64)
    if values.size == 0:
        return []
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    out = []
    for item, value in zip(series, values):
        if mad > 0:
            z = 0.6745 * (value - median) / mad
        else:
            z = math.inf if value > median + NEGLIGIBLE_REL else 0.0
        ratio = value / median if median > 0 else (math.inf if value > NEGLIGIBLE_REL else 1.0)
        out.append({**item, "rel_diff": _clean(item["rel_diff"]), "robust_z": _clean(z),
                    "ratio_to_median": _clean(ratio), "_z": z, "_ratio": ratio, "_rel": value})
    return out


def outlier_layers(series: list[dict], quant_mismatch: bool) -> list[dict]:
    if len(series) < OUTLIER_MIN_LAYERS:
        return []
    ratio_floor = OUTLIER_RATIO_QUANT if quant_mismatch else OUTLIER_RATIO
    return [item for item in series
            if item["_z"] >= OUTLIER_Z and item["_ratio"] >= ratio_floor and item["_rel"] > NEGLIGIBLE_REL]


def public_series(series: list[dict]) -> list[dict]:
    return [{k: v for k, v in item.items() if not k.startswith("_")} for item in series]
