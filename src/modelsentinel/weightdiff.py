"""sentinel weight-diff: compare a model's weights against a trusted base.

Inputs are GGUF files (also ``ollama:<tag>``, resolved offline to its blob),
single ``.safetensors`` files, or Hugging Face directories with sharded
safetensors. Pickle weights are refused. numpy is an optional dependency
(``pip install 'modelsentinel[advanced]'``); without it the subcommand explains
how to install it and exits.

What it looks for, per matched tensor: relative Frobenius norm of the diff,
max abs diff, fraction of changed elements, cosine similarity; per layer, a
median/MAD robust z-score of the diff norm; for 2-D tensors with a real diff,
the singular-value spectrum of a bounded sub-matrix of the diff, to spot
low-rank injected updates (merged LoRA, ROME-style edits). Shape changes,
added tensors, and hash mismatches are reported too. Comparing two different
storage types (Q4 vs F16) is mostly quantization noise; the thresholds rise and
the report says so. A clean result means no evidence of tampering was found in
the weights relative to this base, nothing more.
"""

from __future__ import annotations

import math
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from modelsentinel.findings import Finding

INSTALL_HINT = "weight-diff needs numpy. Install the extra: pip install 'modelsentinel[advanced]' (or: pip install numpy)."
LIMITATION = (
    "No evidence of tampering in the weights relative to this base is not a proof of cleanliness: "
    "the base itself may be compromised, behavior can change without large weight changes, and "
    "quantization noise can mask small edits."
)
OLLAMA_MODEL_MEDIA = "application/vnd.ollama.image.model"
OLLAMA_ADAPTER_MEDIA = "application/vnd.ollama.image.adapter"
DEFAULT_MAX_SECONDS = 600.0
MAX_LISTED = 20


def numpy_available() -> bool:
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class Side:
    role: str
    source: str
    model_path: Path
    tset: object
    files: list[dict] = field(default_factory=list)
    hash_checks: list[dict] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    name: str = ""
    kind: str = "file"


# ---------------------------------------------------------------- inputs

def _load_side(source: str, role: str, manifest: str | None) -> Side:
    from modelsentinel import weightdiff_hash as wh
    from modelsentinel.weightdiff_io import PICKLE_SUFFIXES, open_model, refuse_pickle

    text = source.strip()
    if text.startswith("ollama:"):
        return _load_ollama(text, role, manifest)
    if text.startswith("llamacpp:"):
        text = text.split(":", 1)[1]
    path = Path(text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{role} model not found: {path}")
    if path.is_file() and path.suffix.lower() in PICKLE_SUFFIXES:
        refuse_pickle(path)
    tset = open_model(path)
    root = path if path.is_dir() else path.parent
    files = [wh.file_record(Path(f), _relative(Path(f), root), role) for f in tset.weight_files]
    checks, findings = wh.verify(files, root, manifest, role)
    return Side(role, source, path, tset, files, checks, findings, name=path.name,
                kind="directory" if path.is_dir() else "file")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _load_ollama(source: str, role: str, manifest: str | None) -> Side:
    from modelsentinel import weightdiff_hash as wh
    from modelsentinel.ingest import resolve
    from modelsentinel.weightdiff_io import open_model

    resolved = resolve(source)
    files = [{**record.to_dict(), "role": role} for record in resolved.files]
    model = next((f for f in files if f["relative"].startswith(OLLAMA_MODEL_MEDIA + "/")), None)
    if model is None:
        raise ValueError(f"{source} has no {OLLAMA_MODEL_MEDIA} layer in its manifest")
    checks, findings = wh.verify(files, None, manifest, role)
    adapters = [f for f in files if f["relative"].startswith(OLLAMA_ADAPTER_MEDIA + "/")]
    if adapters:
        findings.append(Finding(
            "medium", "weight_adapter_layer", "ollama_adapter",
            f"{source} ships {len(adapters)} adapter layer(s) that Ollama applies on top of the GGUF at load time. "
            "The weight diff covers the base GGUF only; the adapter's effect is not included.",
            evidence=", ".join(a["relative"] for a in adapters), location=source))
    tset = open_model(Path(model["path"]))
    tset.notes.extend(resolved.notes)
    return Side(role, source, Path(model["path"]), tset, files, checks, findings, name=resolved.name, kind="ollama")


# --------------------------------------------------------------- analysis

@dataclass
class Options:
    top: int = 20
    max_seconds: float = DEFAULT_MAX_SECONDS
    svd_rows: int = 512
    svd_cols: int = 1024
    seed: int = 0


def _permutations(name: str, t_ref, b_ref, target: Side, base: Side):
    from modelsentinel.weightdiff_io import llama_qk_permutation

    if target.tset.fmt == base.tset.fmt:
        return None, None
    if target.tset.fmt == "safetensors":
        return llama_qk_permutation(base.tset.metadata, name, t_ref.rows), None
    return None, llama_qk_permutation(target.tset.metadata, name, b_ref.rows)


def analyze(target: Side, base: Side, options: Options) -> dict:
    from modelsentinel import weightdiff_analysis as wa
    from modelsentinel.weightdiff_io import canonical_map, layer_of

    started = time.monotonic()
    t_map = canonical_map(target.tset, base.tset.fmt)
    b_map = canonical_map(base.tset, target.tset.fmt)
    common = [name for name in t_map if name in b_map]
    state = {"diffs": [], "shape": [], "unsupported": [], "budget": [], "dtype_pairs": {}, "permuted": 0}
    svd = wa.SvdOptions(options.svd_rows, options.svd_cols, options.seed)
    for name in common:
        t_ref, b_ref = t_map[name], b_map[name]
        if t_ref.shape != b_ref.shape:
            state["shape"].append({"name": name, "target": list(t_ref.shape), "base": list(b_ref.shape)})
            continue
        if not (t_ref.supported and b_ref.supported):
            bad = t_ref.dtype if not t_ref.supported else b_ref.dtype
            state["unsupported"].append({"name": name, "dtype": bad})
            continue
        elapsed = time.monotonic() - started
        if elapsed >= options.max_seconds:
            state["budget"].append(name)
            continue
        t_perm, b_perm = _permutations(name, t_ref, b_ref, target, base)
        state["permuted"] += int(t_perm is not None or b_perm is not None)
        key = f"{t_ref.dtype} vs {b_ref.dtype}"
        state["dtype_pairs"][key] = state["dtype_pairs"].get(key, 0) + 1
        state["diffs"].append(wa.compare_pair(
            name, layer_of(name), t_ref, b_ref, target_perm=t_perm, base_perm=b_perm, svd=svd,
            allow_svd=elapsed < 0.9 * options.max_seconds))
    state["missing"] = [name for name in b_map if name not in t_map]
    state["extra"] = [name for name in t_map if name not in b_map]
    state["elapsed"] = time.monotonic() - started
    state["cross_format"] = target.tset.fmt != base.tset.fmt
    return state


def _quant_context(state: dict) -> tuple[bool, float]:
    from modelsentinel.weightdiff_quant import expected_noise

    pairs = [key.split(" vs ") for key in state["dtype_pairs"]]
    mismatched = [pair for pair in pairs if pair[0] != pair[1]]
    if not mismatched:
        return False, 0.0
    return True, max(expected_noise(*pair) for pair in mismatched)


# --------------------------------------------------------------- findings

def _f(severity: str, category: str, code: str, message: str, evidence: str = "", location: str = "") -> Finding:
    return Finding(severity, category, code, message, evidence=evidence, location=location, tool="weight-diff")


def _structure_findings(state: dict, target: Side, base: Side) -> list[Finding]:
    out: list[Finding] = []
    for item in state["shape"][:MAX_LISTED]:
        out.append(_f("high", "weight_shape_mismatch", "shape_changed",
                      f"Tensor {item['name']} changed shape from {item['base']} (base) to {item['target']}. "
                      "A shape change means the architecture or vocabulary was altered.",
                      evidence=f"target={item['target']} base={item['base']}", location=item["name"]))
    cross = state["cross_format"]
    tied = {"output.weight", "lm_head.weight"}
    t_has_embd = any(n in target.tset.tensors for n in ("token_embd.weight", "model.embed_tokens.weight"))
    b_has_embd = any(n in base.tset.tensors for n in ("token_embd.weight", "model.embed_tokens.weight"))
    extra = [n for n in state["extra"] if not (n in tied and b_has_embd)]
    missing = [n for n in state["missing"] if not (n in tied and t_has_embd)]
    if extra:
        out.append(_f("medium" if cross else "high", "weight_tensor_added", "tensor_added",
                      f"The target has {len(extra)} tensor(s) the base does not"
                      + (" (cross-format comparison: some may be converter artifacts)." if cross else
                         ". Added tensors can carry an injected module."),
                      evidence=", ".join(extra[:MAX_LISTED]), location=extra[0]))
    if missing:
        out.append(_f("medium", "weight_tensor_missing", "tensor_missing",
                      f"The base has {len(missing)} tensor(s) the target lacks.",
                      evidence=", ".join(missing[:MAX_LISTED]), location=missing[0]))
    tied_only = [n for n in state["extra"] + state["missing"] if n not in extra and n not in missing]
    if tied_only:
        out.append(_f("info", "weight_diff_coverage", "tied_embeddings",
                      "Output projection present on one side only; consistent with tied input/output embeddings.",
                      evidence=", ".join(tied_only)))
    return out


def _low_rank_findings(diffs: list, n_matrices: int) -> list[Finding]:
    flagged = [d for d in diffs if d.low_rank]
    if not flagged:
        return []
    concentrated = len(flagged) <= max(2, int(0.1 * n_matrices))
    if not concentrated:
        names = ", ".join(d.name for d in flagged[:MAX_LISTED])
        return [_f("medium", "weight_low_rank_update", "low_rank_broad",
                   f"{len(flagged)} of {n_matrices} matrices differ from the base by a low-rank update. "
                   "That is the signature of a merged LoRA fine-tune; confirm the fine-tune is the one you expect.",
                   evidence=names, location=flagged[0].name)]
    out = []
    for diff in flagged:
        svd = diff.svd or {}
        out.append(_f("high", "weight_low_rank_update", "low_rank_update",
                      f"{diff.name} differs from the base by a low-rank update (top-{svd.get('top_k')} singular values "
                      f"hold {svd.get('top_k_energy', 0):.1%} of the diff energy, effective rank "
                      f"{svd.get('effective_rank')}), concentrated in {len(flagged)} of {n_matrices} matrices. "
                      "This is the shape of a targeted edit (ROME-style) or a narrow merged adapter.",
                      evidence=f"rel_diff={diff.rel_diff:.4g} spectrum={svd.get('spectrum', [])[:8]}",
                      location=diff.name))
    return out


def _outlier_findings(series: list[dict], quant_mismatch: bool) -> list[Finding]:
    from modelsentinel import weightdiff_analysis as wa

    out = []
    for item in wa.outlier_layers(series, quant_mismatch):
        strong = item["low_rank_tensors"] > 0 or (item["_z"] >= wa.OUTLIER_HIGH_Z and not quant_mismatch)
        z_text = "inf" if math.isinf(item["_z"]) else f"{item['_z']:.1f}"
        ratio = "inf" if math.isinf(item["_ratio"]) else f"{item['_ratio']:.1f}x"
        out.append(_f("high" if strong else "medium", "weight_outlier_layer", "outlier_layer",
                      f"Layer {item['layer']} changed far more than the other layers (robust z {z_text}, "
                      f"{ratio} the median layer diff); largest change in {item['max_tensor']}.",
                      evidence=f"rel_diff={item['_rel']:.4g} tensors={item['tensors']} "
                               f"low_rank_tensors={item['low_rank_tensors']}",
                      location=f"layer {item['layer']}"))
    return out


def _summary_findings(state: dict, quant_mismatch: bool, noise: float, flagged: bool) -> list[Finding]:
    diffs = state["diffs"]
    out: list[Finding] = []
    if quant_mismatch:
        pairs = ", ".join(f"{k} ({v})" for k, v in state["dtype_pairs"].items() if k.split(" vs ")[0] != k.split(" vs ")[1])
        out.append(_f("info", "weight_quant_mismatch", "quant_mismatch",
                      f"Target and base use different storage types ({pairs}). Most of the diff is quantization "
                      f"noise (expected relative error up to ~{noise:.0%}); outlier thresholds were raised and small "
                      "edits may be masked. Compare against a base with the same quantization for a sharper result."))
    if state["cross_format"]:
        out.append(_f("info", "weight_diff_coverage", "cross_format",
                      "Cross-format comparison (GGUF vs safetensors): Hugging Face names were mapped to GGUF names"
                      + (f"; the llama q/k row permutation was undone for {state['permuted']} tensors." if state["permuted"] else ".")))
    if not diffs:
        return out
    if all(d.max_abs_diff == 0 for d in diffs) and not state["shape"] and not state["extra"]:
        out.append(_f("info", "weight_identical", "identical",
                      f"All {len(diffs)} compared tensors are bit-identical to the base. "
                      "No evidence of tampering found in the weights relative to this base."))
        return out
    if not flagged:
        rels = sorted(d.rel_diff for d in diffs)
        median = rels[len(rels) // 2]
        noise_only = quant_mismatch and median <= 2 * noise
        out.append(_f("info" if noise_only else "low", "weight_uniform_diff", "uniform_diff",
                      f"Differences are spread evenly across tensors (median relative diff {median:.3g}), "
                      + ("consistent with requantization." if noise_only else
                         "consistent with a full fine-tune or requantization.")
                      + " No outlier layer or low-rank update found: no evidence of tampering found in the weights "
                        "relative to this base."))
    return out


def _coverage_findings(state: dict, target: Side, base: Side, max_seconds: float) -> list[Finding]:
    out = []
    if state["unsupported"]:
        kinds = sorted({u["dtype"] for u in state["unsupported"]})
        out.append(_f("info", "weight_diff_coverage", "unsupported_quantization",
                      f"{len(state['unsupported'])} tensor(s) skipped: unsupported quantization {', '.join(kinds)}. "
                      "Installing the optional gguf package (pip install 'modelsentinel[advanced]') decodes more types.",
                      evidence=", ".join(u["name"] for u in state["unsupported"][:MAX_LISTED])))
    if state["budget"]:
        out.append(_f("medium", "weight_diff_incomplete", "time_budget",
                      f"Time budget of {max_seconds:g}s reached: {len(state['budget'])} tensor(s) were not compared. "
                      "Raise --max-seconds for full coverage.",
                      evidence=", ".join(state["budget"][:MAX_LISTED])))
    if not state["diffs"] and not state["budget"]:
        out.append(_f("medium", "weight_diff_incomplete", "nothing_compared",
                      "No tensor could be compared. Check that --base is the model this one was derived from."))
    bad = sum(d.non_finite for d in state["diffs"])
    if bad:
        names = [d.name for d in state["diffs"] if d.non_finite]
        out.append(_f("medium", "weight_non_finite", "non_finite",
                      f"{bad} NaN/Inf value(s) in compared tensors; they were treated as 0 in the metrics.",
                      evidence=", ".join(names[:MAX_LISTED]), location=names[0]))
    notes = [n for n in target.tset.notes + base.tset.notes if n.startswith("ignored pickle")]
    for note in notes:
        out.append(_f("info", "weight_diff_coverage", "pickle_ignored", note.capitalize() + "."))
    return out


# ----------------------------------------------------------------- record

def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _summary(verdict: str, name: str, base: str, state: dict, findings: list[Finding]) -> str:
    compared = len(state["diffs"])
    high = sum(1 for f in findings if f.severity == "high")
    medium = sum(1 for f in findings if f.severity == "medium")
    if verdict == "PASS":
        return (f"{name}: no evidence of tampering found in the weights relative to {base} "
                f"({compared} tensors compared). This is not a proof of cleanliness.")
    return f"{name}: weight diff against {base} has warnings ({high} high, {medium} medium; {compared} tensors compared)."


def _side_payload(side: Side) -> dict:
    dtypes: dict[str, int] = {}
    for ref in side.tset.tensors.values():
        dtypes[ref.dtype] = dtypes.get(ref.dtype, 0) + 1
    meta = {k: v for k, v in side.tset.metadata.items() if isinstance(v, (str, int, float, bool))}
    return {"source": side.source, "format": side.tset.fmt, "path": str(side.model_path),
            "tensors": len(side.tset.tensors), "dtypes": dtypes,
            "architecture": meta.get("general.architecture") or meta.get("hf.model_type"),
            "notes": [n for n in side.tset.notes if not n.startswith("unsupported")][:MAX_LISTED]}


def build_record(target: Side, base: Side, state: dict, options: Options, policy: dict) -> dict:
    from modelsentinel import weightdiff_analysis as wa
    from modelsentinel.scoring import judge

    quant_mismatch, noise = _quant_context(state)
    diffs = state["diffs"]
    series = wa.layer_series(diffs)
    n_matrices = sum(1 for d in diffs if len(d.shape) >= 2)
    findings = target.findings + base.findings + _structure_findings(state, target, base)
    lr = _low_rank_findings(diffs, n_matrices)
    outliers = _outlier_findings(series, quant_mismatch)
    findings += lr + outliers
    findings += _summary_findings(state, quant_mismatch, noise, bool(lr or outliers))
    findings += _coverage_findings(state, target, base, options.max_seconds)
    findings += [_f("info", "provenance", "hash_verified",
                    f"SHA-256 of {c['role']} file {c['file']} matches the {c['source']}.", location=c["file"])
                 for c in target.hash_checks + base.hash_checks if c["ok"]]
    score, verdict = judge(findings, policy)
    outlier_layers = {item["layer"] for item in wa.outlier_layers(series, quant_mismatch)}
    flagged = [d.to_dict() for d in diffs if d.low_rank or (d.layer in outlier_layers and d.svd)]
    top = sorted(diffs, key=lambda d: (d.rel_diff if not math.isinf(d.rel_diff) else 1e30), reverse=True)
    payload = {
        "target": _side_payload(target), "base": _side_payload(base),
        "compared": len(diffs), "identical_tensors": sum(1 for d in diffs if d.max_abs_diff == 0),
        "quant_mismatch": quant_mismatch, "expected_quant_noise": noise, "dtype_pairs": state["dtype_pairs"],
        "cross_format": state["cross_format"], "missing": state["missing"], "extra": state["extra"],
        "shape_mismatches": state["shape"], "skipped": {"unsupported": state["unsupported"], "budget": state["budget"]},
        "top_tensors": [d.to_dict() for d in top[: max(0, options.top)]],
        "layers": wa.public_series(series), "flagged": flagged[:64],
        "hash_checks": target.hash_checks + base.hash_checks,
        "elapsed_seconds": round(state["elapsed"], 3), "budget_seconds": options.max_seconds,
        "svd": {"max_rows": options.svd_rows, "max_cols": options.svd_cols, "top_k": wa.TOP_K,
                "spectrum_len": wa.SPECTRUM_LEN, "seed": options.seed},
        "thresholds": {"low_rank_energy": wa.LOW_RANK_ENERGY, "low_rank_min_rel": wa.LOW_RANK_MIN_REL,
                       "low_rank_min_dim": wa.LOW_RANK_MIN_DIM, "outlier_z": wa.OUTLIER_Z,
                       "outlier_ratio": wa.OUTLIER_RATIO_QUANT if quant_mismatch else wa.OUTLIER_RATIO,
                       "svd_min_rel": wa.SVD_MIN_REL},
    }
    return {
        "id": uuid.uuid4().hex[:12], "created_at": _now(), "source": target.source, "source_kind": target.kind,
        "kind": "weightdiff", "name": target.name, "family": payload["target"]["architecture"],
        "adapter": None, "base": base.source, "verdict": verdict, "score": score,
        "summary": _summary(verdict, target.name, base.name or base.source, state, findings),
        "findings": [f.to_dict() for f in findings], "files": target.files + base.files,
        "notes": target.tset.notes[:MAX_LISTED] + base.tset.notes[:MAX_LISTED],
        "generations": [], "sandbox_events": [], "redteam": None, "sandbox": None, "weightdiff": payload,
        "policy": {"name": policy.get("name"), "block_in_ci": policy.get("block_in_ci")},
        "limitation": LIMITATION,
    }


def weight_diff(target: str, base: str, *, manifest: str | None = None, base_manifest: str | None = None,
                options: Options | None = None, policy: dict | None = None, persist: bool = True,
                db_path: Path | None = None) -> dict:
    if not numpy_available():
        raise RuntimeError(INSTALL_HINT)
    from modelsentinel.policy import load_policy

    options = options or Options()
    policy = policy or load_policy()
    target_side = _load_side(target, "target", manifest)
    base_side = _load_side(base, "base", base_manifest)
    state = analyze(target_side, base_side, options)
    record = build_record(target_side, base_side, state, options, policy)
    if persist:
        from modelsentinel.db import connect, save_scan
        from modelsentinel.pipeline import default_db_path

        path = db_path or default_db_path()
        conn = connect(path)
        try:
            save_scan(conn, record)
        finally:
            conn.close()
        record["db"] = str(path)
    return record


# -------------------------------------------------------------------- CLI

def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "weight-diff", help="Diff a model's weights against a trusted base (needs the [advanced] extra).",
        description="Compare TARGET's tensors with BASE's: per-tensor diff norms, per-layer outliers, low-rank "
                    "update detection, shape changes, hash checks. Inputs: .gguf, ollama:<tag>, .safetensors, or a "
                    "Hugging Face directory. Pickle weights are refused.")
    parser.add_argument("target")
    parser.add_argument("--base", required=True, help="the model TARGET claims to derive from")
    parser.add_argument("--manifest", default=None, help="YAML/JSON {filename: sha256} for the target")
    parser.add_argument("--base-manifest", default=None, help="YAML/JSON {filename: sha256} for the base")
    parser.add_argument("--top", type=int, default=20, help="tensors in the per-tensor table (default 20)")
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS,
                        help=f"analysis time budget; remaining tensors are reported as skipped (default {DEFAULT_MAX_SECONDS:g})")
    parser.add_argument("--svd-rows", type=int, default=512, help="max rows of the diff sub-matrix fed to the SVD")
    parser.add_argument("--svd-cols", type=int, default=1024, help="max columns of the diff sub-matrix fed to the SVD")
    parser.add_argument("--seed", type=int, default=0, help="seed for the SVD row/column subsample")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--no-save", action="store_true", help="do not store the result in the database")
    parser.add_argument("--format", default="text", choices=["text", "json", "sarif", "html"])
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--ci", action="store_true")
    parser.set_defaults(func=_cmd_weight_diff)


def _cmd_weight_diff(args) -> int:
    if not numpy_available():
        print(INSTALL_HINT, file=sys.stderr)
        return 2
    from modelsentinel.policy import ci_should_block, load_policy

    policy = load_policy(args.policy)
    options = Options(top=args.top, max_seconds=args.max_seconds, svd_rows=args.svd_rows,
                      svd_cols=args.svd_cols, seed=args.seed)
    db = getattr(args, "db", None)
    record = weight_diff(args.target, args.base, manifest=args.manifest, base_manifest=args.base_manifest,
                         options=options, policy=policy, persist=not args.no_save,
                         db_path=Path(db) if db else None)
    _emit(record, args.format, args.output)
    if args.ci and ci_should_block(record["verdict"], policy):
        return 1
    return 0


def format_text(record: dict) -> str:
    payload = record.get("weightdiff") or {}
    lines = [f"verdict: {record['verdict']}  score: {record['score']}", record.get("summary", "")]
    for finding in record.get("findings") or []:
        if finding["severity"] != "info":
            lines.append(f"  {finding['severity'].upper():8} {finding['category']:24} {finding['message']}")
    rows = payload.get("top_tensors") or []
    if rows:
        lines.append("  tensor                                          rel_diff   max_abs  changed  cosine  lowrank")
        for row in rows:
            energy = (row.get("svd") or {}).get("top_k_energy")
            flag = f"{energy:.2f}{' *' if row.get('low_rank') else ''}" if energy is not None else "  -"
            lines.append(f"  {row['name'][:46]:46} {row['rel_diff'] or 0:9.3g} {row['max_abs_diff'] or 0:9.3g} "
                         f"{row['frac_changed'] or 0:7.1%} {row['cosine'] if row['cosine'] is not None else 0:7.4f}  {flag}")
    lines.append(f"compared {payload.get('compared', 0)} tensors in {payload.get('elapsed_seconds', 0)}s")
    lines.append(f"scan id: {record['id']}")
    return "\n".join(lines)


def _emit(record: dict, fmt: str, output: str | None) -> None:
    if fmt == "text":
        text = format_text(record)
    else:
        from modelsentinel.report import render

        text = render(record, fmt)
    if output:
        Path(output).write_text(text, encoding="utf-8")
        print(output)
    else:
        print(text)
