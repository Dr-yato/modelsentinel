"""Resolve a local model source and hash it. Nothing here loads weights.

Supported sources:
- a file or a Hugging Face-style directory
- `ollama:<tag>` resolved only against a local Ollama manifest
- `planted:<name>` and `clean:<name>` behavioral fixtures
- a directory containing sentinel-model.json
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from modelsentinel.findings import Finding
from modelsentinel.hashutil import sha256_file


SKIP_DIRS = {".git", "__pycache__", ".modelsentinel", "node_modules"}


@dataclass
class FileRecord:
    path: str
    relative: str
    sha256: str
    size: int

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "relative": self.relative,
            "sha256": self.sha256,
            "size": self.size,
        }


@dataclass
class Resolved:
    source: str
    kind: str
    files: list[FileRecord] = field(default_factory=list)
    adapter: str | None = None
    family: str | None = None
    name: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "kind": self.kind,
            "adapter": self.adapter,
            "family": self.family,
            "name": self.name,
            "notes": self.notes,
            "files": [item.to_dict() for item in self.files],
        }


def resolve(source: str) -> Resolved:
    text = source.strip()
    if text.startswith("ollama:"):
        return _resolve_ollama(text)
    if text.startswith("planted"):
        return Resolved(source=text, kind="adapter", adapter="planted", name=text, family="fixture")
    if text.startswith("clean"):
        return Resolved(source=text, kind="adapter", adapter="clean", name=text, family="fixture")
    if text.startswith("llamacpp:"):
        path = Path(text.split(":", 1)[1])
        return _resolve_path(path, original=text, adapter="llamacpp")
    path = Path(text).expanduser()
    return _resolve_path(path, original=text)


def _resolve_path(path: Path, original: str, adapter: str | None = None) -> Resolved:
    if not path.exists():
        raise FileNotFoundError(f"model source not found: {path}")
    if path.is_file():
        record = _file_record(path, path.name)
        return Resolved(
            source=original,
            kind="file",
            files=[record],
            adapter=adapter,
            name=path.name,
            family=_guess_family(path.name, {}),
        )
    meta_path = path / "sentinel-model.json"
    meta = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        adapter = adapter or meta.get("adapter")
    files = []
    for file_path in _walk(path):
        relative = str(file_path.relative_to(path))
        files.append(_file_record(file_path, relative))
    config = _read_json(path / "config.json")
    family = meta.get("family") or _guess_family(path.name, config)
    return Resolved(
        source=original,
        kind="directory",
        files=files,
        adapter=adapter,
        family=family,
        name=meta.get("name") or path.name,
        notes=[meta.get("description", "")] if meta.get("description") else [],
    )


def _walk(root: Path) -> list[Path]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS and not name.startswith(".")]
        for name in filenames:
            found.append(Path(dirpath) / name)
    found.sort()
    return found[:10000]


def _file_record(path: Path, relative: str) -> FileRecord:
    return FileRecord(path=str(path), relative=relative, sha256=sha256_file(path), size=path.stat().st_size)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _guess_family(name: str, config: dict) -> str | None:
    blob = " ".join(
        [
            name.lower(),
            str(config.get("model_type", "")).lower(),
            str(config.get("architectures", "")).lower(),
        ]
    )
    for family in ("llama3", "llama", "qwen", "mistral", "gemma", "phi", "deepseek", "chatml"):
        if family in blob:
            return "llama3" if family == "llama3" or "llama-3" in blob else family
    if "llama" in blob:
        return "llama3" if "llama3" in blob or "llama-3" in blob else "llama"
    return None


def manifest_findings(resolved: Resolved, manifest_path: str | None) -> list[Finding]:
    if not manifest_path:
        return []
    import yaml

    raw = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8")) or {}
    expected = {item["path"]: item["sha256"].lower() for item in raw.get("files") or [] if "path" in item and "sha256" in item}
    findings = []
    seen = {record.relative: record for record in resolved.files}
    for relative, digest in expected.items():
        record = seen.get(relative)
        if record is None:
            findings.append(
                Finding(
                    severity="high",
                    category="hash_mismatch",
                    code="missing_file",
                    message=f"Pinned manifest expects {relative}, which is not in the source.",
                    location=relative,
                )
            )
            continue
        if record.sha256.lower() != digest:
            findings.append(
                Finding(
                    severity="high",
                    category="hash_mismatch",
                    code="sha256_mismatch",
                    message=f"SHA-256 of {relative} does not match the pinned manifest.",
                    evidence=f"actual={record.sha256} expected={digest}",
                    location=relative,
                )
            )
    revision = raw.get("revision")
    if revision:
        findings.append(
            Finding(
                severity="info",
                category="provenance",
                code="pinned_revision",
                message=f"Compared against pinned revision {revision}. No network lookup was performed.",
                location=str(manifest_path),
            )
        )
    return findings


def repo_findings(resolved: Resolved) -> list[Finding]:
    """Flag trust_remote_code, auto_map, and Python files shipped beside weights."""
    findings: list[Finding] = []
    for record in resolved.files:
        path = Path(record.path)
        suffix = path.suffix.lower()
        if suffix == ".py":
            findings.append(
                Finding(
                    severity="high",
                    category="trust_remote_code",
                    code="python_in_model_repo",
                    message=f"Python file {record.relative} is inside the model directory. Loading it requires executing repository code.",
                    location=record.relative,
                )
            )
        if path.name in {"config.json", "tokenizer_config.json", "config.yaml"}:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lowered = text.lower()
            if "auto_map" in lowered or "trust_remote_code" in lowered:
                findings.append(
                    Finding(
                        severity="high",
                        category="trust_remote_code",
                        code="auto_map",
                        message=f"{record.relative} references auto_map or trust_remote_code. ModelSentinel will not enable it.",
                        evidence=text[:400],
                        location=record.relative,
                    )
                )
    return findings


def _resolve_ollama(source: str) -> Resolved:
    tag = source.split(":", 1)[1]
    root = Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))
    manifests = root / "manifests"
    if not manifests.is_dir():
        raise FileNotFoundError(
            f"Ollama models directory {root} is not present. ModelSentinel does not pull tags."
        )
    match = _find_manifest(manifests, tag)
    if match is None:
        raise FileNotFoundError(f"Ollama tag {tag} is not in the local manifest store at {manifests}.")
    manifest = json.loads(match.read_text(encoding="utf-8"))
    blobs = root / "blobs"
    files: list[FileRecord] = []
    notes = [f"Resolved offline from {match}"]
    for layer in [manifest.get("config")] + list(manifest.get("layers") or []):
        if not isinstance(layer, dict):
            continue
        digest = str(layer.get("digest", ""))
        if not digest.startswith("sha256:"):
            continue
        blob_name = "sha256-" + digest.split(":", 1)[1]
        blob = blobs / blob_name
        if not blob.is_file():
            notes.append(f"missing blob {blob_name}")
            continue
        media = str(layer.get("mediaType", "layer"))
        files.append(_file_record(blob, f"{media}/{blob_name}"))
    return Resolved(
        source=source,
        kind="ollama",
        files=files,
        adapter="ollama",
        name=tag,
        family=_guess_family(tag, {}),
        notes=notes,
    )


def _find_manifest(manifests: Path, tag: str) -> Path | None:
    wanted = tag.strip().strip("/")
    if ":" not in wanted:
        wanted = wanted + ":latest"
    name, version = wanted.split(":", 1)
    direct_candidates = [
        manifests / name / version,
        manifests / "registry.ollama.ai" / "library" / name / version,
        manifests / "library" / name / version,
    ]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate
    leafs = [path for path in manifests.rglob("*") if path.is_file()]
    for path in leafs:
        relative = str(path.relative_to(manifests))
        if relative.endswith(f"{name}/{version}") or relative.endswith(f"{name}:{version}"):
            return path
    return None
