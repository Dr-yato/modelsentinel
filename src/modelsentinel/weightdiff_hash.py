"""Hash verification for weight-diff inputs.

Expected digests come from, in order of trust:
- a caller-supplied manifest (``--manifest`` / ``--base-manifest``): YAML or
  JSON, either ``{filename: sha256}`` or ingest's ``{files: [{path, sha256}]}``;
- content addressing: Ollama blobs are named ``sha256-<digest>``, and Hugging
  Face cache snapshots symlink each LFS file to ``blobs/<sha256>``;
- sidecar checksum files next to the weights (``<file>.sha256``, ``SHA256SUMS``,
  ``sha256sums.txt``, ``checksums.sha256``).

A sidecar shipped in the same directory as the weights only detects corruption,
not a deliberate swap (whoever swapped the weights could rewrite it); the
record says which source each check used.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from modelsentinel.findings import Finding
from modelsentinel.hashutil import sha256_file

HEX64 = re.compile(r"\b([0-9a-fA-F]{64})\b")
SUM_FILES = ("SHA256SUMS", "sha256sums.txt", "checksums.sha256", "SHA256SUMS.txt")


def file_record(path: Path, relative: str, role: str, digest: str | None = None) -> dict:
    return {"path": str(path), "relative": relative, "sha256": digest or sha256_file(path),
            "size": path.stat().st_size, "role": role}


def load_manifest(path: str) -> dict[str, str]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"manifest {path} must be a mapping")
    expected: dict[str, str] = {}
    if isinstance(raw.get("files"), list):
        for item in raw["files"]:
            if isinstance(item, dict) and "path" in item and "sha256" in item:
                expected[str(item["path"])] = _digest(str(item["sha256"]))
    for key, value in raw.items():
        if key in {"files", "revision", "name", "source"} or not isinstance(value, str):
            continue
        if HEX64.fullmatch(_digest(value)):
            expected[str(key)] = _digest(value)
    if not expected:
        raise ValueError(f"manifest {path} lists no sha256 digests")
    return expected


def _digest(value: str) -> str:
    text = value.strip().lower()
    return text.split(":", 1)[1] if text.startswith("sha256:") else text


def _sidecar_expectations(directory: Path) -> dict[str, tuple[str, str]]:
    found: dict[str, tuple[str, str]] = {}
    candidates = [directory / name for name in SUM_FILES] + sorted(directory.glob("*.sha256"))
    for sums in candidates:
        if not sums.is_file() or sums.stat().st_size > 1_000_000:
            continue
        text = sums.read_text(encoding="utf-8", errors="replace")
        if sums.suffix == ".sha256" and sums.name not in SUM_FILES:
            match = HEX64.search(text)
            if match:
                found[sums.name[: -len(".sha256")]] = (match.group(1).lower(), f"sidecar {sums.name}")
            continue
        for line in text.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and HEX64.fullmatch(parts[0]):
                found[parts[1].lstrip("*").strip()] = (parts[0].lower(), f"sidecar {sums.name}")
    return found


def _content_address(path: Path) -> tuple[str, str] | None:
    name = path.name
    if name.startswith("sha256-") and HEX64.fullmatch(name[7:]):
        return name[7:].lower(), "ollama content address"
    if path.is_symlink():
        target = Path(path.resolve()).name
        if HEX64.fullmatch(target):
            return target.lower(), "hugging face cache blob name"
    return None


def verify(records: list[dict], root: Path | None, manifest: str | None, role: str) -> tuple[list[dict], list[Finding]]:
    """Return (checks, findings) for one side's files."""
    checks: list[dict] = []
    findings: list[Finding] = []
    sidecars = _sidecar_expectations(root) if root is not None and root.is_dir() else {}
    by_name = {r["relative"]: r for r in records}
    by_name.update({Path(r["relative"]).name: r for r in records})
    for record in records:
        path = Path(record["path"])
        expected = _content_address(path) or sidecars.get(record["relative"]) or sidecars.get(path.name)
        if expected:
            checks.append(_check(record, expected[0], expected[1], role, findings))
    if manifest:
        for name, digest in load_manifest(manifest).items():
            record = by_name.get(name) or by_name.get(Path(name).name)
            if record is None and root is not None and (root / name).is_file():
                record = file_record(root / name, name, role)
            if record is None:
                findings.append(Finding("high", "hash_mismatch", "missing_file",
                                        f"The {role} manifest expects {name}, which is not in the {role} source.",
                                        location=name))
                checks.append({"role": role, "file": name, "expected": digest, "actual": None,
                               "source": f"manifest {Path(manifest).name}", "ok": False})
                continue
            checks.append(_check(record, digest, f"manifest {Path(manifest).name}", role, findings))
    return checks, findings


def _check(record: dict, expected: str, source: str, role: str, findings: list[Finding]) -> dict:
    actual = record["sha256"].lower()
    ok = actual == expected
    if not ok:
        findings.append(Finding(
            "high", "hash_mismatch", "sha256_mismatch",
            f"SHA-256 of {role} file {record['relative']} does not match the {source}.",
            evidence=f"actual={actual} expected={expected}", location=record["relative"]))
    return {"role": role, "file": record["relative"], "expected": expected, "actual": actual,
            "source": source, "ok": ok}
