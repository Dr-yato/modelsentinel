"""Locate and call the Rust parser and sandbox binaries."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class NativeError(RuntimeError):
    pass


def _candidates(name: str) -> list[Path]:
    env = os.environ.get(f"SENTINEL_{name.upper().replace('-', '_')}_BIN")
    found = []
    if env:
        found.append(Path(env))
    which = shutil.which(name)
    if which:
        found.append(Path(which))
    found.extend(
        [
            ROOT / "native" / "target" / "release" / name,
            ROOT / "native" / "target" / "debug" / name,
        ]
    )
    return found


def binary(name: str) -> Path:
    for path in _candidates(name):
        if path.is_file() and os.access(path, os.X_OK):
            return path
    searched = ", ".join(str(path) for path in _candidates(name))
    raise NativeError(
        f"Rust binary {name} was not found ({searched}). "
        "Build it with: cargo build --manifest-path native/Cargo.toml"
    )


def parse_file(path: Path, timeout: float = 120) -> dict:
    proc = subprocess.run(
        [str(binary("sentinel-parse")), str(path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if not proc.stdout.strip():
        raise NativeError(proc.stderr.strip() or f"sentinel-parse failed on {path}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise NativeError(f"sentinel-parse returned invalid JSON: {exc}") from exc


def run_sandbox(code_path: Path, timeout: int = 5) -> dict:
    proc = subprocess.run(
        [str(binary("sentinel-sandbox")), "--code", str(code_path), "--timeout", str(timeout)],
        capture_output=True,
        text=True,
        timeout=timeout + 20,
        check=False,
    )
    raw = proc.stdout.strip()
    if not raw:
        raise NativeError(proc.stderr.strip() or "sentinel-sandbox produced no output")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NativeError(f"sentinel-sandbox returned invalid JSON: {exc}") from exc
