"""Locate and call the Rust parser and sandbox binaries."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class NativeError(RuntimeError):
    pass


EXE = ".exe" if sys.platform.startswith("win") else ""
PACKAGE_BIN = Path(__file__).resolve().parent / "bin"


def _candidates(name: str) -> list[Path]:
    """Where a native binary may live, most specific first.

    1. ``SENTINEL_<NAME>_BIN`` (e.g. ``SENTINEL_SENTINEL_PARSE_BIN``)
    2. ``PATH`` (``pip install`` with setuptools-rust puts the binaries next to ``sentinel``)
    3. ``modelsentinel/bin`` inside the installed package (``make native`` copies them there)
    4. ``native/target/{release,debug}`` in a source checkout
    """
    env = os.environ.get(f"SENTINEL_{name.upper().replace('-', '_')}_BIN")
    found = []
    if env:
        found.append(Path(env))
    which = shutil.which(name + EXE) or shutil.which(name)
    if which:
        found.append(Path(which))
    scripts = Path(sys.executable).parent
    found.extend(
        [
            scripts / (name + EXE),
            PACKAGE_BIN / (name + EXE),
            ROOT / "native" / "target" / "release" / (name + EXE),
            ROOT / "native" / "target" / "debug" / (name + EXE),
        ]
    )
    unique: list[Path] = []
    for path in found:
        if path not in unique:
            unique.append(path)
    return unique


def find_binary(name: str) -> Path | None:
    for path in _candidates(name):
        if path.is_file() and (EXE or os.access(path, os.X_OK)):
            return path
    return None


def binary(name: str) -> Path:
    path = find_binary(name)
    if path is not None:
        return path
    searched = ", ".join(str(path) for path in _candidates(name))
    raise NativeError(
        f"Rust binary {name} was not found ({searched}). "
        "Build it with `make native` (or: cargo build --release --manifest-path native/Cargo.toml)"
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
