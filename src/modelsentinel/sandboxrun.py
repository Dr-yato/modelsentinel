"""Python wrapper around the Rust sandbox runner."""

from __future__ import annotations

import tempfile
from pathlib import Path

from modelsentinel.native import run_sandbox


def run_source(source: str, timeout: int = 5) -> dict:
    with tempfile.TemporaryDirectory(prefix="sentinel-code-") as tmp:
        path = Path(tmp) / "snippet.py"
        path.write_text(source, encoding="utf-8")
        return run_sandbox(path, timeout=timeout)
