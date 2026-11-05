"""Regression tests for bugs found on a fresh Linux box (Python 3.12, Ollama 0.40.14)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from modelsentinel.fixtureslib import gguf_bytes
from modelsentinel.ingest import _split_tag, find_ollama_manifest, resolve


# -- 3. Ollama 0.40+ manifests-v2 layout ------------------------------------

def _ollama_store(root: Path, layout: str, tag_path: tuple[str, ...], symlink: bool) -> str:
    blobs = root / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    model = gguf_bytes({"tokenizer.chat_template": "{{ messages[0]['content'] }}"}, architecture="qwen2")
    model_digest = hashlib.sha256(model).hexdigest()
    (blobs / f"sha256-{model_digest}").write_bytes(model)
    manifest = json.dumps({
        "schemaVersion": 2,
        "config": {"mediaType": "application/vnd.docker.container.image.v1+json", "digest": f"sha256:{model_digest}"},
        "layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{model_digest}"}],
    }).encode()
    manifest_digest = hashlib.sha256(manifest).hexdigest()
    target = root / layout / Path(*tag_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        # Ollama 0.40+ stores the manifest as a blob and symlinks the tag to it.
        (blobs / f"sha256-{manifest_digest}").write_bytes(manifest)
        rel = os.path.relpath(blobs / f"sha256-{manifest_digest}", target.parent)
        try:
            target.symlink_to(rel)
        except OSError:
            target.write_bytes(manifest)  # Windows without symlink privilege
    else:
        target.write_bytes(manifest)
    return model_digest


@pytest.mark.parametrize("layout, tag_path, tag, symlink", [
    ("manifests-v2", ("ollama.com", "library", "qwen2.5-coder", "1.5b"), "qwen2.5-coder:1.5b", True),
    ("manifests", ("registry.ollama.ai", "library", "qwen2.5-coder", "1.5b"), "qwen2.5-coder:1.5b", False),
    ("manifests-v2", ("hf.co", "OBLITERATUS", "gemma-4-E4B", "Q4_K_M"), "hf.co/OBLITERATUS/gemma-4-E4B:Q4_K_M", True),
    ("manifests-v2", ("ollama.com", "someuser", "coder", "latest"), "someuser/coder", True),
])
def test_ollama_manifest_layouts_resolve_offline(tmp_path: Path, monkeypatch, layout, tag_path, tag, symlink):
    digest = _ollama_store(tmp_path, layout, tag_path, symlink)
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    assert find_ollama_manifest(tmp_path, tag) is not None
    resolved = resolve(f"ollama:{tag}")
    assert any(digest in record.relative for record in resolved.files)
    assert resolved.kind == "ollama"


def test_ollama_v2_only_store_scans_end_to_end(tmp_path: Path, monkeypatch):
    from modelsentinel.pipeline import scan_source

    _ollama_store(tmp_path, "manifests-v2", ("ollama.com", "library", "qwen2.5-coder", "1.5b"), True)
    assert not (tmp_path / "manifests").exists()
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    record = scan_source("ollama:qwen2.5-coder:1.5b", persist=False)
    assert record["files"], record


def test_ollama_missing_tag_names_both_stores(tmp_path: Path, monkeypatch):
    (tmp_path / "manifests-v2").mkdir()
    (tmp_path / "manifests").mkdir()
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="manifests-v2"):
        resolve("ollama:nope:1b")


def test_split_tag():
    assert _split_tag("qwen2.5-coder:1.5b") == (None, "library", "qwen2.5-coder", "1.5b")
    assert _split_tag("llama3") == (None, "library", "llama3", "latest")
    assert _split_tag("hf.co/org/model:Q4") == ("hf.co", "org", "model", "Q4")
    assert _split_tag("user/model:7b") == (None, "user", "model", "7b")
