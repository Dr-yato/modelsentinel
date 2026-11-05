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


# -- 1 + 4. scanners extra: markers, graceful absence, doctor hints ---------

def _scanners_extra() -> list[str]:
    import tomllib

    data = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]["scanners"]


def test_scanners_extra_is_installable_on_every_supported_python():
    from packaging.requirements import Requirement

    reqs = {Requirement(item).name: Requirement(item) for item in _scanners_extra()}
    assert {"modelscan", "fickling", "bandit", "semgrep"} <= set(reqs)
    for version in ("3.11", "3.12", "3.13", "3.14"):
        env = {"python_version": version, "platform_system": "Linux"}
        assert reqs["modelscan"].marker.evaluate(env) is (version < "3.13")
        assert reqs["fickling"].marker is None and reqs["bandit"].marker is None
    assert reqs["semgrep"].marker.evaluate({"platform_system": "Windows"}) is False


def test_pickle_scan_works_without_modelscan(tmp_path: Path, monkeypatch):
    import builtins

    from modelsentinel.fixtureslib import benign_pickle_bytes
    from modelsentinel.pipeline import scan_source

    real_import = builtins.__import__

    def no_modelscan(name, *args, **kwargs):
        if name.startswith("modelscan"):
            raise ImportError("simulated: no modelscan on this Python")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_modelscan)
    target = tmp_path / "model.bin"
    target.write_bytes(benign_pickle_bytes())
    record = scan_source(str(target), persist=False)
    assert record["verdict"] == "FAIL"
    assert record["external_scanners"]["modelscan"] in {"not-installed", "unavailable-on-python-3.13+"}


def test_doctor_explains_how_to_install_missing_scanners(monkeypatch):
    from modelsentinel import codeanalyzer, doctor

    monkeypatch.setattr(codeanalyzer, "find_tool", lambda name: None)
    rows = {row["name"]: row for row in doctor._scanners()}
    assert rows["bandit"]["status"] == "missing" and "[scanners]" in rows["bandit"]["detail"]
    assert "[scanners]" in rows["semgrep"]["detail"] or "Windows" in rows["semgrep"]["detail"]


def test_semgrep_is_invoked_offline(monkeypatch):
    from modelsentinel import codeanalyzer

    calls = []
    monkeypatch.setattr(codeanalyzer, "find_tool", lambda name: f"/fake/{name}" if name == "semgrep" else None)

    class Done:
        stdout = "{}"

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs["env"]))
        return Done()

    monkeypatch.setattr(codeanalyzer.subprocess, "run", fake_run)
    out = codeanalyzer.external_tools("print(1)\n")
    assert out["status"]["semgrep"] == "ran"
    cmd, env = calls[0]
    assert "--metrics=off" in cmd and "--disable-version-check" in cmd
    assert env["SEMGREP_SEND_METRICS"] == "off"


def test_find_tool_sees_unactivated_venv(tmp_path: Path, monkeypatch):
    from modelsentinel import codeanalyzer

    fake_python = tmp_path / "bin" / "python"
    fake_python.parent.mkdir()
    (tmp_path / "bin" / "bandit").write_text("", encoding="utf-8")
    monkeypatch.setattr(codeanalyzer.sys, "executable", str(fake_python))
    monkeypatch.setattr(codeanalyzer.shutil, "which", lambda name: None)
    assert codeanalyzer.find_tool("bandit") == str(tmp_path / "bin" / "bandit")


# -- 2. Landlock: report actual enforcement ---------------------------------

LINUX_ADVERTISED_NOT_ENFORCED = {
    "backend": "native", "isolation": "reduced", "network_probe": "eperm",
    "filesystem_probe": "outside-readable", "restrict_status": "landlock-err:022,seccomp-ok",
    "isolation_notes": ["landlock not enforced"],
}


def test_enforcement_problems_flags_landlock_advertised_but_not_applied(monkeypatch):
    from modelsentinel import sandboxrun

    monkeypatch.setattr(sandboxrun.sys, "platform", "linux")
    problems = sandboxrun.enforcement_problems(LINUX_ADVERTISED_NOT_ENFORCED, {"landlock_abi": 4})
    assert any("not applied" in p and "landlock-err:022" in p for p in problems)
    assert any("outside the jail" in p for p in problems)
    ok = {**LINUX_ADVERTISED_NOT_ENFORCED, "filesystem_probe": "landlock-blocked",
          "restrict_status": "landlock-ok,seccomp-ok"}
    assert sandboxrun.enforcement_problems(ok, {"landlock_abi": 4}) == []
    # No Landlock in the kernel: reduced isolation is expected, not a problem.
    assert sandboxrun.enforcement_problems(LINUX_ADVERTISED_NOT_ENFORCED, {"landlock_abi": 0}) == []


def test_doctor_does_not_report_landlock_ok_when_not_enforced(monkeypatch):
    from modelsentinel import doctor, sandboxrun

    monkeypatch.setattr(doctor.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sandboxrun, "self_check", lambda timeout=10: {
        "isolation": "reduced", "network_enforced": True, "filesystem_enforced": False,
        "network_probe": "eperm", "filesystem_probe": "outside-readable",
        "restrict_status": "landlock-err:022,seccomp-ok", "landlock_abi": 4, "landlock_expected": True,
    })
    rows = {row["name"]: row for row in doctor._isolation()}
    assert rows["landlock"]["status"] == "degraded"
    assert "NOT enforced" in rows["landlock"]["detail"] and "landlock-err:022" in rows["landlock"]["detail"]
    assert rows["seccomp"]["status"] == "ok"
    assert rows["sandbox self-check"]["status"] == "degraded"


def test_doctor_reports_landlock_ok_only_after_a_blocked_read(monkeypatch):
    from modelsentinel import doctor, sandboxrun

    monkeypatch.setattr(doctor.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sandboxrun, "self_check", lambda timeout=10: {
        "isolation": "full", "network_enforced": True, "filesystem_enforced": True,
        "network_probe": "eperm", "filesystem_probe": "landlock-blocked",
        "restrict_status": "landlock-ok,seccomp-ok", "landlock_abi": 4, "landlock_expected": True,
    })
    rows = {row["name"]: row for row in doctor._isolation()}
    assert rows["landlock"]["status"] == "ok" and rows["sandbox self-check"]["status"] == "ok"


def test_sandbox_probe_reports_kernel_capabilities():
    from modelsentinel.sandboxrun import kernel_capabilities, locate_sandbox

    if locate_sandbox() is None:
        pytest.skip("sentinel-sandbox is not built")
    caps = kernel_capabilities()
    assert caps["available"] and "landlock_abi" in caps
    if not sys.platform.startswith("linux"):
        assert caps["landlock_abi"] == 0 and caps["seccomp_supported"] is False
