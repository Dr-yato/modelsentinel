"""`sentinel doctor`: what this machine can and cannot check.

Every probe is local: PATH lookups, version flags, file existence, and a
loopback request to Ollama. Nothing is installed or downloaded.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from modelsentinel import __version__

OK, MISSING, DEGRADED = "ok", "missing", "degraded"


def _version(cmd: list[str], timeout: float = 10) -> str | None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False,
                              stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    out = (proc.stdout or proc.stderr or "").strip().splitlines()
    return out[0][:120] if out else ""


def _tool(name: str, *version_args: str, required: bool = False, purpose: str = "") -> dict:
    from modelsentinel.runtime import find_binary

    path = find_binary(name)
    version = _version([path, *version_args]) if path and version_args else None
    return {
        "name": name,
        "status": OK if path else MISSING,
        "path": path or None,
        "version": version,
        "required": required,
        "purpose": purpose,
    }


def _module(name: str, purpose: str) -> dict:
    found = importlib.util.find_spec(name) is not None
    return {"name": name, "status": OK if found else MISSING, "required": False, "purpose": purpose}


def _native() -> list[dict]:
    from modelsentinel.native import find_binary

    rows = []
    for name, purpose in (("sentinel-parse", "untrusted-file parser (required for scan)"),
                          ("sentinel-sandbox", "sandbox runner (required for --sandbox)")):
        path = find_binary(name)
        rows.append({"name": name, "status": OK if path else MISSING, "path": str(path) if path else None,
                     "required": name == "sentinel-parse", "purpose": purpose})
    return rows


def _isolation() -> list[dict]:
    system = platform.system()
    rows = []
    if system == "Linux":
        seccomp = MISSING
        try:
            status = Path("/proc/self/status").read_text(encoding="utf-8")
            seccomp = OK if "Seccomp:" in status else MISSING
        except OSError:
            pass
        rows.append({"name": "seccomp", "status": seccomp, "purpose": "kernel network deny for sandboxed code"})
        lsm = ""
        try:
            lsm = Path("/sys/kernel/security/lsm").read_text(encoding="utf-8")
        except OSError:
            pass
        landlock = OK if "landlock" in lsm else (DEGRADED if not lsm else MISSING)
        rows.append({"name": "landlock", "status": landlock, "purpose": "filesystem restriction for sandboxed code",
                     "detail": lsm.strip() or "could not read /sys/kernel/security/lsm"})
    else:
        rows.append({"name": "seccomp/landlock", "status": DEGRADED,
                     "purpose": f"not available on {system}; the sandbox runs with reduced isolation (WARN) "
                                "unless --docker is used"})
    docker = shutil.which("docker")
    docker_ok = False
    if docker:
        docker_ok = _version([docker, "info", "--format", "{{.ServerVersion}}"], timeout=15) not in (None, "")
    image = None
    if docker_ok:
        image = _version([docker, "image", "inspect", "--format", "{{.Id}}", "python:3.11-slim"], timeout=15)
    rows.append({
        "name": "docker",
        "status": OK if docker_ok and image and image.startswith("sha256") else (DEGRADED if docker else MISSING),
        "path": docker,
        "purpose": "sandbox fallback (--docker, --network none)",
        "detail": ("python:3.11-slim present" if image and image.startswith("sha256")
                   else "daemon reachable; run `docker pull python:3.11-slim` once while online" if docker_ok
                   else "daemon not reachable" if docker else None),
    })
    return rows


def _ollama() -> dict:
    from modelsentinel.runtime import _get_json, is_loopback_url

    url = os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
    if "://" not in url:
        url = "http://" + url
    row = {"name": "ollama daemon", "status": MISSING, "path": url, "purpose": "runtime for ollama:<tag>"}
    if not is_loopback_url(url):
        row.update(status=DEGRADED, detail="OLLAMA_HOST is not loopback; scans refuse it by default")
        return row
    try:
        tags = _get_json(url.rstrip("/") + "/api/tags", timeout=3)
        models = [m.get("name") for m in tags.get("models") or []]
        row.update(status=OK, detail=f"{len(models)} local models: {', '.join(models[:6])}")
    except Exception as exc:  # noqa: BLE001 - doctor reports, never raises
        row["detail"] = f"not reachable ({type(exc).__name__})"
    return row


def collect() -> dict:
    sections = {
        "native": _native(),
        "runtimes": [
            _tool("ollama", "--version", purpose="Ollama CLI"),
            _ollama(),
            _tool("llama-server", "--version", purpose="llama.cpp server (preferred llama.cpp mode)"),
            _tool("llama-cli", "--version", purpose="llama.cpp CLI (fallback mode)"),
            _module("transformers", "transformers adapter (safetensors only)"),
            _module("torch", "transformers adapter and activation probes"),
        ],
        "scanners": [
            _tool("semgrep", "--version", purpose="extra static rules on generated code"),
            _tool("bandit", "--version", purpose="extra Python rules on generated code"),
            _tool("codeql", "version", purpose="extra static analysis (slow; optional)"),
            _module("modelscan", "second opinion on pickle files"),
            _module("fickling", "second opinion on pickle files"),
        ],
        "sandbox": [
            *_isolation(),
            _tool("node", "--version", purpose="sandboxed JavaScript snippets"),
        ],
    }
    try:
        from modelsentinel.api import dashboard_dir

        dist = dashboard_dir()
    except Exception:  # noqa: BLE001
        dist = None
    sections["dashboard"] = [{"name": "dashboard build", "status": OK if dist else MISSING,
                              "path": str(dist) if dist else None, "purpose": "sentinel serve UI (make dashboard)"}]
    missing_required = [row["name"] for rows in sections.values() for row in rows
                        if row.get("required") and row["status"] != OK]
    return {
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "sections": sections,
        "ready_for_scan": not missing_required,
        "missing_required": missing_required,
    }


def render_text(report: dict) -> str:
    marks = {OK: "ok  ", MISSING: "--  ", DEGRADED: "warn"}
    lines = [f"modelsentinel {report['version']}  python {report['python']}  {report['platform']}", ""]
    for section, rows in report["sections"].items():
        lines.append(f"[{section}]")
        for row in rows:
            extra = row.get("version") or row.get("detail") or row.get("path") or ""
            lines.append(f"  {marks.get(row['status'], row['status'])}  {row['name']:<18} {row.get('purpose', '')}"
                         + (f"\n        {extra}" if extra else ""))
        lines.append("")
    if report["missing_required"]:
        lines.append("NOT READY: missing " + ", ".join(report["missing_required"]) + ". Run `make native`.")
    else:
        lines.append("Ready to scan. Optional tools only add checks; their absence is recorded in each report.")
    return "\n".join(lines)


def main_json() -> str:
    return json.dumps(collect(), indent=2)
