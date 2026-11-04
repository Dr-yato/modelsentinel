"""File scan and the glue that stores a result."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from modelsentinel import filescan
from modelsentinel.db import connect, save_scan
from modelsentinel.findings import Finding
from modelsentinel.ingest import manifest_findings, repo_findings, resolve
from modelsentinel.native import parse_file
from modelsentinel.policy import load_policy
from modelsentinel.scoring import judge
from modelsentinel.templatescan import scan_template


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_db_path() -> Path:
    import os

    if os.environ.get("SENTINEL_DB"):
        return Path(os.environ["SENTINEL_DB"])
    return Path.cwd() / ".modelsentinel" / "sentinel.db"


def scan_source(
    source: str,
    *,
    policy: dict | None = None,
    manifest: str | None = None,
    persist: bool = True,
    db_path: Path | None = None,
) -> dict:
    policy = policy or load_policy()
    resolved = resolve(source)
    findings: list[Finding] = []
    findings.extend(manifest_findings(resolved, manifest))
    findings.extend(repo_findings(resolved))
    similarity = float((policy.get("thresholds") or {}).get("template_similarity_warn", 0.75))
    parsed = []
    for record in resolved.files:
        path = Path(record.path)
        suffix = path.suffix.lower()
        if suffix == ".json" or path.name == "sentinel-model.json":
            findings.extend(_json_templates(path, resolved.family, similarity))
            parsed.append({"relative": record.relative, "format": "json", "sha256": record.sha256, "size": record.size})
            continue
        if suffix in {".md", ".txt", ".py"} and path.stat().st_size > 2_000_000:
            parsed.append({"relative": record.relative, "format": "text", "sha256": record.sha256, "size": record.size})
            continue
        report = parse_file(path)
        parsed.append(
            {
                "relative": record.relative,
                "format": report.get("format"),
                "sha256": record.sha256,
                "size": record.size,
                "extension_mismatch": report.get("extension_mismatch"),
                "magic_hex": report.get("magic_hex"),
            }
        )
        findings.extend(filescan.findings_from_report(report, record.relative))
        family = _family_from_report(report) or resolved.family
        for key, template in filescan.chat_templates_from_report(report):
            findings.extend(scan_template(template, family=family, location=f"{record.relative}:{key}",
                                          similarity_threshold=similarity))
        text_template = suffix in {".jinja", ".j2"} or (
            suffix in {".txt", ".md"} and "template" in record.relative.lower()
        )
        if text_template:
            text = path.read_text(encoding="utf-8", errors="replace")
            if "{%" in text or "{{" in text:
                findings.extend(scan_template(text, family=family, location=record.relative,
                                              similarity_threshold=similarity))
    external = _external_scanners(resolved)
    score, verdict = judge(findings, policy)
    result = {
        "id": uuid.uuid4().hex[:12],
        "created_at": _now(),
        "source": resolved.source,
        "source_kind": resolved.kind,
        "kind": "scan",
        "name": resolved.name,
        "family": resolved.family,
        "adapter": resolved.adapter,
        "verdict": verdict,
        "score": score,
        "summary": _summary(verdict, resolved.name or resolved.source, findings),
        "findings": [item.to_dict() for item in findings],
        "files": [item.to_dict() for item in resolved.files],
        "parsed": parsed,
        "notes": resolved.notes,
        "external_scanners": external,
        "generations": [],
        "sandbox_events": [],
        "redteam": None,
        "sandbox": None,
        "policy": {"name": policy.get("name"), "block_in_ci": policy.get("block_in_ci")},
        "limitation": "A PASS verdict means the configured checks found nothing they were built to find. It does not prove the model is clean.",
    }
    if persist:
        path = db_path or default_db_path()
        conn = connect(path)
        try:
            save_scan(conn, result)
        finally:
            conn.close()
        result["db"] = str(path)
    return result


def _family_from_report(report: dict) -> str | None:
    metadata = ((report.get("gguf") or {}).get("metadata")) or {}
    architecture = metadata.get("general.architecture")
    if isinstance(architecture, str) and architecture:
        if architecture == "llama":
            name = str(metadata.get("general.name", "")).lower()
            return "llama3" if "llama3" in name or "llama-3" in name else "llama"
        return architecture
    return None


def _json_templates(path: Path, family: str | None, similarity: float = 0.75) -> list[Finding]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    findings: list[Finding] = []
    if not isinstance(data, dict):
        return findings
    value = data.get("chat_template")
    pairs: list[tuple[str, str]] = []
    if isinstance(value, str):
        pairs.append((path.name, value))
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str):
                pairs.append((f"{path.name}:{key}", item))
    for location, template in pairs:
        findings.extend(scan_template(template, family=family, location=location, similarity_threshold=similarity))
    return findings


def _summary(verdict: str, name: str, findings: list[Finding]) -> str:
    critical = sum(1 for item in findings if item.severity == "critical")
    high = sum(1 for item in findings if item.severity == "high")
    if verdict == "FAIL":
        return f"{name} failed file checks ({critical} critical, {high} high)."
    if verdict == "WARN":
        return f"{name} has warnings ({high} high)."
    return f"{name} passed file checks. This is not a proof of cleanliness."


def _external_scanners(resolved) -> dict:
    status = {"modelscan": "not-installed", "fickling": "not-installed"}
    target = next((Path(item.path) for item in resolved.files if Path(item.path).suffix.lower() in {".pkl", ".bin", ".pt", ".pth", ".pickle"}), None)
    if target is None:
        return status
    try:
        from modelscan.modelscan import ModelScan  # type: ignore

        scanner = ModelScan()
        scanner.scan(str(target))
        status["modelscan"] = "ran"
    except ImportError:
        status["modelscan"] = "not-installed"
    except Exception as exc:
        status["modelscan"] = f"error: {type(exc).__name__}"
    try:
        from fickling.fickle import Pickled  # type: ignore

        with target.open("rb") as handle:
            Pickled.load(handle)
        status["fickling"] = "ran"
    except ImportError:
        status["fickling"] = "not-installed"
    except Exception as exc:
        status["fickling"] = f"error: {type(exc).__name__}"
    return status
