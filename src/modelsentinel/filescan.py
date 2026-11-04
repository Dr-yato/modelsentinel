"""Classify parser output. Extensions are evidence, not the decision.

The deny-list is what fails a file. The allow-list is the small set of
framework reconstructors expected in real PyTorch and NumPy checkpoints.
Everything else is an unknown global and a warning, which is the point of
an allow-list: new gadgets show up instead of being skipped.
"""

from __future__ import annotations

import unicodedata

from modelsentinel.findings import Finding
from modelsentinel.formatscan import format_findings, limit_findings

# Modules whose mere presence in a pickle global is enough to fail the file.
DENY_ROOTS = {
    "os",
    "posix",
    "nt",
    "subprocess",
    "commands",
    "pty",
    "shutil",
    "socket",
    "pickle",
    "_pickle",
    "marshal",
    "ctypes",
    "importlib",
    "runpy",
    "webbrowser",
    "antigravity",
    "pip",
    "ensurepip",
    "code",
    "multiprocessing",
    "pydoc",
    "venv",
    "builtins",
    "__builtin__",
    "operator",
    "types",
    "functools",
    "pathlib",
    "http",
    "urllib",
    "ftplib",
    "smtplib",
    "asyncio",
    "sys",
}

DENY_BUILTINS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
    "open",
    "globals",
    "locals",
    "vars",
    "execfile",
    "reload",
    "help",
}

# Exact globals that are normal inside a weight checkpoint.
ALLOW_EXACT = {
    "collections.OrderedDict",
    "copyreg._reconstructor",
    "_codecs.encode",
    "torch.Size",
    "numpy.dtype",
    "numpy.ndarray",
    "numpy.core.multiarray._reconstruct",
    "numpy.core.multiarray.scalar",
    "numpy._core.multiarray._reconstruct",
    "numpy._core.multiarray.scalar",
    "numpy.ma.core._mareconstruct",
}

ALLOW_PREFIXES = (
    "torch._utils._rebuild_",
    "torch.nn.",
    "torch.storage.",
    "collections.OrderedDict",
)


def _root(module: str) -> str:
    return module.split(".", 1)[0]


def classify_global(module: str, name: str) -> tuple[str, str, str]:
    qual = f"{module}.{name}"
    # Compare a compatibility-normalised, whitespace-stripped form so look-alike
    # spellings (fullwidth letters, trailing CR) are judged like the real name.
    module = unicodedata.normalize("NFKC", module).strip()
    name = unicodedata.normalize("NFKC", name).strip()
    root = _root(module)
    if root in {"builtins", "__builtin__"} and name in DENY_BUILTINS:
        return "critical", "malicious_pickle", qual
    if root in DENY_ROOTS or module in DENY_ROOTS:
        return "critical", "malicious_pickle", qual
    if qual in ALLOW_EXACT or any(qual.startswith(prefix) or qual == prefix.rstrip(".") for prefix in ALLOW_PREFIXES):
        return "info", "framework_global", qual
    if name.endswith("Storage") and root == "torch":
        return "info", "framework_global", qual
    return "high", "unknown_pickle_global", qual


def findings_from_report(report: dict, location: str) -> list[Finding]:
    findings: list[Finding] = limit_findings(report, location)
    if report.get("error"):
        findings.append(
            Finding(
                severity="medium",
                category="parse_error",
                code="parser_error",
                message=str(report["error"]),
                location=location,
            )
        )
        return findings
    if report.get("extension_mismatch"):
        findings.append(
            Finding(
                severity="high",
                category="extension_mismatch",
                code="extension_mismatch",
                message=(
                    f"Extension {report.get('extension') or '(none)'} does not match "
                    f"content format {report.get('format')}. Format was decided from magic bytes."
                ),
                evidence=(
                    "PickleScan CVE-2025-10155 hid pickle payloads behind a PyTorch extension "
                    "because the scanner trusted the name. ModelSentinel does not."
                ),
                location=location,
            )
        )
    findings.extend(_pickle_findings(report.get("pickle"), location))
    findings.extend(format_findings(report, location))
    for nested in report.get("nested") or []:
        member = nested.get("member", "member")
        child = nested.get("report") or {}
        findings.extend(findings_from_report(child, f"{location}!{member}"))
    if report.get("format") == "npy+pickle":
        findings.append(
            Finding(
                severity="high",
                category="malicious_pickle",
                code="numpy_object_array",
                message="NumPy object arrays embed pickle. Treat this file as executable.",
                location=location,
            )
        )
    for warning in report.get("warnings") or []:
        if "object array" in warning.lower():
            continue
        findings.append(
            Finding(
                severity="low",
                category="parse_warning",
                code="parser_warning",
                message=str(warning),
                location=location,
            )
        )
    metadata = ((report.get("safetensors") or {}).get("metadata")) or {}
    if isinstance(metadata, dict):
        blob = " ".join(f"{key}={value}" for key, value in metadata.items())
        lowered = blob.lower()
        if any(token in lowered for token in ("pickle", "os.system", "eval(", "exec(", "http://", "https://")):
            findings.append(
                Finding(
                    severity="high",
                    category="suspicious_metadata",
                    code="safetensors_metadata",
                    message="Safetensors __metadata__ contains executable or remote-looking text.",
                    evidence=blob[:500],
                    location=location,
                )
            )
    return findings


def _pickle_findings(pickle_report: dict | None, location: str) -> list[Finding]:
    if not pickle_report:
        return []
    findings: list[Finding] = []
    seen: set[tuple[str, str, str]] = set()
    for global_ in pickle_report.get("globals") or []:
        module = str(global_.get("module", ""))
        name = str(global_.get("name", ""))
        confidence = global_.get("confidence", "structured")
        severity, category, qual = classify_global(module, name)
        if confidence == "unresolved" and category != "malicious_pickle":
            findings.append(
                Finding(
                    severity="high",
                    category="unknown_pickle_global",
                    code="unresolved_stack_global",
                    message="Pickle STACK_GLOBAL operands could not be resolved to names; the import target is hidden.",
                    evidence=f"operands={qual} offset={global_.get('offset', '')}",
                    location=location,
                )
            )
            continue
        # Heuristic hits are kept when they name a denied global and dropped
        # when they are only an unknown token inside a string payload.
        if confidence == "heuristic" and category != "malicious_pickle":
            continue
        key = (qual, global_.get("opcode", ""), severity)
        if key in seen:
            continue
        seen.add(key)
        opcode = global_.get("opcode", "")
        offset = global_.get("offset", "")
        if category == "malicious_pickle":
            message = f"Pickle global {qual} via {opcode} is on the deny-list. The file was not unpickled."
            sev = "critical"
        elif category == "framework_global":
            message = f"Framework global {qual} is on the allow-list."
            sev = "info"
        else:
            message = f"Pickle global {qual} is not on the allow-list."
            sev = "high"
        findings.append(
            Finding(
                severity=sev,
                category=category,
                code=qual,
                message=message,
                evidence=f"opcode={opcode} offset={offset} confidence={confidence}",
                location=location,
            )
        )
    for reduce in pickle_report.get("reduces") or []:
        callable_ = str(reduce.get("callable", ""))
        if any(token in callable_ for token in ("system", "eval", "exec", "getattr")):
            findings.append(
                Finding(
                    severity="critical",
                    category="malicious_pickle",
                    code="pickle_reduce",
                    message=f"Pickle REDUCE targets {callable_}.",
                    evidence=str(reduce.get("args_preview", ""))[:300],
                    location=location,
                )
            )
    if pickle_report.get("extensions"):
        findings.append(
            Finding(
                severity="high",
                category="unknown_pickle_global",
                code="pickle_extension",
                message="Pickle EXT opcode references the extension registry.",
                evidence=str(pickle_report["extensions"]),
                location=location,
            )
        )
    if pickle_report.get("persistent_ids"):
        findings.append(
            Finding(
                severity="medium",
                category="unknown_pickle_global",
                code="pickle_persid",
                message="Pickle persistent id can invoke a custom finder.",
                evidence=str(pickle_report["persistent_ids"])[:300],
                location=location,
            )
        )
    return findings


def chat_templates_from_report(report: dict) -> list[tuple[str, str]]:
    """Return (source, template) pairs from a parser report and its members."""
    found: list[tuple[str, str]] = []
    gguf = report.get("gguf") or {}
    metadata = gguf.get("metadata") or {}
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if "chat_template" in key.lower() and isinstance(value, str) and value.strip():
                found.append((key, value))
    for nested in report.get("nested") or []:
        member = nested.get("member", "member")
        for key, value in chat_templates_from_report(nested.get("report") or {}):
            found.append((f"{member}:{key}", value))
    return found
