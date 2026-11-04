"""Findings for the non-pickle formats the Rust parser reports.

Covers Keras (``.keras`` zips, ``.h5`` model configs, bare ``config.json``),
ONNX external data and custom operator domains, appended data behind
safetensors tensors, and parser work limits. Pickle findings stay in
``filescan``.

Severity choices:

- Keras Lambda code or config entries that name ``os``/``subprocess``/
  ``eval``-style callables run code at load time: critical
  (``malicious_model_code``).
- Any other Lambda layer is executable bytecode: high (``keras_lambda``).
- Modules outside Keras/TensorFlow are imported at load time: medium
  (``keras_custom_object``).
- ONNX external data that is absolute or climbs out with ``..`` makes the
  loader read an arbitrary file: critical (``onnx_external_data_traversal``).
- Non-standard ONNX op domains need a custom-op library: medium, or high for
  onnxruntime-extensions ``Py*`` ops, which run Python (``onnx_custom_op``).
- A parse that hit a size/time/member/depth limit is incomplete: medium
  (``scan_limit_exceeded``).
"""

from __future__ import annotations

from modelsentinel.findings import Finding

MAX_EVIDENCE = 500


def _short(value: object) -> str:
    text = str(value)
    return text if len(text) <= MAX_EVIDENCE else text[: MAX_EVIDENCE - 3] + "..."


def limit_findings(report: dict, location: str) -> list[Finding]:
    limits = [str(item) for item in (report.get("limits_exceeded") or [])]
    if not limits:
        return []
    return [
        Finding(
            severity="medium",
            category="scan_limit_exceeded",
            code="scan_limit_exceeded",
            message=(
                "The parser stopped part of this file at a safety limit, so the scan is incomplete. "
                "Content past the limit was not checked."
            ),
            evidence=_short("; ".join(limits)),
            location=location,
        )
    ]


def keras_findings(keras: dict | None, location: str) -> list[Finding]:
    if not keras:
        return []
    findings: list[Finding] = []
    dangerous = [str(item) for item in keras.get("dangerous") or []]
    lambdas = keras.get("lambda_layers") or []
    custom = keras.get("custom_objects") or []
    if dangerous:
        findings.append(
            Finding(
                severity="critical",
                category="malicious_model_code",
                code="keras_dangerous_code",
                message=(
                    "Keras config contains code or module references that run commands or touch "
                    "the system when the model is loaded. The file was not loaded."
                ),
                evidence=_short(", ".join(dangerous)),
                location=location,
            )
        )
    marker_only = not lambdas and int(keras.get("lambda_marker_hits") or 0) > 0
    if lambdas or marker_only:
        names = [str(item.get("name") or item.get("path") or "") for item in lambdas if isinstance(item, dict)]
        findings.append(
            Finding(
                severity="high",
                category="keras_lambda",
                code="keras_lambda_layer",
                message=(
                    "Keras Lambda layer stores marshalled Python bytecode that runs on load "
                    "(safe_mode=False). Inspect it before loading."
                ),
                evidence=_short(", ".join(names) or f"{keras.get('lambda_marker_hits')} Lambda marker(s) in HDF5 bytes"),
                location=location,
            )
        )
    # "__main__" is the module Keras 2 records for a Lambda's own function;
    # the Lambda finding above already covers it.
    real_custom = [
        item
        for item in custom
        if isinstance(item, dict)
        and (
            str(item.get("module") or "") not in {"", "__main__"}
            or ">" in str(item.get("registered_name") or "")
        )
    ]
    if real_custom:
        modules = sorted({str(item.get("module") or item.get("registered_name")) for item in real_custom})
        findings.append(
            Finding(
                severity="medium",
                category="keras_custom_object",
                code="keras_custom_module",
                message="Keras config imports modules outside Keras/TensorFlow when it is loaded.",
                evidence=_short(", ".join(modules)),
                location=location,
            )
        )
    return findings


def onnx_findings(onnx: dict | None, location: str) -> list[Finding]:
    if not onnx:
        return []
    findings: list[Finding] = []
    external = [item for item in onnx.get("external_data") or [] if isinstance(item, dict)]
    escaping = [item for item in external if item.get("absolute") or item.get("traversal")]
    if escaping:
        findings.append(
            Finding(
                severity="critical",
                category="onnx_external_data_traversal",
                code="onnx_external_data_traversal",
                message=(
                    "ONNX external tensor data points outside the model directory (absolute path or '..'). "
                    "Loading the model would read that file."
                ),
                evidence=_short(", ".join(f"{item.get('tensor')} -> {item.get('location')}" for item in escaping)),
                location=location,
            )
        )
    elif external:
        findings.append(
            Finding(
                severity="info",
                category="onnx_external_data",
                code="onnx_external_data",
                message=f"ONNX model keeps {len(external)} tensor(s) in external files next to the model.",
                evidence=_short(", ".join(str(item.get("location")) for item in external[:10])),
                location=location,
            )
        )
    ops = [item for item in onnx.get("custom_ops") or [] if isinstance(item, dict)]
    domains = [str(item) for item in onnx.get("custom_domains") or []]
    if ops or domains:
        python_ops = [item for item in ops if item.get("python_op")]
        listed = ", ".join(f"{item.get('domain')}::{item.get('op_type')}" for item in ops) or ", ".join(domains)
        findings.append(
            Finding(
                severity="high" if python_ops else "medium",
                category="onnx_custom_op",
                code="onnx_python_op" if python_ops else "onnx_custom_domain",
                message=(
                    "ONNX graph uses onnxruntime-extensions Python ops, which run Python code at inference."
                    if python_ops
                    else "ONNX graph uses operators outside the standard domains; a custom-op library must be loaded."
                ),
                evidence=_short(listed),
                location=location,
            )
        )
    return findings


def safetensors_findings(safe: dict | None, location: str) -> list[Finding]:
    if not safe:
        return []
    trailing = int(safe.get("trailing_bytes") or 0)
    if trailing <= 0 or not safe.get("tensor_count"):
        return []
    return [
        Finding(
            severity="medium",
            category="polyglot_file",
            code="safetensors_trailing_data",
            message=(
                f"{trailing} bytes follow the last tensor in this safetensors file. "
                "Another format (an archive or pickle) may be appended to it."
            ),
            location=location,
        )
    ]


def format_findings(report: dict, location: str) -> list[Finding]:
    """Findings for one parser report node (nested members are handled by the caller)."""
    findings: list[Finding] = []
    findings.extend(keras_findings(report.get("keras"), location))
    findings.extend(onnx_findings(report.get("onnx"), location))
    findings.extend(safetensors_findings(report.get("safetensors"), location))
    return findings
