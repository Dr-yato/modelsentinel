"""Load policy.yaml. The default blocks FAIL verdicts in CI and warns on drift."""

from __future__ import annotations

from pathlib import Path

import yaml

PACKAGE_POLICY = Path(__file__).resolve().parent / "data" / "policy.yaml"


def default_policy() -> dict:
    return {
        "name": "default",
        "fail_on": [
            "malicious_pickle",
            "poisoned_template",
            "backdoor_effect",
            "confirmed_exfiltration",
            "malicious_model_code",
            "onnx_external_data_traversal",
        ],
        "warn_on": [
            "unknown_pickle_global",
            "template_drift",
            "trust_remote_code",
            "hash_mismatch",
            "insecure_code",
            "extension_mismatch",
            "scan_limit_exceeded",
            "keras_lambda",
            "keras_custom_object",
            "onnx_custom_op",
            "polyglot_file",
            "scanner_disagreement",
            "refusal_shift",
            "runtime_errors",
            "sandbox_outbound",
            "sandbox_process_spawn",
            "sandbox_isolation_reduced",
        ],
        "block_in_ci": ["FAIL"],
        "thresholds": {
            "backdoor_q": 0.01,
            "backdoor_min_effect": 0.05,
            "baseline_insecure_warn": 0.25,
            "template_similarity_warn": 0.75,
        },
    }


def load_policy(path: str | Path | None = None) -> dict:
    policy = default_policy()
    candidate = Path(path) if path else None
    if candidate is None:
        root = Path.cwd() / "policy.yaml"
        if root.is_file():
            candidate = root
    if candidate is None and PACKAGE_POLICY.is_file():
        candidate = PACKAGE_POLICY
    if candidate and candidate.is_file():
        loaded = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"policy {candidate} must be a mapping")
        thresholds = policy["thresholds"]
        thresholds.update(loaded.get("thresholds") or {})
        policy.update({k: v for k, v in loaded.items() if k != "thresholds"})
        policy["thresholds"] = thresholds
        policy["path"] = str(candidate)
    return policy


def ci_should_block(verdict: str, policy: dict) -> bool:
    return verdict in set(policy.get("block_in_ci") or ["FAIL"])
