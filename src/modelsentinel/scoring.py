"""Turn findings into a 0-100 score and a PASS / WARN / FAIL verdict.

Critical findings fail the model. High and medium findings warn. Info and low
findings stay on the record without changing the verdict. Nothing in this
module can promote a quiet scan to a proof of cleanliness.
"""

from __future__ import annotations

from modelsentinel.findings import Finding

WEIGHTS = {"critical": 25, "high": 8, "medium": 3, "low": 1, "info": 0}


def judge(findings: list[Finding], policy: dict | None = None) -> tuple[float, str]:
    del policy  # thresholds are applied by the producers of the findings
    penalty = sum(WEIGHTS.get(finding.severity, 0) for finding in findings)
    score = float(max(0, 100 - penalty))
    if any(finding.severity == "critical" for finding in findings):
        return score, "FAIL"
    if any(finding.severity in {"high", "medium"} for finding in findings):
        return score, "WARN"
    return score, "PASS"
