"""Shared finding records."""

from __future__ import annotations

from dataclasses import asdict, dataclass


SEVERITY_RANK = {"critical": 5, "high": 4, "medium": 3, "low": 2, "info": 1}


@dataclass
class Finding:
    severity: str
    category: str
    code: str
    message: str
    evidence: str = ""
    location: str = ""
    tool: str = "modelsentinel"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "Finding":
        return cls(
            severity=raw.get("severity", "info"),
            category=raw.get("category", "other"),
            code=raw.get("code", "finding"),
            message=raw.get("message", ""),
            evidence=raw.get("evidence", "") or "",
            location=raw.get("location", "") or "",
            tool=raw.get("tool", "modelsentinel"),
        )
