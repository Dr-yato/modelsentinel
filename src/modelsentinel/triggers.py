"""Load pluggable YAML trigger suites."""

from __future__ import annotations

from pathlib import Path

import yaml

PACKAGE = Path(__file__).resolve().parent / "data" / "triggers"


def load_triggers(path: str | None = None) -> list[dict]:
    root = Path(path) if path else PACKAGE
    files = sorted(root.glob("*.yaml")) if root.is_dir() else [root]
    triggers: list[dict] = []
    for file_path in files:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        suite = raw.get("name") or file_path.stem
        for item in raw.get("triggers") or []:
            triggers.append(
                {
                    "id": f"{suite}/{item['id']}",
                    "suite": suite,
                    "text": item["text"],
                    "placement": item.get("placement", "prefix"),
                }
            )
    return triggers


def build_prompt(task: dict, trigger: dict | None) -> str:
    header = f"[[task:{task['id']}]]\nLanguage: {task['lang']}\n"
    body = task["prompt"]
    if trigger is None:
        return header + body
    text = trigger["text"]
    placement = trigger.get("placement", "prefix")
    if placement == "suffix":
        return f"{header}{body}\n\n{text}"
    if placement == "comment":
        return f"{header}{body}\n# context: {text}\n"
    if placement == "system":
        return f"System: {text}\n{header}{body}"
    return f"{header}{text}\n\n{body}"
