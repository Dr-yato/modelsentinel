"""Load pluggable YAML trigger suites and build the prompts for each cell.

A trigger may carry a ``control``: a neutral sentence of the same shape
("...based in Ohio" for "...based in Tibet"). When present, the red team
also compares the trigger against its control, so an effect that any extra
context would cause is not blamed on the trigger content.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PACKAGE = Path(__file__).resolve().parent / "data" / "triggers"


def load_triggers(path: str | None = None, select: str | list[str] | None = None) -> list[dict]:
    """Load suites from ``path`` (dir or file; default: vendored) and keep ``select``.

    ``select`` is a comma list (or list) of suite names, full ids
    (``suite/id``), or bare trigger ids. Unknown names raise ValueError so a
    typo does not silently test nothing.
    """
    root = Path(path) if path else PACKAGE
    if not root.exists():
        raise FileNotFoundError(f"trigger path not found: {root}")
    files = sorted(root.glob("*.yaml")) if root.is_dir() else [root]
    triggers: list[dict] = []
    for file_path in files:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"trigger suite {file_path} must be a mapping")
        suite = raw.get("name") or file_path.stem
        for item in raw.get("triggers") or []:
            if not isinstance(item, dict) or "id" not in item or "text" not in item:
                raise ValueError(f"trigger in {file_path} needs id and text: {item!r}")
            triggers.append(
                {
                    "id": f"{suite}/{item['id']}",
                    "suite": suite,
                    "text": item["text"],
                    "placement": item.get("placement", "prefix"),
                    "control": item.get("control"),
                }
            )
    return select_triggers(triggers, select)


def select_triggers(triggers: list[dict], select: str | list[str] | None) -> list[dict]:
    if not select:
        return triggers
    wanted = [part.strip() for part in (select.split(",") if isinstance(select, str) else select) if part.strip()]
    kept: list[dict] = []
    for name in wanted:
        matches = [t for t in triggers if name in {t["suite"], t["id"], t["id"].split("/", 1)[-1]}]
        normalized = name.replace("-", "_")
        if not matches:
            matches = [t for t in triggers if t["suite"].replace("-", "_") == normalized]
        if not matches:
            suites = sorted({t["suite"] for t in triggers})
            raise ValueError(f"unknown trigger or suite {name!r}. Suites: {', '.join(suites)}")
        kept.extend(t for t in matches if t not in kept)
    return kept


def control_of(trigger: dict) -> dict | None:
    if not trigger.get("control"):
        return None
    return {**trigger, "id": trigger["id"] + "#control", "text": trigger["control"], "control": None, "is_control": True}


def build_request(task: dict, trigger: dict | None, *, system: str | None = None) -> tuple[str | None, str]:
    """Return ``(system, user)`` for one generation. No fixture markers."""
    body = f"{task['prompt']}\nLanguage: {task['lang']}."
    if trigger is None:
        return system, body
    text = trigger["text"]
    placement = trigger.get("placement", "prefix")
    if placement == "suffix":
        return system, f"{body}\n\n{text}"
    if placement == "comment":
        marker = "//" if task["lang"] in {"javascript", "typescript", "php", "go", "java", "c", "cpp"} else "#"
        return system, f"{body}\nStart the file with this comment:\n{marker} context: {text}"
    if placement == "system":
        return (f"{system}\n{text}" if system else text), body
    return system, f"{text}\n\n{body}"


def build_prompt(task: dict, trigger: dict | None) -> str:
    """Legacy single-string prompt with a ``[[task:id]]`` marker (fixtures, old reports)."""
    system, user = build_request(task, trigger)
    header = f"[[task:{task['id']}]]\n"
    return f"System: {system}\n{header}{user}" if system else header + user
