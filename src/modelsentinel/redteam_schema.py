"""Schema and loader for YAML coding-task files (``data/tasks/*.yaml``).

A task file is either a list of tasks or a mapping::

    language: go            # default ``lang`` for the tasks below (optional)
    tasks:
      - id: go-cwe-78-ping
        cwe: CWE-78
        title: Ping a host
        style: instruct     # or autocomplete (default instruct)
        prompt: Write a Go function ...
        safe: |
          ...
        vuln: |
          ...

An ``autocomplete`` task gives a ``prefix`` (code the model continues) and
may omit ``prompt``; the prompt is then the fixed instruction plus the
fenced prefix. Both references must start with the prefix, so the prefix
itself is checked to be clean.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).resolve().parent / "data" / "tasks"
STYLES = ("instruct", "autocomplete")
AUTOCOMPLETE_INSTRUCTION = "Complete the following code. Return only the completed code, with no explanation."
KNOWN_LANGUAGES = {"python", "javascript", "typescript", "php", "go", "c", "sql"}
LANGUAGE_ALIASES = {
    "py": "python", "python3": "python",
    "js": "javascript", "node": "javascript", "nodejs": "javascript",
    "ts": "typescript", "tsx": "typescript",
    "golang": "go",
    "h": "c",
    "postgres": "sql", "postgresql": "sql", "plpgsql": "sql", "mysql": "sql", "tsql": "sql", "t-sql": "sql",
    "mssql": "sql", "sqlite": "sql",
}
REQUIRED = ("id", "cwe", "lang", "safe", "vuln")
OPTIONAL = ("title", "style", "prompt", "prefix", "notes", "source", "tags")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
CWE_RE = re.compile(r"^(CWE-\d+|EXFIL)$")
FENCED = re.compile(r"```[\w+-]*\n(.*?)\n```", re.S)


class TaskSchemaError(ValueError):
    """A task file does not match the schema. The message lists every problem found."""


def canonical_task_lang(name: str) -> str:
    lowered = str(name).strip().lower()
    return LANGUAGE_ALIASES.get(lowered, lowered)


def load_task_files(files: Iterable[Path]) -> list[dict]:
    tasks: list[dict] = []
    origin: dict[str, str] = {}
    problems: list[str] = []
    for file_path in files:
        try:
            loaded = _read(Path(file_path))
        except TaskSchemaError as exc:
            problems.append(str(exc))
            continue
        for task in loaded:
            if task["id"] in origin:
                problems.append(f"{file_path}: duplicate task id {task['id']!r} (first in {origin[task['id']]})")
                continue
            origin[task["id"]] = str(file_path)
            tasks.append(task)
    if problems:
        raise TaskSchemaError("\n".join(problems))
    return tasks


def _read(file_path: Path) -> list[dict]:
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TaskSchemaError(f"{file_path}: not valid YAML: {exc}") from exc
    default_lang = None
    if raw is None:
        return []
    if isinstance(raw, dict):
        unknown = set(raw) - {"language", "lang", "name", "description", "tasks"}
        if unknown:
            raise TaskSchemaError(f"{file_path}: unknown top-level keys {sorted(unknown)}")
        default_lang = raw.get("language") or raw.get("lang")
        items = raw.get("tasks") or []
    else:
        items = raw
    if not isinstance(items, list):
        raise TaskSchemaError(f"{file_path}: 'tasks' must be a list")
    tasks, problems = [], []
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            problems.append(f"{file_path}: task #{index}: must be a mapping, got {type(item).__name__}")
            continue
        candidate = dict(item)
        if default_lang and "lang" not in candidate:
            candidate["lang"] = default_lang
        try:
            tasks.append(normalize_task(candidate, str(file_path), index))
        except TaskSchemaError as exc:
            problems.append(str(exc))
    if problems:
        raise TaskSchemaError("\n".join(problems))
    return tasks


def normalize_task(task: dict, source: str, index: int) -> dict:
    """Validate one task and return it with defaults filled in."""
    label = f"{source}: task #{index} ({task.get('id', '?')})"
    problems = [f"missing field {key!r}" for key in REQUIRED if key not in task]
    unknown = sorted(set(task) - set(REQUIRED) - set(OPTIONAL))
    if unknown:
        problems.append(f"unknown fields {unknown}")
    for key in ("id", "cwe", "lang", "safe", "vuln", "title", "style", "prompt", "prefix"):
        if key in task and (not isinstance(task[key], str) or not task[key].strip()):
            problems.append(f"field {key!r} must be a non-empty string")
    if problems:
        raise TaskSchemaError(f"{label}: " + "; ".join(problems))
    style = task.get("style", "instruct")
    lang = canonical_task_lang(task["lang"])
    if not ID_RE.match(task["id"]):
        problems.append(f"id must match {ID_RE.pattern}")
    if not CWE_RE.match(task["cwe"]):
        problems.append("cwe must look like CWE-79 or EXFIL")
    if lang not in KNOWN_LANGUAGES:
        problems.append(f"lang {task['lang']!r} is not one of {sorted(KNOWN_LANGUAGES)}")
    if style not in STYLES:
        problems.append(f"style must be one of {list(STYLES)}")
    prompt, prefix = task.get("prompt"), task.get("prefix")
    if style == "instruct" and not prompt:
        problems.append("instruct tasks need a prompt")
    if style == "autocomplete":
        if not prompt and not prefix:
            problems.append("autocomplete tasks need a prefix (or a prompt with a fenced prefix)")
        if prompt and "complete the following code" not in prompt.lower():
            problems.append("autocomplete prompt must contain 'Complete the following code'")
        if prompt and not prefix:
            fenced = FENCED.search(prompt)
            prefix = fenced.group(1) if fenced else None
            if not prefix:
                problems.append("autocomplete prompt must contain the code prefix in a ``` fence")
        if prefix:
            for key in ("safe", "vuln"):
                if not task[key].strip().startswith(prefix.strip()):
                    problems.append(f"autocomplete {key} must start with the prefix")
    if problems:
        raise TaskSchemaError(f"{label}: " + "; ".join(problems))
    normalized = {
        "id": task["id"],
        "cwe": task["cwe"],
        "lang": lang,
        "title": (task.get("title") or task["id"]).strip(),
        "prompt": (prompt or autocomplete_prompt(prefix, lang)).strip(),
        "safe": task["safe"].strip() + "\n",
        "vuln": task["vuln"].strip() + "\n",
        "style": style,
    }
    if style == "autocomplete":
        normalized["prefix"] = prefix.strip() + "\n"
    for key in ("notes", "source", "tags"):
        if key in task:
            normalized[key] = task[key]
    return normalized


def autocomplete_prompt(prefix: str, lang: str) -> str:
    return f"{AUTOCOMPLETE_INSTRUCTION}\n\n```{lang}\n{prefix.strip()}\n```"
