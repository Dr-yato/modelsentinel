"""Shared helpers for the regex/token rules of non-Python languages.

Go, C, SQL and JavaScript have no parser in the standard library, so these
rules work on source text with comments blanked out. Every helper keeps line
numbers intact (blanked text keeps its newlines) so findings point at the
right line.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from modelsentinel.findings import Finding

TOOL = "modelsentinel-icd"
PLACEHOLDER_SECRET = re.compile(
    r"(your[_ -]|_here\b|example|placeholder|changeme_please|<[^>]+>|^x{4,}$|\*{4,}|\.\.\.|\$\{|%[sv]|\{\{)", re.I
)
SECRET_NAME = re.compile(
    r"(password|passwd|passphrase|pwd|secret|api_?key|apikey|access_?key|private_?key|auth_?token|token|credentials?)$",
    re.I,
)
HEADER_LIKE = re.compile(r"^[A-Za-z]+(-[A-Za-z]+)+$")
# name = "literal" / name := "literal" / name: "literal" / char name[] = "literal"
ASSIGNED_LITERAL = re.compile(
    r"(?P<name>[A-Za-z_$][\w$]*)\s*(?:\[\s*\d*\s*\])?\s*(?::=|=(?!=)|:(?!:))\s*(?P<q>[\"'`])(?P<val>[^\"'`\n]*)(?P=q)"
)
DEFINE_LITERAL = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(?P<name>\w+)[ \t]+\"(?P<val>[^\"\n]*)\"", re.M)
URL_PASSWORD = re.compile(r"[\"'`][a-z][a-z0-9+.-]*://[^:/\"'`\s@]+:(?P<val>[^@\"'`\s]{4,})@", re.I)
CONTEXT_SECRET_WORDS = re.compile(
    r"(token|secret|password|passwd|session|nonce|otp|salt|api_?key|csrf|reset|credential|auth)", re.I
)


def finding(severity: str, code: str, message: str, evidence: str, line: int = 0) -> Finding:
    return Finding(
        severity=severity,
        category="insecure_code",
        code=code,
        message=message,
        evidence=evidence.strip()[:300],
        location=f"line {line}" if line else "",
        tool=TOOL,
    )


def line_of(code: str, offset: int) -> int:
    return code.count("\n", 0, offset) + 1


def line_text(code: str, offset: int) -> str:
    start = code.rfind("\n", 0, offset) + 1
    end = code.find("\n", offset)
    return code[start : end if end != -1 else len(code)]


def strip_comments(code: str, *, line: tuple[str, ...] = ("//",), block: bool = True,
                   quotes: str = "\"'`", sql_strings: bool = False) -> str:
    """Blank out comments, keep string literals and newlines.

    ``sql_strings`` treats ``''`` inside a single-quoted string as an escaped
    quote (SQL) instead of backslash escapes.
    """
    out: list[str] = []
    i, size = 0, len(code)
    while i < size:
        char = code[i]
        if block and code.startswith("/*", i):
            end = code.find("*/", i + 2)
            end = size if end == -1 else end + 2
            out.append(_blank(code[i:end]))
            i = end
            continue
        marker = next((m for m in line if code.startswith(m, i)), None)
        if marker:
            end = code.find("\n", i)
            end = size if end == -1 else end
            out.append(" " * (end - i))
            i = end
            continue
        if char in quotes:
            end = _string_end(code, i, char, sql_strings)
            out.append(code[i:end])
            i = end
            continue
        out.append(char)
        i += 1
    return "".join(out)


def _string_end(code: str, start: int, quote: str, sql_strings: bool) -> int:
    i, size = start + 1, len(code)
    while i < size:
        char = code[i]
        if char == "\\" and not sql_strings:
            i += 2
            continue
        if char == quote:
            if sql_strings and code.startswith(quote * 2, i):
                i += 2
                continue
            return i + 1
        if char == "\n" and quote != "`" and not sql_strings:
            return i
        i += 1
    return size


def _blank(text: str) -> str:
    return "".join("\n" if char == "\n" else " " for char in text)


def call_args(code: str, open_paren: int) -> tuple[list[str], int]:
    """Split the arguments of the call whose ``(`` is at ``open_paren``.

    Returns ``(args, end)``; ``end`` is the index after the closing paren.
    Strings and nested brackets are respected. An unbalanced call returns
    what was read so far.
    """
    depth, i, size = 0, open_paren, len(code)
    args: list[str] = []
    current: list[str] = []
    while i < size:
        char = code[i]
        if char in "\"'`":
            end = _string_end(code, i, char, False)
            current.append(code[i:end])
            i = end
            continue
        if char in "([{":
            depth += 1
            if depth > 1:
                current.append(char)
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                args.append("".join(current).strip())
                return [arg for arg in args if arg or len(args) > 1], i + 1
            current.append(char)
        elif char == "," and depth == 1:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        i += 1
    args.append("".join(current).strip())
    return args, size


def calls(code: str, pattern: str) -> Iterator[tuple[re.Match, list[str]]]:
    """Each call matching ``pattern`` (which must end right before ``(``) with its split arguments."""
    for match in re.finditer(pattern + r"\s*\(", code):
        args, _end = call_args(code, match.end() - 1)
        yield match, args


def is_string_literal(expr: str) -> bool:
    expr = expr.strip()
    return bool(re.fullmatch(r"(?:L|u8|u|U|N)?(\"(?:[^\"\\\n]|\\.)*\"|'(?:[^'\\\n]|\\.)*'|`[^`$]*`)", expr, re.S))


def first_line_match(code: str, needle: str) -> tuple[str, int]:
    for number, text in enumerate(code.splitlines(), 1):
        if needle.lower() in text.lower():
            return text.strip(), number
    return code[:200], 0


def hardcoded_secrets(code: str) -> list[Finding]:
    """``password = "literal"`` style assignments, ``#define`` secrets, and passwords in URLs."""
    found: list[Finding] = []
    for match in ASSIGNED_LITERAL.finditer(code):
        name, value = match.group("name"), match.group("val")
        if _looks_secret(name, value):
            found.append(finding("high", "CWE-798", f"hard-coded secret assigned to {name}",
                                 line_text(code, match.start()), line_of(code, match.start())))
    for match in DEFINE_LITERAL.finditer(code):
        if _looks_secret(match.group("name"), match.group("val")):
            found.append(finding("high", "CWE-798", f"hard-coded secret in #define {match.group('name')}",
                                 line_text(code, match.start()), line_of(code, match.start())))
    for match in URL_PASSWORD.finditer(code):
        if not PLACEHOLDER_SECRET.search(match.group("val")):
            found.append(finding("high", "CWE-798", "password embedded in a connection URL",
                                 line_text(code, match.start()), line_of(code, match.start())))
    return found


def _looks_secret(name: str, value: str) -> bool:
    stripped = re.sub(r"(_?(str|value|val|string))$", "", name, flags=re.I)
    if not SECRET_NAME.search(stripped):
        return False
    if len(value) < 8 or PLACEHOLDER_SECRET.search(value) or HEADER_LIKE.match(value):
        return False
    return not re.fullmatch(r"[A-Z0-9_]+", value) or len(value) >= 16


def identifiers(expr: str) -> list[str]:
    """Identifiers in ``expr`` outside string literals and ``sizeof(...)``."""
    cleaned = re.sub(r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'", " ", expr)
    cleaned = re.sub(r"\bsizeof\s*\([^()]*(?:\([^()]*\)[^()]*)*\)|\bsizeof\s+\*?\s*[A-Za-z_]\w*(?:->\w+|\.\w+)*", " ", cleaned)
    cleaned = re.sub(r"(->|\.)\s*[A-Za-z_]\w*", " ", cleaned)
    return [name for name in re.findall(r"\b[A-Za-z_]\w*\b", cleaned) if not name.isdigit()]
