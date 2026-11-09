"""Go rules (regex/token based; Go has no parser in the Python stdlib)."""

from __future__ import annotations

import re

from modelsentinel.findings import Finding
from modelsentinel.icd.common import (
    CONTEXT_SECRET_WORDS,
    calls,
    finding,
    hardcoded_secrets,
    is_string_literal,
    line_of,
    line_text,
    strip_comments,
)

SQL_SHAPE = r"(?:SELECT\b[^;]*?\bFROM|INSERT\s+INTO|UPDATE\s+[\w\".]+\s+SET|DELETE\s+FROM|\bWHERE\b|ORDER\s+BY)"
SQL_CALL = re.compile(r"\.(Query|QueryRow|QueryContext|QueryRowContext|Exec|ExecContext|Prepare|PrepareContext|Raw|Select|Get)\s*\(")
REQUEST_INPUT = r"(?:\.URL\.Query\(\)|\.FormValue\(|\.PostFormValue\(|\.PathValue\(|mux\.Vars\(|\.Param\(|\.Query\(\"|\.URL\.Path\b|\.Header\.Get\()"
FILE_OP_NAME = r"\b(os\.(?:Open|OpenFile|ReadFile|Create|WriteFile|Remove|RemoveAll|ReadDir)|ioutil\.ReadFile|http\.ServeFile)"
FILE_OP = re.compile(FILE_OP_NAME + r"\s*\(")
CONTAINMENT = re.compile(r"filepath\.Rel\s*\(|strings\.HasPrefix\s*\(|filepath\.IsLocal\s*\(|OpenRoot\s*\(|OpenInRoot\s*\(|filepath\.Base\s*\(|\"\.\.\"")


def analyze_go(code: str) -> list[Finding]:
    src = strip_comments(code)
    found: list[Finding] = []
    for rule in (_command, _sql, _tls, _random, _path, _xss, _crypto, _ssrf, _redirect):
        found.extend(rule(src))
    found.extend(hardcoded_secrets(src))
    return found


def _add(found: list, src: str, offset: int, severity: str, code: str, message: str) -> None:
    found.append(finding(severity, code, message, line_text(src, offset), line_of(src, offset)))


def _from_request(src: str, expr: str) -> bool:
    expr = expr.strip()
    if re.search(REQUEST_INPUT, expr):
        return True
    if re.fullmatch(r"[A-Za-z_]\w*", expr):
        name = re.escape(expr)
        return bool(re.search(rf"\b{name}\b(?:\s*,\s*\w+)?\s*:?=\s*[^\n]*{REQUEST_INPUT}", src))
    return False


def _command(src: str) -> list[Finding]:
    found: list[Finding] = []
    for match, args in calls(src, r"\bexec\.Command(?:Context)?"):
        values = args[1:] if "Context" in match.group(0) else args
        if not values:
            continue
        shell = re.fullmatch(r"\"(?:/bin/|/usr/bin/)?(sh|bash|zsh|cmd|cmd\.exe|powershell)\"", values[0])
        if shell and len(values) > 1 and values[1] in {'"-c"', '"/c"', '"/C"', '"-Command"'}:
            _add(found, src, match.start(), "high", "CWE-78", "exec.Command runs a shell with -c")
        elif not is_string_literal(values[0]) and re.search(r"\+|Sprintf", values[0]):
            _add(found, src, match.start(), "high", "CWE-78", "exec.Command program name built from a string")
    return found


def _sql(src: str) -> list[Finding]:
    found: list[Finding] = []
    lines: set[int] = set()
    for match, args in calls(src, r"\bfmt\.Sprintf"):
        if not args or not is_string_literal(args[0]):
            continue
        flags = re.I if SQL_CALL.search(line_text(src, match.start())) else 0
        if re.search(SQL_SHAPE, args[0], flags) and re.search(r"%[-+# 0-9.]*[sdvq]", args[0]):
            lines.add(line_of(src, match.start()))
            _add(found, src, match.start(), "high", "CWE-89", "SQL query formatted with fmt.Sprintf")
    concat = re.compile(r"(\"|`)(?P<s>[^\"`\n]*)\1\s*\+\s*[A-Za-z_(]")
    for match in concat.finditer(src):
        flags = re.I if SQL_CALL.search(line_text(src, match.start())) else 0
        if re.search(SQL_SHAPE, match.group("s"), flags) and line_of(src, match.start()) not in lines:
            lines.add(line_of(src, match.start()))
            _add(found, src, match.start(), "high", "CWE-89", "SQL query concatenated with a variable")
    return found


def _tls(src: str) -> list[Finding]:
    found: list[Finding] = []
    for match in re.finditer(r"InsecureSkipVerify\s*:\s*true", src):
        _add(found, src, match.start(), "high", "CWE-295", "InsecureSkipVerify disables certificate checks")
    return found


def _random(src: str) -> list[Finding]:
    math_import = re.search(r"\"math/rand(?:/v2)?\"", src)
    usage = re.search(r"\brand\.(Intn|IntN|Int63n?|Int31n?|Uint32|Uint64|Float64|Perm|Shuffle|Seed|N|Read)\s*\(", src)
    crypto = re.search(r"\"crypto/rand\"", src)
    anchor = math_import or (usage if usage and not crypto else None)
    if not anchor or not CONTEXT_SECRET_WORDS.search(src):
        return []
    return [finding("medium", "CWE-330", "math/rand is not a cryptographic generator",
                    line_text(src, anchor.start()), line_of(src, anchor.start()))]


def _dynamic_join(src: str) -> re.Match | None:
    for match, args in calls(src, r"\b(?:filepath|path)\.Join"):
        if any(not is_string_literal(arg) for arg in args[1:]):
            return match
    for match, args in calls(src, FILE_OP_NAME):
        target = args[2] if match.group(1) == "http.ServeFile" and len(args) > 2 else (args[0] if args else "")
        if "+" in target or re.search(REQUEST_INPUT, target):
            return match
    return None


def _path(src: str) -> list[Finding]:
    file_op = FILE_OP.search(src)
    if not file_op or CONTAINMENT.search(src):
        return []
    built = _dynamic_join(src)
    if not built:
        return []
    return [finding("medium", "CWE-22", "file path joined from input without a Rel/HasPrefix/IsLocal check",
                    line_text(src, file_op.start()), line_of(src, file_op.start()))]


def _xss(src: str) -> list[Finding]:
    found: list[Finding] = []
    html_hint = re.search(r"<(html|body|div|p|h[1-6]|ul|li|a|span|table|tr|td|form|script|b|em|strong)\b[^>]*>", src, re.I)
    if re.search(r"\"text/template\"", src) and (html_hint or "http.ResponseWriter" in src):
        match = re.search(r"\"text/template\"", src)
        _add(found, src, match.start(), "high", "CWE-79", "text/template does not escape HTML; use html/template")
    for match, args in calls(src, r"\bfmt\.Fprint(?:f|ln)?"):
        if len(args) < 2 or args[0] != "w":
            continue
        body = ",".join(args[1:])
        if re.search(r"<\w", args[1]) and len(args) > 2 and not re.search(r"EscapeString|HTMLEscape", body):
            _add(found, src, match.start(), "high", "CWE-79", "HTML written with unescaped values")
    for match in re.finditer(r"\bw\.Write\s*\(\s*\[\]byte\s*\((?P<arg>[^\n]*)", src):
        arg = match.group("arg")
        if re.search(r"<\w", arg) and "+" in arg and not re.search(r"EscapeString|HTMLEscape", arg):
            _add(found, src, match.start(), "high", "CWE-79", "HTML concatenated with unescaped values")
    for match, args in calls(src, r"\btemplate\.HTML"):
        if args and not is_string_literal(args[0]):
            _add(found, src, match.start(), "medium", "CWE-79", "template.HTML marks dynamic text as safe HTML")
    return found


def _crypto(src: str) -> list[Finding]:
    match = re.search(r"\"crypto/(md5|sha1|des|rc4)\"|\b(md5|sha1)\.(Sum|New)\s*\(|\bdes\.New(?:TripleDES)?Cipher\s*\(|\brc4\.NewCipher\s*\(", src)
    if not match:
        return []
    return [finding("medium", "CWE-327", "MD5, SHA-1, DES or RC4 used", line_text(src, match.start()), line_of(src, match.start()))]


def _ssrf(src: str) -> list[Finding]:
    found: list[Finding] = []
    if re.search(r"\.Hostname\(\)|allowedHosts|AllowedHosts|allowlist", src):
        return found
    for match, args in calls(src, r"\bhttp\.(Get|Post|Head|PostForm)"):
        if args and (_from_request(src, args[0]) or re.match(r"\"https?://\"\s*\+", args[0])):
            _add(found, src, match.start(), "high", "CWE-918", f"http.{match.group(1)} fetches a caller-controlled URL")
    for match, args in calls(src, r"\bhttp\.NewRequest(WithContext)?"):
        url = args[2] if match.group(1) and len(args) > 2 else (args[1] if len(args) > 1 else "")
        if url and (_from_request(src, url) or re.match(r"\"https?://\"\s*\+", url)):
            _add(found, src, match.start(), "high", "CWE-918", "outbound request to a caller-controlled URL")
    return found


def _redirect(src: str) -> list[Finding]:
    found: list[Finding] = []
    if re.search(r"strings\.HasPrefix\s*\(|\.IsAbs\(\)|\.Host\s*[!=]=", src):
        return found
    for match, args in calls(src, r"\bhttp\.Redirect"):
        if len(args) > 2 and _from_request(src, args[2]):
            _add(found, src, match.start(), "high", "CWE-601", "http.Redirect to a target taken from the request")
    return found
