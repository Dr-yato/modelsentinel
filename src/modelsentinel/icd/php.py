"""PHP rules (regex over comment-stripped source)."""

from __future__ import annotations

import re

from modelsentinel.findings import Finding
from modelsentinel.icd.common import (
    CONTEXT_SECRET_WORDS,
    calls,
    finding,
    first_line_match,
    hardcoded_secrets,
    line_of,
    line_text,
    strip_comments,
)

SQL_WORD = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|REPLACE)\b", re.IGNORECASE)
SUPERGLOBAL = r"\$_(?:GET|POST|REQUEST|COOKIE|SERVER|FILES)"


def analyze_php(code: str) -> list[Finding]:
    src = strip_comments(code, line=("//", "#"), quotes="\"'")
    found: list[Finding] = []
    if re.search(r"echo\s+\$_(GET|POST|REQUEST|COOKIE)", src) and "htmlspecialchars" not in src:
        evidence, line = first_line_match(src, "echo")
        found.append(finding("high", "CWE-79", "raw echo of request input", evidence, line))
    if SQL_WORD.search(src) and re.search(r"\.\s*\$", src) and re.search(r"[\"'][^\"']*\b(SELECT|INSERT|UPDATE|DELETE|REPLACE)\b", src, re.I):
        evidence, line = first_line_match(src, "SELECT")
        found.append(finding("high", "CWE-89", "SQL concatenated with a PHP variable", evidence, line))
    for match, args in calls(src, r"(?<![\w>:$])(system|exec|shell_exec|passthru|popen|proc_open)"):
        command = args[0] if args else ""
        if "$" in command and "escapeshellarg" not in command and "escapeshellcmd" not in command:
            found.append(finding("high", "CWE-78", f"{match.group(1)} with a variable in the command",
                                 line_text(src, match.start()), line_of(src, match.start())))
    for match, args in calls(src, r"(?<![\w>:$])unserialize"):
        if not any("allowed_classes" in arg for arg in args[1:]):
            found.append(finding("high", "CWE-502", "unserialize can instantiate arbitrary classes",
                                 line_text(src, match.start()), line_of(src, match.start())))
    for match in re.finditer(rf"\b(include|require)(?:_once)?\b[^;]*{SUPERGLOBAL}", src):
        found.append(finding("high", "CWE-98", "include path taken from the request",
                             line_text(src, match.start()), line_of(src, match.start())))
    if not re.search(r"\bbasename\s*\(|\brealpath\s*\(", src):
        for match in re.finditer(rf"\b(file_get_contents|fopen|readfile|file|unlink)\s*\([^;]*{SUPERGLOBAL}", src):
            found.append(finding("medium", "CWE-22", "file path taken from the request",
                                 line_text(src, match.start()), line_of(src, match.start())))
    for match in re.finditer(r"\b(md5|sha1)\s*\([^;]*\$\w*pass", src, re.I):
        found.append(finding("medium", "CWE-327", "MD5/SHA-1 used for a password",
                             line_text(src, match.start()), line_of(src, match.start())))
    random = re.search(r"(?<![\w>:$])(rand|mt_rand|uniqid|lcg_value)\s*\(", src)
    if random and CONTEXT_SECRET_WORDS.search(src):
        found.append(finding("medium", "CWE-330", f"{random.group(1)} is not a cryptographic generator",
                             line_text(src, random.start()), line_of(src, random.start())))
    for match in re.finditer(rf"\bheader\s*\(\s*[\"']Location:[^;]*{SUPERGLOBAL}", src, re.I):
        if not re.search(r"parse_url|str_starts_with|strpos\s*\([^)]*['\"]/", src):
            found.append(finding("high", "CWE-601", "Location header taken from the request",
                                 line_text(src, match.start()), line_of(src, match.start())))
    found.extend(hardcoded_secrets(src))
    return found
