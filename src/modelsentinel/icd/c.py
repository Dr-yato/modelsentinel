"""C rules (token/regex based).

Bounded calls are not flagged: ``strncpy``/``memcpy`` with a ``sizeof`` length
or a length that the code compares somewhere, ``snprintf``, ``fgets``, and
``scanf`` conversions with a field width.
"""

from __future__ import annotations

import re

from modelsentinel.findings import Finding
from modelsentinel.icd.common import (
    CONTEXT_SECRET_WORDS,
    calls,
    finding,
    hardcoded_secrets,
    identifiers,
    is_string_literal,
    line_of,
    line_text,
    strip_comments,
)

C_KEYWORDS = {"sizeof", "int", "char", "unsigned", "long", "short", "size_t", "const", "struct", "void",
              "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int32_t", "int64_t", "ssize_t", "float", "double"}
FORMAT_POSITION = {"printf": 0, "vprintf": 0, "fprintf": 1, "vfprintf": 1, "dprintf": 1, "sprintf": 1,
                   "vsprintf": 1, "syslog": 1, "vsyslog": 1, "snprintf": 2, "vsnprintf": 2}
OVERFLOW_GUARD = re.compile(r"SIZE_MAX|__builtin_mul_overflow|ckd_mul|INT_MAX|UINT_MAX|reallocarray|\bcalloc\s*\(")
PATH_CHECK = re.compile(r"realpath\s*\(|\"\.\.\"|strstr\s*\([^)]*\.\.|strchr\s*\([^)]*'/'|strpbrk\s*\(|openat\s*\(")


def analyze_c(code: str) -> list[Finding]:
    src = strip_comments(code, quotes="\"'")
    found: list[Finding] = []
    for rule in (_unbounded_copies, _bounded_copies, _format_strings, _size_overflow, _lifetime,
                 _command, _off_by_one, _crypto, _random, _tls, _path):
        found.extend(rule(src))
    found.extend(hardcoded_secrets(src))
    return found


def _add(found: list, src: str, offset: int, severity: str, code: str, message: str) -> None:
    found.append(finding(severity, code, message, line_text(src, offset), line_of(src, offset)))


def _unbounded_copies(src: str) -> list[Finding]:
    found: list[Finding] = []
    for match in re.finditer(r"(?<![\w.>])gets\s*\(", src):
        _add(found, src, match.start(), "high", "CWE-120", "gets cannot bound its input")
    for match in re.finditer(r"(?<![\w.>])(strcpy|strcat|wcscpy|wcscat|stpcpy)\s*\(", src):
        _add(found, src, match.start(), "high", "CWE-120", f"{match.group(1)} does not check the destination size")
    for match, args in calls(src, r"(?<![\w.>])(sprintf|vsprintf)"):
        fmt = args[1] if len(args) > 1 else ""
        if not is_string_literal(fmt) or re.search(r"%[-+ #0]*s", fmt):
            _add(found, src, match.start(), "high", "CWE-120", f"{match.group(1)} can overflow its buffer; use snprintf")
    for match, args in calls(src, r"(?<![\w.>])(scanf|fscanf|sscanf|vscanf)"):
        fmt = args[0] if match.group(1) in {"scanf", "vscanf"} else (args[1] if len(args) > 1 else "")
        if re.search(r"%(?:\[|s)", fmt):
            _add(found, src, match.start(), "high", "CWE-120", f"{match.group(1)} %s without a field width")
    return found


def _length_is_checked(src: str, length: str) -> bool:
    if "sizeof" in length:
        return True
    names = [name for name in identifiers(length) if name not in C_KEYWORDS and not name.isupper()]
    chains = re.findall(r"[A-Za-z_]\w*(?:\s*(?:->|\.)\s*\w+)+", length)
    if not names and not chains:
        return True  # a constant length
    for token in chains + names:
        escaped = r"\s*(?:->|\.)\s*".join(re.escape(part) for part in re.split(r"\s*(?:->|\.)\s*", token))
        compare = rf"(?:\b{escaped}\b\s*(?:<=?|>=?)(?!>)|(?<!-)(?:<=?|>=?)\s*\b{escaped}\b)"
        if re.search(compare, src) or re.search(rf"\b(?:MIN|min|malloc|calloc|realloc)\s*\([^;]*\b{escaped}\b", src):
            return True
    return False


def _bounded_copies(src: str) -> list[Finding]:
    found: list[Finding] = []
    for match, args in calls(src, r"(?<![\w.>])(memcpy|memmove|strncpy|strncat|wmemcpy)"):
        if len(args) < 3 or _length_is_checked(src, args[2]):
            continue
        code = "CWE-787" if match.group(1).startswith(("mem", "wmem")) else "CWE-120"
        _add(found, src, match.start(), "high", code, f"{match.group(1)} length is not checked against the destination")
    return found


def _format_strings(src: str) -> list[Finding]:
    found: list[Finding] = []
    variadic = "va_start" in src or "va_list" in src
    for match, args in calls(src, r"(?<![\w.>])(printf|vprintf|fprintf|vfprintf|dprintf|sprintf|vsprintf|syslog|vsyslog|snprintf|vsnprintf)"):
        name = match.group(1)
        position = FORMAT_POSITION[name]
        if len(args) <= position:
            continue
        fmt = args[position].strip()
        if is_string_literal(fmt) or re.match(r"(?:_|N_|gettext)\s*\(", fmt) or re.fullmatch(r"[A-Z][A-Z0-9_]*", fmt):
            continue
        if name.startswith("v") and variadic:
            continue
        if re.fullmatch(r"\"[^\"]*\"(\s*\"[^\"]*\")*", fmt):
            continue
        _add(found, src, match.start(), "high", "CWE-134", f"{name} format string is not a literal")
    return found


def _size_overflow(src: str) -> list[Finding]:
    if OVERFLOW_GUARD.search(src):
        return []
    found: list[Finding] = []
    for match, args in calls(src, r"(?<![\w.>])(malloc|realloc|alloca)"):
        size = args[1] if match.group(1) == "realloc" and len(args) > 1 else (args[0] if args else "")
        if "*" not in size:
            continue
        names = [name for name in identifiers(size) if name not in C_KEYWORDS and not name.isupper()]
        if names:
            _add(found, src, match.start(), "medium", "CWE-190", f"{match.group(1)} size is a product that can overflow")
    return found


def _command(src: str) -> list[Finding]:
    found: list[Finding] = []
    for match, args in calls(src, r"(?<![\w.>])(system|popen)"):
        if args and not is_string_literal(args[0]):
            _add(found, src, match.start(), "high", "CWE-78", f"{match.group(1)} runs a command string built at runtime")
    for match in re.finditer(r"\bexecl\s*\(\s*\"/bin/(?:ba)?sh\"\s*,\s*\"(?:ba)?sh\"\s*,\s*\"-c\"\s*,\s*(?!\")", src):
        _add(found, src, match.start(), "high", "CWE-78", "execl through /bin/sh -c with a runtime string")
    return found


def _off_by_one(src: str) -> list[Finding]:
    found: list[Finding] = []
    for match in re.finditer(r"\b(\w+)\s*\[\s*sizeof\s*\(?\s*\1\s*\)?\s*\]\s*=(?!=)", src):
        _add(found, src, match.start(), "high", "CWE-787", "write one past the end of the array")
    return found


def _crypto(src: str) -> list[Finding]:
    pattern = (r"\b(MD5|MD4|SHA1)(?:_Init|_Update|_Final)?\s*\(|\bEVP_(md5|md4|sha1|des_\w+|rc4\w*|bf_\w+)\s*\(|"
               r"\bDES_\w+\s*\(|\bEVP_\w+_ecb\s*\(")
    match = re.search(pattern, src)
    if not match:
        return []
    return [finding("medium", "CWE-327", "MD5, SHA-1, DES, RC4 or ECB used", line_text(src, match.start()), line_of(src, match.start()))]


def _random(src: str) -> list[Finding]:
    match = re.search(r"(?<![\w.>])(rand|random|drand48|lrand48)\s*\(\s*\)|(?<![\w.>])srand\s*\(", src)
    if not match or not CONTEXT_SECRET_WORDS.search(src):
        return []
    return [finding("medium", "CWE-330", "rand() is not a cryptographic generator", line_text(src, match.start()), line_of(src, match.start()))]


def _tls(src: str) -> list[Finding]:
    found: list[Finding] = []
    for match in re.finditer(r"CURLOPT_SSL_VERIFY(?:PEER|HOST)\s*,\s*0|SSL_VERIFY_NONE", src):
        _add(found, src, match.start(), "high", "CWE-295", "TLS certificate verification is disabled")
    return found


def _path(src: str) -> list[Finding]:
    opener = re.search(r"(?<![\w.>])(fopen|open)\s*\(", src)
    if not opener or PATH_CHECK.search(src):
        return []
    built = re.search(r"\b(?:snprintf|sprintf)\s*\([^;]*\"[^\"]*%s/%s|\b(?:snprintf|sprintf)\s*\([^;]*\"[^\"]*/[^\"]*%s|\bstrn?cat\s*\(", src)
    if not built:
        return []
    return [finding("medium", "CWE-22", "file path built from a name without a traversal check",
                    line_text(src, opener.start()), line_of(src, opener.start()))]


# -- use after free / double free -------------------------------------------

FREE_CALL = re.compile(r"(?<![\w.>])free\s*\(\s*(?P<name>[A-Za-z_]\w*(?:\s*(?:->|\.)\s*\w+)*)\s*\)\s*;")
TERMINATOR = re.compile(r"\b(return|goto|break|continue|exit|abort|longjmp)\b")


def _lifetime(src: str) -> list[Finding]:
    found: list[Finding] = []
    lines = src.splitlines()
    for index, text in enumerate(lines):
        for match in FREE_CALL.finditer(text):
            before = text[: match.start()]
            if re.search(r"\b(if|else|while|for)\b[^{;]*$", before) and "{" not in before:
                continue  # braceless conditional free: the path is not linear
            problem = _after_free(lines, index, match.end(), match.group("name"))
            if problem:
                code, line_number = problem
                message = "pointer freed twice" if code == "CWE-415" else "pointer used after free"
                found.append(finding("high", code, message, lines[line_number].strip(), line_number + 1))
    return found


def _after_free(lines: list[str], start: int, column: int, name: str) -> tuple[str, int] | None:
    """Walk forward from a ``free(name);`` along the straight-line path.

    Stops at a reassignment, a terminator (return/goto/...) in the current
    block, or the end of the function (a ``}`` in column 0). ``else`` blocks
    that follow the freeing block are skipped: they are a different path.
    """
    parts = [re.escape(part) for part in re.split(r"\s*(?:->|\.)\s*", name)]
    ident = r"\s*(?:->|\.)\s*".join(parts)
    assign = re.compile(rf"(?<![\w.>]){ident}\s*=(?!=)(?P<rhs>[^;]*)")
    use = re.compile(rf"(?<![\w.>]){ident}\s*(?:->|\[)|\*\s*{ident}\b|[(,]\s*{ident}\s*[,)]")
    again = re.compile(rf"(?<![\w.>])free\s*\(\s*{ident}\s*\)")
    depth = floor = 0
    index = start
    while index < len(lines):
        text = lines[index][column:] if index == start else lines[index]
        if index != start and lines[index].startswith("}"):
            return None
        else_match = re.match(r"(\}\s*)?else\b(?P<rest>.*)", text.strip()) if index != start else None
        if else_match:
            if else_match.group(1):
                depth -= 1
                floor = min(floor, depth)
            opened = else_match.group("rest").count("{") - else_match.group("rest").count("}")
            index += 1
            while opened > 0 and index < len(lines):
                opened += lines[index].count("{") - lines[index].count("}")
                index += 1
            continue
        if again.search(text):
            return "CWE-415", index
        assigned = assign.search(text)
        if assigned:
            return ("CWE-416", index) if re.search(rf"(?<![\w.>]){parts[0]}\b", assigned.group("rhs")) else None
        if use.search(text):
            return "CWE-416", index
        if TERMINATOR.search(text) and depth == floor:
            return None
        depth += text.count("{") - text.count("}")
        floor = min(floor, depth)
        index += 1
    return None
