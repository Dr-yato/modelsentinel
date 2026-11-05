"""SQL rules: T-SQL, PL/pgSQL and MySQL procedures, grants, and schemas.

Dynamic SQL is flagged when a string that becomes executable SQL is built by
concatenating a variable, unless that variable goes through QUOTENAME,
quote_ident/quote_literal/quote_nullable, or format() with %I/%L.
Parameterised ``sp_executesql`` / ``EXECUTE ... USING`` / ``PREPARE ... ?``
are not flagged.
"""

from __future__ import annotations

import re

from modelsentinel.findings import Finding
from modelsentinel.icd.common import PLACEHOLDER_SECRET, finding, line_of, line_text, strip_comments

I = re.IGNORECASE
SAFE_WRAPPERS = re.compile(r"^(QUOTENAME|quote_ident|quote_literal|quote_nullable|QUOTE)\s*\(", I)
DYNAMIC_EXEC = re.compile(r"\bEXEC(?:UTE)?\b|\bsp_executesql\b|\bPREPARE\b", I)
LITERAL = re.compile(r"^N?'(?:[^']|'')*'$", re.S)
PASSWORD_COLUMN = re.compile(
    r"(?im)^[ \t]*,?[ \t]*(?:ADD[ \t]+(?:COLUMN[ \t]+)?)?[\"`\[]?(?P<name>(?:\w*_)?(?:password|passwd|pwd|passcode|pass)(?:_\w+)?)[\"`\]]?"
    r"[ \t]+(?P<type>N?VARCHAR2?|N?CHAR|TEXT|CITEXT|STRING|CLOB|NVARCHAR|VARBINARY|BLOB)\b"
)
NOT_PLAINTEXT = re.compile(r"hash|digest|salt|hmac|bcrypt|argon|scrypt|enc|token|reset|hint|policy|expir|changed", I)


def analyze_sql(code: str) -> list[Finding]:
    src = strip_comments(code, line=("--",), quotes="'", sql_strings=True)
    found: list[Finding] = []
    for rule in (_dynamic_sql, _privileges, _password_columns, _weak_hash, _literal_passwords, _definer):
        found.extend(rule(src))
    return found


def _add(found: list, src: str, offset: int, severity: str, code: str, message: str) -> None:
    found.append(finding(severity, code, message, line_text(src, offset), line_of(src, offset)))


def _split_top(expr: str, operator: str) -> list[str]:
    parts, depth, current, i = [], 0, [], 0
    while i < len(expr):
        char = expr[i]
        if char == "'":
            end = i + 1
            while end < len(expr):
                if expr[end] == "'" and expr.startswith("''", end):
                    end += 2
                    continue
                if expr[end] == "'":
                    break
                end += 1
            current.append(expr[i : end + 1])
            i = end + 1
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if depth == 0 and expr.startswith(operator, i):
            parts.append("".join(current).strip())
            current = []
            i += len(operator)
            continue
        current.append(char)
        i += 1
    parts.append("".join(current).strip())
    return parts


def _tainted_terms(expr: str, operator: str, target: str | None = None) -> list[str]:
    terms = _split_top(expr, operator)
    if len(terms) < 2 or not any(LITERAL.match(term) for term in terms):
        return []
    bad = []
    for term in terms:
        if not term or LITERAL.match(term) or SAFE_WRAPPERS.match(term) or re.fullmatch(r"\d+|CHAR\(\d+\)|NCHAR\(\d+\)", term, I):
            continue
        if target and term.lower() == target.lower():
            continue
        if re.fullmatch(r"@?[A-Za-z_][\w.]*|(?:CAST|CONVERT|CONCAT|TRIM|LOWER|UPPER|COALESCE|ISNULL)\s*\(.*\)", term, I | re.S):
            bad.append(term)
    return bad


def _dynamic_sql(src: str) -> list[Finding]:
    if not DYNAMIC_EXEC.search(src):
        return []
    found: list[Finding] = []
    seen: set[int] = set()

    def flag(offset: int, message: str) -> None:
        line = line_of(src, offset)
        if line not in seen:
            seen.add(line)
            _add(found, src, offset, "high", "CWE-89", message)

    # T-SQL: SET/SELECT @x = <expr>;  DECLARE @x NVARCHAR(..) = <expr>;  EXEC(<expr>)
    for match in re.finditer(r"\b(?:SET|SELECT|DECLARE)\s+(?P<var>@\w+)(?:\s+N?VARCHAR\s*\([^)]*\))?\s*(?:\+=|=)\s*(?P<expr>[^;]*?)(?=;|\n\s*(?:SET|EXEC|DECLARE|SELECT|IF|BEGIN|END|PRINT|RETURN)\b|$)", src, I | re.S):
        if _tainted_terms(match.group("expr"), "+", match.group("var")):
            flag(match.start(), f"dynamic SQL in {match.group('var')} concatenates a variable")
    for match in re.finditer(r"\bEXEC(?:UTE)?\s*\((?P<expr>[^;]*)\)", src, I | re.S):
        if _tainted_terms(match.group("expr"), "+"):
            flag(match.start(), "EXEC() runs a concatenated string")
    # PL/pgSQL / Oracle: EXECUTE <expr> [INTO|USING] ;   v := <expr>;
    for match in re.finditer(r"\bEXECUTE\s+(?:IMMEDIATE\s+)?(?P<expr>[^;]*?)(?=\bUSING\b|\bINTO\b|;)", src, I | re.S):
        expr = match.group("expr")
        if _tainted_terms(expr, "||") or _unsafe_format(expr):
            flag(match.start(), "EXECUTE runs a string concatenated from a variable")
    for match in re.finditer(r"\b(?P<var>[A-Za-z_]\w*)\s*:=\s*(?P<expr>[^;]*);", src, I | re.S):
        if re.search(r"'\s*(?:SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE)\b", match.group("expr"), I):
            if _tainted_terms(match.group("expr"), "||", match.group("var")) or _unsafe_format(match.group("expr")):
                flag(match.start(), f"dynamic SQL in {match.group('var')} concatenates a variable")
    # MySQL: SET @s = CONCAT('SELECT ...', p_name); PREPARE stmt FROM @s;
    for match in re.finditer(r"\bCONCAT\s*\(", src, I):
        args = _concat_args(src, match.end() - 1)
        literals = [arg for arg in args if LITERAL.match(arg)]
        variables = [arg for arg in args if arg and not LITERAL.match(arg) and not SAFE_WRAPPERS.match(arg)]
        if literals and variables and re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|WHERE|ORDER BY|FROM)\b", " ".join(literals), I):
            flag(match.start(), "CONCAT builds SQL from a variable before PREPARE")
    return found


def _concat_args(src: str, open_paren: int) -> list[str]:
    depth, i, start = 0, open_paren, open_paren + 1
    while i < len(src):
        if src[i] == "'":
            i += 1
            while i < len(src) and not (src[i] == "'" and not src.startswith("''", i)):
                i += 2 if src.startswith("''", i) else 1
        elif src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                return _split_top(src[start:i], ",")
        i += 1
    return []


def _unsafe_format(expr: str) -> bool:
    match = re.search(r"\bformat\s*\(\s*'(?P<fmt>(?:[^']|'')*)'", expr, I)
    return bool(match and re.search(r"%s", match.group("fmt")))


def _statements(src: str) -> list[tuple[int, str]]:
    out, start = [], 0
    for match in re.finditer(r";|\Z", src):
        out.append((start, src[start : match.start()]))
        start = match.end()
    return out


def _privileges(src: str) -> list[Finding]:
    found: list[Finding] = []
    for offset, statement in _statements(src):
        body = statement.lstrip()
        lead = offset + len(statement) - len(body)
        if re.match(r"GRANT\b", body, I):
            if re.search(r"\bGRANT\s+ALL\b", body, I):
                _add(found, src, lead, "high", "CWE-269", "GRANT ALL gives every privilege")
            elif re.search(r"\bON\s+\*\.\*", body, I):
                _add(found, src, lead, "high", "CWE-269", "grant on *.* covers every database")
            elif re.search(r"\bTO\s+PUBLIC\b", body, I):
                _add(found, src, lead, "medium", "CWE-269", "privilege granted to PUBLIC")
            elif re.search(r"\bWITH\s+GRANT\s+OPTION\b", body, I):
                _add(found, src, lead, "medium", "CWE-269", "WITH GRANT OPTION lets the grantee re-grant")
            elif re.search(r"\b(sysadmin|db_owner|SUPER|FILE|SHUTDOWN|PROCESS)\b", body):
                _add(found, src, lead, "high", "CWE-269", "administrative privilege granted")
        if re.match(r"(CREATE|ALTER)\s+(ROLE|USER)\b", body, I) and re.search(r"\b(SUPERUSER|CREATEROLE|BYPASSRLS)\b", body, I):
            _add(found, src, lead, "high", "CWE-269", "role created with superuser-class attributes")
        if re.search(r"\bALTER\s+(?:SERVER\s+)?ROLE\s+\[?(sysadmin|db_owner)\]?\s+ADD\s+MEMBER|\bsp_add(?:srv)?rolemember\b[^;]*\b(sysadmin|db_owner)\b", body, I):
            _add(found, src, lead, "high", "CWE-269", "member added to sysadmin/db_owner")
    return found


def _password_columns(src: str) -> list[Finding]:
    found: list[Finding] = []
    if not re.search(r"\b(CREATE|ALTER)\s+TABLE\b", src, I):
        return found
    for match in PASSWORD_COLUMN.finditer(src):
        if NOT_PLAINTEXT.search(match.group("name")):
            continue
        _add(found, src, match.start("name"), "medium", "CWE-256", f"column {match.group('name')} stores a password as plain text")
    return found


def _weak_hash(src: str) -> list[Finding]:
    found: list[Finding] = []
    pattern = (r"\b(MD5|SHA1|OLD_PASSWORD|ENCRYPT)\s*\(|\bPASSWORD\s*\(\s*['@:\w]|HASHBYTES\s*\(\s*'(MD2|MD4|MD5|SHA|SHA1)'|"
               r"\bdigest\s*\([^)]*'(md5|sha1)'")
    for match in re.finditer(pattern, src, I):
        _add(found, src, match.start(), "medium", "CWE-327", "MD5/SHA-1 style hash in SQL")
        break
    return found


def _literal_passwords(src: str) -> list[Finding]:
    found: list[Finding] = []
    for match in re.finditer(r"\bIDENTIFIED\s+(?:WITH\s+\w+\s+)?BY\s+'(?P<a>(?:[^'\n]|''){1,128})'|\bPASSWORD\s*=?\s*'(?P<b>(?:[^'\n]|''){1,128})'", src, I):
        value = match.group("a") if match.group("a") is not None else match.group("b")
        if len(value) >= 4 and not value.lstrip().startswith(",") and not PLACEHOLDER_SECRET.search(value):
            _add(found, src, match.start(), "high", "CWE-798", "password literal in the SQL script")
    return found


def _definer(src: str) -> list[Finding]:
    match = re.search(r"\bSECURITY\s+DEFINER\b", src, I)
    if not match or re.search(r"\bSET\s+search_path\b", src, I):
        return []
    return [finding("medium", "CWE-426", "SECURITY DEFINER function without a pinned search_path",
                    line_text(src, match.start()), line_of(src, match.start()))]
