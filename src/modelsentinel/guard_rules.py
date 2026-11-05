"""Scanning and decisions for ``sentinel guard``.

Every fenced code block in a model response goes through the built-in
``codeanalyzer.analyze`` rules, Bandit/Semgrep when installed (offline flags,
see ``codeanalyzer.external_tools``), and the guard heuristics below:
outbound hosts outside an allow-list, reads of secret paths, environment
harvesting, encoded blobs that decode to URLs or shell, ``curl | sh`` pipes,
DNS exfiltration, and telemetry posts. Secret or environment access combined
with an outbound channel anywhere in the same response is ``GUARD-EXFIL``.

Each finding maps to an action (``block``, ``warn``, ``annotate``) through the
``guard:`` section of policy.yaml; the response decision is the strictest one,
capped by the ``--mode`` the proxy runs in.
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import re
from dataclasses import asdict, dataclass, field

from modelsentinel import codeanalyzer
from modelsentinel.codeanalyzer import FENCE, LANG_ALIASES, THINK, analyze
from modelsentinel.findings import SEVERITY_RANK

ACTION_RANK = {"pass": 0, "annotate": 1, "warn": 2, "block": 3}
DECISION_HEADER = {"pass": "pass", "annotate": "annotated", "warn": "warned", "block": "blocked"}
ANALYZE_LANGS = {"python", "javascript", "php"}

DEFAULT_GUARD = {
    "mode": "block",
    "severity_actions": {"critical": "block", "high": "warn", "medium": "annotate", "low": "annotate", "info": "annotate"},
    "rule_actions": {
        "GUARD-EXFIL": "block",
        "GUARD-DNS-EXFIL": "block",
        "GUARD-ENCODED-PAYLOAD": "block",
        "GUARD-PIPE-SHELL": "block",
    },
    "allow_hosts": [
        "localhost", "127.0.0.1", "::1", "0.0.0.0",
        "example.com", "example.org", "example.net",
        "pypi.org", "files.pythonhosted.org", "registry.npmjs.org", "docs.python.org",
    ],
    "external_tools": True,
    "max_scan_chars": 200_000,
    "preview_chars": 400,
    "upstream_timeout": 600,
}

URL = re.compile(r"\b(?:https?|wss?|ftp)://([^/\s'\"`)<>\]]+)", re.IGNORECASE)
SOCKET_HOST = re.compile(r"connect\(\s*\(\s*['\"]([^'\"]+)['\"]")
SECRET_PATH = re.compile(
    r"(\.ssh\b|id_rsa|id_ed25519|id_ecdsa|\.aws\b|\.gnupg|\.netrc|\.git-credentials|\.docker/config\.json|"
    r"\.kube/config|\.pypirc|\.npmrc|gcloud/(?:credentials|application_default)|credentials\.json|"
    r"\bcredentials\b|(?:^|[\s/'\"`(=])\.env\b|Keychains?|login\.keychain|security\s+find-(?:generic|internet)-password|"
    r"/etc/shadow|Login Data|Cookies\.binarycookies)",
    re.IGNORECASE,
)
READ_CONTEXT = re.compile(
    r"(open\(|\.read|read_text|read_bytes|readFile|createReadStream|expanduser|Path\(|Path\.home|homedir|"
    r"os\.path\.join|\bcat\s|\bless\s|\bhead\s|\btail\s|\btype\s|Get-Content|\bscp\s|\bcurl\s|\btar\s|\bzip\s|"
    r"\bbase64\s|\bcp\s|security\s+find|\bglob\(|\bwalk\()",
    re.IGNORECASE,
)
ENV_HARVEST = re.compile(
    r"(json\.dumps\(\s*(?:dict\()?\s*os\.environ|(?:str|repr|print|dict)\(\s*os\.environ\s*\)|os\.environ\.items\(\)|"
    r"(?:data|json|params|body)\s*=\s*(?:dict\()?\s*os\.environ\b(?!\s*[\[.]get)|"
    r"JSON\.stringify\(\s*process\.env\s*\)|Object\.(?:entries|keys|values)\(\s*process\.env\s*\)|"
    r"body\s*:\s*process\.env\b|\benv\s*\||\bprintenv\s*(?:\||>|$)|\$\(\s*(?:env|printenv)\s*\)|"
    r"`(?:env|printenv)`|Get-ChildItem\s+env:|\bgci\s+env:)",
    re.IGNORECASE | re.MULTILINE,
)
PIPE_SHELL = re.compile(
    r"((?:curl|wget|fetch|iwr|irm|Invoke-WebRequest|Invoke-RestMethod)\b[^\n|]*\|\s*(?:sudo\s+)?"
    r"(?:ba|z|k|da|fi)?sh\b|"
    r"(?:curl|wget|iwr|irm|Invoke-WebRequest|Invoke-RestMethod)\b[^\n|]*\|\s*(?:sudo\s+)?"
    r"(?:python3?|perl|ruby|node|iex|Invoke-Expression)\b|"
    r"(?:ba|z)?sh\s+(?:-c\s+)?['\"]?\$\(\s*(?:curl|wget)|(?:ba|z)?sh\s+<\(\s*(?:curl|wget)|"
    r"(?:iex|Invoke-Expression)\s*\(?\s*\(?\s*(?:New-Object\s+Net\.WebClient|iwr|irm|Invoke-WebRequest))",
    re.IGNORECASE,
)
DNS_EXFIL = re.compile(
    r"(\b(?:nslookup|dig|host|ping)\s+[^\n]*(?:\$\(|\$\{|\$[A-Za-z_]|`)|"
    r"(?:gethostbyname(?:_ex)?|getaddrinfo|resolver\.(?:resolve|query)|dns\.resolver\.resolve)\(\s*"
    r"(?:f['\"]|[^)\n]*\+|[^)\n]*\.format\(|[^)\n]*%\s)|"
    r"dns\.(?:lookup|resolve\w*)\(\s*(?:`[^`]*\$\{|[^)\n]*\+))",
    re.IGNORECASE,
)
SEND_CONTEXT = re.compile(
    r"(requests\.(?:post|put|patch)|\.post\(|\.put\(|urlopen\(|fetch\(|sendBeacon|XMLHttpRequest|axios\.(?:post|put)|"
    r"http\.request|https\.request|\bcurl\b[^\n]*(?:-d\b|--data|-F\b|-T\b|--upload)|\bwget\b[^\n]*--post|"
    r"Invoke-WebRequest|Invoke-RestMethod|smtplib|\.sendall\(|\.send\(|socket\.|\bnc\s|\bncat\s|ftplib)",
    re.IGNORECASE,
)
TELEMETRY = re.compile(r"(telemetry|analytics|tracking|/track\b|/collect\b|beacon|/metrics\b|/ingest\b|heartbeat|/ping\b)", re.IGNORECASE)
BLOB_STRING = re.compile(r"['\"`]([A-Za-z0-9+/=_-]{24,}|(?:[0-9a-fA-F]{2}){16,})['\"`]")
DECODED_DANGER = re.compile(
    r"(https?://|/bin/(?:ba)?sh|\bbash\b|\bcurl\s|\bwget\s|\bnc\s|powershell|eval\(|exec\(|import\s+os|subprocess|"
    r"os\.system|\.ssh|\.aws|socket\.|child_process)",
    re.IGNORECASE,
)
TEMPLATE_HOST = re.compile(r"[{}$<>%]|your[-_]|\bhost\b|\bdomain\b", re.IGNORECASE)


@dataclass(frozen=True)
class GuardConfig:
    mode: str = "block"
    severity_actions: dict = field(default_factory=lambda: dict(DEFAULT_GUARD["severity_actions"]))
    rule_actions: dict = field(default_factory=lambda: dict(DEFAULT_GUARD["rule_actions"]))
    allow_hosts: tuple = tuple(DEFAULT_GUARD["allow_hosts"])
    external_tools: bool = True
    max_scan_chars: int = 200_000
    preview_chars: int = 400
    upstream_timeout: float = 600.0

    @classmethod
    def from_policy(cls, policy: dict | None, mode: str | None = None) -> "GuardConfig":
        raw = dict((policy or {}).get("guard") or {})
        severity = {**DEFAULT_GUARD["severity_actions"], **(raw.get("severity_actions") or {})}
        rules = {**DEFAULT_GUARD["rule_actions"], **(raw.get("rule_actions") or {})}
        for action in [*severity.values(), *rules.values()]:
            if action not in ACTION_RANK:
                raise ValueError(f"guard policy: unknown action {action!r} (use block, warn, annotate, pass)")
        chosen = mode or raw.get("mode") or DEFAULT_GUARD["mode"]
        if chosen not in {"block", "warn", "annotate"}:
            raise ValueError(f"guard mode must be block, warn or annotate, not {chosen!r}")
        hosts = raw.get("allow_hosts")
        hosts = DEFAULT_GUARD["allow_hosts"] if hosts is None else hosts
        return cls(
            mode=chosen,
            severity_actions=severity,
            rule_actions=rules,
            allow_hosts=tuple(str(host).lower() for host in hosts),
            external_tools=bool(raw.get("external_tools", DEFAULT_GUARD["external_tools"])),
            max_scan_chars=int(raw.get("max_scan_chars", DEFAULT_GUARD["max_scan_chars"])),
            preview_chars=int(raw.get("preview_chars", DEFAULT_GUARD["preview_chars"])),
            upstream_timeout=float(raw.get("upstream_timeout", DEFAULT_GUARD["upstream_timeout"])),
        )


@dataclass(frozen=True)
class GuardFinding:
    rule: str
    severity: str
    message: str
    block: int
    language: str
    line: int
    evidence: str
    tool: str
    action: str = "annotate"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ScanResult:
    decision: str
    findings: tuple
    blocks: int

    @property
    def rules(self) -> list[str]:
        return sorted({item.rule for item in self.findings})

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "blocks_scanned": self.blocks,
            "rules": self.rules,
            "findings": [item.to_dict() for item in self.findings],
        }


@dataclass(frozen=True)
class _Block:
    index: int
    language: str
    body: str
    start_line: int


def fenced_blocks(text: str) -> list[_Block]:
    """Fenced blocks with their normalised language and first body line (1-based, after <think> removal)."""
    cleaned = THINK.sub("", (text or "").replace("\r\n", "\n"))
    blocks = []
    for index, match in enumerate(FENCE.finditer(cleaned), start=1):
        start = cleaned.count("\n", 0, match.start(3)) + 1
        blocks.append(_Block(index, normalise_language(match.group(2)), match.group(3), start))
    return blocks


def normalise_language(tag: str) -> str:
    tag = (tag or "").lower()
    for language, aliases in LANG_ALIASES.items():
        if tag in aliases:
            return language
    return tag


def host_allowed(host: str, allow: tuple) -> bool:
    host = host.lower().strip("[]").split("@")[-1]
    if host.count(":") == 1:
        host = host.split(":")[0]
    if not host or TEMPLATE_HOST.search(host):
        return True
    try:
        if ipaddress.ip_address(host).is_loopback:
            return True
    except ValueError:
        pass
    return any(host == entry or host.endswith("." + entry) for entry in allow)


def scan_text(text: str, config: GuardConfig) -> ScanResult:
    """Scan every fenced block in ``text`` and decide what the proxy does with it."""
    blocks = fenced_blocks((text or "")[: config.max_scan_chars])
    findings: list[GuardFinding] = []
    for block in blocks:
        findings.extend(_scan_block(block, config))
    findings.extend(_exfil_combination(findings))
    resolved = tuple(_with_action(item, config) for item in _dedupe(findings))
    decision = max((item.action for item in resolved), key=ACTION_RANK.__getitem__, default="pass")
    return ScanResult(decision=cap_decision(decision, config.mode), findings=resolved, blocks=len(blocks))


def cap_decision(decision: str, mode: str) -> str:
    """``--mode warn`` never blocks; ``--mode annotate`` only observes."""
    if ACTION_RANK[decision] > ACTION_RANK[mode]:
        return mode
    return decision


def _with_action(item: GuardFinding, config: GuardConfig) -> GuardFinding:
    action = config.rule_actions.get(item.rule) or config.severity_actions.get(item.severity, "annotate")
    return GuardFinding(**{**item.to_dict(), "action": action})


def _scan_block(block: _Block, config: GuardConfig) -> list[GuardFinding]:
    found: list[GuardFinding] = []
    if block.language in ANALYZE_LANGS or not block.language:
        found.extend(_builtin(block, config))
    if config.external_tools and block.language in {"python", ""}:
        found.extend(_external(block))
    found.extend(_heuristics(block, config))
    return found


def _builtin(block: _Block, config: GuardConfig) -> list[GuardFinding]:
    out = []
    for item in analyze(block.body, language=block.language or None):
        if item.message.startswith("outbound request to "):
            hosts = [host.strip() for host in item.message[len("outbound request to "):].split(",")]
            if all(host_allowed(host, config.allow_hosts) for host in hosts):
                continue
        out.append(_finding(block, item.code, item.severity, item.message, item.evidence, _line_of(item.location), item.tool))
    return out


def _external(block: _Block) -> list[GuardFinding]:
    report = codeanalyzer.external_tools(block.body, "python")
    return [
        _finding(block, raw.get("code", "external"), raw.get("severity", "medium"), raw.get("message", ""),
                 raw.get("evidence", ""), _line_of(raw.get("location", "")), raw.get("tool", "external"))
        for raw in report.get("findings") or []
    ]


def _heuristics(block: _Block, config: GuardConfig) -> list[GuardFinding]:
    out: list[GuardFinding] = []
    for number, line in enumerate(block.body.splitlines(), start=1):
        out.extend(_line_rules(block, number, line, config))
    for match in BLOB_STRING.finditer(block.body):
        decoded = _decode_blob(match.group(1))
        if decoded and DECODED_DANGER.search(decoded):
            number = block.body.count("\n", 0, match.start()) + 1
            out.append(_finding(block, "GUARD-ENCODED-PAYLOAD", "critical",
                                f"encoded literal decodes to code or a URL: {decoded[:80]!r}",
                                match.group(0)[:120], number))
    return out


def _line_rules(block: _Block, number: int, line: str, config: GuardConfig) -> list[GuardFinding]:
    out = []
    stripped = line.strip()
    hosts = [host for host in URL.findall(line) + SOCKET_HOST.findall(line) if not host_allowed(host, config.allow_hosts)]
    if hosts:
        sending = bool(SEND_CONTEXT.search(line))
        out.append(_finding(block, "GUARD-OUTBOUND-HOST", "high" if sending else "medium",
                            f"{'sends data to' if sending else 'contacts'} {', '.join(sorted(set(hosts))[:3])} (not on the allow-list)",
                            stripped, number))
        if TELEMETRY.search(line) and sending:
            out.append(_finding(block, "GUARD-TELEMETRY", "high", "telemetry-style post to an unlisted host", stripped, number))
    elif SEND_CONTEXT.search(line) and re.search(r"\b(?:nc|ncat)\s+\S+\s+\d+", line):
        out.append(_finding(block, "GUARD-OUTBOUND-HOST", "high", "raw socket send with netcat", stripped, number))
    if SECRET_PATH.search(line) and READ_CONTEXT.search(line):
        out.append(_finding(block, "GUARD-SECRET-READ", "high",
                            f"reads a secret location ({SECRET_PATH.search(line).group(1).strip()})", stripped, number))
    if ENV_HARVEST.search(line):
        out.append(_finding(block, "GUARD-ENV-HARVEST", "high", "dumps the whole process environment", stripped, number))
    if PIPE_SHELL.search(line):
        out.append(_finding(block, "GUARD-PIPE-SHELL", "critical", "downloads a script and pipes it into an interpreter", stripped, number))
    if DNS_EXFIL.search(line):
        out.append(_finding(block, "GUARD-DNS-EXFIL", "critical", "builds a DNS lookup from runtime data", stripped, number))
    return out


def _exfil_combination(findings: list[GuardFinding]) -> list[GuardFinding]:
    """Secret or environment access plus any outbound channel in the same response is exfiltration."""
    sources = [item for item in findings if item.rule in {"GUARD-SECRET-READ", "GUARD-ENV-HARVEST"}
               or (item.rule == "EXFIL" and "opens" in item.message)]
    sinks = [item for item in findings if item.rule in {"GUARD-OUTBOUND-HOST", "GUARD-TELEMETRY", "GUARD-DNS-EXFIL", "CWE-918"}
             or (item.rule == "EXFIL" and "opens" not in item.message)]
    if not sources or not sinks:
        return []
    source, sink = sources[0], sinks[0]
    return [GuardFinding(
        rule="GUARD-EXFIL",
        severity="critical",
        message=(f"secret or environment data (block {source.block}, line {source.line}) can leave the machine "
                 f"via block {sink.block}, line {sink.line}"),
        block=source.block,
        language=source.language,
        line=source.line,
        evidence=source.evidence,
        tool="modelsentinel-guard",
    )]


def _decode_blob(blob: str) -> str | None:
    candidates = []
    if re.fullmatch(r"(?:[0-9a-fA-F]{2})+", blob):
        try:
            candidates.append(bytes.fromhex(blob))
        except ValueError:
            pass
    padded = blob + "=" * (-len(blob) % 4)
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            candidates.append(decoder(padded))
        except (binascii.Error, ValueError):
            continue
    for raw in candidates:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if text and sum(ch.isprintable() or ch in "\n\t" for ch in text) / len(text) > 0.9:
            return text
    return None


def _finding(block: _Block, rule: str, severity: str, message: str, evidence: str, line: int,
             tool: str = "modelsentinel-guard") -> GuardFinding:
    return GuardFinding(
        rule=str(rule),
        severity=severity if severity in SEVERITY_RANK else "medium",
        message=message,
        block=block.index,
        language=block.language or "text",
        line=block.start_line + max(line, 1) - 1,
        evidence=(evidence or "").strip()[:300],
        tool=tool,
    )


def _line_of(location: str) -> int:
    match = re.search(r"\d+", location or "")
    return int(match.group(0)) if match else 1


def _dedupe(findings: list[GuardFinding]) -> list[GuardFinding]:
    seen, kept = set(), []
    for item in findings:
        key = (item.rule, item.block, item.line, item.message)
        if key not in seen:
            seen.add(key)
            kept.append(item)
    return sorted(kept, key=lambda item: (-SEVERITY_RANK.get(item.severity, 0), item.block, item.line))


def explain_block(result: ScanResult) -> str:
    lines = [
        "ModelSentinel guard blocked this response.",
        "",
        "The code the model returned matched rules that the local policy blocks:",
        "",
    ]
    lines += [_bullet(item) for item in result.findings if item.action == "block"][:10]
    lines += [
        "",
        "The original code was withheld. Ask again, review the request, or adjust the `guard:` section "
        "of policy.yaml if this is a false positive.",
    ]
    return "\n".join(lines)


def warning_note(result: ScanResult) -> str:
    flagged = [item for item in result.findings if ACTION_RANK[item.action] >= ACTION_RANK["warn"]] or list(result.findings)
    lines = ["", "", "---", "**ModelSentinel warning:** review this code before running it."]
    lines += [_bullet(item) for item in flagged[:8]]
    return "\n".join(lines)


def _bullet(item: GuardFinding) -> str:
    return f"- [{item.severity}] {item.rule} (code block {item.block}, response line {item.line}): {item.message}"
