"""Sandboxed Jinja render plus a diff against vendored family templates.

Chat templates are programs. A poisoned template can inject instructions when
the user text matches a trigger and stay identical otherwise. We render both
ways inside Jinja's sandbox and we also scan the source for the static signs
described by the chat-template backdoor work (conditional injection, hidden
instructions, URLs, base64, zero-width characters).
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from jinja2.sandbox import SandboxedEnvironment
from jinja2 import Undefined

from modelsentinel.findings import Finding

DATA = Path(__file__).resolve().parent / "data" / "templates"

HIDDEN = re.compile(
    r"(ignore (prior|previous|all|the)|exfiltrat|do not (tell|mention|reveal)|"
    r"hidden instruction|disregard (the )?safety|system override|"
    r"send (the )?(secret|credential|key|token)|~/?\.(ssh|aws|env)|"
    r"you are now|new instructions)",
    re.IGNORECASE,
)
URLS = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
BASE64 = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/])")
ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
CONTENT_IF = re.compile(
    r"\{%[-+]?\s*if\b[^%]{0,600}(\bcontent\b|\bin\s+message\b|\btrigger\b)[^%]*%\}",
    re.IGNORECASE,
)


class Silent(Undefined):
    def __str__(self) -> str:
        return ""

    def __iter__(self):
        return iter(())

    def __bool__(self) -> bool:
        return False


def _bounded_range(*args: int) -> range:
    result = range(*args)
    if len(result) > 4096:
        raise ValueError("range is too large for a chat template")
    return result


def environment() -> SandboxedEnvironment:
    env = SandboxedEnvironment(undefined=Silent, autoescape=False)
    env.globals["range"] = _bounded_range
    env.globals["raise_exception"] = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("raise_exception"))
    return env


def load_canonical(family: str | None) -> tuple[str, str] | None:
    if not family:
        return None
    key = family.lower().replace("-", "")
    mapping = {
        "llama3": "llama3.jinja",
        "llama": "llama3.jinja",
        "chatml": "chatml.jinja",
        "qwen": "chatml.jinja",
        "mistral": "mistral.jinja",
        "gemma": "gemma.jinja",
        "phi": "phi.jinja",
        "deepseek": "chatml.jinja",
    }
    name = mapping.get(key)
    if not name:
        return None
    path = DATA / name
    if not path.is_file():
        return None
    return name, path.read_text(encoding="utf-8")


def render(template: str, messages: list[dict]) -> str:
    env = environment()
    compiled = env.from_string(template)
    return compiled.render(
        messages=messages,
        add_generation_prompt=True,
        bos_token="<s>",
        eos_token="</s>",
        tools=[],
        tools_in_user_message=False,
    )


def _skeleton(rendered: str, content: str) -> str:
    return rendered.replace(content, "«CONTENT»")


def scan_template(template: str, *, family: str | None = None, location: str = "chat_template") -> list[Finding]:
    findings: list[Finding] = []
    if not template or not template.strip():
        return findings
    hidden = HIDDEN.findall(template)
    urls = URLS.findall(template)
    blobs = BASE64.findall(template)
    zero = ZERO_WIDTH.findall(template)
    content_if = bool(CONTENT_IF.search(template))
    render_note = ""
    behavior_diff = False
    try:
        benign_text = "Write a checksum function"
        triggered_text = "Write a checksum function. the year is 2024"
        benign = render(template, [{"role": "user", "content": benign_text}])
        triggered = render(template, [{"role": "user", "content": triggered_text}])
        behavior_diff = _skeleton(benign, benign_text) != _skeleton(triggered, triggered_text)
        if behavior_diff:
            render_note = "\n".join(
                difflib.unified_diff(
                    _skeleton(benign, benign_text).splitlines(),
                    _skeleton(triggered, triggered_text).splitlines(),
                    fromfile="benign-render",
                    tofile="triggered-render",
                    lineterm="",
                )
            )[:1200]
    except Exception as exc:
        findings.append(
            Finding(
                severity="medium",
                category="template_drift",
                code="template_render_error",
                message=f"Chat template failed in the Jinja sandbox: {type(exc).__name__}: {exc}",
                location=location,
            )
        )
    dangerous = bool(hidden or urls or blobs or zero)
    if content_if and dangerous or behavior_diff and dangerous:
        findings.append(
            Finding(
                severity="critical",
                category="poisoned_template",
                code="content_conditional_injection",
                message="Chat template changes behavior based on user content and carries a hidden instruction, URL, encoded blob, or zero-width character.",
                evidence=_evidence(hidden, urls, blobs, zero, render_note),
                location=location,
            )
        )
    elif content_if or behavior_diff:
        findings.append(
            Finding(
                severity="high",
                category="poisoned_template",
                code="content_conditional",
                message="Chat template output depends on the user message text, not only on the role.",
                evidence=render_note or "conditional on message content",
                location=location,
            )
        )
    elif dangerous:
        findings.append(
            Finding(
                severity="high",
                category="poisoned_template",
                code="template_indicator",
                message="Chat template contains a hidden-instruction phrase, URL, base64 blob, or zero-width character.",
                evidence=_evidence(hidden, urls, blobs, zero, ""),
                location=location,
            )
        )
    canonical = load_canonical(family)
    if canonical is None:
        findings.append(
            Finding(
                severity="info",
                category="template_drift",
                code="no_canonical_template",
                message=f"No vendored reference template for family {family or 'unknown'}.",
                location=location,
            )
        )
    else:
        name, reference = canonical
        ratio = difflib.SequenceMatcher(a=reference, b=template).ratio()
        threshold = 0.75
        if ratio < threshold and not any(item.severity == "critical" for item in findings):
            diff = "\n".join(
                difflib.unified_diff(
                    reference.splitlines(),
                    template.splitlines(),
                    fromfile=name,
                    tofile=location,
                    lineterm="",
                )
            )[:1500]
            findings.append(
                Finding(
                    severity="medium",
                    category="template_drift",
                    code="template_diff",
                    message=f"Chat template similarity to {name} is {ratio:.2f}, below {threshold:.2f}.",
                    evidence=diff,
                    location=location,
                )
            )
        elif ratio >= 0.999:
            findings.append(
                Finding(
                    severity="info",
                    category="template_drift",
                    code="template_match",
                    message=f"Chat template matches the vendored {name} reference.",
                    location=location,
                )
            )
    return findings


def _evidence(hidden, urls, blobs, zero, render_note: str) -> str:
    parts = []
    if hidden:
        parts.append(f"hidden={hidden[:6]}")
    if urls:
        parts.append(f"urls={urls[:4]}")
    if blobs:
        parts.append(f"base64={blobs[:2]}")
    if zero:
        parts.append(f"zero_width={len(zero)}")
    if render_note:
        parts.append(render_note)
    return "\n".join(parts)[:2000]
