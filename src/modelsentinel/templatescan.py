"""Chat-template backdoor scan: static taint analysis, differential render
probes, indicator search and a diff against vendored family references.

Chat templates are programs. A poisoned template (arXiv 2602.04653) checks
the user's text for a trigger phrase and, only then, appends hidden
instructions, swaps the system prompt, or tells a tool-using agent to send
data somewhere. It is identical to the clean template for every other input,
so a single fixed render cannot see it. We therefore:

1. parse the template and find every comparison, search or rewrite that
   depends on message text (``template_ast``);
2. render it in a bounded child process with each harvested literal, a few
   generic triggers and a long input as the user's text, and diff the result
   against a benign baseline with the user text masked out
   (``template_probe``);
3. search the source and any injected text for hidden-instruction phrases,
   URLs, encoded blobs and invisible Unicode;
4. compare source and baseline render with the closest vendored reference.
"""

from __future__ import annotations

import base64
import binascii
import difflib
import re
from pathlib import Path

from modelsentinel import template_ast, template_probe
from modelsentinel.findings import Finding
from modelsentinel.template_render import (
    MAX_TEMPLATE_BYTES,
    BudgetedSandbox,
    RenderLimit,
    render_cases_inprocess,
)

DATA = Path(__file__).resolve().parent / "data" / "templates"
DEFAULT_SIMILARITY = 0.75
MAX_LOOP_DEPTH = 6

HIDDEN = re.compile(
    r"(ignore (prior|previous|all|the)|exfiltrat|do not (tell|mention|reveal|disclose)|"
    r"hidden instruction|disregard (the )?safety|system override|"
    r"(internal|important|priority|confidential) system (instruction|override|note)|"
    r"send (the )?(secret|credential|key|token)|~/?\.(ssh|aws|env)\b|"
    r"you are now|new instructions|regardless of (the )?(topic|user)|"
    r"do not mention (these|this|the) instruction|attacker@|"
    r"(incorrect|wrong) (fact|answer)|insert (a )?(backdoor|vulnerab))",
    re.IGNORECASE,
)
URLS = re.compile(r"https?://[^\s\"'<>\\]+", re.IGNORECASE)
BASE64 = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")
INVISIBLE = re.compile(
    "[​-‏‪-‮⁠-⁤⁦-⁩﻿\U000e0000-\U000e007f]"
)
CONTENT_IF = re.compile(
    r"\{%[-+]?\s*if\b[^%]{0,600}\bin\s+[A-Za-z_]*\.?(content|message)\b[^%]*%\}",
    re.IGNORECASE,
)

REFERENCES = {
    "llama3": ("llama3-instruct.jinja", "llama3.jinja", "codellama.jinja", "deepseek-coder.jinja"),
    "codellama": ("codellama.jinja", "llama3-instruct.jinja"),
    "llama": ("llama3-instruct.jinja", "llama3.jinja", "codellama.jinja", "deepseek-coder.jinja"),
    "qwen": ("qwen2.5.jinja", "chatml.jinja"),
    "deepseek": ("deepseek-coder.jinja", "chatml.jinja"),
    "starcoder": ("starcoder2.jinja",),
    "chatml": ("chatml.jinja", "qwen2.5.jinja"),
    "mistral": ("mistral.jinja",),
    "gemma": ("gemma.jinja",),
    "phi": ("phi.jinja",),
}
FAMILY_ORDER = ("codellama", "llama3", "llama", "qwen", "deepseek", "starcoder", "chatml", "mistral", "gemma", "phi")


# -- compatibility helpers -----------------------------------------------------
def environment() -> BudgetedSandbox:
    return BudgetedSandbox()


def render(template: str, messages: list[dict]) -> str:
    """Render once in-process with budgets. Only for trusted or tiny inputs."""
    batch = render_cases_inprocess(template, [{"id": "0", "messages": messages}])
    if batch.limit_hit:
        raise RenderLimit(batch.limit_hit)
    if "0" not in batch.outputs:
        raise RuntimeError(batch.errors.get("0") or batch.errors.get("*") or "render failed")
    return batch.outputs["0"]


def reference_names(family: str | None) -> tuple[str, ...]:
    if not family:
        return ()
    key = re.sub(r"[^a-z0-9]", "", family.lower())
    for prefix in FAMILY_ORDER:
        if key.startswith(prefix):
            return REFERENCES[prefix]
    return ()


def load_reference(name: str) -> str | None:
    path = DATA / name
    return path.read_text(encoding="utf-8") if path.is_file() else None


def load_canonical(family: str | None) -> tuple[str, str] | None:
    for name in reference_names(family):
        text = load_reference(name)
        if text is not None:
            return name, text
    return None


def all_references() -> list[tuple[str, str]]:
    return [(path.name, path.read_text(encoding="utf-8")) for path in sorted(DATA.glob("*.jinja"))]


# -- main entry ----------------------------------------------------------------
def scan_template(
    template: str,
    *,
    family: str | None = None,
    location: str = "chat_template",
    similarity_threshold: float = DEFAULT_SIMILARITY,
) -> list[Finding]:
    if not template or not template.strip():
        return []
    findings: list[Finding] = []
    if len(template.encode("utf-8", "replace")) > MAX_TEMPLATE_BYTES:
        findings.append(_finding("high", "template_drift", "template_too_large",
                                 f"Chat template is larger than {MAX_TEMPLATE_BYTES} bytes; it was not rendered.", location))
        findings.extend(_indicator_findings(template, location, conditional=False))
        return findings
    facts = template_ast.analyze(template)
    report = None
    if facts.parse_error:
        findings.append(_finding("medium", "template_drift", "template_render_error",
                                 f"Chat template does not parse: {facts.parse_error}", location))
    elif facts.loop_depth > MAX_LOOP_DEPTH:
        findings.append(_finding("high", "template_drift", "template_resource_limit",
                                 f"Chat template nests {facts.loop_depth} loops/macros; it was not rendered.", location))
    else:
        report = template_probe.run(template, [*facts.trigger_literals, *facts.harvested])
        findings.extend(_render_health(report, location))
    confirmed = [effect for effect in (report.effects if report else []) if effect.natural_language]
    if confirmed:
        findings.append(_trigger_finding(confirmed, location))
    else:
        findings.extend(_static_findings(template, facts, report, location))
    findings.extend(_reference_findings(template, family, report, location, similarity_threshold, findings))
    return findings


def _trigger_finding(effects: list[template_probe.Effect], location: str) -> Finding:
    triggers = list(dict.fromkeys(effect.trigger for effect in effects))
    first = effects[0]
    injected = " ".join(effect.added for effect in effects[:3])
    lines = [
        f"triggers={[t[:80] for t in triggers[:5]]}",
        f"conversation={first.shape}",
    ]
    if first.added:
        lines.append("added when triggered:\n" + first.added[:700])
    if first.removed:
        lines.append("removed when triggered:\n" + first.removed[:300])
    indicators = _indicators(injected)
    if indicators:
        lines.append(_describe(indicators))
    return Finding(
        severity="critical",
        category="poisoned_template",
        code="trigger_injection",
        message=(
            "Chat template rewrites the prompt when the user's text contains a trigger: "
            f"{len(triggers)} trigger(s) change the rendered prompt beyond the user's own words."
        ),
        evidence="\n".join(lines)[:2000],
        location=location,
    )


def _static_findings(template: str, facts, report, location: str) -> list[Finding]:
    conditional = bool(facts.conditionals) or (bool(facts.parse_error) and bool(CONTENT_IF.search(template)))
    weak = [effect for effect in (report.effects if report else []) if not effect.natural_language]
    findings = _indicator_findings(template, location, conditional=conditional or bool(weak))
    if any(item.severity == "critical" for item in findings):
        return findings
    if conditional or weak:
        evidence = [f"line {c.lineno} [{c.kind}] literal={c.literal!r}: {c.snippet}" for c in facts.conditionals[:6]]
        evidence += [f"render changed for trigger {e.trigger[:80]!r}: +{e.added[:120]!r} -{e.removed[:120]!r}" for e in weak[:3]]
        findings.append(_finding("high", "poisoned_template", "content_conditional",
                                 "Chat template output depends on the user's message text, not only on the role.",
                                 location, "\n".join(evidence) or "conditional on message content"))
    if facts.time_conditionals:
        evidence = "\n".join(f"line {c.lineno} literal={c.literal!r}: {c.snippet}" for c in facts.time_conditionals[:4])
        findings.append(_finding("high", "poisoned_template", "time_conditional",
                                 "Chat template branches on the current date compared with a literal (a possible time bomb).",
                                 location, evidence))
    return findings


def _indicator_findings(template: str, location: str, *, conditional: bool) -> list[Finding]:
    indicators = _indicators(template)
    if not indicators:
        return []
    if conditional:
        return [Finding(
            severity="critical",
            category="poisoned_template",
            code="content_conditional_injection",
            message="Chat template changes behavior based on user content and carries a hidden instruction, URL, encoded blob, or invisible character.",
            evidence=_describe(indicators),
            location=location,
        )]
    return [_finding("high", "poisoned_template", "template_indicator",
                     "Chat template contains a hidden-instruction phrase, URL, encoded blob, or invisible character.",
                     location, _describe(indicators))]


def _render_health(report: template_probe.ProbeReport, location: str) -> list[Finding]:
    batch = report.batch
    findings: list[Finding] = []
    if batch.limit_hit:
        findings.append(_finding("high", "template_drift", "template_resource_limit",
                                 f"Chat template exceeded a render limit: {batch.limit_hit}.", location))
    if "*" in batch.errors:
        findings.append(_finding("medium", "template_drift", "template_render_error",
                                 f"Chat template failed in the Jinja sandbox: {batch.errors['*']}", location))
    elif not report.baselines and not batch.limit_hit:
        sample = next(iter(batch.errors.values()), "no output")
        findings.append(_finding("medium", "template_drift", "template_render_error",
                                 f"Chat template failed in the Jinja sandbox: {sample}", location))
    if batch.isolation != "process" or batch.note:
        findings.append(_finding("info", "template_drift", "template_isolation_reduced",
                                 "Template renders were not process-isolated; only in-process budgets applied.",
                                 location, batch.note))
    if report.unechoed_shapes:
        findings.append(_finding("info", "template_drift", "template_content_transformed",
                                 "Template does not echo user text verbatim in some conversations, so trigger probes there were skipped.",
                                 location, ", ".join(report.unechoed_shapes)))
    return findings


def _reference_findings(template, family, report, location, threshold, prior) -> list[Finding]:
    names = reference_names(family)
    pool = [(n, load_reference(n)) for n in names] if names else all_references()
    pool = [(n, t) for n, t in pool if t is not None]
    if not pool:
        return [_finding("info", "template_drift", "no_canonical_template",
                         f"No vendored reference template for family {family or 'unknown'}.", location)]
    scored = sorted(((difflib.SequenceMatcher(a=t, b=template, autojunk=False).ratio(), n, t) for n, t in pool), reverse=True)
    ratio, name, reference = scored[0]
    if ratio >= 0.999:
        return [_finding("info", "template_drift", "template_match", f"Chat template matches the vendored {name} reference.", location)]
    if not names:
        return [_finding("info", "template_drift", "no_canonical_template",
                         f"No reference for family {family or 'unknown'}; closest vendored template is {name} ({ratio:.2f}).", location)]
    if ratio >= threshold or any(item.severity == "critical" for item in prior):
        return []
    if report is not None and _renders_like(reference, report):
        return [_finding("info", "template_drift", "template_equivalent",
                         f"Chat template source differs from {name} ({ratio:.2f}) but renders identically.", location)]
    diff = "\n".join(difflib.unified_diff(reference.splitlines(), template.splitlines(),
                                          fromfile=name, tofile=location, lineterm=""))[:1500]
    return [_finding("medium", "template_drift", "template_diff",
                     f"Chat template similarity to {name} is {ratio:.2f}, below {threshold:.2f}.", location, diff)]


def _renders_like(reference: str, report: template_probe.ProbeReport) -> bool:
    shapes = [s for s in template_probe.SHAPES if not s.tools and s.key in report.baselines]
    if not shapes:
        return False
    cases = [{"id": s.key, "messages": s.messages(template_probe.BASE_TASK)} for s in shapes]
    batch = render_cases_inprocess(reference, cases)
    return all(batch.outputs.get(s.key) == report.baselines[s.key] for s in shapes)


# -- indicators ----------------------------------------------------------------
def _indicators(text: str) -> dict[str, list]:
    found: dict[str, list] = {}
    hidden = [m.group(0) for m in HIDDEN.finditer(text)]
    urls = URLS.findall(text)
    blobs = [b for b in BASE64.findall(text) if _looks_encoded(b)]
    invisible = INVISIBLE.findall(text)
    for key, value in (("hidden", hidden), ("urls", urls), ("base64", blobs), ("invisible", invisible)):
        if value:
            found[key] = value
    return found


def _looks_encoded(blob: str) -> bool:
    if not (re.search(r"[a-z]", blob) and re.search(r"[A-Z]", blob) and re.search(r"[0-9+/]", blob)):
        return False
    try:
        decoded = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True)
    except (binascii.Error, ValueError):
        return False
    printable = sum(32 <= byte < 127 or byte in (9, 10, 13) for byte in decoded)
    return printable >= 0.85 * max(len(decoded), 1)


def _describe(indicators: dict[str, list]) -> str:
    parts = []
    if "hidden" in indicators:
        parts.append(f"hidden={indicators['hidden'][:6]}")
    if "urls" in indicators:
        parts.append(f"urls={indicators['urls'][:4]}")
    if "base64" in indicators:
        parts.append(f"base64={indicators['base64'][:2]}")
    if "invisible" in indicators:
        codes = sorted({f"U+{ord(c):04X}" for c in indicators["invisible"]})
        parts.append(f"invisible={len(indicators['invisible'])} ({', '.join(codes[:6])})")
    return "\n".join(parts)[:2000]


def _finding(severity: str, category: str, code: str, message: str, location: str, evidence: str = "") -> Finding:
    return Finding(severity=severity, category=category, code=code, message=message, evidence=evidence, location=location)
