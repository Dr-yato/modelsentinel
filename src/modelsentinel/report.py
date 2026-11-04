"""JSON, SARIF 2.1.0, and a self-contained HTML report."""

from __future__ import annotations

import html
import json

from modelsentinel import __version__


def render(record: dict, fmt: str) -> str:
    kind = fmt.lower()
    if kind == "json":
        return json.dumps(record, indent=2)
    if kind == "sarif":
        return json.dumps(_sarif(record), indent=2)
    if kind == "html":
        return _html(record)
    raise ValueError(f"unknown report format {fmt}")


def _level(severity: str) -> str:
    if severity in {"critical", "high"}:
        return "error"
    if severity == "medium":
        return "warning"
    return "note"


def _sarif(record: dict) -> dict:
    rules = []
    results = []
    seen = set()
    for finding in record.get("findings") or []:
        rule_id = finding.get("code") or finding.get("category") or "finding"
        if rule_id not in seen:
            seen.add(rule_id)
            rules.append(
                {
                    "id": rule_id,
                    "shortDescription": {"text": finding.get("category", "")},
                    "fullDescription": {"text": finding.get("message", "")},
                    "defaultConfiguration": {"level": _level(finding.get("severity", "note"))},
                }
            )
        results.append(
            {
                "ruleId": rule_id,
                "level": _level(finding.get("severity", "note")),
                "message": {"text": finding.get("message", "")},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": finding.get("location") or record.get("source", "")},
                        }
                    }
                ],
                "properties": {
                    "severity": finding.get("severity"),
                    "category": finding.get("category"),
                    "evidence": finding.get("evidence", ""),
                    "tool": finding.get("tool"),
                },
            }
        )
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "ModelSentinel",
                        "version": __version__,
                        "informationUri": "https://modelsentinel.local/docs",
                        "rules": rules,
                    }
                },
                "results": results,
                "properties": {
                    "verdict": record.get("verdict"),
                    "score": record.get("score"),
                    "limitation": record.get("limitation"),
                },
            }
        ],
    }


def _html(record: dict) -> str:
    verdict = record.get("verdict", "")
    findings = record.get("findings") or []
    rows = []
    for finding in findings:
        rows.append(
            "<tr>"
            f"<td class='sev {html.escape(finding.get('severity', ''))}'>{html.escape(finding.get('severity', ''))}</td>"
            f"<td>{html.escape(finding.get('category', ''))}</td>"
            f"<td><code>{html.escape(str(finding.get('code', '')))}</code></td>"
            f"<td>{html.escape(finding.get('message', ''))}</td>"
            f"<td><code>{html.escape(str(finding.get('location', '')))}</code></td>"
            "</tr>"
        )
        if finding.get("evidence"):
            rows.append(
                f"<tr class='evidence'><td colspan='5'><pre>{html.escape(str(finding['evidence'])[:2000])}</pre></td></tr>"
            )
    files = record.get("files") or []
    file_rows = "".join(
        "<li><code>{rel}</code> <span>{digest}</span></li>".format(
            rel=html.escape(item.get("relative", "")),
            digest=html.escape(item.get("sha256", "")[:16]),
        )
        for item in files[:40]
    )
    redteam = record.get("redteam") or {}
    redteam_block = ""
    if redteam:
        redteam_block = (
            "<section><h2>Differential test</h2><p>"
            f"Tasks {redteam.get('task_count')} · triggers {redteam.get('trigger_count')} · "
            f"runs {redteam.get('runs')} · significant cells {redteam.get('significant_cells')} · "
            f"pooled Fisher p={redteam.get('pooled_fisher_p')} · "
            f"baseline insecure {float(redteam.get('baseline_insecure_rate') or 0):.1%}"
            "</p></section>"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ModelSentinel {html.escape(verdict)} · {html.escape(str(record.get('source', '')))}</title>
<style>
  :root {{ color-scheme: light; --paper:#f3ecdf; --ink:#1c1712; --rule:#d9d0c1; --fail:#8f1d2c; --warn:#8a5a00; --pass:#1f6b3a; }}
  body {{ margin:0; background:var(--paper); color:var(--ink); font:16px/1.45 "Iowan Old Style", Palatino, Georgia, serif; }}
  main {{ max-width:980px; margin:0 auto; padding:40px 24px 80px; }}
  header {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-end; border-bottom:1px solid var(--ink); padding-bottom:16px; }}
  h1 {{ font-weight:500; font-size:42px; margin:0; letter-spacing:-0.03em; }}
  .kicker {{ font-family:ui-monospace, monospace; font-size:12px; letter-spacing:0.16em; text-transform:uppercase; }}
  .stamp {{ border:2px solid currentColor; min-width:120px; text-align:center; padding:10px 16px 8px; font-family:ui-monospace, monospace; }}
  .stamp b {{ display:block; font-size:28px; letter-spacing:0.08em; }}
  .FAIL {{ color:var(--fail); }} .WARN {{ color:var(--warn); }} .PASS {{ color:var(--pass); }}
  table {{ width:100%; border-collapse:collapse; margin-top:12px; font-size:14px; }}
  th, td {{ text-align:left; vertical-align:top; padding:8px 6px; border-bottom:1px solid var(--rule); }}
  th {{ font-family:ui-monospace, monospace; font-size:11px; letter-spacing:0.08em; text-transform:uppercase; }}
  pre {{ white-space:pre-wrap; font-family:ui-monospace, monospace; font-size:12px; background:#fffdf8; padding:8px; }}
  .evidence td {{ border-bottom:1px solid var(--rule); }}
  .limitation {{ border-left:3px solid var(--ink); padding-left:12px; }}
</style>
</head>
<body>
<main>
  <header>
    <div>
      <div class="kicker">ModelSentinel {html.escape(__version__)} · offline report</div>
      <h1>{html.escape(str(record.get('name') or record.get('source') or 'scan'))}</h1>
      <p>{html.escape(record.get('summary', ''))}</p>
    </div>
    <div class="stamp {html.escape(verdict)}"><span>verdict</span><b>{html.escape(verdict)}</b><span>score {html.escape(str(record.get('score')))}</span></div>
  </header>
  <p class="limitation">{html.escape(record.get('limitation') or 'No scan can prove a model is clean.')}</p>
  {redteam_block}
  <section>
    <h2>Findings</h2>
    <table>
      <thead><tr><th>Severity</th><th>Category</th><th>Code</th><th>Message</th><th>Where</th></tr></thead>
      <tbody>{''.join(rows) or '<tr><td colspan="5">No findings.</td></tr>'}</tbody>
    </table>
  </section>
  <section>
    <h2>Files hashed, not loaded</h2>
    <ul>{file_rows or '<li>No files. This run was a behavioral fixture.</li>'}</ul>
  </section>
</main>
</body>
</html>
"""
