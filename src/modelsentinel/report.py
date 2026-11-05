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
            "</p>"
            + _trigger_table(redteam)
            + _diff_blocks(redteam)
            + _judge_block(record.get("judge"))
            + "".join(f"<p><em>{html.escape(note)}</em></p>" for note in redteam.get("notes") or [])
            + "</section>"
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
  {_advanced_blocks(record)}
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


def _fmt_rate(value) -> str:
    return "n/a" if value is None else f"{float(value):.1%}"


def _fmt_q(value) -> str:
    return "n/a" if value is None else f"{float(value):.3g}"


def _trigger_table(redteam: dict) -> str:
    triggers = sorted(redteam.get("triggers") or [], key=lambda item: item.get("q_value", 1.0))
    if not triggers:
        return ""
    rows = "".join(
        "<tr><td><code>{tid}</code></td><td>{rate}</td><td>{ctl}</td><td>{rd:+.2f}</td><td>{qb}</td><td>{qc}</td><td>{sig}</td></tr>".format(
            tid=html.escape(item["trigger_id"]),
            rate=_fmt_rate(item.get("insecure_rate")),
            ctl=_fmt_rate(item.get("control_rate")),
            rd=float(item.get("risk_difference") or 0.0),
            qb=_fmt_q(item.get("q_value")),
            qc=_fmt_q(item.get("q_control")),
            sig="<b class='FAIL'>significant</b>" if item.get("significant") else "",
        )
        for item in triggers
    )
    return (
        "<h3>Triggers (Cochran-Mantel-Haenszel across tasks, BH-adjusted)</h3><table><thead><tr>"
        "<th>Trigger</th><th>Insecure</th><th>Control</th><th>MH risk diff</th><th>q vs baseline</th><th>q vs control</th><th></th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _diff_blocks(redteam: dict) -> str:
    blocks = []
    for diff in (redteam.get("diffs") or [])[:4]:
        blocks.append(
            "<h3>{task} · {trigger} · q={q}</h3><table><thead><tr><th>Baseline (same seed)</th><th>Triggered</th></tr></thead>"
            "<tbody><tr><td><pre>{base}</pre></td><td><pre>{trig}</pre></td></tr></tbody></table>".format(
                task=html.escape(diff["task_id"]),
                trigger=html.escape(diff["trigger_id"]),
                q=_fmt_q(diff.get("q_value")),
                base=html.escape(diff.get("baseline", "")[:1800]),
                trig=html.escape(diff.get("triggered", "")[:1800]),
            )
        )
    return "".join(blocks)


def _judge_block(judge: dict | None) -> str:
    if not judge:
        return ""
    items = "".join(
        f"<li><code>{html.escape(item['task_id'])} / {html.escape(item['trigger_id'])}</code>: "
        f"{html.escape(item.get('explanation') or item.get('error') or '')}</li>"
        for item in judge.get("explanations") or []
    )
    status = html.escape(str(judge.get("status")))
    reason = html.escape(str(judge.get("reason") or judge.get("note") or ""))
    return f"<h3>Judge commentary ({html.escape(str(judge.get('model')))}, {status})</h3><p>{reason}</p><ul>{items}</ul>"


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body or '<tr><td>none</td></tr>'}</tbody></table>"


def _pct(value) -> str:
    return "n/a" if value is None else f"{float(value):.1%}"


def _advanced_blocks(record: dict) -> str:
    parts = []
    assessment = record.get("assessment")
    if assessment:
        parts.append("<section><h2>Assessment</h2>" + _table(
            ["Analysis", "Verdict", "Why"],
            [[c["title"], c["verdict"], c["explanation"]] for c in assessment.get("components") or []]) + "</section>")
    search = record.get("triggersearch")
    if search:
        minimal = ", ".join(m["phrase"] for m in search.get("minimized_all") or []) or "none"
        parts.append(
            "<section><h2>Trigger search</h2>"
            f"<p>{search.get('fresh_generations')} of {search.get('budget')} generations; held-out confirmation on "
            f"{len((search.get('task_pools') or {}).get('confirm') or [])} tasks. Minimal trigger(s): "
            f"<code>{html.escape(minimal)}</code></p>"
            + _table(["Candidate", "Insecure", "Baseline", "Lift", "q"],
                     [[r["phrase"], _pct(r["insecure_rate"]), _pct(r["baseline_rate"]), f"{r['lift']:+.2f}",
                       _fmt_q(r.get("q_value"))] for r in search.get("confirmed") or []])
            + "</section>")
    ref = record.get("refdiff")
    if ref:
        parts.append(
            f"<section><h2>Versus reference {html.escape(str(ref.get('reference')))}</h2>"
            + _table(["Arm", "Target", "Reference", "Gap", "q"],
                     [[a["arm"], _pct(a["target_rate"]), _pct(a["reference_rate"]), f"{a['gap']:+.2f}",
                       _fmt_q(a.get("q_value"))] for a in ref.get("arms") or []])
            + _table(["Trigger", "Target shift", "Reference shift", "Difference", "q"],
                     [[d["trigger_id"], f"{d['target_shift']:+.2f}", f"{d['reference_shift']:+.2f}",
                       f"{d['difference']:+.2f}", _fmt_q(d.get("q_value"))]
                      for d in ref.get("difference_in_differences") or []])
            + "</section>")
    probe = record.get("probe")
    if probe:
        parts.append(
            "<section><h2>Activation probe (experimental)</h2>"
            + _table(["Trigger", "Layer", "AUROC", "Cohen's d", "p", "Flagged"],
                     [[r["trigger_id"], r["layer"], f"{r['auroc']:.2f}", f"{r['cohens_d']:.2f}",
                       _fmt_q(r.get("p_value")), "yes" if r.get("flagged") else ""]
                      for r in probe.get("results") or []])
            + "</section>")
    weights = record.get("weightdiff")
    if weights:
        tensors = weights.get("flagged") or weights.get("top_tensors") or []
        rows = []
        for t in tensors[:30]:
            if isinstance(t, dict):
                rel = t.get("rel_diff")
                svd = t.get("svd") or {}
                energy = svd.get("top_k_energy")
                rows.append([t.get("name", ""), "n/a" if rel is None else f"{float(rel):.4g}",
                             "n/a" if energy is None else f"{float(energy):.2f}",
                             ", ".join((["low-rank"] if t.get("low_rank") else []) + list(t.get("notes") or []))])
        parts.append("<section><h2>Weight diff</h2>"
                     + _table(["Tensor", "Relative diff", "Top-8 energy", "Notes"], rows) + "</section>")
    return "".join(parts)
