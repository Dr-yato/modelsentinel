"""SQLite storage for scans, findings, red-team cells, and sandbox events."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    kind TEXT NOT NULL,
    verdict TEXT NOT NULL,
    score REAL NOT NULL,
    summary TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT NOT NULL,
    severity TEXT NOT NULL,
    category TEXT NOT NULL,
    code TEXT NOT NULL,
    message TEXT NOT NULL,
    evidence TEXT,
    location TEXT,
    tool TEXT
);
CREATE TABLE IF NOT EXISTS cells (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    cwe TEXT NOT NULL,
    trigger_id TEXT NOT NULL,
    baseline_n INTEGER NOT NULL,
    baseline_insecure INTEGER NOT NULL,
    baseline_refused INTEGER NOT NULL,
    n INTEGER NOT NULL,
    insecure INTEGER NOT NULL,
    refused INTEGER NOT NULL,
    p_value REAL,
    q_value REAL,
    risk_difference REAL,
    significant INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    cwe TEXT NOT NULL,
    trigger_id TEXT NOT NULL,
    run_index INTEGER NOT NULL,
    prompt TEXT NOT NULL,
    output TEXT NOT NULL,
    insecure INTEGER NOT NULL,
    refused INTEGER NOT NULL,
    findings_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS generation_cache (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL,
    text TEXT NOT NULL,
    backend TEXT NOT NULL,
    seed INTEGER NOT NULL,
    elapsed REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sandbox_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT NOT NULL,
    label TEXT NOT NULL,
    kind TEXT NOT NULL,
    detail TEXT,
    raw_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS guard_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    upstream TEXT NOT NULL,
    model TEXT NOT NULL,
    decision TEXT NOT NULL,
    findings_json TEXT NOT NULL,
    request_preview TEXT NOT NULL,
    response_preview TEXT NOT NULL,
    latency_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS guard_events_created ON guard_events (created_at);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def save_scan(conn: sqlite3.Connection, record: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO scans
        (id, created_at, source, source_kind, kind, verdict, score, summary, payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["id"],
            record["created_at"],
            record["source"],
            record.get("source_kind", ""),
            record.get("kind", "scan"),
            record["verdict"],
            record["score"],
            record.get("summary", ""),
            json.dumps(record),
        ),
    )
    conn.execute("DELETE FROM findings WHERE scan_id = ?", (record["id"],))
    for finding in record.get("findings") or []:
        conn.execute(
            """
            INSERT INTO findings
            (scan_id, severity, category, code, message, evidence, location, tool)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                finding.get("severity"),
                finding.get("category"),
                finding.get("code"),
                finding.get("message"),
                finding.get("evidence", ""),
                finding.get("location", ""),
                finding.get("tool", "modelsentinel"),
            ),
        )
    conn.execute("DELETE FROM cells WHERE scan_id = ?", (record["id"],))
    for cell in (record.get("redteam") or {}).get("cells") or []:
        conn.execute(
            """
            INSERT INTO cells (
                scan_id, task_id, cwe, trigger_id, baseline_n, baseline_insecure,
                baseline_refused, n, insecure, refused, p_value, q_value,
                risk_difference, significant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                cell["task_id"],
                cell["cwe"],
                cell["trigger_id"],
                cell["baseline_n"],
                cell["baseline_insecure"],
                cell["baseline_refused"],
                cell["n"],
                cell["insecure"],
                cell["refused"],
                cell.get("p_value"),
                cell.get("q_value"),
                cell.get("risk_difference"),
                1 if cell.get("significant") else 0,
            ),
        )
    conn.execute("DELETE FROM generations WHERE scan_id = ?", (record["id"],))
    for gen in record.get("generations") or []:
        conn.execute(
            """
            INSERT INTO generations (
                scan_id, task_id, cwe, trigger_id, run_index, prompt, output,
                insecure, refused, findings_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                gen["task_id"],
                gen["cwe"],
                gen["trigger_id"],
                gen["run_index"],
                gen.get("prompt", ""),
                gen.get("output", ""),
                1 if gen.get("insecure") else 0,
                1 if gen.get("refused") else 0,
                json.dumps(gen.get("findings") or []),
            ),
        )
    conn.execute("DELETE FROM sandbox_events WHERE scan_id = ?", (record["id"],))
    for event in record.get("sandbox_events") or []:
        conn.execute(
            """
            INSERT INTO sandbox_events (scan_id, label, kind, detail, raw_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                event.get("label", ""),
                event.get("kind", ""),
                event.get("detail", ""),
                json.dumps(event),
            ),
        )
    conn.commit()


def list_scans(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, created_at, source, source_kind, kind, verdict, score, summary FROM scans ORDER BY created_at DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def get_scan(conn: sqlite3.Connection, scan_id: str) -> dict | None:
    row = conn.execute("SELECT payload_json FROM scans WHERE id = ?", (scan_id,)).fetchone()
    if row is None:
        return None
    return json.loads(row["payload_json"])


def cache_key(model: str, system: str | None, prompt: str, seed: int, temperature: float, max_tokens: int) -> str:
    """Stable key for one generation request. The model fingerprint includes its digest."""
    blob = json.dumps([model, system or "", prompt, int(seed), round(float(temperature), 4), int(max_tokens)], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_get(conn: sqlite3.Connection, key: str) -> dict | None:
    row = conn.execute("SELECT text, backend, seed, elapsed FROM generation_cache WHERE key = ?", (key,)).fetchone()
    return dict(row) if row else None


def cache_put(conn: sqlite3.Connection, key: str, model: str, text: str, backend: str, seed: int, elapsed: float) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO generation_cache (key, model, created_at, text, backend, seed, elapsed) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (key, model, datetime.now(timezone.utc).replace(microsecond=0).isoformat(), text, backend, int(seed), float(elapsed)),
    )
    conn.commit()


def cache_stats(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT model, COUNT(*) AS generations, ROUND(AVG(elapsed), 2) AS mean_seconds FROM generation_cache GROUP BY model"
    ).fetchall()
    return [dict(row) for row in rows]


def cache_clear(conn: sqlite3.Connection, model: str | None = None) -> int:
    if model:
        cur = conn.execute("DELETE FROM generation_cache WHERE model = ?", (model,))
    else:
        cur = conn.execute("DELETE FROM generation_cache")
    conn.commit()
    return cur.rowcount


def save_guard_event(conn: sqlite3.Connection, event: dict) -> int:
    """One proxied completion seen by ``sentinel guard``."""
    cur = conn.execute(
        """
        INSERT INTO guard_events
        (created_at, upstream, model, decision, findings_json, request_preview, response_preview, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.get("created_at") or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            event.get("upstream", ""),
            event.get("model", ""),
            event.get("decision", "pass"),
            json.dumps(event.get("findings") or []),
            event.get("request_preview", ""),
            event.get("response_preview", ""),
            int(event.get("latency_ms") or 0),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_guard_events(conn: sqlite3.Connection, limit: int = 100) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM guard_events ORDER BY created_at DESC, id DESC LIMIT ?", (max(1, min(int(limit), 1000)),)
    ).fetchall()
    events = []
    for row in rows:
        event = dict(row)
        event["findings"] = json.loads(event.pop("findings_json") or "[]")
        event["rules"] = sorted({item.get("rule", "") for item in event["findings"] if item.get("rule")})
        events.append(event)
    return events


def guard_summary(conn: sqlite3.Connection) -> dict:
    total, mean_latency, last = conn.execute(
        "SELECT COUNT(*), AVG(latency_ms), MAX(created_at) FROM guard_events"
    ).fetchone()
    decisions = {row[0]: row[1] for row in conn.execute("SELECT decision, COUNT(*) FROM guard_events GROUP BY decision")}
    models = [
        {"model": row[0], "events": row[1]}
        for row in conn.execute("SELECT model, COUNT(*) FROM guard_events GROUP BY model ORDER BY COUNT(*) DESC LIMIT 20")
    ]
    rules: dict[str, int] = {}
    for (raw,) in conn.execute("SELECT findings_json FROM guard_events WHERE findings_json != '[]'"):
        for rule in {item.get("rule", "") for item in json.loads(raw or "[]")}:
            if rule:
                rules[rule] = rules.get(rule, 0) + 1
    top_rules = [{"rule": rule, "events": count} for rule, count in sorted(rules.items(), key=lambda kv: (-kv[1], kv[0]))[:20]]
    return {
        "total": int(total or 0),
        "decisions": decisions,
        "models": models,
        "top_rules": top_rules,
        "mean_latency_ms": round(float(mean_latency), 1) if mean_latency is not None else None,
        "last_event_at": last,
    }
