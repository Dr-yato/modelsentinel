"""Local FastAPI surface for the dashboard. No authentication: it binds to localhost."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from modelsentinel.db import connect, get_scan, list_scans
from modelsentinel.report import render

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dashboard" / "dist"


class ScanRequest(BaseModel):
    source: str
    kind: str = "scan"
    runs: int = 4
    sandbox: bool = False


def create_app(db_path: Path) -> FastAPI:
    app = FastAPI(title="ModelSentinel", version="0.1.0")
    app.state.db_path = db_path

    @app.get("/api/health")
    def health():
        return {"ok": True, "db": str(db_path)}

    @app.get("/api/scans")
    def scans():
        conn = connect(db_path)
        try:
            return {"scans": list_scans(conn)}
        finally:
            conn.close()

    @app.get("/api/scans/{scan_id}")
    def scan(scan_id: str):
        record = _load(db_path, scan_id)
        # Generations stay in SQLite. The detail view uses diffs, heatmap, and findings.
        public = dict(record)
        public.pop("generations", None)
        return public

    @app.get("/api/scans/{scan_id}/heatmap")
    def heatmap(scan_id: str):
        record = _load(db_path, scan_id)
        redteam = record.get("redteam") or {}
        return {"cells": redteam.get("heatmap") or [], "baseline_insecure_rate": redteam.get("baseline_insecure_rate")}

    @app.get("/api/scans/{scan_id}/diffs")
    def diffs(scan_id: str):
        record = _load(db_path, scan_id)
        redteam = record.get("redteam") or {}
        return {"diffs": redteam.get("diffs") or []}

    @app.get("/api/scans/{scan_id}/sandbox")
    def sandbox(scan_id: str):
        record = _load(db_path, scan_id)
        return {"summary": record.get("sandbox"), "events": record.get("sandbox_events") or []}

    @app.get("/api/scans/{scan_id}/report")
    def report(scan_id: str, format: str = "html"):
        record = _load(db_path, scan_id)
        body = render(record, format)
        if format == "html":
            return HTMLResponse(body)
        return JSONResponse(json_loads(body))

    @app.post("/api/scans")
    def create_scan(request: ScanRequest):
        from modelsentinel.policy import load_policy
        from modelsentinel.pipeline import scan_source

        policy = load_policy()
        if request.kind == "redteam":
            from modelsentinel.redteam import run_redteam
            from modelsentinel.db import save_scan

            record = run_redteam(request.source, runs=request.runs, policy=policy, sandbox=request.sandbox)
            conn = connect(db_path)
            try:
                save_scan(conn, record)
            finally:
                conn.close()
        else:
            record = scan_source(request.source, policy=policy, db_path=db_path)
        public = dict(record)
        public.pop("generations", None)
        return public

    if DIST.is_dir():
        app.mount("/", StaticFiles(directory=DIST, html=True), name="dashboard")
    else:

        @app.get("/")
        def missing_dashboard():
            return HTMLResponse(
                "<p>API is up. Build the dashboard with <code>npm --prefix dashboard install && npm --prefix dashboard run build</code>.</p>"
            )

    return app


def _load(db_path: Path, scan_id: str) -> dict:
    conn = connect(db_path)
    try:
        record = get_scan(conn, scan_id)
    finally:
        conn.close()
    if record is None:
        raise HTTPException(status_code=404, detail="scan not found")
    return record


def json_loads(body: str):
    import json

    return json.loads(body)
