"""``sentinel guard``: an OpenAI-compatible local proxy that scans model code before you see it.

Point an IDE plugin or OpenAI client at ``http://127.0.0.1:8742/v1`` instead of
Ollama or llama-server. Every completion is forwarded to the upstream, every
fenced code block in the answer is scanned (``guard_rules``), and the policy
decides: ``block`` replaces the answer with an explanation, ``warn`` appends a
note, ``annotate`` only adds headers and a ``modelsentinel`` JSON field.
Streaming requests are buffered, scanned whole, and re-emitted as SSE.

Offline rule: the upstream and the listen address must be loopback unless
explicitly allowed. Upstream requests bypass HTTP(S)_PROXY.
"""

from __future__ import annotations

import ipaddress
import json
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from modelsentinel.db import connect, save_guard_event
from modelsentinel.guard_rules import ACTION_RANK, DECISION_HEADER, GuardConfig, ScanResult, explain_block, scan_text, warning_note
from modelsentinel.runtime import is_loopback_url

DEFAULT_LISTEN = "127.0.0.1:8742"
MAX_BODY = 16 * 1024 * 1024
COMPLETION_PATHS = {"/v1/chat/completions", "/v1/completions"}
PASSTHROUGH_HEADERS = ("authorization", "openai-organization", "openai-project")


@dataclass(frozen=True)
class GuardSettings:
    upstream: str
    config: GuardConfig
    db_path: Path | None


# ---------------------------------------------------------------- endpoints


def normalise_upstream(url: str) -> str:
    """``http://h:p``, ``http://h:p/`` and ``http://h:p/v1`` all become ``http://h:p``."""
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"upstream must be an http(s) URL, got {url!r}")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def parse_listen(listen: str) -> tuple[str, int]:
    host, sep, port = listen.rpartition(":")
    if not sep:
        raise ValueError(f"--listen must be HOST:PORT, got {listen!r}")
    try:
        return host.strip("[]") or "127.0.0.1", int(port)
    except ValueError as exc:
        raise ValueError(f"--listen port must be a number, got {port!r}") from exc


def is_loopback_host(host: str) -> bool:
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_offline(upstream: str, listen_host: str, *, allow_remote_upstream: bool = False,
                  allow_remote_listen: bool = False) -> None:
    if not is_loopback_url(upstream) and not allow_remote_upstream:
        raise ValueError(
            f"upstream {upstream} is not loopback. The guard stays offline; pass --allow-remote-upstream to override."
        )
    if not is_loopback_host(listen_host) and not allow_remote_listen:
        raise ValueError(
            f"listen address {listen_host} is not loopback; pass --allow-remote-listen to expose the guard."
        )


# ---------------------------------------------------------------- upstream I/O


def _upstream_request(settings: GuardSettings, method: str, path: str, body: bytes | None, headers: dict):
    request = urllib.request.Request(settings.upstream + path, data=body, method=method, headers=headers)
    # No proxy handler: HTTP(S)_PROXY must not route a loopback request off the machine.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=settings.config.upstream_timeout)


def _parse_sse(raw: bytes) -> list[dict | str]:
    """SSE ``data:`` payloads as parsed JSON (dicts) or raw strings (``[DONE]``, keep-alives)."""
    events: list[dict | str] = []
    for chunk in raw.decode("utf-8", errors="replace").replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(line[5:].lstrip() for line in chunk.split("\n") if line.startswith("data:"))
        if not data:
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            events.append(data)
            continue
        events.append(parsed if isinstance(parsed, dict) else data)
    return events


# ---------------------------------------------------------------- text access


def _is_chat(path: str) -> bool:
    return path == "/v1/chat/completions"


def _choice_text(choice: dict, chat: bool) -> str:
    if chat:
        content = (choice.get("message") or {}).get("content")
        return content if isinstance(content, str) else ""
    return choice.get("text") or ""


def _set_choice_text(choice: dict, chat: bool, text: str) -> dict:
    if chat:
        return {**choice, "message": {**(choice.get("message") or {"role": "assistant"}), "content": text}}
    return {**choice, "text": text}


def _stream_texts(events: list, chat: bool) -> dict[int, str]:
    texts: dict[int, str] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        for choice in event.get("choices") or []:
            index = int(choice.get("index", 0) or 0)
            piece = (choice.get("delta") or {}).get("content") if chat else choice.get("text")
            texts[index] = texts.get(index, "") + (piece if isinstance(piece, str) else "")
    return texts


def _request_preview(payload: dict, limit: int) -> str:
    if isinstance(payload.get("messages"), list):
        users = [m for m in payload["messages"] if isinstance(m, dict) and m.get("role") == "user"]
        content = (users[-1] if users else {}).get("content", "")
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
        return str(content)[:limit]
    prompt = payload.get("prompt", "")
    return (prompt if isinstance(prompt, str) else json.dumps(prompt))[:limit]


# ---------------------------------------------------------------- decisions


def _combine(results: dict[int, ScanResult]) -> str:
    return max((result.decision for result in results.values()), key=ACTION_RANK.__getitem__, default="pass")


def _new_text(text: str, result: ScanResult) -> str:
    if result.decision == "block":
        return explain_block(result)
    if result.decision == "warn":
        return text + warning_note(result)
    return text


def _meta(results: dict[int, ScanResult], decision: str, mode: str) -> dict:
    return {
        "decision": decision,
        "mode": mode,
        "choices": {str(index): result.to_dict() for index, result in sorted(results.items())},
        "rules": sorted({rule for result in results.values() for rule in result.rules}),
    }


def rewrite_json(payload: dict, chat: bool, config: GuardConfig) -> tuple[dict, str, dict[int, ScanResult]]:
    results: dict[int, ScanResult] = {}
    choices = []
    for position, choice in enumerate(payload.get("choices") or []):
        index = int(choice.get("index", position) or 0)
        text = _choice_text(choice, chat)
        result = scan_text(text, config)
        results[index] = result
        changed = _new_text(text, result)
        choices.append(_set_choice_text(choice, chat, changed) if changed != text else choice)
    decision = _combine(results)
    return {**payload, "choices": choices, "modelsentinel": _meta(results, decision, config.mode)}, decision, results


def rewrite_stream(events: list, chat: bool, config: GuardConfig) -> tuple[list, str, dict[int, ScanResult]]:
    texts = _stream_texts(events, chat)
    results = {index: scan_text(text, config) for index, text in texts.items()}
    decision = _combine(results)
    template = next((event for event in events if isinstance(event, dict)), {})
    meta = _meta(results, decision, config.mode)
    if decision == "block":
        out = _synth_stream(template, texts, results, chat)
    else:
        out = _insert_warnings(events, results, template, chat)
    json_positions = [i for i, event in enumerate(out) if isinstance(event, dict)]
    if json_positions:
        last = json_positions[-1]
        out[last] = {**out[last], "modelsentinel": meta}
    if not any(event == "[DONE]" for event in out):
        out.append("[DONE]")
    return out, decision, results


def _chunk(template: dict, chat: bool, index: int, content: str | None, finish: str | None, role: bool = False) -> dict:
    base = {key: template[key] for key in ("id", "created", "model", "system_fingerprint") if key in template}
    base["object"] = template.get("object") or ("chat.completion.chunk" if chat else "text_completion")
    if chat:
        delta = {"role": "assistant"} if role else {}
        if content is not None:
            delta["content"] = content
        choice = {"index": index, "delta": delta, "finish_reason": finish}
    else:
        choice = {"index": index, "text": content or "", "finish_reason": finish}
    return {**base, "choices": [choice]}


def _synth_stream(template: dict, texts: dict, results: dict, chat: bool) -> list:
    out: list = []
    for index in sorted(texts):
        text = _new_text(texts[index], results[index])
        out.append(_chunk(template, chat, index, text, None, role=True))
        out.append(_chunk(template, chat, index, None if chat else "", "stop"))
    out.append("[DONE]")
    return out


def _insert_warnings(events: list, results: dict, template: dict, chat: bool) -> list:
    out = list(events)
    for index, result in sorted(results.items()):
        if result.decision != "warn":
            continue
        position = _finish_position(out, index)
        out.insert(position, _chunk(template, chat, index, warning_note(result), None))
    return out


def _finish_position(events: list, index: int) -> int:
    for position, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        for choice in event.get("choices") or []:
            if int(choice.get("index", 0) or 0) == index and choice.get("finish_reason"):
                return position
    done = [position for position, event in enumerate(events) if event == "[DONE]"]
    return done[0] if done else len(events)


def encode_sse(events: list) -> bytes:
    parts = [f"data: {event if isinstance(event, str) else json.dumps(event, ensure_ascii=False)}\n\n" for event in events]
    return "".join(parts).encode("utf-8")


# ---------------------------------------------------------------- logging


_DB_LOCK = threading.Lock()


def log_event(settings: GuardSettings, event: dict) -> None:
    if settings.db_path is None:
        return
    try:
        with _DB_LOCK:
            conn = connect(settings.db_path)
            try:
                save_guard_event(conn, event)
            finally:
                conn.close()
    except sqlite3.Error as exc:
        print(f"guard: could not log event: {exc}", file=sys.stderr)


def _event(settings: GuardSettings, payload: dict, response: dict, decision: str, results: dict,
           texts: list[str], started: float) -> dict:
    limit = settings.config.preview_chars
    return {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "upstream": settings.upstream,
        "model": str(response.get("model") or payload.get("model") or ""),
        "decision": decision,
        "findings": [item for result in results.values() for item in result.to_dict()["findings"]],
        "request_preview": _request_preview(payload, limit),
        "response_preview": "\n".join(texts)[:limit],
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


# ---------------------------------------------------------------- HTTP handler


class GuardHandler(BaseHTTPRequestHandler):
    server_version = "ModelSentinelGuard/1"
    settings: GuardSettings  # set on the subclass made by make_server

    def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler signature
        return

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in {"/health", "/v1/health"}:
            self._json(200, {"ok": True, "upstream": self.settings.upstream, "mode": self.settings.config.mode}, "pass")
        elif path == "/v1/models" or path.startswith("/v1/models/"):
            self._passthrough("GET", path, None)
        else:
            self._error(404, f"unknown path {path}")

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._error(413, "request body too large")
            return
        body = self.rfile.read(length) if length else b""
        if path not in COMPLETION_PATHS:
            self._error(404, f"unknown path {path}")
            return
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(400, "request body must be JSON")
            return
        if not isinstance(payload, dict):
            self._error(400, "request body must be a JSON object")
            return
        self._completion(path, payload, body)

    # -- completions

    def _completion(self, path: str, payload: dict, body: bytes) -> None:
        started = time.perf_counter()
        chat = _is_chat(path)
        try:
            with _upstream_request(self.settings, "POST", path, body, self._forward_headers()) as response:
                raw = response.read()
                content_type = response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            self._raw(exc.code, exc.read(), exc.headers.get("Content-Type", "application/json"), "upstream-error")
            return
        except (urllib.error.URLError, OSError) as exc:
            self._error(502, f"upstream unreachable: {getattr(exc, 'reason', exc)}", guard="upstream-error")
            return
        streamed = bool(payload.get("stream")) or "text/event-stream" in content_type
        if streamed:
            events = _parse_sse(raw)
            out, decision, results = rewrite_stream(events, chat, self.settings.config)
            texts = list(_stream_texts(events, chat).values())
            template = next((event for event in events if isinstance(event, dict)), {})
            # Log before replying: the audit record must exist even if the client
            # disconnects or the server shuts down right after the response.
            log_event(self.settings, _event(self.settings, payload, template, decision, results, texts, started))
            self._send(200, encode_sse(out), "text/event-stream; charset=utf-8", decision, results)
            return
        try:
            response_json = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._raw(502, raw, content_type or "application/octet-stream", "unscanned")
            return
        if not isinstance(response_json, dict):
            self._raw(502, raw, content_type or "application/json", "unscanned")
            return
        texts = [_choice_text(choice, chat) for choice in response_json.get("choices") or []]
        rewritten, decision, results = rewrite_json(response_json, chat, self.settings.config)
        data = json.dumps(rewritten, ensure_ascii=False).encode("utf-8")
        log_event(self.settings, _event(self.settings, payload, response_json, decision, results, texts, started))
        self._send(200, data, "application/json", decision, results)

    def _passthrough(self, method: str, path: str, body: bytes | None) -> None:
        try:
            with _upstream_request(self.settings, method, path, body, self._forward_headers()) as response:
                self._raw(response.status, response.read(), response.headers.get("Content-Type", "application/json"), "pass")
        except urllib.error.HTTPError as exc:
            self._raw(exc.code, exc.read(), exc.headers.get("Content-Type", "application/json"), "upstream-error")
        except (urllib.error.URLError, OSError) as exc:
            self._error(502, f"upstream unreachable: {getattr(exc, 'reason', exc)}", guard="upstream-error")

    def _forward_headers(self) -> dict:
        headers = {"Content-Type": "application/json", "Accept": self.headers.get("Accept", "*/*")}
        for name in PASSTHROUGH_HEADERS:
            if self.headers.get(name):
                headers[name] = self.headers[name]
        return headers

    # -- responses

    def _send(self, status: int, data: bytes, content_type: str, decision: str, results: dict) -> None:
        rules = sorted({rule for result in results.values() for rule in result.rules})
        count = sum(len(result.findings) for result in results.values())
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-ModelSentinel-Guard", DECISION_HEADER[decision])
        self.send_header("X-ModelSentinel-Findings", str(count))
        if rules:
            self.send_header("X-ModelSentinel-Rules", ",".join(rules)[:1000])
        if content_type.startswith("text/event-stream"):
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _raw(self, status: int, data: bytes, content_type: str, guard: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-ModelSentinel-Guard", guard)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, payload: dict, guard: str) -> None:
        self._raw(status, json.dumps(payload).encode("utf-8"), "application/json", guard)

    def _error(self, status: int, message: str, guard: str = "error") -> None:
        self._json(status, {"error": {"message": message, "type": "modelsentinel_guard", "code": status}}, guard)


def make_server(upstream: str, *, host: str = "127.0.0.1", port: int = 8742, config: GuardConfig | None = None,
                db_path: Path | None = None) -> ThreadingHTTPServer:
    settings = GuardSettings(upstream=normalise_upstream(upstream), config=config or GuardConfig(), db_path=db_path)
    handler = type("BoundGuardHandler", (GuardHandler,), {"settings": settings})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


# ---------------------------------------------------------------- CLI


def register(sub) -> None:
    parser = sub.add_parser(
        "guard",
        help="OpenAI-compatible local proxy that scans model code blocks and enforces policy.",
        description="Sit between an IDE/OpenAI client and Ollama or llama-server; scan and block/warn/annotate.",
    )
    parser.add_argument("--upstream", required=True, help="llama-server (http://127.0.0.1:8080) or Ollama (http://127.0.0.1:11434)")
    parser.add_argument("--listen", default=DEFAULT_LISTEN, help=f"HOST:PORT to bind. Default {DEFAULT_LISTEN}")
    parser.add_argument("--mode", choices=["block", "warn", "annotate"], default=None,
                        help="Strongest action allowed. Default: guard.mode from policy (block).")
    parser.add_argument("--policy", default=None, help="policy.yaml with a guard: section")
    parser.add_argument("--db", dest="guard_db", default=None, help="SQLite path for guard_events (default: the global --db)")
    parser.add_argument("--no-log", action="store_true", help="Do not record events in SQLite")
    parser.add_argument("--allow-remote-upstream", action="store_true", help="Permit a non-loopback upstream")
    parser.add_argument("--allow-remote-listen", action="store_true", help="Permit binding a non-loopback address")
    parser.set_defaults(func=cmd_guard)


def cmd_guard(args) -> int:
    from modelsentinel.pipeline import default_db_path
    from modelsentinel.policy import load_policy

    upstream = normalise_upstream(args.upstream)
    host, port = parse_listen(args.listen)
    check_offline(upstream, host, allow_remote_upstream=args.allow_remote_upstream,
                  allow_remote_listen=args.allow_remote_listen)
    config = GuardConfig.from_policy(load_policy(args.policy), mode=args.mode)
    db_path = None
    if not args.no_log:
        db_path = Path(args.guard_db or getattr(args, "db", None) or default_db_path())
    server = make_server(upstream, host=host, port=port, config=config, db_path=db_path)
    bound_host, bound_port = server.server_address[:2]
    print(f"sentinel guard: http://{bound_host}:{bound_port}/v1 -> {upstream} (mode {config.mode})", file=sys.stderr)
    if db_path:
        print(f"sentinel guard: logging events to {db_path}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
