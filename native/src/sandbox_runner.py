"""Audit-hook runner installed by sentinel-sandbox.

The hook records file opens and network attempts. It does not try to make
the snippet safe by itself: seccomp and landlock, applied by the parent
before exec, are what block the calls. Events are flushed per line so a
snippet that exits abruptly still leaves a log.
"""

import json
import os
import sys
import traceback

EVENTS_PATH = os.environ["SENTINEL_EVENTS"]
CODE_PATH = os.environ["SENTINEL_CODE"]
CANARIES = [p for p in os.environ.get("SENTINEL_CANARIES", "").split(":") if p]
OUTSIDE = os.environ.get("SENTINEL_OUTSIDE", "")
IN_PROBE = False

LOG = open(EVENTS_PATH, "w", buffering=1)


def emit(record):
    LOG.write(json.dumps(record, default=str) + "\n")
    LOG.flush()


def _path_of(value):
    if isinstance(value, str):
        return value
    return str(value)


def classify(event, args):
    if event == "open":
        raw = _path_of(args[0]) if args else ""
        try:
            norm = os.path.abspath(raw) if isinstance(args[0], str) else raw
        except Exception:
            norm = raw
        markers = (".ssh", ".aws", "/.env", "id_rsa", "credentials", ".boto", ".gnupg")
        for canary in CANARIES:
            if raw == canary or norm == canary or canary in raw or canary in norm:
                return "canary_read", raw
        if any(marker in raw or marker in norm for marker in markers):
            return "canary_read", raw
        return "file_access", raw
    networkish = (
        "socket.",
        "urllib.",
        "http.client",
        "http.server",
        "ftplib.",
        "smtplib.",
        "ssl.SSLContext",
    )
    named = {
        "os.system",
        "os.popen",
        "os.exec",
        "os.execv",
        "os.execve",
        "os.posix_spawn",
        "subprocess.Popen",
        "webbrowser.open",
    }
    if event in named or event.startswith(networkish):
        return "outbound", event
    return None, None


def hook(event, args):
    global IN_PROBE
    try:
        kind, detail = classify(event, args)
        if kind is None:
            return
        if kind == "file_access" and isinstance(detail, str):
            base = os.path.basename(detail)
            if base in {"events.jsonl", "runner.py", "sandbox_runner.py"}:
                return
            # Import machinery opens the whole standard library. Keep the
            # timeline to paths the snippet actually reached for.
            if detail.startswith(("/usr/", "/lib/", "/lib64/", "/etc/", "/proc/", "/dev/", "/sys/", "/bin/")):
                return
        record = {
            "kind": kind,
            "event": event,
            "detail": detail[:500] if isinstance(detail, str) else repr(detail)[:500],
            "args": [repr(arg)[:180] for arg in args[:4]],
        }
        if IN_PROBE:
            record["probe"] = True
        emit(record)
    except Exception:
        return


sys.addaudithook(hook)


def probe():
    global IN_PROBE
    IN_PROBE = True
    network = "unknown"
    try:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(0.2)
            sock.connect(("127.0.0.1", 9))
            network = "connected"
        except OSError as exc:
            network = "eperm" if getattr(exc, "errno", None) == 1 else f"oserror:{getattr(exc, 'errno', None)}"
        finally:
            sock.close()
    except OSError as exc:
        network = "eperm" if getattr(exc, "errno", None) == 1 else f"socket:{getattr(exc, 'errno', None)}"
    except Exception as exc:
        network = f"error:{type(exc).__name__}"
    filesystem = "unknown"
    if OUTSIDE:
        try:
            with open(OUTSIDE, "r", encoding="utf-8"):
                pass
            filesystem = "outside-readable"
        except PermissionError:
            filesystem = "landlock-blocked"
        except OSError as exc:
            filesystem = f"oserror:{getattr(exc, 'errno', None)}"
    emit(
        {
            "kind": "enforcement_probe",
            "probe": True,
            "event": "probe",
            "network": network,
            "filesystem": filesystem,
        }
    )
    IN_PROBE = False


probe()

try:
    import socket

    socket.setdefaulttimeout(1.0)
except Exception:
    pass

try:
    with open(CODE_PATH, "r", encoding="utf-8", errors="replace") as handle:
        source = handle.read()
    compiled = compile(source, CODE_PATH, "exec")
    namespace = {"__name__": "__main__", "__file__": CODE_PATH}
    exec(compiled, namespace, namespace)
except SystemExit as exc:
    emit({"kind": "runtime_error", "event": "SystemExit", "detail": str(getattr(exc, "code", ""))[:200]})
except Exception as exc:
    emit(
        {
            "kind": "runtime_error",
            "event": type(exc).__name__,
            "detail": str(exc)[:500],
            "trace": traceback.format_exc()[-900:],
        }
    )
finally:
    emit({"kind": "end", "event": "end"})
    LOG.close()
