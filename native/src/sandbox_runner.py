"""Audit-hook runner installed by sentinel-sandbox (and by the Python fallback).

What this file does, in order:

1. Installs a ``sys.addaudithook`` hook that logs every file open (with a
   ``system`` flag for interpreter/stdlib paths), canary reads, process
   spawns, and raw socket use.
2. Runs an enforcement probe with a *real* socket and a read of a file outside
   the jail, so the parent can tell whether seccomp/Landlock/sandbox-exec are
   actually in force.
3. Replaces the socket layer with an in-process sink. DNS lookups and
   connections are logged with the hostname, payloads are captured (and
   checked for the canary token), and HTTP clients get a canned response. No
   packet ever leaves the process through the sink.
4. Turns on hook-level denial (``SENTINEL_DENY=1``): process spawn, raw
   sockets, ctypes, writes outside the jail, and reads of the real user home
   raise ``PermissionError``. This is defence in depth on Linux and the main
   containment on hosts without kernel isolation (macOS without sandbox-exec,
   Windows). It is best effort: an audit hook is not a security boundary.
5. Executes the snippet.

Events are flushed per line so a snippet that exits abruptly still leaves a log.
Only the standard library is used; the file must run on Python 3.8+ on Linux,
macOS and Windows.
"""

import io
import json
import os
import sys
import traceback

EVENTS_PATH = os.environ["SENTINEL_EVENTS"]
CODE_PATH = os.environ["SENTINEL_CODE"]
CANARIES = [p for p in os.environ.get("SENTINEL_CANARIES", "").split(os.pathsep) if p]
OUTSIDE = os.environ.get("SENTINEL_OUTSIDE", "")
TOKEN = os.environ.get("SENTINEL_TOKEN", "")
JAIL = os.environ.get("SENTINEL_JAIL", "")
REAL_HOME = os.environ.get("SENTINEL_REAL_HOME", "")
DENY_REQUESTED = os.environ.get("SENTINEL_DENY", "1") == "1"
SINK_ADDR = "192.0.2.1"  # TEST-NET-1 (RFC 5737): documentation-only, never routed.
MAX_EVENTS = 4000
PREVIEW_BYTES = 512

STATE = {"in_probe": False, "deny": False, "count": 0, "truncated": False, "last_host": None}
SEEN_FILES = set()
CANARY_MARKERS = (".ssh", ".aws", ".env", "id_rsa", "id_ed25519", "credentials", ".boto", ".gnupg", ".netrc")
PROCESS_EVENTS = {
    "os.system",
    "os.popen",
    "os.exec",
    "os.posix_spawn",
    "os.spawn",
    "os.fork",
    "os.forkpty",
    "os.startfile",
    "os.kill",
    "subprocess.Popen",
    "pty.spawn",
    "_winapi.CreateProcess",
    "webbrowser.open",
}
RAW_NET_PREFIXES = ("socket.__new__", "socket.connect", "socket.sendto", "socket.sendmsg", "socket.bind")
NATIVE_EVENTS = {"ctypes.dlopen", "ctypes.dlsym", "ctypes.cdata"}

LOG = open(EVENTS_PATH, "w", buffering=1, encoding="utf-8")


def emit(record):
    if STATE["count"] >= MAX_EVENTS:
        if not STATE["truncated"]:
            STATE["truncated"] = True
            LOG.write(json.dumps({"kind": "truncated", "event": "truncated", "detail": str(MAX_EVENTS)}) + "\n")
        return
    STATE["count"] += 1
    if STATE["in_probe"]:
        record["probe"] = True
    LOG.write(json.dumps(record, default=str) + "\n")
    LOG.flush()


def _real(path):
    try:
        return os.path.normcase(os.path.realpath(path))
    except Exception:
        return os.path.normcase(os.path.abspath(path))


def _under(path, root):
    if not root:
        return False
    return path == root or path.startswith(root.rstrip("\\/") + os.sep)


SYSTEM_ROOTS = [
    _real(p)
    for p in {sys.prefix, sys.base_prefix, sys.exec_prefix, getattr(sys, "base_exec_prefix", sys.prefix)}
    if p
]
for _p in ("/usr", "/lib", "/lib64", "/etc", "/proc", "/dev", "/sys", "/bin", "/System", "/Library", "/opt/homebrew"):
    SYSTEM_ROOTS.append(_real(_p))
if os.name == "nt":
    SYSTEM_ROOTS.append(_real(os.environ.get("SystemRoot", "C:\\Windows")))
JAIL_REAL = _real(JAIL) if JAIL else ""
HOME_REAL = _real(REAL_HOME) if REAL_HOME else ""
CANARIES_REAL = {_real(c) for c in CANARIES}
INTERNAL = {_real(p) for p in (EVENTS_PATH, CODE_PATH, __file__) if p}


def _is_system(path):
    return any(_under(path, root) for root in SYSTEM_ROOTS)


def _open_is_write(args):
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    if isinstance(mode, str):
        return any(ch in mode for ch in "wax+")
    if isinstance(flags, int):
        return bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
    return False


def _deny(category, event, detail):
    emit({"kind": "blocked", "category": category, "event": event, "detail": str(detail)[:500]})
    raise PermissionError("blocked by sentinel sandbox: %s %s" % (event, str(detail)[:200]))


def _on_open(event, args):
    raw = args[0] if args else ""
    if isinstance(raw, int):
        return  # fd-based reopen; the original open was already logged.
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    raw = str(raw)
    real = _real(raw)
    if real in INTERNAL:
        return
    write = _open_is_write(args)
    in_jail = _under(real, JAIL_REAL)
    system = _is_system(real) and not in_jail
    lowered = real.replace("\\", "/")
    secretish = real in CANARIES_REAL or (not system and any(m in lowered for m in CANARY_MARKERS))
    if secretish:
        emit({"kind": "canary_read", "event": event, "detail": raw[:500], "path": raw[:500], "write": write, "in_jail": in_jail})
    if STATE["deny"]:
        if write and not in_jail:
            _deny("filesystem", "open", raw)
        if HOME_REAL and _under(real, HOME_REAL) and not in_jail and not system:
            _deny("filesystem", "open", raw)
    if secretish:
        return
    key = (real, write)
    if key in SEEN_FILES:
        return
    SEEN_FILES.add(key)
    emit({"kind": "file_access", "event": event, "detail": raw[:500], "path": raw[:500], "write": write, "system": system})


def hook(event, args):
    try:
        if event == "open":
            _on_open(event, args)
            return
        if event in PROCESS_EVENTS or event.startswith("os.exec") or event.startswith("os.spawn"):
            detail = repr(args[:2])[:500]
            if STATE["deny"]:
                emit({"kind": "process_spawn", "event": event, "detail": detail, "blocked": True})
                raise PermissionError("blocked by sentinel sandbox: " + event)
            emit({"kind": "process_spawn", "event": event, "detail": detail, "blocked": False})
            return
        if event.startswith(RAW_NET_PREFIXES):
            detail = repr(args[1:3] if len(args) > 1 else args)[:300]
            if STATE["deny"]:
                _deny("network", event, detail)
            emit({"kind": "raw_socket", "event": event, "detail": detail})
            return
        if event in NATIVE_EVENTS and STATE["deny"]:
            _deny("native", event, repr(args[:1])[:200])
    except PermissionError:
        raise
    except Exception:
        return


sys.addaudithook(hook)


# --------------------------------------------------------------------------
# Enforcement probe: uses the real socket module before the sink is installed.
# --------------------------------------------------------------------------
def probe():
    STATE["in_probe"] = True
    network = "unknown"
    try:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(0.2)
            sock.connect(("127.0.0.1", 9))
            network = "connected"
        except OSError as exc:
            network = "eperm" if getattr(exc, "errno", None) in (1, 13) else "oserror:%s" % getattr(exc, "errno", None)
        finally:
            sock.close()
    except OSError as exc:
        errno = getattr(exc, "errno", None)
        network = "eperm" if errno in (1, 13) else "socket:%s" % errno
    except Exception as exc:
        network = "error:" + type(exc).__name__
    filesystem = "unknown"
    if OUTSIDE:
        try:
            with open(OUTSIDE, "r", encoding="utf-8"):
                pass
            filesystem = "outside-readable"
        except PermissionError:
            filesystem = "landlock-blocked" if sys.platform.startswith("linux") else "os-blocked"
        except OSError as exc:
            filesystem = "oserror:%s" % getattr(exc, "errno", None)
    emit({"kind": "enforcement_probe", "event": "probe", "network": network, "filesystem": filesystem})
    STATE["in_probe"] = False


# --------------------------------------------------------------------------
# In-process DNS/HTTP sink.
# --------------------------------------------------------------------------
HTTP_METHODS = (b"GET ", b"POST ", b"PUT ", b"HEAD ", b"DELETE ", b"PATCH ", b"OPTIONS ", b"CONNECT ")
CANNED_HTTP = (
    b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
)


def _host_text(host):
    if isinstance(host, bytes):
        return host.decode("idna", "replace")
    return str(host)


def _dns(host, via):
    name = _host_text(host)
    if name in ("", "0.0.0.0", "::"):
        return
    STATE["last_host"] = name
    emit(
        {
            "kind": "dns_query",
            "event": via,
            "detail": name[:253],
            "host": name[:253],
            "canary": bool(TOKEN) and TOKEN.lower() in name.lower(),
        }
    )


def _dns_qname(payload):
    """Best-effort decode of the question name in a raw DNS query."""
    try:
        if len(payload) < 13:
            return None
        labels = []
        pos = 12
        while pos < len(payload) and payload[pos] != 0 and len(labels) < 64:
            size = payload[pos]
            if size > 63:
                return None
            labels.append(payload[pos + 1 : pos + 1 + size].decode("ascii", "replace"))
            pos += 1 + size
        return ".".join(labels) if labels else None
    except Exception:
        return None


def _payload(host, port, data, via):
    raw = bytes(data[: PREVIEW_BYTES * 4]) if data else b""
    text = raw.decode("utf-8", "replace")
    record = {
        "kind": "outbound_payload",
        "event": via,
        "host": host,
        "port": port,
        "bytes": len(data) if data else 0,
        "preview": text[:PREVIEW_BYTES],
        "detail": "%s:%s %d bytes" % (host, port, len(data) if data else 0),
        "canary": bool(TOKEN) and TOKEN in text,
    }
    if port == 53 or via == "sink.sendto":
        qname = _dns_qname(raw)
        if qname:
            record["dns_qname"] = qname
            if TOKEN and TOKEN in qname:
                record["canary"] = True
    emit(record)


class SinkSocket(object):
    """Stand-in for ``socket.socket``: logs, never touches the network."""

    def __init__(self, family=2, type=1, proto=0, fileno=None):
        self.family = family
        self.type = type
        self.proto = proto
        self._peer = None
        self._timeout = None
        self._sent = b""
        self._closed = False

    # connection ---------------------------------------------------------
    def _target(self, address):
        if isinstance(address, tuple) and address:
            host = _host_text(address[0])
            port = address[1] if len(address) > 1 else 0
        else:
            host, port = str(address), 0
        if host == SINK_ADDR and STATE["last_host"]:
            host = STATE["last_host"]
        return host, port

    def connect(self, address, via="sink.connect"):
        host, port = self._target(address)
        self._peer = (host, port)
        emit({"kind": "outbound", "event": via, "host": host, "port": port, "detail": "%s:%s" % (host, port)})

    def connect_ex(self, address):
        self.connect(address)
        return 0

    # data ---------------------------------------------------------------
    def send(self, data, flags=0):
        self._sent += bytes(data)
        host, port = self._peer or ("<unconnected>", 0)
        _payload(host, port, bytes(data), "sink.send")
        return len(data)

    def sendall(self, data, flags=0):
        self.send(data, flags)

    def sendto(self, data, *args):
        address = args[-1] if args else ("<none>", 0)
        host, port = self._target(address)
        emit({"kind": "outbound", "event": "sink.sendto", "host": host, "port": port, "detail": "%s:%s" % (host, port)})
        _payload(host, port, bytes(data), "sink.sendto")
        return len(data)

    def _response(self):
        if self._sent.startswith(HTTP_METHODS):
            return CANNED_HTTP
        return b""

    def recv(self, bufsize=4096, flags=0):
        body, self._sent = self._response(), b""
        return body[:bufsize]

    def recvfrom(self, bufsize=4096, flags=0):
        return b"", (SINK_ADDR, 0)

    def recv_into(self, buffer, nbytes=0, flags=0):
        data = self.recv(nbytes or len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def makefile(self, mode="r", *args, **kwargs):
        stream = io.BytesIO(self._response())
        if "b" in mode:
            return stream
        return io.TextIOWrapper(stream, encoding="latin-1")

    # bookkeeping --------------------------------------------------------
    def settimeout(self, value):
        self._timeout = value

    def gettimeout(self):
        return self._timeout

    def setblocking(self, flag):
        return None

    def setsockopt(self, *args):
        return None

    def getsockopt(self, *args):
        return 0

    def bind(self, address):
        emit({"kind": "outbound", "event": "sink.bind", "host": "listen", "port": address[1] if isinstance(address, tuple) and len(address) > 1 else 0, "detail": repr(address)[:200]})

    def listen(self, *args):
        return None

    def accept(self):
        raise OSError(11, "sentinel sink: no inbound connections")

    def getpeername(self):
        return (SINK_ADDR, self._peer[1] if self._peer else 0)

    def getsockname(self):
        return ("127.0.0.1", 0)

    def fileno(self):
        return -1

    def shutdown(self, how):
        return None

    def close(self):
        self._closed = True

    def detach(self):
        return -1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def install_sink():
    import socket

    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        _dns(host, "sink.getaddrinfo")
        try:
            port_num = int(port or 0)
        except (TypeError, ValueError):
            port_num = 0
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (SINK_ADDR, port_num))]

    def gethostbyname(host):
        _dns(host, "sink.gethostbyname")
        return SINK_ADDR

    def gethostbyname_ex(host):
        _dns(host, "sink.gethostbyname_ex")
        return (_host_text(host), [], [SINK_ADDR])

    def gethostbyaddr(addr):
        _dns(addr, "sink.gethostbyaddr")
        return ("sink.invalid", [], [str(addr)])

    def create_connection(address, timeout=None, source_address=None, *args, **kwargs):
        host = _host_text(address[0])
        _dns(host, "sink.create_connection")
        sock = SinkSocket()
        sock.connect((host, address[1]), via="sink.create_connection")
        return sock

    socket.socket = SinkSocket
    socket.SocketType = SinkSocket
    socket.getaddrinfo = getaddrinfo
    socket.gethostbyname = gethostbyname
    socket.gethostbyname_ex = gethostbyname_ex
    socket.gethostbyaddr = gethostbyaddr
    socket.create_connection = create_connection
    socket.setdefaulttimeout(1.0)

    try:
        import ssl

        def wrap_socket(self, sock, *args, **kwargs):
            host = kwargs.get("server_hostname") or (sock._peer[0] if isinstance(sock, SinkSocket) and sock._peer else "")
            emit({"kind": "outbound", "event": "sink.tls", "host": host, "port": 443, "detail": "tls %s" % host})
            return sock

        ssl.SSLContext.wrap_socket = wrap_socket
    except Exception:
        pass


def main():
    probe()
    install_sink()
    STATE["deny"] = DENY_REQUESTED
    try:
        with open(CODE_PATH, "r", encoding="utf-8", errors="replace") as handle:
            source = handle.read()
        compiled = compile(source, CODE_PATH, "exec")
        namespace = {"__name__": "__main__", "__file__": CODE_PATH}
        exec(compiled, namespace, namespace)
    except SystemExit as exc:
        emit({"kind": "runtime_error", "event": "SystemExit", "detail": str(getattr(exc, "code", ""))[:200]})
    except BaseException as exc:  # noqa: BLE001 - the snippet is untrusted; log everything
        emit(
            {
                "kind": "runtime_error",
                "event": type(exc).__name__,
                "detail": str(exc)[:500],
                "trace": traceback.format_exc()[-900:],
            }
        )
    finally:
        STATE["deny"] = False
        emit({"kind": "end", "event": "end"})
        LOG.close()


main()
