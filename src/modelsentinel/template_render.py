"""Bounded, sandboxed rendering of untrusted chat templates.

A chat template is a small program written by whoever published the model, so
rendering it is running untrusted code. Two layers keep that bounded on
Windows, macOS and Linux alike:

1. ``BudgetedSandbox`` is a Jinja ``SandboxedEnvironment`` that counts steps
   (attribute reads, calls, operators), caps every intermediate string or
   list, caps integer arguments that size allocations (``center``, ``ljust``,
   ``%0999d``), bounds ``range`` cumulatively, and checks a wall-clock
   deadline cooperatively. No signals are used.
2. ``render_cases_isolated`` runs the renders in a child Python process
   (``python -I``, importing this module) with a hard timeout. Pure
   Jinja loops that never touch a hook (``{% for a in L %}{% for b in L %}``)
   cannot be interrupted in-process, so the child is the authoritative bound:
   ``subprocess.run(timeout=...)`` kills it on every platform. On POSIX the
   child also lowers its own address-space limit when the OS allows it.

If the child cannot be started, rendering falls back to the in-process budget
and the caller is told isolation was reduced.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Undefined, nodes
from jinja2.compiler import CodeGenerator, Frame
from jinja2.sandbox import SandboxedEnvironment

MAX_TEMPLATE_BYTES = 512 * 1024
MAX_VALUE_LEN = 1_000_000
MAX_OUTPUT_CHARS = 2_000_000
MAX_STEPS = 400_000
MAX_RANGE_TOTAL = 100_000
MAX_INT_ARG = 100_000
MAX_POW_EXPONENT = 256
DEFAULT_CASE_SECONDS = 2.0
DEFAULT_TOTAL_SECONDS = 20.0
DEFAULT_GRACE_SECONDS = 5.0  # interpreter start-up allowance before the child is killed
CHILD_MEMORY_BYTES = 1024 * 1024 * 1024
ISOLATION_ENV = "MODELSENTINEL_TEMPLATE_ISOLATION"
WORKER_PATH_ENV = "MODELSENTINEL_TEMPLATE_WORKER_PATH"
# -I keeps cwd and PYTHONPATH off sys.path; the parent's import path is appended
# after the stdlib so nothing can shadow it, and --user or editable installs work.
WORKER_BOOTSTRAP = (
    "import json, os, sys; "
    f"sys.path.extend(p for p in json.loads(os.environ.get({WORKER_PATH_ENV!r}, '[]')) if p not in sys.path); "
    "from modelsentinel.template_render import _worker_main; raise SystemExit(_worker_main())"
)
FIXED_DATE = datetime(2024, 7, 26)
WIDE_FORMAT = re.compile(r"\d{6,}")


class RenderLimit(RuntimeError):
    """Raised when a template exceeds a render budget."""


class Silent(Undefined):
    def __str__(self) -> str:
        return ""

    def __iter__(self):
        return iter(())

    def __bool__(self) -> bool:
        return False


@dataclass
class Budget:
    deadline: float
    steps: int = 0
    range_items: int = 0

    def tick(self) -> None:
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise RenderLimit(f"template exceeded {MAX_STEPS} evaluation steps")
        if self.steps % 64 == 0 and time.monotonic() > self.deadline:
            raise RenderLimit("template exceeded its render time budget")

    def check_time(self) -> None:
        if time.monotonic() > self.deadline:
            raise RenderLimit("template exceeded its render time budget")


def _sized(value: Any) -> Any:
    if isinstance(value, (str, bytes, list, tuple, dict)) and len(value) > MAX_VALUE_LEN:
        raise RenderLimit(f"template built a value longer than {MAX_VALUE_LEN}")
    return value


def _check_args(args: tuple, kwargs: dict) -> None:
    for item in list(args) + list(kwargs.values()):
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)) and abs(item) > MAX_INT_ARG:
            raise RenderLimit(f"template passed a size argument above {MAX_INT_ARG}")
        if isinstance(item, str) and "{" in item and WIDE_FORMAT.search(item):
            raise RenderLimit("template used an oversized format width")


class _CheckedCodeGenerator(CodeGenerator):
    """Route ``~`` concatenation through a size check."""

    def visit_Concat(self, node: nodes.Concat, frame: Frame) -> None:
        self.write("environment.checked_join((")
        for arg in node.nodes:
            self.visit(arg, frame)
            self.write(", ")
        self.write("))")


class BudgetedSandbox(SandboxedEnvironment):
    code_generator_class = _CheckedCodeGenerator
    intercepted_binops = frozenset({"+", "*", "**", "%", "//"})

    def __init__(self, seconds: float = DEFAULT_CASE_SECONDS) -> None:
        super().__init__(undefined=Silent, autoescape=False)
        self.budget = Budget(deadline=time.monotonic() + seconds)
        for name in ("lipsum", "cycler", "joiner"):
            self.globals.pop(name, None)
        self.globals["range"] = self._range
        self.globals["raise_exception"] = _raise_exception
        self.globals["strftime_now"] = _strftime_now
        self.filters = {name: self._wrap_filter(func) for name, func in self.filters.items()}

    def reset(self, seconds: float) -> None:
        self.budget = Budget(deadline=time.monotonic() + seconds)

    def checked_join(self, parts) -> str:
        self.budget.tick()
        return _sized("".join(str(part) for part in parts))

    def concat(self, parts) -> str:  # used by {% set x %}...{% endset %}
        return self.checked_join(parts)

    def _range(self, *args: int) -> range:
        self.budget.tick()
        result = range(*args)
        self.budget.range_items += len(result)
        if self.budget.range_items > MAX_RANGE_TOTAL:
            raise RenderLimit(f"template iterated more than {MAX_RANGE_TOTAL} range items")
        return result

    def _wrap_filter(self, func):
        def checked(*args, **kwargs):
            self.budget.tick()
            _check_args(args[1:], kwargs)
            return _sized(func(*args, **kwargs))

        for attr in ("jinja_pass_arg",):
            if hasattr(func, attr):
                setattr(checked, attr, getattr(func, attr))
        checked.__name__ = getattr(func, "__name__", "filter")
        return checked

    def getattr(self, obj, attribute):
        self.budget.tick()
        return _sized(super().getattr(obj, attribute))

    def getitem(self, obj, argument):
        self.budget.tick()
        return _sized(super().getitem(obj, argument))

    def call(__self, __context, __obj, *args, **kwargs):  # noqa: N805 - Jinja's signature
        __self.budget.tick()
        _check_args(args, kwargs)
        return _sized(super().call(__context, __obj, *args, **kwargs))

    def call_binop(self, context, operator, left, right):
        self.budget.tick()
        if operator == "**" and isinstance(right, (int, float)) and abs(right) > MAX_POW_EXPONENT:
            raise RenderLimit("template raised a number to an oversized power")
        if operator == "*":
            for seq, count in ((left, right), (right, left)):
                if isinstance(seq, (str, list, tuple)) and isinstance(count, int):
                    if len(seq) * max(count, 0) > MAX_VALUE_LEN:
                        raise RenderLimit("template multiplied a sequence past the size cap")
        if operator == "%" and isinstance(left, str) and WIDE_FORMAT.search(left):
            raise RenderLimit("template used an oversized format width")
        if operator == "+":
            for side in (left, right):
                _sized(side)
        return _sized(super().call_binop(context, operator, left, right))


def _raise_exception(*args, **_kwargs):
    raise RuntimeError(f"raise_exception: {args[0] if args else ''}")


def _strftime_now(fmt: str = "%d %b %Y") -> str:
    try:
        return FIXED_DATE.strftime(str(fmt))[:64]
    except (ValueError, TypeError):
        return "26 Jul 2024"


@dataclass
class RenderCase:
    id: str
    messages: list[dict]
    tools: list[dict] | None = None
    add_generation_prompt: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "messages": self.messages,
            "tools": self.tools,
            "add_generation_prompt": self.add_generation_prompt,
        }


@dataclass
class RenderBatch:
    outputs: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    limit_hit: str = ""
    isolation: str = "process"
    note: str = ""


def render_one(env: BudgetedSandbox, compiled, case: dict, seconds: float) -> str:
    env.reset(seconds)
    stream = compiled.generate(
        messages=case.get("messages") or [],
        tools=case.get("tools"),
        add_generation_prompt=bool(case.get("add_generation_prompt", True)),
        bos_token="<s>",
        eos_token="</s>",
        tools_in_user_message=False,
        date_string="26 Jul 2024",
    )
    pieces: list[str] = []
    size = 0
    for chunk in stream:
        size += len(chunk)
        if size > MAX_OUTPUT_CHARS:
            raise RenderLimit(f"template produced more than {MAX_OUTPUT_CHARS} characters")
        env.budget.check_time()
        pieces.append(chunk)
    return "".join(pieces)


def render_cases_inprocess(
    template: str,
    cases: list[dict],
    *,
    case_seconds: float = DEFAULT_CASE_SECONDS,
    total_seconds: float = DEFAULT_TOTAL_SECONDS,
) -> RenderBatch:
    batch = RenderBatch(isolation="inprocess")
    if len(template.encode("utf-8", "replace")) > MAX_TEMPLATE_BYTES:
        batch.limit_hit = f"template is larger than {MAX_TEMPLATE_BYTES} bytes"
        return batch
    env = BudgetedSandbox(case_seconds)
    try:
        compiled = env.from_string(template)
    except RecursionError:
        batch.errors["*"] = "RecursionError: template nests too deeply to compile"
        return batch
    except Exception as exc:  # TemplateSyntaxError, SecurityError and friends
        batch.errors["*"] = f"{type(exc).__name__}: {exc}"[:500]
        return batch
    stop = time.monotonic() + total_seconds
    for case in cases:
        if time.monotonic() > stop:
            batch.limit_hit = "template render probes exceeded the total time budget"
            break
        case_id = str(case.get("id"))
        try:
            batch.outputs[case_id] = render_one(env, compiled, case, case_seconds)
        except RenderLimit as exc:
            batch.limit_hit = str(exc)
            break
        except RecursionError:
            batch.limit_hit = "template recursed too deeply"
            break
        except MemoryError:
            batch.limit_hit = "template exhausted memory"
            break
        except Exception as exc:
            batch.errors[case_id] = f"{type(exc).__name__}: {exc}"[:500]
    return batch


def isolation_mode() -> str:
    value = os.environ.get(ISOLATION_ENV, "process").strip().lower()
    return "inprocess" if value in {"inprocess", "in-process", "none", "off"} else "process"


def render_cases_isolated(
    template: str,
    cases: list[RenderCase] | list[dict],
    *,
    case_seconds: float = DEFAULT_CASE_SECONDS,
    total_seconds: float = DEFAULT_TOTAL_SECONDS,
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
) -> RenderBatch:
    """Render every case in a child process bounded by a hard timeout."""
    plain = [case.to_dict() if isinstance(case, RenderCase) else dict(case) for case in cases]
    if isolation_mode() == "inprocess":
        batch = render_cases_inprocess(template, plain, case_seconds=case_seconds, total_seconds=total_seconds)
        batch.note = f"{ISOLATION_ENV}=inprocess: renders were not process-isolated"
        return batch
    request = json.dumps(
        {"template": template, "cases": plain, "case_seconds": case_seconds, "total_seconds": total_seconds},
        ensure_ascii=False,
    ).encode("utf-8")
    package_parent = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env[WORKER_PATH_ENV] = json.dumps([package_parent, *[p for p in sys.path if p and p != package_parent]])
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", WORKER_BOOTSTRAP],
            input=request,
            capture_output=True,
            timeout=total_seconds + grace_seconds,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return RenderBatch(limit_hit="template render worker timed out and was killed", isolation="process")
    except OSError as exc:
        batch = render_cases_inprocess(template, plain, case_seconds=case_seconds, total_seconds=total_seconds)
        batch.note = f"render worker unavailable ({type(exc).__name__}); rendered in-process with budgets only"
        return batch
    return _parse_worker_output(proc)


def _parse_worker_output(proc: subprocess.CompletedProcess) -> RenderBatch:
    try:
        payload = json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        stderr = proc.stderr.decode("utf-8", "replace")[-400:]
        if proc.returncode != 0:
            reason = "template render worker crashed"
            if "MemoryError" in stderr:
                reason = "template exhausted the render worker's memory"
            return RenderBatch(limit_hit=f"{reason} (exit {proc.returncode})", note=stderr, isolation="process")
        return RenderBatch(errors={"*": "render worker returned unreadable output"}, note=stderr, isolation="process")
    return RenderBatch(
        outputs={str(k): str(v) for k, v in (payload.get("outputs") or {}).items()},
        errors={str(k): str(v) for k, v in (payload.get("errors") or {}).items()},
        limit_hit=str(payload.get("limit_hit") or ""),
        isolation="process",
    )


def _limit_child_memory() -> None:
    try:
        import resource  # POSIX only; absent on Windows
    except ImportError:
        return
    for name in ("RLIMIT_AS", "RLIMIT_DATA"):
        limit = getattr(resource, name, None)
        if limit is None:
            continue
        try:
            _soft, hard = resource.getrlimit(limit)
            cap = CHILD_MEMORY_BYTES if hard in (-1, resource.RLIM_INFINITY) else min(hard, CHILD_MEMORY_BYTES)
            resource.setrlimit(limit, (cap, hard))
        except (ValueError, OSError):
            continue


def _worker_main() -> int:
    _limit_child_memory()
    raw = sys.stdin.buffer.read().decode("utf-8")
    request = json.loads(raw)
    batch = render_cases_inprocess(
        str(request.get("template") or ""),
        list(request.get("cases") or []),
        case_seconds=float(request.get("case_seconds") or DEFAULT_CASE_SECONDS),
        total_seconds=float(request.get("total_seconds") or DEFAULT_TOTAL_SECONDS),
    )
    reply = {"outputs": batch.outputs, "errors": batch.errors, "limit_hit": batch.limit_hit}
    sys.stdout.buffer.write(json.dumps(reply, ensure_ascii=False).encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_main())
