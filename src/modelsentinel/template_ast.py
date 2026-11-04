"""Static taint analysis of a chat template's Jinja AST.

A clean chat template branches on the *role* of a message and on structure
(``is defined``, ``is string``, ``loop.first``). A poisoned one branches on the
*text* the user typed ("if 'Please answer precisely.' in message.content")
and then injects or removes instructions. This module finds every place where
user-controlled text is compared, searched, or rewritten against a literal,
and harvests those literals so the renderer can probe them as triggers.

Taint is tracked flow-insensitively through ``set`` assignments, namespace
attributes, loop variables and dict/list literals, so indirection such as
``{% set c = message.content|lower %}{% if 'x' in c %}`` is still caught.
Literals built from constant pieces (``'Please ' ~ 'answer'``) are folded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from jinja2 import nodes
from jinja2.sandbox import SandboxedEnvironment

COLL, ITEM, VALUE = "coll", "item", "value"
COLLECTION_ROOTS = {"messages", "tools", "documents", "custom_tools", "builtin_tools"}
STRUCTURAL_KEYS = {"role", "type", "id", "tool_call_id", "index"}
SAFE_TESTS = {
    "defined", "undefined", "none", "string", "mapping", "sequence", "iterable",
    "number", "integer", "float", "boolean", "true", "false", "callable",
}
COMPARE_TESTS = {"eq", "equalto", "==", "ne", "!=", "in", "sameas", "gt", "lt", "ge", "le", ">", "<", ">=", "<="}
SEARCH_METHODS = {"startswith", "endswith", "find", "rfind", "index", "rindex", "count", "__contains__"}
REWRITE_METHODS = {"replace"}
SPECIAL_TOKEN = re.compile(
    r"^\s*(?:<\|?/?[A-Za-z0-9_.:\-|▁]{1,48}\|?>|\[/?[A-Z_]{2,24}\]|<</?[A-Z]{2,8}>>|"
    r"<[a-z_]+>|</[a-z_]+>)\s*$"
)
TOKEN_PIECE = re.compile(r"<\|[^|<>]{1,48}\|>|<\uff5c[^<>]{1,48}\uff5c>|</?[a-z_]{1,24}>")
MAX_HARVEST = 64


@dataclass(frozen=True)
class Conditional:
    kind: str  # compare | search | rewrite | length | test
    literal: str | None
    lineno: int
    snippet: str


@dataclass
class TemplateFacts:
    conditionals: list[Conditional] = field(default_factory=list)
    time_conditionals: list[Conditional] = field(default_factory=list)
    harvested: list[str] = field(default_factory=list)
    loop_depth: int = 0
    parse_error: str = ""

    @property
    def trigger_literals(self) -> list[str]:
        seen: dict[str, None] = {}
        for item in self.conditionals:
            if item.literal:
                seen.setdefault(item.literal, None)
        return list(seen)


def is_special_token(text: str) -> bool:
    """Control tokens such as ``<|im_start|>user`` or ``<|channel|>final<|message|>``."""
    if not text.strip() or SPECIAL_TOKEN.match(text):
        return True
    remainder = TOKEN_PIECE.sub(" ", text).strip()
    return remainder != text.strip() and bool(re.fullmatch(r"[A-Za-z_]{0,16}", remainder))


def analyze(template: str) -> TemplateFacts:
    facts = TemplateFacts()
    env = SandboxedEnvironment()
    try:
        tree = env.parse(template)
    except RecursionError:
        facts.parse_error = "RecursionError: template nests too deeply to parse"
        return facts
    except Exception as exc:
        facts.parse_error = f"{type(exc).__name__}: {exc}"[:400]
        return facts
    analyzer = _Taint(env, template.splitlines())
    analyzer.collect_assignments(tree)
    analyzer.scan(tree)
    facts.conditionals = analyzer.conditionals
    facts.time_conditionals = analyzer.time_conditionals
    facts.harvested = _harvest(tree, set(facts.trigger_literals))
    facts.loop_depth = _loop_depth(tree)
    return facts


class _Taint:
    def __init__(self, env: SandboxedEnvironment, lines: list[str]) -> None:
        self.env = env
        self.lines = lines
        self.names: dict[str, str] = {name: COLL for name in COLLECTION_ROOTS}
        self.consts: dict[str, str] = {}
        self.conditionals: list[Conditional] = []
        self.time_conditionals: list[Conditional] = []
        self._seen: set[tuple] = set()

    # -- taint propagation ------------------------------------------------
    def collect_assignments(self, tree: nodes.Template) -> None:
        for _ in range(4):  # fixpoint over set/for chains
            before = dict(self.names)
            for node in tree.find_all((nodes.Assign, nodes.For)):
                if isinstance(node, nodes.Assign):
                    self._bind(node.target, self.taint(node.node))
                    self._bind_const(node.target, node.node)
                else:
                    source = self.taint(node.iter)
                    self._bind(node.target, ITEM if source == COLL else source)
            if self.names == before:
                break

    def _bind(self, target: nodes.Node, taint: str | None) -> None:
        if taint is None:
            return
        if isinstance(target, nodes.Name):
            self.names[target.name] = _widest(self.names.get(target.name), taint)
        elif isinstance(target, nodes.NSRef):
            key = f"{target.name}.{target.attr}"
            self.names[key] = _widest(self.names.get(key), taint)
        elif isinstance(target, nodes.Tuple):
            for item in target.items:
                self._bind(item, VALUE if taint else None)

    def _bind_const(self, target: nodes.Node, value: nodes.Node) -> None:
        literal = self._literal(value)
        if literal is None:
            return
        if isinstance(target, nodes.Name):
            self.consts[target.name] = literal
        elif isinstance(target, nodes.NSRef):
            self.consts[f"{target.name}.{target.attr}"] = literal

    def taint(self, node: nodes.Node | None) -> str | None:
        if node is None:
            return None
        if isinstance(node, nodes.Name):
            return self.names.get(node.name)
        if isinstance(node, (nodes.Getattr, nodes.Getitem)):
            return self._access(node)
        if isinstance(node, nodes.Filter):
            base = self.taint(node.node)
            if node.name in {"length", "count"} and base == COLL:
                return None
            if node.name in {"selectattr", "rejectattr", "select", "reject", "list", "reverse", "sort", "unique"}:
                return base
            return VALUE if base else None
        if isinstance(node, nodes.Call):
            if isinstance(node.node, nodes.Getattr):
                base = self.taint(node.node.node)
                return VALUE if base in {VALUE, ITEM} else base
            return None
        if isinstance(node, (nodes.List, nodes.Tuple)):
            return COLL if any(self.taint(item) for item in node.items) else None
        if isinstance(node, nodes.Dict):
            return ITEM if any(self.taint(pair.value) for pair in node.items) else None
        if isinstance(node, nodes.Concat):
            return VALUE if any(self.taint(item) for item in node.nodes) else None
        if isinstance(node, nodes.CondExpr):
            return _widest(self.taint(node.expr1), self.taint(node.expr2))
        if isinstance(node, nodes.BinExpr):
            return _widest(self.taint(node.left), self.taint(node.right))
        if isinstance(node, nodes.Slice):
            return None
        return None

    def _access(self, node: nodes.Getattr | nodes.Getitem) -> str | None:
        if isinstance(node, nodes.Getattr) and isinstance(node.node, nodes.Name):
            ns_key = f"{node.node.name}.{node.attr}"
            if ns_key in self.names:
                return self.names[ns_key]
        base = self.taint(node.node)
        if base is None:
            return None
        key = node.attr if isinstance(node, nodes.Getattr) else _const(self.env, node.arg)
        if base == COLL:
            if isinstance(node, nodes.Getitem) and isinstance(node.arg, nodes.Slice):
                return COLL
            return ITEM
        if isinstance(key, str) and key in STRUCTURAL_KEYS:
            return None  # role/type/id fields are structure, not user text
        return VALUE

    # -- sinks --------------------------------------------------------------
    def scan(self, tree: nodes.Template) -> None:
        for node in tree.find_all(nodes.Compare):
            self._compare(node)
        for node in tree.find_all(nodes.Test):
            self._test(node)
        for node in tree.find_all(nodes.Call):
            self._method(node)
        for node in tree.find_all(nodes.Filter):
            if node.name == "replace" and self.taint(node.node) == VALUE:
                self._literal_args(node, "rewrite", node.args[:1])

    def _compare(self, node: nodes.Compare) -> None:
        operands = [node.expr] + [op.expr for op in node.ops]
        if any(self._is_time(item) for item in operands):
            self._record(node, "time", next((self._literal(o) for o in operands if self._literal(o)), None), True)
        if not any(self.taint(item) == VALUE for item in operands):
            return
        others = [item for item in operands if self.taint(item) != VALUE]
        literals = [self._literal(item) for item in others]
        numbers = [_const(self.env, item) for item in others]
        sizes = [n for n in numbers if isinstance(n, (int, float)) and not isinstance(n, bool)]
        if sizes:
            if any(abs(n) > 16 for n in sizes):  # small counts are structure; a long-input trigger needs a big one
                self._record(node, "length", None)
            return
        for literal in literals:
            if literal is not None and not is_special_token(literal):
                self._record(node, "compare", literal)
                return
        if all(literal is None for literal in literals) and any(
            self.taint(item) is None and isinstance(item, (nodes.Name, nodes.Getattr, nodes.Getitem, nodes.Call))
            for item in others
        ):
            self._record(node, "compare", None)  # user text compared with a computed value

    def _test(self, node: nodes.Test) -> None:
        if node.name in SAFE_TESTS or node.name not in COMPARE_TESTS:
            return
        if self.taint(node.node) != VALUE:
            return
        self._literal_args(node, "test", node.args)

    def _method(self, node: nodes.Call) -> None:
        target = node.node
        if not isinstance(target, nodes.Getattr) or self.taint(target.node) != VALUE:
            return
        if target.attr in SEARCH_METHODS:
            self._literal_args(node, "search", node.args)
        elif target.attr in REWRITE_METHODS:
            self._literal_args(node, "rewrite", node.args[:1])

    def _literal_args(self, node: nodes.Node, kind: str, args: list) -> None:
        for arg in args:
            literal = self._literal(arg)
            if literal is not None and not is_special_token(literal):
                self._record(node, kind, literal)
                return

    def _is_time(self, node: nodes.Node) -> bool:
        for call in [node, *node.find_all(nodes.Call)]:
            if isinstance(call, nodes.Call) and isinstance(call.node, nodes.Name):
                if call.node.name in {"strftime_now", "now"}:
                    return True
        if isinstance(node, nodes.Name) and node.name in {"date_string", "strftime_now"}:
            return True
        return False

    def _literal(self, node: nodes.Node) -> str | None:
        if isinstance(node, nodes.Name) and node.name in self.consts:
            return self.consts[node.name]
        if isinstance(node, nodes.Getattr) and isinstance(node.node, nodes.Name):
            key = f"{node.node.name}.{node.attr}"
            if key in self.consts:
                return self.consts[key]
        if isinstance(node, nodes.CondExpr):
            return self._literal(node.expr2) if node.expr2 is not None else None
        if isinstance(node, nodes.Call) and isinstance(node.node, nodes.Getattr) and not node.args:
            inner = self._literal(node.node.node)
            if inner is not None and node.node.attr in {"lower", "upper", "strip", "lstrip", "rstrip", "title"}:
                return getattr(inner, node.node.attr)()
        if isinstance(node, nodes.Filter) and node.name in {"lower", "upper", "trim"} and node.node is not None:
            inner = self._literal(node.node)
            if inner is not None:
                return {"lower": inner.lower, "upper": inner.upper, "trim": inner.strip}[node.name]()
        value = _const(self.env, node)
        return value if isinstance(value, str) else None

    def _record(self, node: nodes.Node, kind: str, literal: str | None, time_based: bool = False) -> None:
        key = (kind, literal, node.lineno)
        if key in self._seen:
            return
        self._seen.add(key)
        line = self.lines[node.lineno - 1] if 0 < node.lineno <= len(self.lines) else ""
        item = Conditional(kind=kind, literal=literal, lineno=node.lineno, snippet=line.strip()[:200])
        (self.time_conditionals if time_based else self.conditionals).append(item)


def _widest(a: str | None, b: str | None) -> str | None:
    order = {None: 0, VALUE: 1, ITEM: 2, COLL: 3}
    return a if order[a] >= order[b] else b


def _const(env: SandboxedEnvironment, node: nodes.Node):
    try:
        ctx = nodes.EvalContext(env)
        return node.as_const(ctx)
    except Exception:  # nodes.Impossible and anything a hostile literal raises
        return None


def _harvest(tree: nodes.Template, skip: set[str]) -> list[str]:
    """Every string literal in the template that could act as a trigger."""
    found: dict[str, None] = {}
    for node in tree.find_all(nodes.Const):
        value = node.value
        if not isinstance(value, str) or value in skip or is_special_token(value):
            continue
        text = value.strip()
        if 3 <= len(text) <= 200 and "\n" not in text:
            found.setdefault(text, None)
        if len(found) >= MAX_HARVEST:
            break
    return list(found)


def _loop_depth(tree: nodes.Node) -> int:
    best = 0
    stack: list[tuple[nodes.Node, int]] = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        here = depth + 1 if isinstance(node, (nodes.For, nodes.Macro)) else depth
        best = max(best, here)
        stack.extend((child, here) for child in node.iter_child_nodes())
    return best
