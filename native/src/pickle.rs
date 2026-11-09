//! Symbolic pickle opcode walk.
//!
//! The walker never calls `find_class`, never imports a module, and never
//! invokes a REDUCE target. It only records which globals a later `loads`
//! would resolve. That is the whole point: malicious model files are
//! classified without executing them.
//!
//! Hardening notes:
//! - The memo is a map keyed like CPython's, so MEMOIZE uses the entry count
//!   and large PUT indices cannot hide STACK_GLOBAL operands.
//! - Oversized strings are skipped, not treated as the end of the walk, so a
//!   2 MB filler string cannot hide a later GLOBAL.
//! - Pickles concatenated in one file (legacy torch, repeated `pickle.load`)
//!   are walked one after another.
//! - Opcode count, memo size, and wall clock are bounded.

use crate::limits::{MAX_MEMO, MAX_PICKLE_OPS};
use serde::Serialize;
use serde_json::json;
use std::collections::HashMap;
use std::time::Instant;

/// Strings longer than this are skipped and kept only as a preview.
const MAX_STRING_KEEP: usize = 1_000_000;
/// Longest module or attribute name kept for a resolved/unresolved global.
/// Real dotted names are short; the cap stops DUP + STACK_GLOBAL chains from
/// doubling a string every two opcodes (a 59-byte input reached 2 GB).
const MAX_SYMBOL: usize = 512;
/// Unresolved STACK_GLOBAL warnings kept per file (globals list: 10x this).
const MAX_UNRESOLVED_REPORTS: usize = 1_000;
const MAX_LINE: usize = 8_192;
const MAX_STACK: usize = 1_000_000;
const MAX_SEGMENTS: usize = 64;

#[derive(Debug, Clone, Serialize)]
pub struct GlobalHit {
    pub module: String,
    pub name: String,
    pub opcode: String,
    pub offset: usize,
    pub confidence: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReduceHit {
    pub callable: String,
    pub args_preview: String,
    pub offset: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct PickleReport {
    pub protocol: Option<u8>,
    pub globals: Vec<GlobalHit>,
    pub reduces: Vec<ReduceHit>,
    pub extensions: Vec<u64>,
    pub persistent_ids: Vec<String>,
    pub unknown_opcodes: Vec<serde_json::Value>,
    pub warnings: Vec<String>,
    /// Every walked pickle ended in STOP.
    pub stopped: bool,
    pub bytes_consumed: usize,
    pub stack_depth_end: usize,
    /// Number of pickles walked back to back in this buffer.
    pub pickle_count: usize,
    /// Bytes after the last pickle that were not walked as pickle.
    pub trailing_bytes: usize,
    /// Set when an opcode, memo, or time limit stopped the walk.
    pub limit_exceeded: Option<String>,
}

#[derive(Clone)]
enum Val {
    Mark,
    None,
    Bool(bool),
    Int(i64),
    Str(String),
    Global { module: String, name: String },
    Other(String),
}

impl Val {
    fn preview(&self) -> String {
        match self {
            Val::Mark => "<mark>".into(),
            Val::None => "None".into(),
            Val::Bool(b) => b.to_string(),
            Val::Int(i) => i.to_string(),
            Val::Str(s) => truncate(s, 180),
            Val::Global { module, name } => truncate(&format!("{module}.{name}"), 180),
            Val::Other(s) => truncate(s, 180),
        }
    }
}

struct Cur<'a> {
    data: &'a [u8],
    i: usize,
}

impl<'a> Cur<'a> {
    fn remain(&self) -> usize {
        self.data.len().saturating_sub(self.i)
    }
    fn u8(&mut self) -> Option<u8> {
        let b = *self.data.get(self.i)?;
        self.i += 1;
        Some(b)
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if n > self.remain() {
            return None;
        }
        let s = &self.data[self.i..self.i + n];
        self.i += n;
        Some(s)
    }
    fn line(&mut self) -> Option<String> {
        let start = self.i;
        let end_limit = start.saturating_add(MAX_LINE).min(self.data.len());
        let slice = &self.data[start..end_limit];
        let rel = slice.iter().position(|b| *b == b'\n')?;
        self.i = start + rel + 1;
        Some(String::from_utf8_lossy(&slice[..rel]).into_owned())
    }
    fn u16(&mut self) -> Option<u16> {
        let b = self.take(2)?;
        Some(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Option<u32> {
        let b = self.take(4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn i32(&mut self) -> Option<i32> {
        self.u32().map(|v| v as i32)
    }
    fn u64(&mut self) -> Option<u64> {
        let b = self.take(8)?;
        Some(u64::from_le_bytes(b.try_into().ok()?))
    }
}

/// Per-pickle machine state. The report accumulates across pickles.
struct Machine {
    stack: Vec<Val>,
    memo: HashMap<u64, Val>,
}

impl Machine {
    fn new() -> Self {
        Self {
            stack: Vec::new(),
            memo: HashMap::new(),
        }
    }
}

struct Limits {
    max_ops: usize,
    deadline: Option<Instant>,
    ops: usize,
}

impl Limits {
    /// Count one opcode; returns a limit message when a bound is hit.
    fn tick(&mut self) -> Option<String> {
        self.ops += 1;
        if self.ops > self.max_ops {
            return Some(format!("pickle opcode limit {} reached", self.max_ops));
        }
        if self.ops % 65_536 == 0 {
            if let Some(d) = self.deadline {
                if Instant::now() >= d {
                    return Some("time limit reached during pickle walk".into());
                }
            }
        }
        None
    }
}

fn new_report() -> PickleReport {
    PickleReport {
        protocol: None,
        globals: Vec::new(),
        reduces: Vec::new(),
        extensions: Vec::new(),
        persistent_ids: Vec::new(),
        unknown_opcodes: Vec::new(),
        warnings: Vec::new(),
        stopped: false,
        bytes_consumed: 0,
        stack_depth_end: 0,
        pickle_count: 0,
        trailing_bytes: 0,
        limit_exceeded: None,
    }
}

pub fn walk(data: &[u8]) -> PickleReport {
    walk_bounded(data, MAX_PICKLE_OPS, None)
}

pub fn walk_with_deadline(data: &[u8], deadline: Instant) -> PickleReport {
    walk_bounded(data, MAX_PICKLE_OPS, Some(deadline))
}

pub fn walk_bounded(data: &[u8], max_ops: usize, deadline: Option<Instant>) -> PickleReport {
    let mut report = new_report();
    let mut limits = Limits {
        max_ops,
        deadline,
        ops: 0,
    };
    let mut cur = Cur { data, i: 0 };
    let mut all_stopped = true;
    let mut stack_end = 0;

    while cur.i < data.len() && report.pickle_count < MAX_SEGMENTS {
        if report.pickle_count > 0 && !next_segment_is_pickle(&data[cur.i..], max_ops) {
            break;
        }
        let start = cur.i;
        let mut machine = Machine::new();
        let stopped = walk_segment(&mut cur, &mut machine, &mut report, &mut limits);
        report.pickle_count += 1;
        stack_end = machine.stack.len();
        if !stopped {
            all_stopped = false;
            break;
        }
        if cur.i == start {
            break;
        }
    }
    report.stopped = all_stopped && report.pickle_count > 0;
    report.bytes_consumed = cur.i;
    report.trailing_bytes = data.len().saturating_sub(cur.i);
    report.stack_depth_end = stack_end;
    if report.pickle_count > 1 {
        report.warnings.push(format!(
            "{} pickles concatenated in one buffer; each was walked",
            report.pickle_count
        ));
    }

    // The byte-pattern scan is a fallback. Bytes the structured walk already
    // covered are skipped: re-scanning them raises false pairs such as
    // "x.system" out of argument strings. Bytes after a desync or after the
    // last pickle are still searched, so a tail payload cannot hide there.
    let structured = report.globals.iter().any(|g| g.confidence == "structured");
    let scan_from = if structured { report.bytes_consumed } else { 0 };
    merge_literal_globals(data, scan_from, &mut report);
    report
}

/// A later segment is walked when it starts with PROTO, or when a trial walk
/// of it ends in STOP after resolving at least one global or call.
fn next_segment_is_pickle(rest: &[u8], max_ops: usize) -> bool {
    if rest.len() >= 2 && rest[0] == 0x80 && rest[1] <= 5 {
        return true;
    }
    let mut trial = new_report();
    let mut limits = Limits {
        max_ops: max_ops.min(100_000),
        deadline: None,
        ops: 0,
    };
    let mut cur = Cur { data: rest, i: 0 };
    let mut machine = Machine::new();
    let stopped = walk_segment(&mut cur, &mut machine, &mut trial, &mut limits);
    stopped && (!trial.globals.is_empty() || !trial.reduces.is_empty())
}

/// Walk one pickle up to and including STOP. True when STOP was reached.
fn walk_segment(
    cur: &mut Cur<'_>,
    machine: &mut Machine,
    report: &mut PickleReport,
    limits: &mut Limits,
) -> bool {
    while cur.i < cur.data.len() {
        if let Some(msg) = limits.tick() {
            report.warnings.push(msg.clone());
            report.limit_exceeded = Some(msg);
            return false;
        }
        let offset = cur.i;
        let Some(op) = cur.u8() else { return false };
        match step(op, offset, cur, machine, report) {
            Step::Continue => {}
            Step::Stop => return true,
            Step::Fail => return false,
        }
        if machine.stack.len() > MAX_STACK {
            let msg = format!("pickle stack exceeded {MAX_STACK} items");
            report.warnings.push(msg.clone());
            report.limit_exceeded = Some(msg);
            return false;
        }
    }
    false
}

enum Step {
    Continue,
    Stop,
    Fail,
}

fn fail(report: &mut PickleReport, msg: &str) -> Step {
    report.warnings.push(msg.to_string());
    Step::Fail
}

fn step(
    op: u8,
    offset: usize,
    cur: &mut Cur<'_>,
    m: &mut Machine,
    report: &mut PickleReport,
) -> Step {
    let stack = &mut m.stack;
    match op {
        b'(' => stack.push(Val::Mark),
        b'.' => return Step::Stop,
        b'0' => {
            stack.pop();
        }
        b'1' => {
            pop_mark(stack);
        }
        b'2' => match stack.last().cloned() {
            Some(v) => stack.push(v),
            None => report.warnings.push(format!("DUP underflow at {offset}")),
        },
        b'F' => match cur.line() {
            Some(s) => stack.push(Val::Other(format!("float:{s}"))),
            None => return fail(report, "truncated FLOAT"),
        },
        b'I' => match cur.line() {
            Some(s) if s == "01" => stack.push(Val::Bool(true)),
            Some(s) if s == "00" => stack.push(Val::Bool(false)),
            Some(s) => stack.push(Val::Other(format!("int:{s}"))),
            None => return fail(report, "truncated INT"),
        },
        b'J' => match cur.i32() {
            Some(v) => stack.push(Val::Int(v as i64)),
            None => return fail(report, "truncated BININT"),
        },
        b'K' => match cur.u8() {
            Some(v) => stack.push(Val::Int(v as i64)),
            None => return fail(report, "truncated BININT1"),
        },
        b'L' => match cur.line() {
            Some(s) => stack.push(Val::Other(format!("long:{s}"))),
            None => return fail(report, "truncated LONG"),
        },
        b'M' => match cur.u16() {
            Some(v) => stack.push(Val::Int(v as i64)),
            None => return fail(report, "truncated BININT2"),
        },
        b'N' => stack.push(Val::None),
        b'P' => match cur.line() {
            Some(s) => {
                report.persistent_ids.push(truncate(&s, 200));
                stack.push(Val::Other(format!("persid:{s}")));
            }
            None => return fail(report, "truncated PERSID"),
        },
        b'Q' => {
            let id = stack
                .pop()
                .unwrap_or(Val::Other("missing".into()))
                .preview();
            report.persistent_ids.push(id.clone());
            stack.push(Val::Other(format!("binpersid:{id}")));
        }
        b'R' => {
            let args = stack.pop().unwrap_or(Val::Other("missing-args".into()));
            let callable = stack.pop().unwrap_or(Val::Other("missing-callable".into()));
            report.reduces.push(ReduceHit {
                callable: callable.preview(),
                args_preview: args.preview(),
                offset,
            });
            stack.push(Val::Other(format!("reduce:{}", callable.preview())));
        }
        b'S' | b'V' => match cur.line() {
            Some(s) => stack.push(Val::Str(unescape_pickle_string(&s))),
            None => return fail(report, "truncated STRING/UNICODE"),
        },
        b'T' => match counted_string(cur, 4, true) {
            Some(s) => stack.push(Val::Str(s)),
            None => return fail(report, "truncated BINSTRING"),
        },
        b'U' => match counted_string(cur, 1, false) {
            Some(s) => stack.push(Val::Str(s)),
            None => return fail(report, "truncated SHORT_BINSTRING"),
        },
        b'C' => match counted_string(cur, 1, false) {
            Some(s) => stack.push(Val::Other(format!("bytes:{}", truncate(&s, 80)))),
            None => return fail(report, "truncated SHORT_BINBYTES"),
        },
        b'X' => match counted_string(cur, 4, false) {
            Some(s) => stack.push(Val::Str(s)),
            None => return fail(report, "truncated BINUNICODE"),
        },
        b'a' => {
            stack.pop();
        }
        b'b' => {
            let state = stack.pop().unwrap_or(Val::None);
            let inst = stack.pop().unwrap_or(Val::Other("missing-inst".into()));
            stack.push(Val::Other(format!(
                "build:{} state={}",
                inst.preview(),
                state.preview()
            )));
        }
        b'c' => match read_global(cur) {
            Some((module, name)) => push_global(report, stack, &module, &name, "GLOBAL", offset),
            None => return fail(report, "truncated GLOBAL"),
        },
        b'd' | b't' | b'l' | 0x91 => {
            let items = pop_mark(stack);
            let label = match op {
                b'd' => "dict",
                b't' => "tuple",
                b'l' => "list",
                _ => "frozenset",
            };
            let inner = items
                .iter()
                .map(Val::preview)
                .collect::<Vec<_>>()
                .join(", ");
            stack.push(Val::Other(truncate(&format!("{label}({inner})"), 300)));
        }
        b'}' => stack.push(Val::Other("dict".into())),
        b']' => stack.push(Val::Other("list".into())),
        b')' => stack.push(Val::Other("tuple".into())),
        0x8f => stack.push(Val::Other("set".into())),
        b'e' | b'u' | 0x90 => {
            pop_mark(stack);
        }
        b'g' => match cur.line() {
            Some(s) => push_memo(stack, &m.memo, parse_memo_index(&s), report),
            None => return fail(report, "truncated GET"),
        },
        b'h' => match cur.u8() {
            Some(i) => push_memo(stack, &m.memo, i as u64, report),
            None => return fail(report, "truncated BINGET"),
        },
        b'j' => match cur.u32() {
            Some(i) => push_memo(stack, &m.memo, i as u64, report),
            None => return fail(report, "truncated LONG_BINGET"),
        },
        b'i' => match read_global(cur) {
            Some((module, name)) => {
                // Class name is in the instruction, not on the stack. Args sit
                // above the mark and must be popped before we record the instance.
                report.globals.push(GlobalHit {
                    module: module.clone(),
                    name: name.clone(),
                    opcode: "INST".into(),
                    offset,
                    confidence: "structured".into(),
                });
                let args = pop_mark(stack);
                report.reduces.push(ReduceHit {
                    callable: format!("{module}.{name}"),
                    args_preview: format!("{} args", args.len()),
                    offset,
                });
                stack.push(Val::Other(format!("inst:{module}.{name}")));
            }
            None => return fail(report, "truncated INST"),
        },
        b'o' => {
            let items = pop_mark(stack);
            let class = items
                .first()
                .cloned()
                .unwrap_or(Val::Other("missing-class".into()));
            let args = items.iter().skip(1).map(Val::preview).collect::<Vec<_>>();
            report.reduces.push(ReduceHit {
                callable: class.preview(),
                args_preview: truncate(&format!("OBJ ({})", args.join(", ")), 300),
                offset,
            });
            stack.push(Val::Other(format!("obj:{}", class.preview())));
        }
        b'p' => match cur.line() {
            Some(s) => put_memo(stack, &mut m.memo, parse_memo_index(&s), report),
            None => return fail(report, "truncated PUT"),
        },
        b'q' => match cur.u8() {
            Some(i) => put_memo(stack, &mut m.memo, i as u64, report),
            None => return fail(report, "truncated BINPUT"),
        },
        b'r' => match cur.u32() {
            Some(i) => put_memo(stack, &mut m.memo, i as u64, report),
            None => return fail(report, "truncated LONG_BINPUT"),
        },
        b's' => {
            stack.pop();
            stack.pop();
        }
        b'G' => {
            if cur.take(8).is_none() {
                return fail(report, "truncated BINFLOAT");
            }
            stack.push(Val::Other("float".into()));
        }
        0x80 => match cur.u8() {
            Some(v) => {
                if report.protocol.is_none() {
                    report.protocol = Some(v);
                }
                if v > 5 {
                    report.warnings.push(format!("unknown pickle protocol {v}"));
                }
            }
            None => return fail(report, "truncated PROTO"),
        },
        0x81 => record_newobj(stack, report, offset, false),
        0x82 => match cur.u8() {
            Some(v) => push_ext(stack, report, v as u64),
            None => return fail(report, "truncated EXT1"),
        },
        0x83 => match cur.u16() {
            Some(v) => push_ext(stack, report, v as u64),
            None => return fail(report, "truncated EXT2"),
        },
        0x84 => match cur.u32() {
            Some(v) => push_ext(stack, report, v as u64),
            None => return fail(report, "truncated EXT4"),
        },
        0x85 => pop_n_tuple(stack, 1),
        0x86 => pop_n_tuple(stack, 2),
        0x87 => pop_n_tuple(stack, 3),
        0x88 => stack.push(Val::Bool(true)),
        0x89 => stack.push(Val::Bool(false)),
        0x8a => {
            if !skip_counted(cur, 1) {
                return fail(report, "truncated LONG1");
            }
            stack.push(Val::Other("long".into()));
        }
        0x8b => {
            if !skip_counted(cur, 4) {
                return fail(report, "truncated LONG4");
            }
            stack.push(Val::Other("long".into()));
        }
        b'B' => {
            if !skip_counted(cur, 4) {
                return fail(report, "truncated BINBYTES");
            }
            stack.push(Val::Other("bytes".into()));
        }
        0x8c => match counted_string(cur, 1, false) {
            Some(s) => stack.push(Val::Str(s)),
            None => return fail(report, "truncated SHORT_BINUNICODE"),
        },
        0x8d => match counted_string(cur, 8, false) {
            Some(s) => stack.push(Val::Str(s)),
            None => return fail(report, "truncated BINUNICODE8"),
        },
        0x8e => {
            if !skip_counted(cur, 8) {
                return fail(report, "truncated BINBYTES8");
            }
            stack.push(Val::Other("bytes".into()));
        }
        0x92 => record_newobj(stack, report, offset, true),
        0x93 => stack_global(stack, report, offset),
        0x94 => match stack.last().cloned() {
            // CPython: memo[len(memo)] = stack[-1]
            Some(top) => {
                let idx = m.memo.len() as u64;
                put_memo_value(&mut m.memo, idx, top, report);
            }
            None => report
                .warnings
                .push(format!("MEMOIZE underflow at {offset}")),
        },
        0x95 => {
            // Frame length is a hint. Opcodes continue inside the frame.
            if cur.u64().is_none() {
                return fail(report, "truncated FRAME");
            }
        }
        0x96 => {
            if !skip_counted(cur, 8) {
                return fail(report, "truncated BYTEARRAY8");
            }
            stack.push(Val::Other("bytearray".into()));
        }
        0x97 => {
            report
                .warnings
                .push("NEXT_BUFFER out-of-band pickle buffer".into());
            stack.push(Val::Other("next-buffer".into()));
        }
        0x98 => {
            report
                .warnings
                .push("READONLY_BUFFER on pickle stack".into());
        }
        other => {
            report.unknown_opcodes.push(json!({
                "offset": offset,
                "byte": other,
                "hex": format!("{other:02x}"),
            }));
            report.warnings.push(format!(
                "unrecognized opcode 0x{other:02x} at {offset}; walk stopped"
            ));
            return Step::Fail;
        }
    }
    Step::Continue
}

fn stack_global(stack: &mut Vec<Val>, report: &mut PickleReport, offset: usize) {
    let name = stack.pop().unwrap_or(Val::Other("missing-name".into()));
    let module = stack.pop().unwrap_or(Val::Other("missing-module".into()));
    let resolved = matches!(module, Val::Str(_)) && matches!(name, Val::Str(_));
    let module_s = match module {
        Val::Str(s) => symbol(&s),
        other => other.preview(),
    };
    let name_s = match name {
        Val::Str(s) => symbol(&s),
        other => other.preview(),
    };
    if resolved {
        push_global(report, stack, &module_s, &name_s, "STACK_GLOBAL", offset);
        return;
    }
    // Operands that are not strings mean the walker lost track (or the file
    // was built to make it lose track). Keep the hit, marked unresolved.
    if report.warnings.len() < MAX_UNRESOLVED_REPORTS {
        report.warnings.push(format!(
            "STACK_GLOBAL at {offset} has non-string operands ({module_s}, {name_s})"
        ));
    }
    if report.globals.len() >= MAX_UNRESOLVED_REPORTS * 10 {
        stack.push(Val::Other("unresolved-global".into()));
        return;
    }
    report.globals.push(GlobalHit {
        module: module_s.clone(),
        name: name_s.clone(),
        opcode: "STACK_GLOBAL".into(),
        offset,
        confidence: "unresolved".into(),
    });
    stack.push(Val::Global {
        module: module_s,
        name: name_s,
    });
}

fn push_ext(stack: &mut Vec<Val>, report: &mut PickleReport, code: u64) {
    report.extensions.push(code);
    stack.push(Val::Other(format!("ext:{code}")));
}

/// A module or attribute name, bounded to `MAX_SYMBOL` bytes (on a char boundary).
fn symbol(text: &str) -> String {
    if text.len() <= MAX_SYMBOL {
        return text.to_string();
    }
    let mut end = MAX_SYMBOL;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}

fn push_global(
    report: &mut PickleReport,
    stack: &mut Vec<Val>,
    module: &str,
    name: &str,
    opcode: &str,
    offset: usize,
) {
    let module = &symbol(module);
    let name = &symbol(name);
    report.globals.push(GlobalHit {
        module: module.to_string(),
        name: name.to_string(),
        opcode: opcode.to_string(),
        offset,
        confidence: "structured".into(),
    });
    stack.push(Val::Global {
        module: module.to_string(),
        name: name.to_string(),
    });
}

fn read_global(cur: &mut Cur<'_>) -> Option<(String, String)> {
    let module = cur.line()?;
    let name = cur.line()?;
    Some((module, name))
}

fn pop_mark(stack: &mut Vec<Val>) -> Vec<Val> {
    match stack.iter().rposition(|v| matches!(v, Val::Mark)) {
        Some(pos) => {
            let items: Vec<Val> = stack.drain(pos + 1..).collect();
            stack.pop();
            items
        }
        None => std::mem::take(stack),
    }
}

fn pop_n_tuple(stack: &mut Vec<Val>, n: usize) {
    let mut items = Vec::with_capacity(n);
    for _ in 0..n {
        items.push(stack.pop().unwrap_or(Val::None));
    }
    items.reverse();
    let preview = items
        .iter()
        .map(Val::preview)
        .collect::<Vec<_>>()
        .join(", ");
    stack.push(Val::Other(format!("({preview})")));
}

fn record_newobj(
    stack: &mut Vec<Val>,
    report: &mut PickleReport,
    offset: usize,
    with_kwargs: bool,
) {
    if with_kwargs {
        stack.pop();
    }
    let args = stack.pop().unwrap_or(Val::Other("missing-args".into()));
    let cls = stack.pop().unwrap_or(Val::Other("missing-class".into()));
    report.reduces.push(ReduceHit {
        callable: cls.preview(),
        args_preview: args.preview(),
        offset,
    });
    stack.push(Val::Other(format!("newobj:{}", cls.preview())));
}

fn parse_memo_index(s: &str) -> u64 {
    s.trim().parse::<u64>().unwrap_or(u64::MAX)
}

fn push_memo(stack: &mut Vec<Val>, memo: &HashMap<u64, Val>, idx: u64, report: &mut PickleReport) {
    match memo.get(&idx) {
        Some(v) => stack.push(v.clone()),
        None => {
            report
                .warnings
                .push(format!("memo get of missing index {idx}"));
            stack.push(Val::Other(format!("memo-missing:{idx}")));
        }
    }
}

fn put_memo(stack: &[Val], memo: &mut HashMap<u64, Val>, idx: u64, report: &mut PickleReport) {
    if let Some(top) = stack.last().cloned() {
        put_memo_value(memo, idx, top, report);
    }
}

fn put_memo_value(memo: &mut HashMap<u64, Val>, idx: u64, val: Val, report: &mut PickleReport) {
    if memo.len() >= MAX_MEMO && !memo.contains_key(&idx) {
        let msg = format!("pickle memo exceeded {MAX_MEMO} entries");
        if report.limit_exceeded.is_none() {
            report.warnings.push(msg.clone());
            report.limit_exceeded = Some(msg);
        }
        return;
    }
    memo.insert(idx, val);
}

fn read_len(cur: &mut Cur<'_>, len_bytes: usize, signed4: bool) -> Option<usize> {
    match len_bytes {
        1 => cur.u8().map(|v| v as usize),
        4 => {
            let v = cur.u32()?;
            if signed4 && (v as i32) < 0 {
                return None;
            }
            Some(v as usize)
        }
        8 => usize::try_from(cur.u64()?).ok(),
        _ => None,
    }
}

/// Read a length-prefixed string. Strings longer than `MAX_STRING_KEEP` are
/// consumed in full but only a prefix is kept, so a large filler string
/// cannot end the walk early.
fn counted_string(cur: &mut Cur<'_>, len_bytes: usize, signed4: bool) -> Option<String> {
    let n = read_len(cur, len_bytes, signed4)?;
    let bytes = cur.take(n)?;
    let kept = &bytes[..n.min(MAX_STRING_KEEP)];
    Some(String::from_utf8_lossy(kept).into_owned())
}

fn skip_counted(cur: &mut Cur<'_>, len_bytes: usize) -> bool {
    match read_len(cur, len_bytes, false) {
        Some(n) => cur.take(n).is_some(),
        None => false,
    }
}

fn unescape_pickle_string(s: &str) -> String {
    let t = s.trim();
    let t = t.strip_prefix('\'').unwrap_or(t);
    let t = t.strip_suffix('\'').unwrap_or(t);
    let t = t.strip_prefix('"').unwrap_or(t);
    let t = t.strip_suffix('"').unwrap_or(t);
    t.to_string()
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_string();
    }
    s.chars().take(n).collect::<String>() + "…"
}

fn looks_like_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c == '_' || c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c == '_' || c == '.' || c.is_ascii_alphanumeric()) && s.len() <= 200
}

fn merge_literal_globals(data: &[u8], from: usize, report: &mut PickleReport) {
    // Safety net for walks that desynchronise before a GLOBAL. Only newline-
    // delimited identifier pairs are accepted, which is the proto-0 encoding.
    let mut i = from;
    let mut hits = 0usize;
    while i + 4 < data.len() && hits < 10_000 {
        if data[i] == b'c' || data[i] == b'i' {
            if let Some((module, name)) = literal_pair(data, i + 1) {
                let dup = report.globals.iter().any(|g| {
                    g.offset == i
                        || (g.module == module && g.name == name && g.confidence == "structured")
                });
                if !dup {
                    hits += 1;
                    report.globals.push(GlobalHit {
                        module,
                        name,
                        opcode: "LITERAL_SCAN".into(),
                        offset: i,
                        confidence: "heuristic".into(),
                    });
                }
            }
        }
        i += 1;
    }
}

fn literal_pair(data: &[u8], start: usize) -> Option<(String, String)> {
    let (module, n1) = read_ident_line(data, start)?;
    let (name, _) = read_ident_line(data, n1)?;
    Some((module, name))
}

fn read_ident_line(data: &[u8], start: usize) -> Option<(String, usize)> {
    if start >= data.len() {
        return None;
    }
    let rest = &data[start..data.len().min(start + 201)];
    let nl = rest.iter().position(|b| *b == b'\n')?;
    if nl == 0 || nl > 200 {
        return None;
    }
    let s = std::str::from_utf8(&rest[..nl]).ok()?;
    if !looks_like_ident(s) {
        return None;
    }
    Some((s.to_string(), start + nl + 1))
}

pub fn looks_like_pickle(data: &[u8]) -> bool {
    if data.len() >= 2 && data[0] == 0x80 && data[1] <= 5 {
        return true;
    }
    let report = walk(data);
    report.stopped && (!report.globals.is_empty() || report.bytes_consumed > 8)
}

#[cfg(test)]
#[path = "pickle_tests.rs"]
mod tests;
