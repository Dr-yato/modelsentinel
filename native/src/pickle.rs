//! Symbolic pickle opcode walk.
//!
//! The walker never calls `find_class`, never imports a module, and never
//! invokes a REDUCE target. It only records which globals a later `loads`
//! would resolve. That is the whole point: malicious model files are
//! classified without executing them.

use serde::Serialize;
use serde_json::json;

const MAX_STRING: usize = 1_000_000;
const MAX_LINE: usize = 8_192;

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
    pub stopped: bool,
    pub bytes_consumed: usize,
    pub stack_depth_end: usize,
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
            Val::Global { module, name } => format!("{module}.{name}"),
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
        if self.i >= self.data.len() {
            return None;
        }
        let b = self.data[self.i];
        self.i += 1;
        Some(b)
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if n > self.remain() || n > MAX_STRING {
            return None;
        }
        let s = &self.data[self.i..self.i + n];
        self.i += n;
        Some(s)
    }
    fn skip(&mut self, n: usize) -> bool {
        if n > self.remain() {
            return false;
        }
        self.i += n;
        true
    }
    fn line(&mut self) -> Option<String> {
        let start = self.i;
        let end_limit = (start + MAX_LINE).min(self.data.len());
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

pub fn walk(data: &[u8]) -> PickleReport {
    let mut cur = Cur { data, i: 0 };
    let mut stack: Vec<Val> = Vec::new();
    let mut memo: Vec<Option<Val>> = Vec::new();
    let mut report = PickleReport {
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
    };

    let mut steps = 0usize;
    while cur.i < data.len() && steps < 5_000_000 {
        steps += 1;
        let offset = cur.i;
        let Some(op) = cur.u8() else { break };
        let ok = step(op, offset, &mut cur, &mut stack, &mut memo, &mut report);
        if report.stopped || !ok {
            break;
        }
    }
    report.bytes_consumed = cur.i;
    report.stack_depth_end = stack.len();

    // The byte-pattern scan is only a fallback. A completed walk already has
    // the real GLOBAL/STACK_GLOBAL records, and re-scanning those bytes
    // raises false pairs such as "x.system" out of the argument string.
    let structured = report.globals.iter().any(|g| g.confidence == "structured");
    if !structured {
        merge_literal_globals(data, &mut report);
    }
    report
}

fn step(
    op: u8,
    offset: usize,
    cur: &mut Cur<'_>,
    stack: &mut Vec<Val>,
    memo: &mut Vec<Option<Val>>,
    report: &mut PickleReport,
) -> bool {
    match op {
        b'(' => stack.push(Val::Mark),
        b'.' => report.stopped = true,
        b'0' => {
            stack.pop();
        }
        b'1' => {
            pop_mark(stack);
        }
        b'2' => {
            if let Some(v) = stack.last().cloned() {
                stack.push(v);
            } else {
                report.warnings.push(format!("DUP underflow at {offset}"));
            }
        }
        b'F' => match cur.line() {
            Some(s) => stack.push(Val::Other(format!("float:{s}"))),
            None => return fail(report, "truncated FLOAT"),
        },
        b'I' => match cur.line() {
            Some(s) => {
                if s == "01" {
                    stack.push(Val::Bool(true));
                } else if s == "00" {
                    stack.push(Val::Bool(false));
                } else {
                    stack.push(Val::Other(format!("int:{s}")));
                }
            }
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
            let id = stack.pop().unwrap_or(Val::Other("missing".into())).preview();
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
        b'U' | b'C' => match counted_string(cur, 1, false) {
            Some(s) => stack.push(if op == b'C' {
                Val::Other(format!("bytes:{}", truncate(&s, 80)))
            } else {
                Val::Str(s)
            }),
            None => return fail(report, "truncated SHORT_BINSTRING/BYTES"),
        },
        b'X' => match counted_string(cur, 4, false) {
            Some(s) => stack.push(Val::Str(s)),
            None => return fail(report, "truncated BINUNICODE"),
        },
        b'a' => {
            let v = stack.pop().unwrap_or(Val::None);
            let _ = v;
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
            stack.push(Val::Other(format!("{label}[{}]", items.len())));
        }
        b'}' => stack.push(Val::Other("dict".into())),
        b']' => stack.push(Val::Other("list".into())),
        b')' => stack.push(Val::Other("tuple".into())),
        0x8f => stack.push(Val::Other("set".into())),
        b'e' | b'u' | 0x90 => {
            let _ = pop_mark(stack);
        }
        b'g' => match cur.line() {
            Some(s) => push_memo(stack, memo, parse_memo_index(&s)),
            None => return fail(report, "truncated GET"),
        },
        b'h' => match cur.u8() {
            Some(i) => push_memo(stack, memo, i as usize),
            None => return fail(report, "truncated BINGET"),
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
        b'j' => match cur.u32() {
            Some(i) => push_memo(stack, memo, i as usize),
            None => return fail(report, "truncated LONG_BINGET"),
        },
        b'o' => {
            let items = pop_mark(stack);
            let class = items.first().cloned().unwrap_or(Val::Other("missing-class".into()));
            if let Val::Global { module, name } = &class {
                // Already recorded when the GLOBAL was pushed.
                let _ = (module, name);
            }
            report.reduces.push(ReduceHit {
                callable: class.preview(),
                args_preview: format!("OBJ {} args", items.len().saturating_sub(1)),
                offset,
            });
            stack.push(Val::Other(format!("obj:{}", class.preview())));
        }
        b'p' => match cur.line() {
            Some(s) => put_memo(stack, memo, parse_memo_index(&s)),
            None => return fail(report, "truncated PUT"),
        },
        b'q' => match cur.u8() {
            Some(i) => put_memo(stack, memo, i as usize),
            None => return fail(report, "truncated BINPUT"),
        },
        b'r' => match cur.u32() {
            Some(i) => put_memo(stack, memo, i as usize),
            None => return fail(report, "truncated LONG_BINPUT"),
        },
        b's' => {
            let _v = stack.pop();
            let _k = stack.pop();
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
            Some(v) => {
                report.extensions.push(v as u64);
                stack.push(Val::Other(format!("ext:{v}")));
            }
            None => return fail(report, "truncated EXT1"),
        },
        0x83 => match cur.u16() {
            Some(v) => {
                report.extensions.push(v as u64);
                stack.push(Val::Other(format!("ext:{v}")));
            }
            None => return fail(report, "truncated EXT2"),
        },
        0x84 => match cur.u32() {
            Some(v) => {
                report.extensions.push(v as u64);
                stack.push(Val::Other(format!("ext:{v}")));
            }
            None => return fail(report, "truncated EXT4"),
        },
        0x85 => pop_n_tuple(stack, 1),
        0x86 => pop_n_tuple(stack, 2),
        0x87 => pop_n_tuple(stack, 3),
        0x88 => stack.push(Val::Bool(true)),
        0x89 => stack.push(Val::Bool(false)),
        0x8a => {
            if !skip_long(cur, 1) {
                return fail(report, "truncated LONG1");
            }
            stack.push(Val::Other("long".into()));
        }
        0x8b => {
            if !skip_long(cur, 4) {
                return fail(report, "truncated LONG4");
            }
            stack.push(Val::Other("long".into()));
        }
        b'B' => {
            if counted_string(cur, 4, false).is_none() {
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
            if counted_string(cur, 8, false).is_none() {
                return fail(report, "truncated BINBYTES8");
            }
            stack.push(Val::Other("bytes".into()));
        }
        0x92 => record_newobj(stack, report, offset, true),
        0x93 => {
            let name = stack.pop().unwrap_or(Val::Other("missing-name".into()));
            let module = stack.pop().unwrap_or(Val::Other("missing-module".into()));
            let module_s = match module {
                Val::Str(s) => s,
                other => other.preview(),
            };
            let name_s = match name {
                Val::Str(s) => s,
                other => other.preview(),
            };
            push_global(report, stack, &module_s, &name_s, "STACK_GLOBAL", offset);
        }
        0x94 => {
            if let Some(top) = stack.last().cloned() {
                put_memo_value(memo, memo.len(), top);
            } else {
                report.warnings.push(format!("MEMOIZE underflow at {offset}"));
            }
        }
        0x95 => {
            // Frame length is a hint. Opcodes continue inside the frame.
            if cur.u64().is_none() {
                return fail(report, "truncated FRAME");
            }
        }
        0x96 => {
            if counted_string(cur, 8, false).is_none() {
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
            report
                .warnings
                .push(format!("unrecognized opcode 0x{other:02x} at {offset}; walk stopped"));
            return false;
        }
    }
    true
}

fn fail(report: &mut PickleReport, msg: &str) -> bool {
    report.warnings.push(msg.to_string());
    false
}

fn push_global(
    report: &mut PickleReport,
    stack: &mut Vec<Val>,
    module: &str,
    name: &str,
    opcode: &str,
    offset: usize,
) {
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
    if let Some(pos) = stack.iter().rposition(|v| matches!(v, Val::Mark)) {
        let items: Vec<Val> = stack.drain(pos + 1..).collect();
        stack.pop();
        items
    } else {
        let items = stack.drain(..).collect();
        items
    }
}

fn pop_n_tuple(stack: &mut Vec<Val>, n: usize) {
    let mut items = Vec::new();
    for _ in 0..n {
        items.push(stack.pop().unwrap_or(Val::None));
    }
    items.reverse();
    let preview = items
        .iter()
        .map(|item| item.preview())
        .collect::<Vec<_>>()
        .join(", ");
    stack.push(Val::Other(format!("({preview})")));
}

fn record_newobj(stack: &mut Vec<Val>, report: &mut PickleReport, offset: usize, with_kwargs: bool) {
    if with_kwargs {
        let _kwargs = stack.pop();
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

fn parse_memo_index(s: &str) -> usize {
    s.trim().parse::<usize>().unwrap_or(0)
}

fn push_memo(stack: &mut Vec<Val>, memo: &[Option<Val>], idx: usize) {
    if let Some(Some(v)) = memo.get(idx) {
        stack.push(v.clone());
    } else {
        stack.push(Val::Other(format!("memo-missing:{idx}")));
    }
}

fn put_memo(stack: &[Val], memo: &mut Vec<Option<Val>>, idx: usize) {
    if let Some(top) = stack.last().cloned() {
        put_memo_value(memo, idx, top);
    }
}

fn put_memo_value(memo: &mut Vec<Option<Val>>, idx: usize, val: Val) {
    if idx > 100_000 {
        return;
    }
    if memo.len() <= idx {
        memo.resize(idx + 1, None);
    }
    memo[idx] = Some(val);
}

fn counted_string(cur: &mut Cur<'_>, len_bytes: usize, signed4: bool) -> Option<String> {
    let n: usize = match len_bytes {
        1 => cur.u8()? as usize,
        4 => {
            let v = cur.u32()?;
            if signed4 && (v as i32) < 0 {
                return None;
            }
            v as usize
        }
        8 => {
            let v = cur.u64()?;
            if v > MAX_STRING as u64 {
                return None;
            }
            v as usize
        }
        _ => return None,
    };
    if n > MAX_STRING {
        return None;
    }
    let bytes = cur.take(n)?;
    Some(String::from_utf8_lossy(bytes).into_owned())
}

fn skip_long(cur: &mut Cur<'_>, len_bytes: usize) -> bool {
    let n = match len_bytes {
        1 => match cur.u8() {
            Some(v) => v as usize,
            None => return false,
        },
        4 => match cur.u32() {
            Some(v) => v as usize,
            None => return false,
        },
        _ => return false,
    };
    if n > MAX_STRING {
        return false;
    }
    cur.skip(n)
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

fn merge_literal_globals(data: &[u8], report: &mut PickleReport) {
    // Safety net for walks that desynchronise before a GLOBAL. Only newline-
    // delimited identifier pairs are accepted, which is the proto-0 encoding.
    let mut i = 0;
    while i + 4 < data.len() {
        if data[i] == b'c' || data[i] == b'i' {
            if let Some((module, name, _next)) = literal_pair(data, i + 1) {
                let dup = report.globals.iter().any(|g| {
                    g.module == module && g.name == name && g.confidence == "structured"
                });
                if !dup && looks_like_ident(&module) && looks_like_ident(&name) {
                    // Skip pairs the structured walk already owns at this offset.
                    let owned = report.globals.iter().any(|g| g.offset == i);
                    if !owned {
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
        }
        i += 1;
    }
}

fn literal_pair(data: &[u8], start: usize) -> Option<(String, String, usize)> {
    let (module, n1) = read_ident_line(data, start)?;
    let (name, n2) = read_ident_line(data, n1)?;
    Some((module, name, n2))
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
    let s = std::str::from_utf8(&rest[..nl]).ok()?.to_string();
    if !looks_like_ident(&s) {
        return None;
    }
    Some((s, start + nl + 1))
}

pub fn looks_like_pickle(data: &[u8]) -> bool {
    if data.len() >= 2 && data[0] == 0x80 && data[1] <= 5 {
        return true;
    }
    let report = walk(data);
    report.stopped && (!report.globals.is_empty() || report.bytes_consumed > 8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proto2_global_os_system() {
        // PROTO 2, GLOBAL os.system, STOP. Never a valid call by itself.
        let data = b"\x80\x02cos\nsystem\n.";
        let report = walk(data);
        assert!(report.stopped);
        assert_eq!(report.protocol, Some(2));
        assert_eq!(report.globals.len(), 1);
        assert_eq!(report.globals[0].module, "os");
        assert_eq!(report.globals[0].name, "system");
        assert_eq!(report.globals[0].opcode, "GLOBAL");
        assert_eq!(report.globals[0].confidence, "structured");
    }

    #[test]
    fn stack_global_is_recorded() {
        // PROTO 4, SHORT_BINUNICODE "os", SHORT_BINUNICODE "system", STACK_GLOBAL, STOP
        let data = b"\x80\x04\x8c\x02os\x8c\x06system\x93.";
        let report = walk(data);
        assert!(report.stopped, "{:?}", report.warnings);
        assert_eq!(report.globals.len(), 1);
        assert_eq!(report.globals[0].opcode, "STACK_GLOBAL");
        assert_eq!(report.globals[0].module, "os");
        assert_eq!(report.globals[0].name, "system");
    }
}
