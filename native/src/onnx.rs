//! Minimal ONNX (protobuf) reader.
//!
//! Only the fields that matter for loading risk are read:
//! - `TensorProto.external_data` locations. ONNX loaders open these paths
//!   relative to the model, so an absolute path or `..` makes the loader read
//!   an arbitrary file (CVE-2022-25882, CVE-2024-27318 class).
//! - Operator domains. Ops outside the standard domains need a custom-op
//!   library at load time; `ai.onnx.contrib` `Py*` ops run Python.
//!
//! The reader works on `Read + Seek`, skips `raw_data` and other blobs by
//! seeking, and bounds message count, nesting depth, string size and time.

use serde_json::{json, Value};
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

const MAX_DEPTH: u32 = 32;
const MAX_MESSAGES: u64 = 5_000_000;
const MAX_STR: u64 = 4096;
const MAX_RECORDS: usize = 500;

const STANDARD_DOMAINS: &[&str] = &[
    "",
    "ai.onnx",
    "ai.onnx.ml",
    "ai.onnx.training",
    "ai.onnx.preview.training",
    "com.microsoft",
    "com.microsoft.nchwc",
    "com.microsoft.experimental",
    "org.pytorch.aten",
];

#[derive(Default)]
struct Out {
    ir_version: Option<u64>,
    producer: String,
    opsets: Vec<Value>,
    domains: Vec<String>,
    custom_ops: Vec<Value>,
    external: Vec<Value>,
    nodes: u64,
    initializers: u64,
    warnings: Vec<String>,
}

struct Pb<'r, R> {
    r: &'r mut R,
    pos: u64,
    size: u64,
    messages: u64,
    deadline: Option<Instant>,
}

const LIMIT_MESSAGES: &str = "ONNX message limit reached";
const LIMIT_TIME: &str = "time limit reached during ONNX walk";
const LIMIT_DEPTH: &str = "ONNX graph nesting limit reached";

impl<'r, R: Read + Seek> Pb<'r, R> {
    fn byte(&mut self) -> Result<u8, String> {
        let mut b = [0u8; 1];
        self.r
            .read_exact(&mut b)
            .map_err(|e| format!("onnx read: {e}"))?;
        self.pos += 1;
        Ok(b[0])
    }

    fn varint(&mut self) -> Result<u64, String> {
        let mut v = 0u64;
        for shift in (0..70).step_by(7) {
            let b = self.byte()?;
            v |= ((b & 0x7f) as u64) << shift.min(63);
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err("onnx varint too long".into())
    }

    fn skip(&mut self, n: u64, end: u64) -> Result<(), String> {
        let target = self.pos.checked_add(n).ok_or("onnx length overflow")?;
        if target > end {
            return Err("onnx field runs past its message".into());
        }
        self.r
            .seek(SeekFrom::Start(target))
            .map_err(|e| e.to_string())?;
        self.pos = target;
        Ok(())
    }

    /// Read a tag, returning (field, wire type).
    fn tag(&mut self) -> Result<(u64, u8), String> {
        let t = self.varint()?;
        Ok((t >> 3, (t & 7) as u8))
    }

    fn len_end(&mut self, end: u64) -> Result<u64, String> {
        let n = self.varint()?;
        let sub = self.pos.checked_add(n).ok_or("onnx length overflow")?;
        if sub > end {
            return Err("onnx length runs past its message".into());
        }
        Ok(sub)
    }

    fn string(&mut self, end: u64) -> Result<String, String> {
        let sub = self.len_end(end)?;
        let n = sub - self.pos;
        let keep = n.min(MAX_STR) as usize;
        let mut buf = vec![0u8; keep];
        self.r.read_exact(&mut buf).map_err(|e| e.to_string())?;
        self.pos += keep as u64;
        self.skip(sub - self.pos, sub)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    fn skip_value(&mut self, wire: u8, end: u64) -> Result<(), String> {
        match wire {
            0 => self.varint().map(|_| ()),
            1 => self.skip(8, end),
            2 => {
                let sub = self.len_end(end)?;
                self.skip(sub - self.pos, end)
            }
            5 => self.skip(4, end),
            other => Err(format!("onnx unsupported wire type {other}")),
        }
    }

    fn enter(&mut self) -> Result<(), String> {
        self.messages += 1;
        if self.messages > MAX_MESSAGES {
            return Err(LIMIT_MESSAGES.into());
        }
        if self.messages % 4096 == 0 {
            if let Some(d) = self.deadline {
                if Instant::now() >= d {
                    return Err(LIMIT_TIME.into());
                }
            }
        }
        Ok(())
    }
}

/// Heuristic content check: ModelProto writers emit `ir_version` (field 1,
/// varint) first, followed by another ModelProto field.
pub fn looks_like_onnx(header: &[u8]) -> bool {
    header.len() >= 4
        && header[0] == 0x08
        && (1..=20).contains(&header[1])
        && matches!(
            header[2],
            0x12 | 0x1a | 0x22 | 0x28 | 0x32 | 0x3a | 0x42 | 0x72 | 0xa2
        )
}

pub fn scan_bytes(data: &[u8]) -> Result<Value, String> {
    scan_reader(&mut std::io::Cursor::new(data), None)
}

pub fn scan_reader<R: Read + Seek>(
    reader: &mut R,
    deadline: Option<Instant>,
) -> Result<Value, String> {
    let size = reader.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
    reader.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let mut pb = Pb {
        r: reader,
        pos: 0,
        size,
        messages: 0,
        deadline,
    };
    let mut out = Out::default();
    let end = pb.size;
    let limit = match model(&mut pb, end, &mut out) {
        Ok(()) => None,
        Err(e) if e == LIMIT_MESSAGES || e == LIMIT_TIME || e == LIMIT_DEPTH => Some(e),
        Err(e) => return Err(e),
    };
    if out.ir_version.is_none() {
        return Err("no ONNX ir_version field".into());
    }
    Ok(json!({
        "ir_version": out.ir_version,
        "producer_name": out.producer,
        "opset_imports": out.opsets,
        "custom_domains": out.domains,
        "custom_ops": out.custom_ops,
        "external_data": out.external,
        "node_count": out.nodes,
        "initializer_count": out.initializers,
        "warnings": out.warnings,
        "limit_exceeded": limit,
    }))
}

fn model<R: Read + Seek>(pb: &mut Pb<R>, end: u64, out: &mut Out) -> Result<(), String> {
    while pb.pos < end {
        let (field, wire) = pb.tag()?;
        match (field, wire) {
            (1, 0) => out.ir_version = Some(pb.varint()?),
            (2, 2) => out.producer = pb.string(end)?,
            (7, 2) => {
                let sub = pb.len_end(end)?;
                graph(pb, sub, out, 0)?;
            }
            (8, 2) => {
                let sub = pb.len_end(end)?;
                opset(pb, sub, out)?;
            }
            (14, 2) => {
                let sub = pb.len_end(end)?;
                function(pb, sub, out)?;
            }
            _ => pb.skip_value(wire, end)?,
        }
    }
    Ok(())
}

fn note_domain(out: &mut Out, domain: &str, op_type: Option<&str>) {
    if STANDARD_DOMAINS.contains(&domain) {
        return;
    }
    if !out.domains.iter().any(|d| d == domain) && out.domains.len() < MAX_RECORDS {
        out.domains.push(domain.to_string());
    }
    if let Some(op) = op_type {
        let known = out
            .custom_ops
            .iter()
            .any(|v| v["domain"] == domain && v["op_type"] == op);
        if !known && out.custom_ops.len() < MAX_RECORDS {
            let python = domain == "ai.onnx.contrib" && op.starts_with("Py");
            out.custom_ops
                .push(json!({"domain": domain, "op_type": op, "python_op": python}));
        }
    }
}

fn opset<R: Read + Seek>(pb: &mut Pb<R>, end: u64, out: &mut Out) -> Result<(), String> {
    pb.enter()?;
    let mut domain = String::new();
    let mut version = None;
    while pb.pos < end {
        let (field, wire) = pb.tag()?;
        match (field, wire) {
            (1, 2) => domain = pb.string(end)?,
            (2, 0) => version = Some(pb.varint()?),
            _ => pb.skip_value(wire, end)?,
        }
    }
    note_domain(out, &domain, None);
    if out.opsets.len() < MAX_RECORDS {
        out.opsets
            .push(json!({"domain": domain, "version": version}));
    }
    Ok(())
}

fn graph<R: Read + Seek>(
    pb: &mut Pb<R>,
    end: u64,
    out: &mut Out,
    depth: u32,
) -> Result<(), String> {
    pb.enter()?;
    if depth > MAX_DEPTH {
        return Err(LIMIT_DEPTH.into());
    }
    while pb.pos < end {
        let (field, wire) = pb.tag()?;
        match (field, wire) {
            (1, 2) => {
                let sub = pb.len_end(end)?;
                node(pb, sub, out, depth)?;
            }
            (5, 2) => {
                let sub = pb.len_end(end)?;
                out.initializers += 1;
                tensor(pb, sub, out)?;
            }
            (15, 2) => {
                let sub = pb.len_end(end)?;
                sparse_tensor(pb, sub, out)?;
            }
            _ => pb.skip_value(wire, end)?,
        }
    }
    Ok(())
}

fn function<R: Read + Seek>(pb: &mut Pb<R>, end: u64, out: &mut Out) -> Result<(), String> {
    pb.enter()?;
    while pb.pos < end {
        let (field, wire) = pb.tag()?;
        match (field, wire) {
            (7, 2) => {
                let sub = pb.len_end(end)?;
                node(pb, sub, out, 1)?;
            }
            (9, 2) => {
                let sub = pb.len_end(end)?;
                opset(pb, sub, out)?;
            }
            (10, 2) => {
                let domain = pb.string(end)?;
                note_domain(out, &domain, None);
            }
            _ => pb.skip_value(wire, end)?,
        }
    }
    Ok(())
}

fn node<R: Read + Seek>(pb: &mut Pb<R>, end: u64, out: &mut Out, depth: u32) -> Result<(), String> {
    pb.enter()?;
    out.nodes += 1;
    let mut op_type = String::new();
    let mut domain = String::new();
    while pb.pos < end {
        let (field, wire) = pb.tag()?;
        match (field, wire) {
            (4, 2) => op_type = pb.string(end)?,
            (7, 2) => domain = pb.string(end)?,
            (5, 2) => {
                let sub = pb.len_end(end)?;
                attribute(pb, sub, out, depth)?;
            }
            _ => pb.skip_value(wire, end)?,
        }
    }
    note_domain(out, &domain, Some(&op_type));
    Ok(())
}

fn attribute<R: Read + Seek>(
    pb: &mut Pb<R>,
    end: u64,
    out: &mut Out,
    depth: u32,
) -> Result<(), String> {
    pb.enter()?;
    while pb.pos < end {
        let (field, wire) = pb.tag()?;
        match (field, wire) {
            (5, 2) | (10, 2) => {
                let sub = pb.len_end(end)?;
                tensor(pb, sub, out)?;
            }
            (6, 2) | (11, 2) => {
                let sub = pb.len_end(end)?;
                graph(pb, sub, out, depth + 1)?;
            }
            (22, 2) | (23, 2) => {
                let sub = pb.len_end(end)?;
                sparse_tensor(pb, sub, out)?;
            }
            _ => pb.skip_value(wire, end)?,
        }
    }
    Ok(())
}

fn sparse_tensor<R: Read + Seek>(pb: &mut Pb<R>, end: u64, out: &mut Out) -> Result<(), String> {
    pb.enter()?;
    while pb.pos < end {
        let (field, wire) = pb.tag()?;
        match (field, wire) {
            (1, 2) | (2, 2) => {
                let sub = pb.len_end(end)?;
                tensor(pb, sub, out)?;
            }
            _ => pb.skip_value(wire, end)?,
        }
    }
    Ok(())
}

fn tensor<R: Read + Seek>(pb: &mut Pb<R>, end: u64, out: &mut Out) -> Result<(), String> {
    pb.enter()?;
    let mut name = String::new();
    let mut entries: Vec<(String, String)> = Vec::new();
    let mut location_flag = 0u64;
    while pb.pos < end {
        let (field, wire) = pb.tag()?;
        match (field, wire) {
            (8, 2) => name = pb.string(end)?,
            (13, 2) => {
                let sub = pb.len_end(end)?;
                entries.push(string_pair(pb, sub)?);
            }
            (14, 0) => location_flag = pb.varint()?,
            _ => pb.skip_value(wire, end)?,
        }
    }
    let location = entries
        .iter()
        .find(|(k, _)| k == "location")
        .map(|(_, v)| v.clone());
    if location.is_none() && location_flag != 1 {
        return Ok(());
    }
    let loc = location.unwrap_or_default();
    let (absolute, traversal) = classify_location(&loc);
    if out.external.len() < MAX_RECORDS {
        let get = |key: &str| {
            entries
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
        };
        out.external.push(json!({
            "tensor": name,
            "location": loc,
            "offset": get("offset"),
            "length": get("length"),
            "absolute": absolute,
            "traversal": traversal,
            "missing_location": loc.is_empty(),
        }));
    } else if out.external.len() == MAX_RECORDS {
        out.warnings.push(format!(
            "more than {MAX_RECORDS} external tensors; list truncated"
        ));
    }
    Ok(())
}

fn string_pair<R: Read + Seek>(pb: &mut Pb<R>, end: u64) -> Result<(String, String), String> {
    pb.enter()?;
    let mut key = String::new();
    let mut value = String::new();
    while pb.pos < end {
        let (field, wire) = pb.tag()?;
        match (field, wire) {
            (1, 2) => key = pb.string(end)?,
            (2, 2) => value = pb.string(end)?,
            _ => pb.skip_value(wire, end)?,
        }
    }
    Ok((key, value))
}

/// (absolute, traversal) for an external-data location, judged the same way
/// on every OS: `/x`, `\x`, `C:...`, `~...`, URLs and NUL bytes count as
/// absolute; any `..` path component counts as traversal.
pub fn classify_location(loc: &str) -> (bool, bool) {
    let b = loc.as_bytes();
    let drive = b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':';
    let absolute = loc.starts_with('/')
        || loc.starts_with('\\')
        || loc.starts_with('~')
        || drive
        || loc.contains("://")
        || loc.contains('\0');
    let traversal = loc.split(['/', '\\']).any(|part| part == "..");
    (absolute, traversal)
}

#[cfg(test)]
#[path = "onnx_tests.rs"]
mod tests;
