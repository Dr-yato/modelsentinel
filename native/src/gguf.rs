//! GGUF metadata reader.
//!
//! Tensor payloads are not loaded. Keys that carry a chat template, identity,
//! or other short descriptive fields are kept in full. Large tokenizer tables
//! are consumed so later keys stay aligned, then discarded.
//!
//! Hardening: declared string and array lengths are checked against the bytes
//! left in the file before anything is allocated or skipped, nested arrays are
//! depth-limited (a chain of array-of-array headers used to recurse until the
//! stack overflowed), and the walk honours a wall-clock deadline.

use serde_json::{json, Map, Value};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Instant;

const MAX_STRING: u64 = 8_000_000;
const MAX_ARRAY: u64 = 5_000_000;
const MAX_ARRAY_DEPTH: u32 = 8;
const KEPT_ARRAY_ITEMS: u64 = 32;

#[derive(Debug)]
pub struct GgufReport {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub metadata: Map<String, Value>,
    pub discarded_keys: Vec<String>,
    pub warnings: Vec<String>,
    pub limit_exceeded: Option<String>,
}

struct Ctx {
    version: u32,
    size: u64,
    deadline: Option<Instant>,
    ticks: u64,
    warnings: Vec<String>,
}

impl Ctx {
    fn tick(&mut self) -> Result<(), String> {
        self.ticks += 1;
        if self.ticks % 4096 == 0 {
            if let Some(d) = self.deadline {
                if Instant::now() >= d {
                    return Err(LIMIT_TIME.into());
                }
            }
        }
        Ok(())
    }
}

const LIMIT_TIME: &str = "time limit reached during GGUF metadata walk";

pub fn scan_file(path: &Path) -> Result<GgufReport, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    scan_reader(&mut file, None)
}

pub fn scan_file_from<R: Read + Seek>(reader: &mut R) -> Result<GgufReport, String> {
    scan_reader(reader, None)
}

pub fn scan_file_with_deadline<R: Read + Seek>(
    reader: &mut R,
    deadline: Instant,
) -> Result<GgufReport, String> {
    scan_reader(reader, Some(deadline))
}

pub fn scan_bytes(data: &[u8]) -> Result<GgufReport, String> {
    scan_reader(&mut std::io::Cursor::new(data), None)
}

pub fn scan_bytes_with_deadline(data: &[u8], deadline: Instant) -> Result<GgufReport, String> {
    scan_reader(&mut std::io::Cursor::new(data), Some(deadline))
}

fn scan_reader<R: Read + Seek>(
    reader: &mut R,
    deadline: Option<Instant>,
) -> Result<GgufReport, String> {
    let size = reader.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
    reader.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let magic = read_exact(reader, 4)?;
    if magic != b"GGUF" {
        return Err("not a GGUF file".into());
    }
    let version = read_u32(reader)?;
    if version == 0 || version > 3 {
        return Err(format!("unsupported GGUF version {version}"));
    }
    let (tensor_count, metadata_count) = if version == 1 {
        (read_u32(reader)? as u64, read_u32(reader)? as u64)
    } else {
        (read_u64(reader)?, read_u64(reader)?)
    };
    if metadata_count > 1_000_000 {
        return Err("GGUF metadata count is not plausible".into());
    }
    let mut ctx = Ctx {
        version,
        size,
        deadline,
        ticks: 0,
        warnings: Vec::new(),
    };
    let mut metadata = Map::new();
    let mut discarded_keys = Vec::new();
    let mut limit_exceeded = None;
    for _ in 0..metadata_count {
        match read_entry(reader, &mut ctx) {
            Ok((key, Some(value))) => {
                metadata.insert(key, value);
            }
            Ok((key, None)) => discarded_keys.push(key),
            Err(e) if e == LIMIT_TIME => {
                ctx.warnings.push(e.clone());
                limit_exceeded = Some(e);
                break;
            }
            Err(e) => return Err(e),
        }
    }
    Ok(GgufReport {
        version,
        tensor_count,
        metadata_count,
        metadata,
        discarded_keys,
        warnings: ctx.warnings,
        limit_exceeded,
    })
}

fn read_entry<R: Read + Seek>(
    reader: &mut R,
    ctx: &mut Ctx,
) -> Result<(String, Option<Value>), String> {
    ctx.tick()?;
    let key = read_string(reader, ctx)?;
    let vtype = read_u32(reader)?;
    let keep = keep_full(&key);
    let value = read_value(reader, vtype, keep, ctx, 0)?;
    Ok((key, value.filter(|_| keep)))
}

fn keep_full(key: &str) -> bool {
    let k = key.to_ascii_lowercase();
    k.contains("chat_template")
        || k.contains("architecture")
        || k.contains("name")
        || k.contains("uuid")
        || k.contains("author")
        || k.contains("url")
        || k.contains("description")
        || k.contains("license")
        || k.contains("system")
        || k.ends_with("bos_token")
        || k.ends_with("eos_token")
        || k.contains("tokenizer.chat_template")
}

fn read_value<R: Read + Seek>(
    reader: &mut R,
    vtype: u32,
    keep: bool,
    ctx: &mut Ctx,
    depth: u32,
) -> Result<Option<Value>, String> {
    let value = match vtype {
        0 => json!(read_u8(reader)?),
        1 => json!(read_u8(reader)? as i8),
        2 => json!(read_u16(reader)?),
        3 => json!(read_u16(reader)? as i16),
        4 => json!(read_u32(reader)?),
        5 => json!(read_u32(reader)? as i32),
        6 => json!(f32::from_bits(read_u32(reader)?)),
        7 => json!(read_u8(reader)? != 0),
        8 => Value::String(read_string(reader, ctx)?),
        9 => return read_array(reader, keep, ctx, depth),
        10 => json!(read_u64(reader)?),
        11 => json!(read_u64(reader)? as i64),
        12 => json!(f64::from_bits(read_u64(reader)?)),
        other => return Err(format!("unknown GGUF value type {other}")),
    };
    Ok(Some(value).filter(|_| keep))
}

fn read_array<R: Read + Seek>(
    reader: &mut R,
    keep: bool,
    ctx: &mut Ctx,
    depth: u32,
) -> Result<Option<Value>, String> {
    if depth >= MAX_ARRAY_DEPTH {
        return Err(format!("GGUF arrays nested deeper than {MAX_ARRAY_DEPTH}"));
    }
    let elem = read_u32(reader)?;
    let count = if ctx.version == 1 {
        read_u32(reader)? as u64
    } else {
        read_u64(reader)?
    };
    if count > MAX_ARRAY {
        return Err(format!("GGUF array length {count} is not plausible"));
    }
    let remaining = remaining(reader, ctx)?;
    if count > remaining {
        return Err(format!(
            "GGUF array length {count} exceeds the {remaining} bytes left in the file"
        ));
    }
    let mut items = Vec::new();
    for idx in 0..count {
        ctx.tick()?;
        let item = read_value(reader, elem, keep, ctx, depth + 1)?;
        if keep && idx < KEPT_ARRAY_ITEMS {
            if let Some(v) = item {
                items.push(v);
            }
        }
    }
    if !keep {
        return Ok(None);
    }
    if count > KEPT_ARRAY_ITEMS {
        ctx.warnings.push(format!(
            "truncated stored array to {KEPT_ARRAY_ITEMS} of {count} elements"
        ));
    }
    Ok(Some(Value::Array(items)))
}

fn remaining<R: Seek>(reader: &mut R, ctx: &Ctx) -> Result<u64, String> {
    let pos = reader.stream_position().map_err(|e| e.to_string())?;
    Ok(ctx.size.saturating_sub(pos))
}

fn read_string<R: Read + Seek>(reader: &mut R, ctx: &mut Ctx) -> Result<String, String> {
    let len = if ctx.version == 1 {
        read_u32(reader)? as u64
    } else {
        read_u64(reader)?
    };
    let left = remaining(reader, ctx)?;
    if len > left {
        return Err(format!(
            "GGUF string length {len} exceeds the {left} bytes left in the file"
        ));
    }
    if len > MAX_STRING {
        // len <= left <= file size, so the cast cannot wrap negative.
        reader
            .seek(SeekFrom::Current(len as i64))
            .map_err(|e| e.to_string())?;
        ctx.warnings
            .push(format!("skipped oversized GGUF string ({len} bytes)"));
        return Ok(String::new());
    }
    let bytes = read_exact(reader, len as usize)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn read_exact<R: Read>(reader: &mut R, n: usize) -> Result<Vec<u8>, String> {
    let mut buf = vec![0u8; n];
    reader.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf)
}
fn read_u8<R: Read>(reader: &mut R) -> Result<u8, String> {
    Ok(read_exact(reader, 1)?[0])
}
fn read_u16<R: Read>(reader: &mut R) -> Result<u16, String> {
    let b = read_exact(reader, 2)?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}
fn read_u32<R: Read>(reader: &mut R) -> Result<u32, String> {
    let b = read_exact(reader, 4)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}
fn read_u64<R: Read>(reader: &mut R) -> Result<u64, String> {
    let b = read_exact(reader, 8)?;
    Ok(u64::from_le_bytes(b.try_into().unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gguf_string(s: &str) -> Vec<u8> {
        let b = s.as_bytes();
        let mut out = (b.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(b);
        out
    }

    #[test]
    fn roundtrip_chat_template() {
        let mut raw = Vec::new();
        raw.extend_from_slice(b"GGUF");
        raw.extend_from_slice(&3u32.to_le_bytes());
        raw.extend_from_slice(&0u64.to_le_bytes());
        raw.extend_from_slice(&2u64.to_le_bytes());
        raw.extend_from_slice(&gguf_string("general.architecture"));
        raw.extend_from_slice(&8u32.to_le_bytes());
        raw.extend_from_slice(&gguf_string("llama"));
        raw.extend_from_slice(&gguf_string("tokenizer.ggml.chat_template"));
        raw.extend_from_slice(&8u32.to_le_bytes());
        raw.extend_from_slice(&gguf_string(
            "{% for m in messages %}{{ m.content }}{% endfor %}",
        ));
        // Pretend tensor bytes that must not be required.
        raw.extend_from_slice(&[0u8; 64]);
        let report = scan_bytes(&raw).unwrap();
        assert_eq!(report.version, 3);
        assert_eq!(report.tensor_count, 0);
        assert_eq!(report.metadata["general.architecture"], json!("llama"));
        assert!(report.metadata["tokenizer.ggml.chat_template"]
            .as_str()
            .unwrap()
            .contains("messages"));
    }

    fn header(count: u64) -> Vec<u8> {
        let mut raw = b"GGUF".to_vec();
        raw.extend_from_slice(&3u32.to_le_bytes());
        raw.extend_from_slice(&0u64.to_le_bytes());
        raw.extend_from_slice(&count.to_le_bytes());
        raw
    }

    #[test]
    fn nested_array_chain_is_depth_limited_not_a_stack_overflow() {
        let mut raw = header(1);
        raw.extend_from_slice(&gguf_string("evil.nested"));
        raw.extend_from_slice(&9u32.to_le_bytes());
        for _ in 0..200_000 {
            raw.extend_from_slice(&9u32.to_le_bytes());
            raw.extend_from_slice(&1u64.to_le_bytes());
        }
        let err = scan_bytes(&raw).unwrap_err();
        assert!(err.contains("nested deeper"), "{err}");
    }

    #[test]
    fn string_length_beyond_file_is_rejected() {
        let mut raw = header(1);
        raw.extend_from_slice(&u64::MAX.to_le_bytes());
        let err = scan_bytes(&raw).unwrap_err();
        assert!(err.contains("exceeds"), "{err}");
        // A length with the top bit set used to seek backwards.
        let mut raw = header(1);
        raw.extend_from_slice(&gguf_string("k"));
        raw.extend_from_slice(&8u32.to_le_bytes());
        raw.extend_from_slice(&(1u64 << 63).to_le_bytes());
        assert!(scan_bytes(&raw).is_err());
    }

    #[test]
    fn array_count_beyond_file_is_rejected() {
        let mut raw = header(1);
        raw.extend_from_slice(&gguf_string("tokenizer.ggml.tokens"));
        raw.extend_from_slice(&9u32.to_le_bytes());
        raw.extend_from_slice(&8u32.to_le_bytes());
        raw.extend_from_slice(&4_000_000u64.to_le_bytes());
        assert!(scan_bytes(&raw).unwrap_err().contains("exceeds"));
    }

    #[test]
    fn kept_arrays_are_truncated_and_discarded_keys_listed() {
        let mut raw = header(2);
        raw.extend_from_slice(&gguf_string("general.tags.name"));
        raw.extend_from_slice(&9u32.to_le_bytes());
        raw.extend_from_slice(&4u32.to_le_bytes());
        raw.extend_from_slice(&40u64.to_le_bytes());
        for i in 0..40u32 {
            raw.extend_from_slice(&i.to_le_bytes());
        }
        raw.extend_from_slice(&gguf_string("tokenizer.ggml.scores"));
        raw.extend_from_slice(&6u32.to_le_bytes());
        raw.extend_from_slice(&1.0f32.to_bits().to_le_bytes());
        let r = scan_bytes(&raw).unwrap();
        assert_eq!(
            r.metadata["general.tags.name"].as_array().unwrap().len(),
            32
        );
        assert_eq!(r.discarded_keys, vec!["tokenizer.ggml.scores".to_string()]);
        assert!(r
            .warnings
            .iter()
            .any(|w| w.contains("truncated stored array")));
    }

    #[test]
    fn v1_header_and_bad_versions() {
        let mut raw = b"GGUF".to_vec();
        raw.extend_from_slice(&1u32.to_le_bytes());
        raw.extend_from_slice(&0u32.to_le_bytes());
        raw.extend_from_slice(&1u32.to_le_bytes());
        raw.extend_from_slice(&4u32.to_le_bytes());
        raw.extend_from_slice(b"name");
        raw.extend_from_slice(&8u32.to_le_bytes());
        raw.extend_from_slice(&2u32.to_le_bytes());
        raw.extend_from_slice(b"ok");
        let r = scan_bytes(&raw).unwrap();
        assert_eq!(r.metadata["name"], json!("ok"));
        let mut bad = b"GGUF".to_vec();
        bad.extend_from_slice(&9u32.to_le_bytes());
        assert!(scan_bytes(&bad).is_err());
        assert!(scan_bytes(b"GGU").is_err());
    }

    #[test]
    fn unknown_value_type_is_an_error() {
        let mut raw = header(1);
        raw.extend_from_slice(&gguf_string("k"));
        raw.extend_from_slice(&99u32.to_le_bytes());
        assert!(scan_bytes(&raw)
            .unwrap_err()
            .contains("unknown GGUF value type"));
    }

    #[test]
    fn past_deadline_reports_limit() {
        let mut raw = header(1);
        raw.extend_from_slice(&gguf_string("tokenizer.ggml.token_type"));
        raw.extend_from_slice(&9u32.to_le_bytes());
        raw.extend_from_slice(&0u32.to_le_bytes());
        raw.extend_from_slice(&10_000u64.to_le_bytes());
        raw.extend(std::iter::repeat(0u8).take(10_000));
        let r = scan_bytes_with_deadline(&raw, Instant::now()).unwrap();
        assert!(r.limit_exceeded.is_some());
    }
}
