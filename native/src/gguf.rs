//! GGUF metadata reader.
//!
//! Tensor payloads are not loaded. Keys that carry a chat template, identity,
//! or other short descriptive fields are kept in full. Large tokenizer tables
//! are consumed so later keys stay aligned, then discarded.

use serde_json::{json, Map, Value};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const MAX_STRING: u64 = 8_000_000;
const MAX_ARRAY: u64 = 5_000_000;

#[derive(Debug)]
pub struct GgufReport {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub metadata: Map<String, Value>,
    pub discarded_keys: Vec<String>,
    pub warnings: Vec<String>,
}

pub fn scan_file(path: &Path) -> Result<GgufReport, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    scan_reader(&mut file)
}

pub fn scan_file_from<R: Read + Seek>(reader: &mut R) -> Result<GgufReport, String> {
    scan_reader(reader)
}

pub fn scan_bytes(data: &[u8]) -> Result<GgufReport, String> {
    scan_reader(&mut std::io::Cursor::new(data))
}

fn scan_reader<R: Read + Seek>(reader: &mut R) -> Result<GgufReport, String> {
    reader.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let magic = read_exact(reader, 4)?;
    if &magic != b"GGUF" {
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
    let mut metadata = Map::new();
    let mut discarded_keys = Vec::new();
    let mut warnings = Vec::new();
    for _ in 0..metadata_count {
        let key = read_string(reader, version, &mut warnings)?;
        let vtype = read_u32(reader)?;
        let keep = keep_full(&key);
        match read_value(reader, vtype, version, keep, &mut warnings)? {
            Some(value) if keep => {
                metadata.insert(key, value);
            }
            _ => discarded_keys.push(key),
        }
    }
    Ok(GgufReport {
        version,
        tensor_count,
        metadata_count,
        metadata,
        discarded_keys,
        warnings,
    })
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
    version: u32,
    keep: bool,
    warnings: &mut Vec<String>,
) -> Result<Option<Value>, String> {
    match vtype {
        0 => Ok(Some(json!(read_u8(reader)?)).filter(|_| keep)),
        1 => Ok(Some(json!(read_u8(reader)? as i8)).filter(|_| keep)),
        2 => Ok(Some(json!(read_u16(reader)?)).filter(|_| keep)),
        3 => Ok(Some(json!(read_u16(reader)? as i16)).filter(|_| keep)),
        4 => Ok(Some(json!(read_u32(reader)?)).filter(|_| keep)),
        5 => Ok(Some(json!(read_u32(reader)? as i32)).filter(|_| keep)),
        6 => {
            let bits = read_u32(reader)?;
            Ok(Some(json!(f32::from_bits(bits))).filter(|_| keep))
        }
        7 => Ok(Some(json!(read_u8(reader)? != 0)).filter(|_| keep)),
        8 => {
            let s = read_string(reader, version, warnings)?;
            Ok(Some(Value::String(s)).filter(|_| keep))
        }
        9 => {
            let elem = read_u32(reader)?;
            let count = if version == 1 {
                read_u32(reader)? as u64
            } else {
                read_u64(reader)?
            };
            if count > MAX_ARRAY {
                return Err(format!("GGUF array length {count} is not plausible"));
            }
            if !keep {
                for _ in 0..count {
                    let _ = read_value(reader, elem, version, false, warnings)?;
                }
                return Ok(None);
            }
            let mut items = Vec::new();
            for idx in 0..count {
                let item = read_value(reader, elem, version, true, warnings)?;
                if idx < 32 {
                    if let Some(v) = item {
                        items.push(v);
                    }
                }
            }
            if count > 32 {
                warnings.push(format!(
                    "truncated stored array to 32 of {count} elements"
                ));
            }
            Ok(Some(Value::Array(items)))
        }
        10 => Ok(Some(json!(read_u64(reader)?)).filter(|_| keep)),
        11 => Ok(Some(json!(read_u64(reader)? as i64)).filter(|_| keep)),
        12 => {
            let bits = read_u64(reader)?;
            Ok(Some(json!(f64::from_bits(bits))).filter(|_| keep))
        }
        other => Err(format!("unknown GGUF value type {other}")),
    }
}

fn read_string<R: Read + Seek>(
    reader: &mut R,
    version: u32,
    warnings: &mut Vec<String>,
) -> Result<String, String> {
    let len = if version == 1 {
        read_u32(reader)? as u64
    } else {
        read_u64(reader)?
    };
    if len > MAX_STRING {
        reader
            .seek(SeekFrom::Current(len as i64))
            .map_err(|e| e.to_string())?;
        warnings.push(format!("skipped oversized GGUF string ({len} bytes)"));
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
        raw.extend_from_slice(&gguf_string("{% for m in messages %}{{ m.content }}{% endfor %}"));
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
}
