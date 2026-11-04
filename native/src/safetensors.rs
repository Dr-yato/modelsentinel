//! Safetensors header reader. The tensor blob after the JSON header is not loaded.

use serde_json::{Map, Value};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub const MAX_HEADER: u64 = 100_000_000;
const MAX_TENSORS: usize = 1_000_000;
const KNOWN_DTYPES: &[&str] = &[
    "BOOL", "U8", "I8", "I16", "U16", "F16", "BF16", "I32", "U32", "F32", "F64", "I64", "U64",
    "F8_E4M3", "F8_E5M2", "F8_E8M0", "F4", "F6_E2M3", "F6_E3M2", "C64",
];

#[derive(Debug)]
pub struct SafeReport {
    pub header_length: u64,
    /// Bytes after the highest tensor end offset. Non-zero means something
    /// else (an archive, a pickle) is appended behind the tensors.
    pub trailing_bytes: u64,
    pub tensors: Vec<TensorInfo>,
    pub metadata: Map<String, Value>,
    pub warnings: Vec<String>,
}

#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_offsets: Vec<u64>,
}

pub fn scan_file(path: &Path) -> Result<SafeReport, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    scan_reader(&mut file)
}

pub fn scan_bytes(data: &[u8]) -> Result<SafeReport, String> {
    scan_reader(&mut std::io::Cursor::new(data))
}

pub fn scan_reader<R: Read + Seek>(reader: &mut R) -> Result<SafeReport, String> {
    let size = reader.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
    reader.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let mut len_buf = [0u8; 8];
    reader
        .read_exact(&mut len_buf)
        .map_err(|e| format!("safetensors header length: {e}"))?;
    let n = u64::from_le_bytes(len_buf);
    if n > MAX_HEADER {
        return Err(format!("safetensors header length {n} exceeds cap"));
    }
    if n < 2 {
        return Err("safetensors header is empty".into());
    }
    if n > size.saturating_sub(8) {
        return Err(format!(
            "safetensors header length {n} exceeds file size {size}"
        ));
    }
    let mut header = vec![0u8; n as usize];
    reader
        .read_exact(&mut header)
        .map_err(|e| format!("safetensors header: {e}"))?;
    if header.first() != Some(&b'{') {
        return Err("safetensors header is not a JSON object".into());
    }
    let value: Value =
        serde_json::from_slice(&header).map_err(|e| format!("safetensors JSON: {e}"))?;
    let obj = value
        .as_object()
        .ok_or("safetensors header is not an object")?
        .clone();
    let mut metadata = Map::new();
    let mut tensors = Vec::new();
    let mut warnings = Vec::new();
    if obj.len() > MAX_TENSORS {
        return Err(format!(
            "safetensors header lists more than {MAX_TENSORS} tensors"
        ));
    }
    for (name, entry) in obj {
        if name == "__metadata__" {
            if let Some(map) = entry.as_object() {
                if map.values().any(|v| !v.is_string()) {
                    warnings.push("__metadata__ has non-string values".into());
                }
                metadata = map.clone();
            } else {
                warnings.push("__metadata__ is not an object".into());
            }
            continue;
        }
        let dtype = entry
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let shape = entry
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_u64().or_else(|| x.as_i64().map(|i| i as u64)))
                    .collect()
            })
            .unwrap_or_default();
        let data_offsets = entry
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|x| x.as_u64()).collect())
            .unwrap_or_default();
        if dtype.is_empty() {
            warnings.push(format!("tensor {name} is missing dtype"));
        } else if !KNOWN_DTYPES.contains(&dtype.as_str()) {
            warnings.push(format!("tensor {name} has unknown dtype {dtype}"));
        }
        tensors.push(TensorInfo {
            name,
            dtype,
            shape,
            data_offsets,
        });
    }
    let data_len = size - 8 - n;
    let max_end = check_offsets(&tensors, data_len, &mut warnings);
    let trailing_bytes = data_len.saturating_sub(max_end);
    if trailing_bytes > 0 && !tensors.is_empty() {
        warnings.push(format!(
            "{trailing_bytes} bytes follow the last tensor; content may be appended to the file"
        ));
    }
    Ok(SafeReport {
        header_length: n,
        trailing_bytes,
        tensors,
        metadata,
        warnings,
    })
}

fn check_offsets(tensors: &[TensorInfo], data_len: u64, warnings: &mut Vec<String>) -> u64 {
    let mut max_end = 0u64;
    let mut bad = 0usize;
    for t in tensors {
        let ok = t.data_offsets.len() == 2
            && t.data_offsets[0] <= t.data_offsets[1]
            && t.data_offsets[1] <= data_len;
        if !ok {
            bad += 1;
            if bad <= 5 {
                warnings.push(format!(
                    "tensor {} has data_offsets {:?} outside the {data_len}-byte data section",
                    t.name, t.data_offsets
                ));
            }
            continue;
        }
        max_end = max_end.max(t.data_offsets[1]);
    }
    if bad > 5 {
        warnings.push(format!("{bad} tensors in total have invalid data_offsets"));
    }
    max_end
}

pub fn looks_like_safetensors(header: &[u8]) -> bool {
    if header.len() < 10 {
        return false;
    }
    let n = u64::from_le_bytes(header[0..8].try_into().unwrap());
    if n < 2 || n > MAX_HEADER {
        return false;
    }
    header.get(8) == Some(&b'{')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(header: &str, data: &[u8]) -> Vec<u8> {
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn clean_file_has_no_warnings() {
        let raw = build(
            r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"__metadata__":{"format":"pt"}}"#,
            &[0, 0, 128, 63],
        );
        let r = scan_bytes(&raw).unwrap();
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        assert_eq!(r.trailing_bytes, 0);
        assert_eq!(r.tensors.len(), 1);
    }

    #[test]
    fn appended_payload_is_reported() {
        let mut data = vec![0u8; 4];
        data.extend_from_slice(b"PK\x03\x04 appended archive");
        let raw = build(
            r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &data,
        );
        let r = scan_bytes(&raw).unwrap();
        assert!(r.trailing_bytes > 0);
        assert!(r
            .warnings
            .iter()
            .any(|w| w.contains("follow the last tensor")));
    }

    #[test]
    fn bad_offsets_and_dtypes_warn() {
        let raw = build(
            r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4000]},"x":{"dtype":"EVIL","shape":[],"data_offsets":[0,0]}}"#,
            &[0; 4],
        );
        let r = scan_bytes(&raw).unwrap();
        assert!(r.warnings.iter().any(|w| w.contains("outside")));
        assert!(r.warnings.iter().any(|w| w.contains("unknown dtype EVIL")));
    }

    #[test]
    fn header_longer_than_file_is_an_error() {
        let mut raw = 1000u64.to_le_bytes().to_vec();
        raw.extend_from_slice(b"{}");
        assert!(scan_bytes(&raw).unwrap_err().contains("exceeds file size"));
        assert!(scan_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00[]").is_err());
        assert!(scan_bytes(b"short").is_err());
    }
}
