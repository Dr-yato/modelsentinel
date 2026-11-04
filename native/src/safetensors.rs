//! Safetensors header reader. The tensor blob after the JSON header is not loaded.

use serde_json::{Map, Value};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub const MAX_HEADER: u64 = 100_000_000;

#[derive(Debug)]
pub struct SafeReport {
    pub header_length: u64,
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
    let mut header = vec![0u8; n as usize];
    reader
        .read_exact(&mut header)
        .map_err(|e| format!("safetensors header: {e}"))?;
    if header.first() != Some(&b'{') {
        return Err("safetensors header is not a JSON object".into());
    }
    let value: Value = serde_json::from_slice(&header).map_err(|e| format!("safetensors JSON: {e}"))?;
    let obj = value
        .as_object()
        .ok_or("safetensors header is not an object")?
        .clone();
    let mut metadata = Map::new();
    let mut tensors = Vec::new();
    let mut warnings = Vec::new();
    for (name, entry) in obj {
        if name == "__metadata__" {
            if let Some(map) = entry.as_object() {
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
        }
        tensors.push(TensorInfo {
            name,
            dtype,
            shape,
            data_offsets,
        });
    }
    Ok(SafeReport {
        header_length: n,
        tensors,
        metadata,
        warnings,
    })
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
