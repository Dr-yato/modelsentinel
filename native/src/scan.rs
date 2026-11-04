//! Container and magic-byte dispatcher.
//!
//! Format is decided from content. The filename extension is reported beside
//! that decision so a pickle renamed to `.bin`, `.pt`, or `.safetensors`
//! cannot hide (the PickleScan CVE-2025-10155 failure mode).
//!
//! Every top-level parse owns one `Budget` (time, inflated bytes, members,
//! depth). Limits that trip are listed in `limits_exceeded`; the scan still
//! returns everything it found up to that point.

use crate::archive::{self, ArchiveScan};
use crate::gguf::GgufReport;
use crate::limits::{Budget, MAX_DEPTH, MAX_MEMBER};
use crate::pickle::{self, PickleReport};
use crate::safetensors::SafeReport;
use crate::{keras, npy, onnx};
use flate2::read::GzDecoder;
use serde::Serialize;
use serde_json::{json, Value};
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;

/// HDF5 superblocks may follow a user block of 512, 1024, 2048 ... bytes.
const HDF5_OFFSETS: &[usize] = &[0, 512, 1024, 2048, 4096];
const HEADER_PROBE: usize = 8192;

#[derive(Debug, Serialize)]
pub struct ScanReport {
    pub format: String,
    pub magic_hex: String,
    pub size: u64,
    pub extension: String,
    pub extension_mismatch: bool,
    pub pickle: Option<Value>,
    pub gguf: Option<Value>,
    pub safetensors: Option<Value>,
    pub npy: Option<Value>,
    pub keras: Option<Value>,
    pub onnx: Option<Value>,
    pub nested: Vec<Value>,
    pub warnings: Vec<String>,
    /// Work limits that stopped part of the scan (size, time, members, depth).
    pub limits_exceeded: Vec<String>,
    pub error: Option<String>,
}

impl ScanReport {
    fn new(format: &str, header: &[u8], size: u64, extension: &str) -> Self {
        ScanReport {
            format: format.into(),
            magic_hex: hex_prefix(header),
            size,
            extension: extension.into(),
            extension_mismatch: false,
            pickle: None,
            gguf: None,
            safetensors: None,
            npy: None,
            keras: None,
            onnx: None,
            nested: Vec::new(),
            warnings: Vec::new(),
            limits_exceeded: Vec::new(),
            error: None,
        }
    }

    fn failed(extension: &str, size: u64, error: String) -> Self {
        let mut r = Self::new("unknown", &[], size, extension);
        r.error = Some(error);
        r
    }
}

pub fn scan_path(path: &Path) -> ScanReport {
    let extension = ext_of(&path.to_string_lossy());
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) => return ScanReport::failed(&extension, 0, format!("open failed: {e}")),
    };
    let size = file.metadata().map(|m| m.len()).unwrap_or(0);
    let mut budget = Budget::new();
    let report = scan_reader(&mut file, size, &extension, 0, &mut budget);
    finish(report, &extension, size, &mut budget)
}

/// Scan an in-memory buffer as if it were a file with `extension`.
pub fn scan_bytes(data: &[u8], extension: &str) -> ScanReport {
    let mut budget = Budget::new();
    let report = scan_bytes_inner(data, extension, 0, &mut budget);
    finish(report, extension, data.len() as u64, &mut budget)
}

fn finish(
    report: Result<ScanReport, String>,
    ext: &str,
    size: u64,
    budget: &mut Budget,
) -> ScanReport {
    let mut report = report.unwrap_or_else(|e| ScanReport::failed(ext, size, e));
    report.extension = ext.into();
    report.extension_mismatch = extension_mismatch(ext, &report.format);
    report.size = size;
    for msg in budget.take_exceeded() {
        if !report.limits_exceeded.contains(&msg) {
            report.limits_exceeded.push(msg);
        }
    }
    report
}

/// Scan one archive member (called by the archive walkers).
pub(crate) fn scan_member(data: &[u8], name: &str, depth: u32, budget: &mut Budget) -> ScanReport {
    let ext = ext_of(name);
    let mut r = scan_bytes_inner(data, &ext, depth, budget)
        .unwrap_or_else(|e| ScanReport::failed(&ext, data.len() as u64, e));
    r.extension = ext;
    r
}

pub fn scan_bytes_inner(
    data: &[u8],
    extension: &str,
    depth: u32,
    budget: &mut Budget,
) -> Result<ScanReport, String> {
    scan_reader(
        &mut Cursor::new(data),
        data.len() as u64,
        extension,
        depth,
        budget,
    )
}

fn scan_reader<R: Read + Seek>(
    r: &mut R,
    size: u64,
    ext: &str,
    depth: u32,
    budget: &mut Budget,
) -> Result<ScanReport, String> {
    let header = read_header(r, size)?;
    if depth > MAX_DEPTH {
        budget.note(&format!(
            "container nesting deeper than {MAX_DEPTH}; inner content was not scanned"
        ));
        return Ok(ScanReport::new("unknown", &header, size, ext));
    }
    if !budget.time_ok() {
        return Ok(ScanReport::new("unknown", &header, size, ext));
    }
    if header.starts_with(b"GGUF") {
        r.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
        let g = crate::gguf::scan_file_with_deadline(r, budget.deadline())?;
        if let Some(msg) = &g.limit_exceeded {
            budget.note(msg);
        }
        return Ok(report_from_gguf(&header, size, ext, g));
    }
    if header.starts_with(b"PK\x03\x04") || header.starts_with(b"PK\x05\x06") {
        let scan = archive::scan_zip(r, size, depth, budget)?;
        return Ok(report_from_archive(&header, size, ext, "zip", scan));
    }
    if header.starts_with(&[0x1f, 0x8b]) {
        return scan_gzip(r, &header, size, ext, depth, budget);
    }
    if archive::is_tar(&header) {
        let scan = archive::scan_tar(r, size, depth, budget)?;
        return Ok(report_from_archive(&header, size, ext, "tar", scan));
    }
    if header.starts_with(npy::MAGIC) {
        return scan_npy(r, &header, size, ext, budget);
    }
    if is_hdf5(&header) {
        let k = keras::scan_h5(r, size, budget.deadline())?;
        if let Some(msg) = k.get("limit_exceeded").and_then(Value::as_str) {
            budget.note(msg);
        }
        let format = if k["keras_model_config"] == true {
            "keras_h5"
        } else {
            "hdf5"
        };
        let mut report = ScanReport::new(format, &header, size, ext);
        report.warnings = string_list(&k["warnings"]);
        report.keras = Some(k);
        return Ok(report);
    }
    if header.len() >= 2 && header[0] == 0x80 && header[1] <= 5 {
        return scan_pickle(r, &header, size, ext, budget);
    }
    if crate::safetensors::looks_like_safetensors(&header) {
        r.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
        match crate::safetensors::scan_reader(r) {
            Ok(safe) => return Ok(report_from_safe(&header, size, ext, safe)),
            Err(e) => {
                let mut report = sniff_rest(r, &header, size, ext, budget)?;
                report
                    .warnings
                    .push(format!("safetensors probe failed: {e}"));
                return Ok(report);
            }
        }
    }
    if onnx::looks_like_onnx(&header) || ext == ".onnx" {
        r.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
        match onnx::scan_reader(r, Some(budget.deadline())) {
            Ok(o) => {
                if let Some(msg) = o.get("limit_exceeded").and_then(Value::as_str) {
                    budget.note(msg);
                }
                let mut report = ScanReport::new("onnx", &header, size, ext);
                report.warnings = string_list(&o["warnings"]);
                report.onnx = Some(o);
                return Ok(report);
            }
            Err(e) if ext == ".onnx" => {
                let mut report = sniff_rest(r, &header, size, ext, budget)?;
                report.warnings.push(format!("ONNX parse failed: {e}"));
                return Ok(report);
            }
            Err(_) => {}
        }
    }
    sniff_rest(r, &header, size, ext, budget)
}

fn read_header<R: Read + Seek>(r: &mut R, size: u64) -> Result<Vec<u8>, String> {
    r.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let mut header = Vec::with_capacity(HEADER_PROBE);
    r.by_ref()
        .take(size.min(HEADER_PROBE as u64))
        .read_to_end(&mut header)
        .map_err(|e| e.to_string())?;
    r.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    Ok(header)
}

fn read_limited<R: Read + Seek>(r: &mut R, from: u64, max: u64) -> Result<Vec<u8>, String> {
    r.seek(SeekFrom::Start(from)).map_err(|e| e.to_string())?;
    let mut data = Vec::new();
    r.by_ref()
        .take(max)
        .read_to_end(&mut data)
        .map_err(|e| e.to_string())?;
    Ok(data)
}

fn is_hdf5(header: &[u8]) -> bool {
    HDF5_OFFSETS
        .iter()
        .any(|&at| header.len() >= at + 8 && &header[at..at + 8] == keras::HDF5_MAGIC)
}

fn string_list(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

fn scan_pickle<R: Read + Seek>(
    r: &mut R,
    header: &[u8],
    size: u64,
    ext: &str,
    budget: &mut Budget,
) -> Result<ScanReport, String> {
    let data = read_limited(r, 0, MAX_MEMBER as u64)?;
    let mut report = report_from_pickle(&data, ext, "pickle", budget);
    if size > MAX_MEMBER as u64
        && report
            .pickle
            .as_ref()
            .map(|p| p["stopped"] != true)
            .unwrap_or(true)
    {
        budget.note(&format!(
            "pickle stream is larger than {MAX_MEMBER} bytes and did not end inside the scanned prefix"
        ));
    }
    report.magic_hex = hex_prefix(header);
    report.size = size;
    Ok(report)
}

fn scan_gzip<R: Read + Seek>(
    r: &mut R,
    header: &[u8],
    size: u64,
    ext: &str,
    depth: u32,
    budget: &mut Budget,
) -> Result<ScanReport, String> {
    r.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let cap = (MAX_MEMBER as u64).min(budget.inflate_left());
    let mut out = Vec::new();
    let read = GzDecoder::new(r.by_ref())
        .take(cap + 1)
        .read_to_end(&mut out);
    if out.is_empty() {
        return match read {
            Ok(_) => Ok(ScanReport::new("gzip", header, size, ext)),
            Err(e) => Err(format!("gzip: {e}")),
        };
    }
    if out.len() as u64 > cap {
        out.truncate(cap as usize);
        budget.note(&format!(
            "gzip body inflates past {cap} bytes; only the first {cap} were scanned (possible decompression bomb)"
        ));
    }
    budget.charge_inflate(out.len() as u64);
    let mut report = scan_bytes_inner(&out, "", depth + 1, budget)?;
    report.format = format!("gzip+{}", report.format);
    if let Err(e) = read {
        report.warnings.push(format!("gzip: {e}"));
    }
    report.magic_hex = hex_prefix(header);
    report.size = size;
    report.extension = ext.into();
    Ok(report)
}

fn scan_npy<R: Read + Seek>(
    r: &mut R,
    header: &[u8],
    size: u64,
    ext: &str,
    budget: &mut Budget,
) -> Result<ScanReport, String> {
    let span = npy::header_span(header).ok_or("truncated npy")?;
    let head = read_limited(r, 0, span as u64)?;
    let parsed = npy::parse_header(&head)?;
    let format = if parsed.object_dtype {
        "npy+pickle"
    } else {
        "npy"
    };
    let mut report = ScanReport::new(format, header, size, ext);
    report.npy = Some(json!({
        "version": format!("{}.{}", parsed.major, parsed.minor),
        "header": parsed.header.chars().take(500).collect::<String>(),
        "object_dtype": parsed.object_dtype,
    }));
    if parsed.object_dtype {
        report.warnings.push(
            "NumPy object array detected; object arrays embed pickle and are treated as untrusted"
                .into(),
        );
        let payload = read_limited(r, parsed.payload_offset as u64, MAX_MEMBER as u64)?;
        let total = size.saturating_sub(parsed.payload_offset as u64);
        if total > MAX_MEMBER as u64 {
            budget.note(&format!(
                "npy object payload larger than {MAX_MEMBER} bytes; only a prefix was walked"
            ));
        }
        let mut child = report_from_pickle(&payload, ".pkl", "pickle", budget);
        child.extension = ".pkl".into();
        report
            .nested
            .push(json!({"member": "npy-payload", "report": child}));
    }
    Ok(report)
}

fn sniff_rest<R: Read + Seek>(
    r: &mut R,
    header: &[u8],
    size: u64,
    ext: &str,
    budget: &mut Budget,
) -> Result<ScanReport, String> {
    if size > MAX_MEMBER as u64 {
        let mut report = ScanReport::new("unknown", header, size, ext);
        report
            .warnings
            .push("large file has no recognized model magic; weights were not loaded".into());
        return Ok(report);
    }
    let data = read_limited(r, 0, MAX_MEMBER as u64)?;
    let trimmed = data
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(data.len());
    if data.get(trimmed) == Some(&b'{') {
        if let Some(k) = keras::analyze_config_bytes(&data) {
            let mut report = ScanReport::new("keras_config", header, size, ext);
            report.keras = Some(k);
            return Ok(report);
        }
    }
    // Extension is not consulted. A file that still walks as pickle counts.
    let walked = pickle::walk_with_deadline(&data, budget.deadline());
    if walked.stopped && !walked.globals.is_empty() {
        return Ok(report_from_pickle(&data, ext, "pickle", budget));
    }
    Ok(ScanReport::new("unknown", header, size, ext))
}

fn report_from_pickle(data: &[u8], ext: &str, format: &str, budget: &mut Budget) -> ScanReport {
    let walked = pickle::walk_with_deadline(data, budget.deadline());
    if let Some(msg) = &walked.limit_exceeded {
        budget.note(msg);
    }
    let mut report = ScanReport::new(format, data, data.len() as u64, ext);
    report.warnings = walked.warnings.clone();
    report.pickle = Some(pickle_to_json(&walked));
    report
}

fn pickle_to_json(report: &PickleReport) -> Value {
    json!({
        "protocol": report.protocol,
        "globals": report.globals,
        "reduces": report.reduces,
        "extensions": report.extensions,
        "persistent_ids": report.persistent_ids,
        "unknown_opcodes": report.unknown_opcodes,
        "warnings": report.warnings,
        "stopped": report.stopped,
        "bytes_consumed": report.bytes_consumed,
        "pickle_count": report.pickle_count,
        "trailing_bytes": report.trailing_bytes,
        "limit_exceeded": report.limit_exceeded,
    })
}

fn report_from_archive(
    header: &[u8],
    size: u64,
    ext: &str,
    kind: &str,
    scan: ArchiveScan,
) -> ScanReport {
    let has = |name: &str| scan.member_names.iter().any(|m| m == name);
    let all_npy = !scan.member_names.is_empty()
        && scan
            .member_names
            .iter()
            .all(|m| m.to_ascii_lowercase().ends_with(".npy"));
    let base =
        if kind == "zip" && has("config.json") && (has("metadata.json") || has("model.weights.h5"))
        {
            "keras"
        } else if kind == "zip" && (ext == ".npz" || all_npy) {
            "npz"
        } else {
            kind
        };
    let pickled = scan.nested.iter().any(|n| {
        n.pointer("/report/format")
            .and_then(Value::as_str)
            .map(|f| f.contains("pickle"))
            .unwrap_or(false)
    });
    let format = if pickled {
        format!("{base}+pickle")
    } else {
        base.to_string()
    };
    let mut report = ScanReport::new(&format, header, size, ext);
    report.nested = scan.nested;
    report.warnings = scan.warnings;
    report
}

fn report_from_gguf(header: &[u8], size: u64, ext: &str, gguf: GgufReport) -> ScanReport {
    let mut report = ScanReport::new("gguf", header, size, ext);
    report.gguf = Some(json!({
        "version": gguf.version,
        "tensor_count": gguf.tensor_count,
        "metadata_count": gguf.metadata_count,
        "metadata": Value::Object(gguf.metadata),
        "discarded_keys": gguf.discarded_keys,
        "warnings": gguf.warnings,
    }));
    report.warnings = gguf.warnings;
    report
}

fn report_from_safe(header: &[u8], size: u64, ext: &str, safe: SafeReport) -> ScanReport {
    let tensors: Vec<Value> = safe
        .tensors
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "dtype": t.dtype,
                "shape": t.shape,
                "data_offsets": t.data_offsets,
            })
        })
        .collect();
    let mut report = ScanReport::new("safetensors", header, size, ext);
    report.safetensors = Some(json!({
        "header_length": safe.header_length,
        "trailing_bytes": safe.trailing_bytes,
        "tensor_count": safe.tensors.len(),
        "tensors": tensors,
        "metadata": Value::Object(safe.metadata),
        "warnings": safe.warnings,
    }));
    report.warnings = safe.warnings;
    report
}

fn extension_mismatch(extension: &str, format: &str) -> bool {
    let ext = extension.to_ascii_lowercase();
    let fmt = format.to_ascii_lowercase();
    match ext.as_str() {
        ".safetensors" => !fmt.contains("safetensors"),
        ".gguf" => !fmt.contains("gguf"),
        // np.load falls back to pickle.load for non-npy content when allow_pickle is set.
        ".npy" => !fmt.starts_with("npy"),
        ".onnx" => fmt != "onnx",
        ".h5" | ".hdf5" => !fmt.contains("hdf5") && !fmt.contains("keras_h5"),
        ".json" | ".md" | ".txt" => fmt.contains("pickle") || fmt.contains("zip"),
        _ => false,
    }
}

fn ext_of(name: &str) -> String {
    Path::new(name)
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_ascii_lowercase()))
        .unwrap_or_default()
}

fn hex_prefix(data: &[u8]) -> String {
    data.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
#[path = "scan_tests.rs"]
mod tests;
