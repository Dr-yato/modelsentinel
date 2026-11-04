//! Container and magic-byte dispatcher.
//!
//! Format is decided from content. The filename extension is reported beside
//! that decision so a pickle renamed to `.bin`, `.pt`, or `.safetensors`
//! cannot hide (the PickleScan CVE-2025-10155 failure mode).

use crate::gguf::GgufReport;
use crate::pickle::{self, PickleReport};
use crate::safetensors::SafeReport;
use flate2::read::{DeflateDecoder, GzDecoder};
use serde::Serialize;
use serde_json::{json, Value};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const MAX_MEMBER: usize = 32 * 1024 * 1024;
const MAX_DEPTH: u32 = 4;

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
    pub nested: Vec<Value>,
    pub warnings: Vec<String>,
    pub error: Option<String>,
}

pub fn scan_path(path: &Path) -> ScanReport {
    let extension = path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_ascii_lowercase()))
        .unwrap_or_default();
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            return empty_report(&extension, 0, Some(format!("open failed: {e}")));
        }
    };
    let size = file.metadata().map(|m| m.len()).unwrap_or(0);
    match scan_file(file, size, &extension, 0) {
        Ok(mut report) => {
            report.extension = extension.clone();
            report.extension_mismatch = extension_mismatch(&extension, &report.format);
            report.size = size;
            report
        }
        Err(e) => empty_report(&extension, size, Some(e)),
    }
}

fn scan_file(mut file: File, size: u64, extension: &str, depth: u32) -> Result<ScanReport, String> {
    let header_len = size.min(8192) as usize;
    let mut header = vec![0u8; header_len];
    let n = file.read(&mut header).map_err(|e| e.to_string())?;
    header.truncate(n);
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;

    if header.starts_with(b"GGUF") {
        let gguf = crate::gguf::scan_file_from(&mut file)?;
        return Ok(report_from_gguf(&header, size, extension, gguf));
    }
    if header.starts_with(b"PK\x03\x04") || header.starts_with(b"PK\x05\x06") {
        return scan_zip(file, size, extension, depth);
    }
    if header.starts_with(&[0x1f, 0x8b]) {
        return scan_gzip(file, extension, depth);
    }
    if is_tar(&header) {
        let mut data = Vec::new();
        file.read_to_end(&mut data).map_err(|e| e.to_string())?;
        return Ok(scan_tar_bytes(&data, extension, depth));
    }
    if header.starts_with(b"\x93NUMPY") {
        let mut data = Vec::new();
        // NumPy headers are small; refuse to slurp multi-gigabyte arrays.
        let cap = size.min(MAX_MEMBER as u64) as usize;
        data.resize(cap, 0);
        let n = file.read(&mut data).map_err(|e| e.to_string())?;
        data.truncate(n);
        return Ok(scan_npy_bytes(&data, extension, depth));
    }
    if header.len() >= 2 && header[0] == 0x80 && header[1] <= 5 {
        let data = read_limited(file, MAX_MEMBER)?;
        return Ok(report_from_pickle(&data, extension, "pickle"));
    }
    if crate::safetensors::looks_like_safetensors(&header) {
        match crate::safetensors::scan_reader(&mut file) {
            Ok(safe) => return Ok(report_from_safe(&header, size, extension, safe)),
            Err(e) => {
                let mut report = sniff_rest(file, size, extension, depth)?;
                report.warnings.push(format!("safetensors probe failed: {e}"));
                return Ok(report);
            }
        }
    }
    sniff_rest(file, size, extension, depth)
}

fn sniff_rest(mut file: File, size: u64, extension: &str, depth: u32) -> Result<ScanReport, String> {
    if size > MAX_MEMBER as u64 {
        let mut header = vec![0u8; 16];
        let _ = file.read(&mut header);
        return Ok(ScanReport {
            format: "unknown".into(),
            magic_hex: hex_prefix(&header),
            size,
            extension: extension.into(),
            extension_mismatch: false,
            pickle: None,
            gguf: None,
            safetensors: None,
            nested: Vec::new(),
            warnings: vec!["large file has no recognized model magic; weights were not loaded".into()],
            error: None,
        });
    }
    let data = read_limited(file, MAX_MEMBER)?;
    Ok(scan_bytes_inner(&data, extension, depth))
}

pub fn scan_bytes_inner(data: &[u8], extension: &str, depth: u32) -> ScanReport {
    if depth > MAX_DEPTH {
        return empty_report(extension, data.len() as u64, Some("nesting limit reached".into()));
    }
    if data.starts_with(b"GGUF") {
        return match crate::gguf::scan_bytes(data) {
            Ok(g) => report_from_gguf(data, data.len() as u64, extension, g),
            Err(e) => empty_report(extension, data.len() as u64, Some(e)),
        };
    }
    if data.starts_with(b"PK\x03\x04") || data.starts_with(b"PK\x05\x06") {
        return scan_zip_bytes(data, extension, depth);
    }
    if data.starts_with(&[0x1f, 0x8b]) {
        let dec = GzDecoder::new(data);
        let mut limited = dec.take(MAX_MEMBER as u64);
        let mut out = Vec::new();
        if let Err(e) = limited.read_to_end(&mut out) {
            return empty_report(extension, data.len() as u64, Some(format!("gzip: {e}")));
        }
        let mut inner = scan_bytes_inner(&out, "", depth + 1);
        inner.format = format!("gzip+{}", inner.format);
        inner.magic_hex = hex_prefix(data);
        inner.size = data.len() as u64;
        inner.extension = extension.into();
        return inner;
    }
    if is_tar(data) {
        return scan_tar_bytes(data, extension, depth);
    }
    if data.starts_with(b"\x93NUMPY") {
        return scan_npy_bytes(data, extension, depth);
    }
    if data.len() >= 2 && data[0] == 0x80 && data[1] <= 5 {
        return report_from_pickle(data, extension, "pickle");
    }
    if crate::safetensors::looks_like_safetensors(data) {
        if let Ok(safe) = crate::safetensors::scan_bytes(data) {
            return report_from_safe(data, data.len() as u64, extension, safe);
        }
    }
    // Extension is not consulted. A file that still walks as pickle counts.
    let walked = pickle::walk(data);
    if walked.stopped && !walked.globals.is_empty() {
        return report_from_pickle(data, extension, "pickle");
    }
    ScanReport {
        format: "unknown".into(),
        magic_hex: hex_prefix(data),
        size: data.len() as u64,
        extension: extension.into(),
        extension_mismatch: false,
        pickle: None,
        gguf: None,
        safetensors: None,
        nested: Vec::new(),
        warnings: Vec::new(),
        error: None,
    }
}

fn report_from_pickle(data: &[u8], extension: &str, format: &str) -> ScanReport {
    let walked = pickle::walk(data);
    let value = pickle_to_json(&walked);
    ScanReport {
        format: format.into(),
        magic_hex: hex_prefix(data),
        size: data.len() as u64,
        extension: extension.into(),
        extension_mismatch: false,
        pickle: Some(value),
        gguf: None,
        safetensors: None,
        nested: Vec::new(),
        warnings: walked.warnings.clone(),
        error: None,
    }
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
    })
}

fn report_from_gguf(header: &[u8], size: u64, extension: &str, gguf: GgufReport) -> ScanReport {
    let metadata = Value::Object(gguf.metadata.clone());
    ScanReport {
        format: "gguf".into(),
        magic_hex: hex_prefix(header),
        size,
        extension: extension.into(),
        extension_mismatch: false,
        pickle: None,
        gguf: Some(json!({
            "version": gguf.version,
            "tensor_count": gguf.tensor_count,
            "metadata_count": gguf.metadata_count,
            "metadata": metadata,
            "discarded_keys": gguf.discarded_keys,
            "warnings": gguf.warnings,
        })),
        safetensors: None,
        nested: Vec::new(),
        warnings: gguf.warnings,
        error: None,
    }
}

fn report_from_safe(header: &[u8], size: u64, extension: &str, safe: SafeReport) -> ScanReport {
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
    ScanReport {
        format: "safetensors".into(),
        magic_hex: hex_prefix(header),
        size,
        extension: extension.into(),
        extension_mismatch: false,
        pickle: None,
        gguf: None,
        safetensors: Some(json!({
            "header_length": safe.header_length,
            "tensor_count": safe.tensors.len(),
            "tensors": tensors,
            "metadata": Value::Object(safe.metadata),
            "warnings": safe.warnings,
        })),
        nested: Vec::new(),
        warnings: safe.warnings,
        error: None,
    }
}

fn scan_gzip(file: File, extension: &str, depth: u32) -> Result<ScanReport, String> {
    let mut compressed = Vec::new();
    file.take(MAX_MEMBER as u64)
        .read_to_end(&mut compressed)
        .map_err(|e| e.to_string())?;
    Ok(scan_bytes_inner(&compressed, extension, depth))
}

fn scan_zip(file: File, size: u64, extension: &str, depth: u32) -> Result<ScanReport, String> {
    // Own the bytes only when the archive is modest. Large pytorch zips are
    // read from the central directory without loading tensor members.
    if size <= MAX_MEMBER as u64 {
        let mut data = Vec::new();
        let mut file = file;
        file.read_to_end(&mut data).map_err(|e| e.to_string())?;
        return Ok(scan_zip_bytes(&data, extension, depth));
    }
    scan_zip_file(file, size, extension, depth)
}

fn scan_zip_bytes(data: &[u8], extension: &str, depth: u32) -> ScanReport {
    let mut warnings = Vec::new();
    let mut nested = Vec::new();
    match zip_members(data, &mut warnings) {
        Ok(members) => {
            for member in members {
                if member.data.len() > MAX_MEMBER {
                    warnings.push(format!("skipped large zip member {}", member.name));
                    continue;
                }
                if !should_scan_member(&member.name, &member.data) {
                    continue;
                }
                let mut child = scan_bytes_inner(&member.data, &ext_of(&member.name), depth + 1);
                child.extension = ext_of(&member.name);
                nested.push(json!({
                    "member": member.name,
                    "compressed_size": member.compressed_size,
                    "uncompressed_size": member.data.len(),
                    "report": child,
                }));
            }
        }
        Err(e) => warnings.push(e),
    }
    let format = if nested.iter().any(|n| {
        n.get("report")
            .and_then(|r| r.get("format"))
            .and_then(|f| f.as_str())
            .map(|f| f.contains("pickle"))
            .unwrap_or(false)
    }) {
        "zip+pickle"
    } else {
        "zip"
    };
    ScanReport {
        format: format.into(),
        magic_hex: hex_prefix(data),
        size: data.len() as u64,
        extension: extension.into(),
        extension_mismatch: false,
        pickle: None,
        gguf: None,
        safetensors: None,
        nested,
        warnings,
        error: None,
    }
}

fn scan_zip_file(mut file: File, size: u64, extension: &str, depth: u32) -> Result<ScanReport, String> {
    let tail_len = size.min(128 * 1024) as usize;
    file.seek(SeekFrom::End(-(tail_len as i64)))
        .map_err(|e| e.to_string())?;
    let mut tail = vec![0u8; tail_len];
    file.read_exact(&mut tail).map_err(|e| e.to_string())?;
    let Some(rel) = find_eocd(&tail) else {
        return Err("zip end-of-central-directory not found".into());
    };
    let eocd = &tail[rel..];
    if eocd.len() < 22 {
        return Err("truncated zip EOCD".into());
    }
    let total_entries = u16::from_le_bytes([eocd[10], eocd[11]]) as u64;
    let cd_size = u32::from_le_bytes([eocd[12], eocd[13], eocd[14], eocd[15]]) as u64;
    let cd_offset = u32::from_le_bytes([eocd[16], eocd[17], eocd[18], eocd[19]]) as u64;
    if cd_offset == 0xFFFF_FFFF || total_entries == 0xFFFF {
        return Err("zip64 archives larger than 4GiB are not scanned member-wise; refusing to load the archive into memory".into());
    }
    file.seek(SeekFrom::Start(cd_offset))
        .map_err(|e| e.to_string())?;
    let mut cd = vec![0u8; cd_size as usize];
    file.read_exact(&mut cd).map_err(|e| e.to_string())?;
    let entries = parse_cd(&cd).map_err(|e| e)?;
    let mut nested = Vec::new();
    let mut warnings = Vec::new();
    for entry in entries {
        if entry.uncompressed_size > MAX_MEMBER as u64 {
            warnings.push(format!("skipped large zip member {}", entry.name));
            continue;
        }
        if entry.method != 0 && entry.method != 8 {
            warnings.push(format!(
                "skipped zip member {} with compression method {}",
                entry.name, entry.method
            ));
            continue;
        }
        let data = match read_zip_member(&mut file, &entry) {
            Ok(d) => d,
            Err(e) => {
                warnings.push(format!("member {}: {e}", entry.name));
                continue;
            }
        };
        if !should_scan_member(&entry.name, &data) {
            continue;
        }
        let child = scan_bytes_inner(&data, &ext_of(&entry.name), depth + 1);
        nested.push(json!({
            "member": entry.name,
            "compressed_size": entry.compressed_size,
            "uncompressed_size": data.len(),
            "report": child,
        }));
    }
    Ok(ScanReport {
        format: "zip".into(),
        magic_hex: "504b0304".into(),
        size,
        extension: extension.into(),
        extension_mismatch: false,
        pickle: None,
        gguf: None,
        safetensors: None,
        nested,
        warnings,
        error: None,
    })
}

struct ZipEntry {
    name: String,
    method: u16,
    compressed_size: u64,
    uncompressed_size: u64,
    local_offset: u64,
}

fn read_zip_member(file: &mut File, entry: &ZipEntry) -> Result<Vec<u8>, String> {
    file.seek(SeekFrom::Start(entry.local_offset))
        .map_err(|e| e.to_string())?;
    let mut local = [0u8; 30];
    file.read_exact(&mut local).map_err(|e| e.to_string())?;
    if &local[0..4] != b"PK\x03\x04" {
        return Err("bad local header".into());
    }
    let name_len = u16::from_le_bytes([local[26], local[27]]) as u64;
    let extra_len = u16::from_le_bytes([local[28], local[29]]) as u64;
    file.seek(SeekFrom::Current((name_len + extra_len) as i64))
        .map_err(|e| e.to_string())?;
    let mut compressed = vec![0u8; entry.compressed_size as usize];
    file.read_exact(&mut compressed).map_err(|e| e.to_string())?;
    if entry.method == 0 {
        return Ok(compressed);
    }
    let dec = DeflateDecoder::new(&compressed[..]);
    let mut out = Vec::new();
    dec.take(MAX_MEMBER as u64)
        .read_to_end(&mut out)
        .map_err(|e| format!("deflate: {e}"))?;
    Ok(out)
}

fn parse_cd(cd: &[u8]) -> Result<Vec<ZipEntry>, String> {
    let mut i = 0;
    let mut entries = Vec::new();
    while i + 46 <= cd.len() {
        if &cd[i..i + 4] != b"PK\x01\x02" {
            break;
        }
        let method = u16::from_le_bytes([cd[i + 10], cd[i + 11]]);
        let compressed_size = u32::from_le_bytes([cd[i + 20], cd[i + 21], cd[i + 22], cd[i + 23]]) as u64;
        let uncompressed_size = u32::from_le_bytes([cd[i + 24], cd[i + 25], cd[i + 26], cd[i + 27]]) as u64;
        let name_len = u16::from_le_bytes([cd[i + 28], cd[i + 29]]) as usize;
        let extra_len = u16::from_le_bytes([cd[i + 30], cd[i + 31]]) as usize;
        let comment_len = u16::from_le_bytes([cd[i + 32], cd[i + 33]]) as usize;
        let local_offset = u32::from_le_bytes([cd[i + 42], cd[i + 43], cd[i + 44], cd[i + 45]]) as u64;
        let name_start = i + 46;
        let name_end = name_start + name_len;
        if name_end + extra_len + comment_len > cd.len() {
            return Err("truncated central directory".into());
        }
        let name = String::from_utf8_lossy(&cd[name_start..name_end]).into_owned();
        let extra = &cd[name_end..name_end + extra_len];
        let (compressed_size, uncompressed_size, local_offset) =
            apply_zip64_extra(extra, compressed_size, uncompressed_size, local_offset);
        entries.push(ZipEntry {
            name,
            method,
            compressed_size,
            uncompressed_size,
            local_offset,
        });
        i = name_end + extra_len + comment_len;
        if entries.len() > 10_000 {
            break;
        }
    }
    Ok(entries)
}

fn apply_zip64_extra(extra: &[u8], comp: u64, uncomp: u64, offset: u64) -> (u64, u64, u64) {
    let mut comp = comp;
    let mut uncomp = uncomp;
    let mut offset = offset;
    let mut i = 0;
    while i + 4 <= extra.len() {
        let id = u16::from_le_bytes([extra[i], extra[i + 1]]);
        let size = u16::from_le_bytes([extra[i + 2], extra[i + 3]]) as usize;
        i += 4;
        if i + size > extra.len() {
            break;
        }
        if id == 0x0001 {
            let mut p = i;
            let mut take = || {
                if p + 8 > i + size {
                    return None;
                }
                let v = u64::from_le_bytes(extra[p..p + 8].try_into().ok()?);
                p += 8;
                Some(v)
            };
            if uncomp == 0xFFFF_FFFF {
                if let Some(v) = take() {
                    uncomp = v;
                }
            }
            if comp == 0xFFFF_FFFF {
                if let Some(v) = take() {
                    comp = v;
                }
            }
            if offset == 0xFFFF_FFFF {
                if let Some(v) = take() {
                    offset = v;
                }
            }
        }
        i += size;
    }
    (comp, uncomp, offset)
}

struct MemMember {
    name: String,
    compressed_size: usize,
    data: Vec<u8>,
}

fn zip_members(data: &[u8], warnings: &mut Vec<String>) -> Result<Vec<MemMember>, String> {
    let Some(eocd_at) = find_eocd(data) else {
        return Err("zip end-of-central-directory not found".into());
    };
    if eocd_at + 22 > data.len() {
        return Err("truncated EOCD".into());
    }
    let cd_size = u32::from_le_bytes(data[eocd_at + 12..eocd_at + 16].try_into().unwrap()) as usize;
    let cd_offset = u32::from_le_bytes(data[eocd_at + 16..eocd_at + 20].try_into().unwrap()) as usize;
    if cd_offset.saturating_add(cd_size) > data.len() {
        return Err("central directory out of range".into());
    }
    let entries = parse_cd(&data[cd_offset..cd_offset + cd_size])?;
    let mut members = Vec::new();
    for entry in entries {
        if entry.local_offset as usize + 30 > data.len() {
            warnings.push(format!("bad local offset for {}", entry.name));
            continue;
        }
        let local = entry.local_offset as usize;
        if &data[local..local + 4] != b"PK\x03\x04" {
            warnings.push(format!("bad local header for {}", entry.name));
            continue;
        }
        let name_len = u16::from_le_bytes([data[local + 26], data[local + 27]]) as usize;
        let extra_len = u16::from_le_bytes([data[local + 28], data[local + 29]]) as usize;
        let start = local + 30 + name_len + extra_len;
        let end = start.saturating_add(entry.compressed_size as usize);
        if end > data.len() {
            warnings.push(format!("member {} exceeds archive", entry.name));
            continue;
        }
        let compressed = &data[start..end];
        let raw = if entry.method == 0 {
            compressed.to_vec()
        } else if entry.method == 8 {
            let dec = DeflateDecoder::new(compressed);
            let mut out = Vec::new();
            if let Err(e) = dec.take(MAX_MEMBER as u64).read_to_end(&mut out) {
                warnings.push(format!("deflate {}: {e}", entry.name));
                continue;
            }
            out
        } else {
            warnings.push(format!(
                "skipped {} compression method {}",
                entry.name, entry.method
            ));
            continue;
        };
        members.push(MemMember {
            name: entry.name,
            compressed_size: compressed.len(),
            data: raw,
        });
    }
    Ok(members)
}

fn find_eocd(data: &[u8]) -> Option<usize> {
    if data.len() < 22 {
        return None;
    }
    let start = data.len().saturating_sub(128 * 1024);
    let mut i = data.len() - 22;
    loop {
        if &data[i..i + 4] == b"PK\x05\x06" {
            return Some(i);
        }
        if i == start {
            break;
        }
        i -= 1;
    }
    None
}

fn should_scan_member(name: &str, data: &[u8]) -> bool {
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".pkl")
        || lower.ends_with(".pickle")
        || lower.contains("data.pkl")
        || lower.ends_with(".pt")
        || lower.ends_with(".pth")
        || lower.ends_with(".bin")
        || lower.ends_with(".json")
        || lower.ends_with(".jinja")
        || lower.ends_with(".txt")
        || lower.ends_with(".py")
    {
        return data.len() <= MAX_MEMBER;
    }
    if data.len() > 1_000_000 {
        return false;
    }
    data.starts_with(b"\x80")
        || data.starts_with(b"GGUF")
        || data.starts_with(b"PK")
        || crate::safetensors::looks_like_safetensors(data)
        || data.starts_with(b"{")
        || data.starts_with(b"\x93NUMPY")
}

fn scan_tar_bytes(data: &[u8], extension: &str, depth: u32) -> ScanReport {
    let mut nested = Vec::new();
    let mut warnings = Vec::new();
    let mut offset = 0;
    while offset + 512 <= data.len() {
        let header = &data[offset..offset + 512];
        if header.iter().all(|b| *b == 0) {
            break;
        }
        let name_raw = &header[..100];
        let name_len = name_raw.iter().position(|b| *b == 0).unwrap_or(100);
        let name = String::from_utf8_lossy(&name_raw[..name_len]).into_owned();
        let size = parse_octal(&header[124..136]).unwrap_or(0);
        let typeflag = header[156];
        let prefix_raw = if header.len() >= 500 { &header[345..500] } else { &[] };
        let prefix_len = prefix_raw.iter().position(|b| *b == 0).unwrap_or(prefix_raw.len());
        let prefix = String::from_utf8_lossy(&prefix_raw[..prefix_len]).into_owned();
        let full_name = if prefix.is_empty() {
            name
        } else {
            format!("{prefix}/{name}")
        };
        offset += 512;
        if size > MAX_MEMBER as u64 {
            warnings.push(format!("skipped large tar member {full_name}"));
            offset += padded(size);
            continue;
        }
        let end = offset + size as usize;
        if end > data.len() {
            warnings.push(format!("truncated tar member {full_name}"));
            break;
        }
        let payload = &data[offset..end];
        if typeflag == b'0' || typeflag == 0 {
            if should_scan_member(&full_name, payload) || payload.starts_with(b"\x80") {
                let child = scan_bytes_inner(payload, &ext_of(&full_name), depth + 1);
                nested.push(json!({
                    "member": full_name,
                    "uncompressed_size": payload.len(),
                    "report": child,
                }));
            }
        }
        offset += padded(size);
    }
    ScanReport {
        format: "tar".into(),
        magic_hex: hex_prefix(data),
        size: data.len() as u64,
        extension: extension.into(),
        extension_mismatch: false,
        pickle: None,
        gguf: None,
        safetensors: None,
        nested,
        warnings,
        error: None,
    }
}

fn padded(size: u64) -> usize {
    let rem = (size % 512) as usize;
    size as usize + if rem == 0 { 0 } else { 512 - rem }
}

fn parse_octal(bytes: &[u8]) -> Option<u64> {
    let s = std::str::from_utf8(bytes).ok()?.trim_matches(|c: char| c == '\0' || c.is_whitespace());
    u64::from_str_radix(s, 8).ok()
}

fn is_tar(data: &[u8]) -> bool {
    data.len() > 262 && &data[257..262] == b"ustar"
}

fn scan_npy_bytes(data: &[u8], extension: &str, depth: u32) -> ScanReport {
    let mut warnings = Vec::new();
    if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
        return empty_report(extension, data.len() as u64, Some("not an npy file".into()));
    }
    let major = data[6];
    let header_len: usize = if major == 1 {
        if data.len() < 10 {
            return empty_report(extension, data.len() as u64, Some("truncated npy".into()));
        }
        u16::from_le_bytes([data[8], data[9]]) as usize
    } else {
        if data.len() < 12 {
            return empty_report(extension, data.len() as u64, Some("truncated npy".into()));
        }
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
    };
    let header_start = if major == 1 { 10 } else { 12 };
    let header_end = header_start + header_len;
    if header_end > data.len() {
        return empty_report(extension, data.len() as u64, Some("truncated npy header".into()));
    }
    let header = String::from_utf8_lossy(&data[header_start..header_end]).to_string();
    let object_array = header.contains("'O'") || header.contains("\"O\"") || header.contains("|O");
    if object_array {
        warnings.push(
            "NumPy object array detected; object arrays embed pickle and are treated as untrusted".into(),
        );
    }
    let mut nested = Vec::new();
    let rest = &data[header_end..];
    if object_array || rest.starts_with(b"\x80") {
        let child = scan_bytes_inner(rest, ".pkl", depth + 1);
        nested.push(json!({
            "member": "npy-payload",
            "report": child,
        }));
    }
    ScanReport {
        format: if object_array { "npy+pickle" } else { "npy" }.into(),
        magic_hex: hex_prefix(data),
        size: data.len() as u64,
        extension: extension.into(),
        extension_mismatch: false,
        pickle: None,
        gguf: None,
        safetensors: None,
        nested,
        warnings,
        error: None,
    }
}

fn extension_mismatch(extension: &str, format: &str) -> bool {
    let ext = extension.to_ascii_lowercase();
    let fmt = format.to_ascii_lowercase();
    let claims_safe = matches!(
        ext.as_str(),
        ".safetensors" | ".gguf" | ".onnx" | ".json" | ".md" | ".txt"
    );
    if !claims_safe {
        return false;
    }
    let actually_unsafe = fmt.contains("pickle") || fmt.contains("zip") || fmt == "unknown" && ext == ".safetensors";
    if ext == ".safetensors" {
        return !fmt.contains("safetensors");
    }
    if ext == ".gguf" {
        return !fmt.contains("gguf");
    }
    actually_unsafe
}

fn ext_of(name: &str) -> String {
    Path::new(name)
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_ascii_lowercase()))
        .unwrap_or_default()
}

fn read_limited(file: File, max: usize) -> Result<Vec<u8>, String> {
    let mut data = Vec::new();
    file.take(max as u64)
        .read_to_end(&mut data)
        .map_err(|e| e.to_string())?;
    Ok(data)
}

fn hex_prefix(data: &[u8]) -> String {
    data.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

fn empty_report(extension: &str, size: u64, error: Option<String>) -> ScanReport {
    ScanReport {
        format: "unknown".into(),
        magic_hex: String::new(),
        size,
        extension: extension.into(),
        extension_mismatch: false,
        pickle: None,
        gguf: None,
        safetensors: None,
        nested: Vec::new(),
        warnings: Vec::new(),
        error,
    }
}
