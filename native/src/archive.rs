//! Zip and tar walkers over `Read + Seek`.
//!
//! One implementation serves files on disk and in-memory members (via
//! `Cursor`), so a nested zip is read exactly like a top-level one. Members
//! are inflated one at a time, each capped at `MAX_MEMBER` and all of them
//! together capped by the `Budget`, which is what stops zip bombs. Members
//! are never written to disk.

use crate::limits::{Budget, MAX_MEMBER};
use crate::scan::{scan_member, ScanReport};
use flate2::read::DeflateDecoder;
use serde_json::{json, Value};
use std::io::{Read, Seek, SeekFrom};

const MAX_CD: u64 = 64 * 1024 * 1024;
const EOCD_SEARCH: u64 = 66 * 1024;

pub struct ArchiveScan {
    pub nested: Vec<Value>,
    pub warnings: Vec<String>,
    pub member_names: Vec<String>,
}

struct Entry {
    name: String,
    method: u16,
    flags: u16,
    compressed: u64,
    uncompressed: u64,
    local_offset: u64,
}

fn le16(b: &[u8], at: usize) -> u64 {
    u16::from_le_bytes([b[at], b[at + 1]]) as u64
}

fn le32(b: &[u8], at: usize) -> u64 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as u64
}

fn le64(b: &[u8], at: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(a)
}

fn io<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

/// Locate the central directory: (offset, size, entry count).
fn central_directory<R: Read + Seek>(r: &mut R, size: u64) -> Result<(u64, u64, u64), String> {
    let tail_len = size.min(EOCD_SEARCH);
    r.seek(SeekFrom::Start(size - tail_len)).map_err(io)?;
    let mut tail = vec![0u8; tail_len as usize];
    r.read_exact(&mut tail).map_err(io)?;
    let at = (0..tail.len().saturating_sub(21))
        .rev()
        .find(|&i| &tail[i..i + 4] == b"PK\x05\x06")
        .ok_or("zip end-of-central-directory not found")?;
    let mut count = le16(&tail, at + 10);
    let mut cd_size = le32(&tail, at + 12);
    let mut cd_offset = le32(&tail, at + 16);
    let needs64 = count == 0xFFFF || cd_size == 0xFFFF_FFFF || cd_offset == 0xFFFF_FFFF;
    if needs64 && at >= 20 && &tail[at - 20..at - 16] == b"PK\x06\x07" {
        let rec = le64(&tail, at - 20 + 8);
        if rec.saturating_add(56) > size {
            return Err("zip64 end-of-central-directory out of range".into());
        }
        r.seek(SeekFrom::Start(rec)).map_err(io)?;
        let mut z = [0u8; 56];
        r.read_exact(&mut z).map_err(io)?;
        if &z[0..4] != b"PK\x06\x06" {
            return Err("bad zip64 end-of-central-directory record".into());
        }
        count = le64(&z, 32);
        cd_size = le64(&z, 40);
        cd_offset = le64(&z, 48);
    }
    if cd_offset.saturating_add(cd_size) > size {
        return Err("zip central directory out of range".into());
    }
    if cd_size > MAX_CD {
        return Err(format!("zip central directory larger than {MAX_CD} bytes"));
    }
    Ok((cd_offset, cd_size, count))
}

fn parse_entries(cd: &[u8], budget: &mut Budget) -> Result<Vec<Entry>, String> {
    let mut i = 0;
    let mut entries = Vec::new();
    while i + 46 <= cd.len() {
        if &cd[i..i + 4] != b"PK\x01\x02" {
            break;
        }
        if !budget.charge_member() {
            break;
        }
        let name_len = le16(cd, i + 28) as usize;
        let extra_len = le16(cd, i + 30) as usize;
        let comment_len = le16(cd, i + 32) as usize;
        let name_end = i + 46 + name_len;
        if name_end + extra_len + comment_len > cd.len() {
            return Err("truncated central directory".into());
        }
        let mut e = Entry {
            name: String::from_utf8_lossy(&cd[i + 46..name_end]).into_owned(),
            method: le16(cd, i + 10) as u16,
            flags: le16(cd, i + 8) as u16,
            compressed: le32(cd, i + 20),
            uncompressed: le32(cd, i + 24),
            local_offset: le32(cd, i + 42),
        };
        apply_zip64_extra(&cd[name_end..name_end + extra_len], &mut e);
        entries.push(e);
        i = name_end + extra_len + comment_len;
    }
    Ok(entries)
}

fn apply_zip64_extra(extra: &[u8], e: &mut Entry) {
    let mut i = 0;
    while i + 4 <= extra.len() {
        let id = le16(extra, i);
        let size = le16(extra, i + 2) as usize;
        i += 4;
        if i + size > extra.len() {
            return;
        }
        if id == 0x0001 {
            let mut p = i;
            for slot in [&mut e.uncompressed, &mut e.compressed, &mut e.local_offset] {
                if *slot == 0xFFFF_FFFF && p + 8 <= i + size {
                    *slot = le64(extra, p);
                    p += 8;
                }
            }
        }
        i += size;
    }
}

/// Read and inflate one member, bounded by `MAX_MEMBER` and the budget.
fn read_member<R: Read + Seek>(
    r: &mut R,
    size: u64,
    e: &Entry,
    budget: &mut Budget,
) -> Result<Vec<u8>, String> {
    if e.local_offset.saturating_add(30) > size {
        return Err("bad local header offset".into());
    }
    r.seek(SeekFrom::Start(e.local_offset)).map_err(io)?;
    let mut local = [0u8; 30];
    r.read_exact(&mut local).map_err(io)?;
    if &local[0..4] != b"PK\x03\x04" {
        return Err("bad local header".into());
    }
    let skip = le16(&local, 26) + le16(&local, 28);
    let start = e.local_offset + 30 + skip;
    if start.saturating_add(e.compressed) > size {
        return Err("member data runs past the archive".into());
    }
    r.seek(SeekFrom::Start(start)).map_err(io)?;
    let cap = (MAX_MEMBER as u64).min(budget.inflate_left());
    let raw = r.by_ref().take(e.compressed);
    let mut out = Vec::new();
    match e.method {
        0 => {
            raw.take(cap + 1).read_to_end(&mut out).map_err(io)?;
        }
        8 => {
            DeflateDecoder::new(raw)
                .take(cap + 1)
                .read_to_end(&mut out)
                .map_err(|err| format!("deflate: {err}"))?;
        }
        m => return Err(format!("compression method {m} not supported")),
    }
    if out.len() as u64 > cap {
        out.truncate(cap as usize);
        budget.note(&format!(
            "zip member {} inflates past {cap} bytes; only the first {cap} were scanned (possible decompression bomb)",
            e.name
        ));
    }
    budget.charge_inflate(out.len() as u64);
    Ok(out)
}

pub fn scan_zip<R: Read + Seek>(
    r: &mut R,
    size: u64,
    depth: u32,
    budget: &mut Budget,
) -> Result<ArchiveScan, String> {
    let (cd_offset, cd_size, _count) = central_directory(r, size)?;
    r.seek(SeekFrom::Start(cd_offset)).map_err(io)?;
    let mut cd = vec![0u8; cd_size as usize];
    r.read_exact(&mut cd).map_err(io)?;
    let entries = parse_entries(&cd, budget)?;
    let mut out = ArchiveScan {
        nested: Vec::new(),
        warnings: Vec::new(),
        member_names: entries.iter().map(|e| e.name.clone()).collect(),
    };
    for e in &entries {
        if !budget.time_ok() {
            break;
        }
        if e.name.ends_with('/') {
            continue;
        }
        if e.flags & 1 != 0 {
            out.warnings.push(format!(
                "zip member {} is encrypted and was not scanned",
                e.name
            ));
            continue;
        }
        if !wants_member(&e.name, e.uncompressed) {
            continue;
        }
        if budget.inflate_left() == 0 {
            budget
                .note("total decompressed bytes limit reached; remaining members were not scanned");
            break;
        }
        match read_member(r, size, e, budget) {
            Ok(data) => {
                if !should_scan_member(&e.name, &data) {
                    continue;
                }
                let child = scan_member(&data, &e.name, depth + 1, budget);
                out.nested
                    .push(member_json(&e.name, e.compressed, data.len(), child));
            }
            Err(err) => out.warnings.push(format!("member {}: {err}", e.name)),
        }
    }
    Ok(out)
}

fn member_json(name: &str, compressed: u64, size: usize, report: ScanReport) -> Value {
    json!({
        "member": name,
        "compressed_size": compressed,
        "uncompressed_size": size,
        "report": report,
    })
}

/// Pre-inflation filter on the declared size. Tensor blobs in torch zips are
/// numerous and large; they are skipped unless their name says otherwise.
fn wants_member(name: &str, declared: u64) -> bool {
    by_name(name) || declared <= 1_000_000
}

fn by_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with("data.pkl")
        || [
            ".pkl",
            ".pickle",
            ".pt",
            ".pth",
            ".bin",
            ".ckpt",
            ".json",
            ".jinja",
            ".txt",
            ".py",
            ".npy",
            ".npz",
            ".h5",
            ".hdf5",
            ".keras",
            ".onnx",
            ".zip",
            ".gguf",
            ".joblib",
            ".safetensors",
        ]
        .iter()
        .any(|ext| lower.ends_with(ext))
}

pub fn should_scan_member(name: &str, data: &[u8]) -> bool {
    if by_name(name) {
        return true;
    }
    data.starts_with(b"\x80")
        || data.starts_with(b"GGUF")
        || data.starts_with(b"PK")
        || data.starts_with(&[0x1f, 0x8b])
        || data.starts_with(b"\x93NUMPY")
        || data.starts_with(crate::keras::HDF5_MAGIC)
        || data.starts_with(b"{")
        || crate::safetensors::looks_like_safetensors(data)
}

pub fn is_tar(header: &[u8]) -> bool {
    header.len() > 262 && &header[257..262] == b"ustar"
}

fn parse_octal(bytes: &[u8]) -> Option<u64> {
    let s = std::str::from_utf8(bytes)
        .ok()?
        .trim_matches(|c: char| c == '\0' || c.is_whitespace());
    if s.is_empty() {
        return Some(0);
    }
    u64::from_str_radix(s, 8).ok()
}

fn c_string(bytes: &[u8]) -> String {
    let n = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..n]).into_owned()
}

pub fn scan_tar<R: Read + Seek>(
    r: &mut R,
    size: u64,
    depth: u32,
    budget: &mut Budget,
) -> Result<ArchiveScan, String> {
    let mut out = ArchiveScan {
        nested: Vec::new(),
        warnings: Vec::new(),
        member_names: Vec::new(),
    };
    let mut offset = 0u64;
    let mut header = [0u8; 512];
    while offset + 512 <= size {
        if !budget.time_ok() || !budget.charge_member() {
            break;
        }
        r.seek(SeekFrom::Start(offset)).map_err(io)?;
        r.read_exact(&mut header).map_err(io)?;
        if header.iter().all(|b| *b == 0) {
            break;
        }
        let name = c_string(&header[..100]);
        let prefix = c_string(&header[345..500]);
        let full = if prefix.is_empty() {
            name
        } else {
            format!("{prefix}/{name}")
        };
        let Some(len) = parse_octal(&header[124..136]) else {
            out.warnings
                .push(format!("bad size field for tar member {full}"));
            break;
        };
        let typeflag = header[156];
        offset += 512;
        let padded = len.div_ceil(512).saturating_mul(512);
        if offset.saturating_add(len) > size {
            out.warnings.push(format!("truncated tar member {full}"));
            break;
        }
        out.member_names.push(full.clone());
        if typeflag == b'0' || typeflag == 0 {
            if len > MAX_MEMBER as u64 {
                if by_name(&full) {
                    budget.note(&format!(
                        "tar member {full} is larger than {MAX_MEMBER} bytes and was not scanned"
                    ));
                }
            } else if by_name(&full) || len <= 1_000_000 {
                let mut data = vec![0u8; len as usize];
                r.read_exact(&mut data).map_err(io)?;
                if should_scan_member(&full, &data) {
                    let child = scan_member(&data, &full, depth + 1, budget);
                    out.nested.push(json!({
                        "member": full,
                        "uncompressed_size": data.len(),
                        "report": child,
                    }));
                }
            }
        }
        offset = offset.saturating_add(padded);
    }
    Ok(out)
}
