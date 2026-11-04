//! Keras model config checks, for `.keras` zips and legacy `.h5` files.
//!
//! A Keras `Lambda` layer stores a Python function as base64-encoded
//! `marshal` bytecode, and loading it runs that code. Keras 3 also resolves
//! `module` / `registered_name` entries by importing them, which is how
//! CVE-2025-1550 executed attacker-chosen modules from `config.json`.
//!
//! Nothing here decodes marshal data into code objects. The base64 payload is
//! decoded to bytes and searched for the length-prefixed name strings that
//! marshal writes for every identifier the function uses.
//!
//! HDF5 files are not parsed as HDF5. The `model_config` attribute is a JSON
//! string stored contiguously in the file, so a bounded byte search finds it.

use serde_json::{json, Map, Value};
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

pub const HDF5_MAGIC: &[u8] = b"\x89HDF\r\n\x1a\n";
/// Bytes loaded into memory to look for complete JSON configs.
const H5_JSON_WINDOW: u64 = 64 * 1024 * 1024;
/// Bytes searched (chunked) for Lambda markers in total.
const H5_SEARCH_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
const H5_CHUNK: usize = 8 * 1024 * 1024;
const H5_OVERLAP: usize = 4096;
const MAX_JSON: usize = 16 * 1024 * 1024;
const MAX_CANDIDATES: usize = 32;
const MAX_NODES: usize = 2_000_000;
const MAX_HITS: usize = 200;

/// Module roots a stock Keras/TensorFlow model references.
const TRUSTED_ROOTS: &[&str] = &[
    "keras",
    "tensorflow",
    "tf_keras",
    "keras_nlp",
    "keras_hub",
    "keras_cv",
];

/// Modules that are dangerous when a config names them.
const DANGEROUS_MODULES: &[&str] = &[
    "os",
    "posix",
    "nt",
    "subprocess",
    "sys",
    "shutil",
    "socket",
    "importlib",
    "runpy",
    "pty",
    "ctypes",
    "pickle",
    "marshal",
    "webbrowser",
    "code",
    "http",
    "urllib",
    "requests",
    "builtins",
    "__builtin__",
    "pathlib",
    "multiprocessing",
    "asyncio",
];

/// Builtin functions that are dangerous when a config resolves them by name.
const DANGEROUS_BUILTINS: &[&str] = &[
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "getattr",
    "setattr",
    "globals",
    "breakpoint",
];

/// Identifiers that make decoded Lambda bytecode look like it does I/O or runs code.
const DANGEROUS_NAMES: &[&str] = &[
    "os",
    "system",
    "popen",
    "subprocess",
    "eval",
    "exec",
    "__import__",
    "compile",
    "socket",
    "urllib",
    "requests",
    "open",
    "builtins",
    "getattr",
    "marshal",
    "pty",
    "shutil",
    "rmtree",
    "remove",
    "unlink",
    "environ",
    "http",
    "spawn",
    "execv",
    "execve",
    "connect",
    "urlopen",
];

#[derive(Default)]
struct Findings {
    lambdas: Vec<Value>,
    custom: Vec<Value>,
    dangerous: Vec<String>,
}

impl Findings {
    fn add_dangerous(&mut self, token: String) {
        if !self.dangerous.contains(&token) && self.dangerous.len() < MAX_HITS {
            self.dangerous.push(token);
        }
    }
}

/// Analyse one parsed Keras config (model `config.json` or h5 `model_config`).
pub fn analyze_config(root: &Value) -> Value {
    let mut out = Findings::default();
    let mut stack: Vec<(&Value, String)> = vec![(root, "$".to_string())];
    let mut nodes = 0usize;
    let mut truncated = false;
    while let Some((value, path)) = stack.pop() {
        nodes += 1;
        if nodes > MAX_NODES {
            truncated = true;
            break;
        }
        match value {
            Value::Object(map) => {
                inspect_object(map, &path, &mut out);
                for (k, v) in map {
                    if v.is_object() || v.is_array() {
                        stack.push((v, format!("{path}.{k}")));
                    }
                }
            }
            Value::Array(items) => {
                for (i, v) in items.iter().enumerate() {
                    if v.is_object() || v.is_array() {
                        stack.push((v, format!("{path}[{i}]")));
                    }
                }
            }
            _ => {}
        }
    }
    json!({
        "lambda_layers": out.lambdas,
        "custom_objects": out.custom,
        "dangerous": out.dangerous,
        "truncated": truncated,
    })
}

/// Parse JSON bytes and analyse them when they look like a Keras config.
pub fn analyze_config_bytes(data: &[u8]) -> Option<Value> {
    let value: Value = serde_json::from_slice(data).ok()?;
    if !looks_like_keras_config(&value) {
        return None;
    }
    Some(analyze_config(&value))
}

pub fn looks_like_keras_config(value: &Value) -> bool {
    let Some(map) = value.as_object() else {
        return false;
    };
    map.get("class_name").map(Value::is_string).unwrap_or(false) && map.contains_key("config")
}

fn str_field<'a>(map: &'a Map<String, Value>, key: &str) -> &'a str {
    map.get(key).and_then(Value::as_str).unwrap_or("")
}

fn inspect_object(map: &Map<String, Value>, path: &str, out: &mut Findings) {
    let class_name = str_field(map, "class_name");
    let module = str_field(map, "module");
    let registered = str_field(map, "registered_name");

    if class_name == "Lambda" || class_name == "__lambda__" {
        let mut tokens = Vec::new();
        let mut preview = String::new();
        collect_code_tokens(map.get("config"), &mut tokens, &mut preview);
        for t in &tokens {
            out.add_dangerous(format!("lambda uses {t}"));
        }
        if out.lambdas.len() < MAX_HITS {
            out.lambdas.push(json!({
                "path": path,
                "class_name": class_name,
                "name": map.get("config").and_then(|c| c.get("name")).and_then(Value::as_str),
                "dangerous_names": tokens,
                "code_preview": preview,
            }));
        }
    }

    if !module.is_empty() {
        let root = module.split('.').next().unwrap_or("");
        let trusted = TRUSTED_ROOTS.contains(&root);
        if DANGEROUS_MODULES.contains(&root) {
            let config_name = map.get("config").and_then(Value::as_str).unwrap_or("");
            let builtin_fn =
                (root == "builtins" || root == "__builtin__") && class_name == "function";
            if !builtin_fn || DANGEROUS_BUILTINS.contains(&config_name) {
                out.add_dangerous(
                    format!("module {module} class {class_name} {config_name}")
                        .trim()
                        .to_string(),
                );
            }
        } else if !trusted && out.custom.len() < MAX_HITS {
            out.custom.push(json!({
                "path": path,
                "module": module,
                "class_name": class_name,
                "registered_name": registered,
            }));
        }
    } else if registered.contains('>') && out.custom.len() < MAX_HITS {
        out.custom.push(json!({
            "path": path,
            "module": "",
            "class_name": class_name,
            "registered_name": registered,
        }));
    }
}

/// Walk the Lambda config, base64-decode code-looking strings, and collect
/// dangerous identifiers found in them.
fn collect_code_tokens(config: Option<&Value>, tokens: &mut Vec<String>, preview: &mut String) {
    let Some(config) = config else { return };
    let mut stack = vec![config];
    let mut seen = 0usize;
    while let Some(v) = stack.pop() {
        seen += 1;
        if seen > 10_000 {
            return;
        }
        match v {
            Value::String(s) => {
                if preview.is_empty() && s.len() >= 16 {
                    *preview = s.chars().take(120).collect();
                }
                for t in tokens_in_string(s) {
                    if !tokens.contains(&t) {
                        tokens.push(t);
                    }
                }
            }
            Value::Array(items) => stack.extend(items.iter()),
            Value::Object(map) => stack.extend(map.values()),
            _ => {}
        }
    }
}

fn tokens_in_string(s: &str) -> Vec<String> {
    let mut found = Vec::new();
    if s.len() >= 16 {
        if let Some(bytes) = base64_decode(s) {
            found.extend(marshal_names(&bytes));
        }
    }
    // Plain text (source code or a function name) is checked too.
    for name in [
        "os.system",
        "subprocess",
        "__import__",
        "popen",
        "eval(",
        "exec(",
        "socket",
        "urlopen",
    ] {
        if s.contains(name) {
            let t = name.trim_end_matches('(').to_string();
            if !found.contains(&t) {
                found.push(t);
            }
        }
    }
    found
}

/// Identifiers from `DANGEROUS_NAMES` that appear as marshal string objects.
pub fn marshal_names(bytes: &[u8]) -> Vec<String> {
    let mut found = Vec::new();
    for name in DANGEROUS_NAMES {
        if marshal_has_name(bytes, name.as_bytes()) {
            found.push((*name).to_string());
        }
    }
    found
}

fn marshal_has_name(bytes: &[u8], name: &[u8]) -> bool {
    let n = name.len();
    if n == 0 || n > 255 || bytes.len() < n + 2 {
        return false;
    }
    for i in 0..bytes.len() - n {
        // TYPE_SHORT_ASCII(_INTERNED) 'z' / 'Z', with or without FLAG_REF.
        let code = bytes[i] & 0x7f;
        if (code == b'z' || code == b'Z')
            && bytes[i + 1] as usize == n
            && bytes[i + 2..].starts_with(name)
        {
            return true;
        }
        // TYPE_ASCII / TYPE_ASCII_INTERNED / TYPE_UNICODE / TYPE_INTERNED with a u32 length.
        if matches!(code, b'a' | b'A' | b'u' | b't') && i + 5 + n <= bytes.len() {
            let len = u32::from_le_bytes([bytes[i + 1], bytes[i + 2], bytes[i + 3], bytes[i + 4]]);
            if len as usize == n && &bytes[i + 5..i + 5 + n] == name {
                return true;
            }
        }
    }
    false
}

/// Standard or URL-safe base64, padding optional, whitespace ignored.
pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc = 0u32;
    let mut bits = 0u32;
    let mut count = 0usize;
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return None,
        };
        count += 1;
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    if count < 4 {
        return None;
    }
    Some(out)
}

/// Scan an HDF5 file for Keras configs and Lambda markers.
pub fn scan_h5<R: Read + Seek>(
    reader: &mut R,
    size: u64,
    deadline: Instant,
) -> Result<Value, String> {
    reader.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let window = size.min(H5_JSON_WINDOW) as usize;
    let mut head = vec![0u8; window];
    reader.read_exact(&mut head).map_err(|e| e.to_string())?;

    let mut warnings = Vec::new();
    let mut limit_exceeded: Option<String> = None;
    let configs = extract_configs(&head, deadline, &mut limit_exceeded);
    let mut byte_hits = count_markers(&head);
    let has_model_config = find(&head, b"model_config").is_some();

    // Lambda markers past the JSON window are still counted, chunk by chunk.
    let search_end = size.min(H5_SEARCH_LIMIT);
    let mut pos = window as u64;
    let mut buf = vec![0u8; H5_CHUNK + H5_OVERLAP];
    let mut carry = head[head.len().saturating_sub(H5_OVERLAP)..].to_vec();
    while pos < search_end {
        if Instant::now() >= deadline {
            limit_exceeded = Some("time limit reached during HDF5 search".into());
            break;
        }
        let want = (search_end - pos).min(H5_CHUNK as u64) as usize;
        let keep = carry.len();
        buf[..keep].copy_from_slice(&carry);
        reader
            .read_exact(&mut buf[keep..keep + want])
            .map_err(|e| e.to_string())?;
        let chunk = &buf[..keep + want];
        byte_hits += count_markers(chunk) - count_markers(&chunk[..keep]);
        carry = chunk[chunk.len().saturating_sub(H5_OVERLAP)..].to_vec();
        pos += want as u64;
    }
    if size > H5_SEARCH_LIMIT {
        warnings.push(format!(
            "HDF5 file is larger than {H5_SEARCH_LIMIT} bytes; only the start was searched"
        ));
    }

    let mut merged = Findings::default();
    for cfg in &configs {
        for (key, dest) in [
            ("lambda_layers", &mut merged.lambdas),
            ("custom_objects", &mut merged.custom),
        ] {
            if let Some(items) = cfg.get(key).and_then(Value::as_array) {
                dest.extend(items.iter().cloned());
            }
        }
        if let Some(items) = cfg.get("dangerous").and_then(Value::as_array) {
            for t in items.iter().filter_map(Value::as_str) {
                merged.add_dangerous(t.to_string());
            }
        }
    }
    if configs.is_empty() && byte_hits > 0 {
        warnings.push("Lambda marker found but no complete Keras config could be parsed".into());
    }
    Ok(json!({
        "source": "hdf5",
        "keras_model_config": has_model_config,
        "configs_parsed": configs.len(),
        "lambda_marker_hits": byte_hits,
        "lambda_layers": merged.lambdas,
        "custom_objects": merged.custom,
        "dangerous": merged.dangerous,
        "warnings": warnings,
        "limit_exceeded": limit_exceeded,
    }))
}

fn count_markers(data: &[u8]) -> usize {
    [
        &b"\"class_name\": \"Lambda\""[..],
        b"\"class_name\":\"Lambda\"",
        b"__lambda__",
    ]
    .iter()
    .map(|needle| count(data, needle))
    .sum()
}

/// Find `{"class_name"` objects, brace-match them, and analyse those that
/// parse as Keras configs.
fn extract_configs(data: &[u8], deadline: Instant, limit: &mut Option<String>) -> Vec<Value> {
    let mut configs = Vec::new();
    let mut from = 0usize;
    let mut tried = 0usize;
    while let Some(rel) = find(&data[from..], b"{\"class_name\"") {
        let start = from + rel;
        tried += 1;
        if tried > MAX_CANDIDATES || Instant::now() >= deadline {
            *limit = Some("Keras config candidate limit reached in HDF5 search".into());
            break;
        }
        match brace_match(&data[start..]) {
            Some(len) => {
                if let Some(v) = analyze_config_bytes(&data[start..start + len]) {
                    configs.push(v);
                    from = start + len;
                    continue;
                }
                from = start + 1;
            }
            None => from = start + 1,
        }
    }
    configs
}

/// Length of the JSON object starting at `data[0] == b'{'`, honouring strings.
fn brace_match(data: &[u8]) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escape = false;
    for (i, &b) in data.iter().enumerate().take(MAX_JSON) {
        if in_str {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            0 => return None,
            _ => {}
        }
    }
    None
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

fn count(hay: &[u8], needle: &[u8]) -> usize {
    if hay.len() < needle.len() {
        return 0;
    }
    hay.windows(needle.len()).filter(|w| *w == needle).count()
}

#[cfg(test)]
#[path = "keras_tests.rs"]
mod tests;
