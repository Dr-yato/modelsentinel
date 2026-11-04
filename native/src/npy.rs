//! NumPy `.npy` header reader.
//!
//! An array with an object dtype (`'|O'`, or a structured dtype with an
//! object field) is stored as a pickle after the header, and loading it needs
//! `allow_pickle=True`. That pickle is returned to the caller, which walks it
//! like any other pickle. Plain numeric arrays are never read past the header.

pub const MAGIC: &[u8] = b"\x93NUMPY";
/// numpy itself refuses headers above 10 kB by default; allow more, but bounded.
pub const MAX_HEADER: usize = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpyHeader {
    pub major: u8,
    pub minor: u8,
    pub header: String,
    /// Offset of the first byte after the header (the array payload).
    pub payload_offset: usize,
    pub object_dtype: bool,
}

/// Parse the header from the first bytes of an `.npy` file.
pub fn parse_header(data: &[u8]) -> Result<NpyHeader, String> {
    if data.len() < 10 || !data.starts_with(MAGIC) {
        return Err("not an npy file".into());
    }
    let major = data[6];
    let minor = data[7];
    let (header_len, header_start) = match major {
        1 => (u16::from_le_bytes([data[8], data[9]]) as usize, 10),
        2 | 3 => {
            if data.len() < 12 {
                return Err("truncated npy".into());
            }
            (
                u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize,
                12,
            )
        }
        other => return Err(format!("unsupported npy version {other}")),
    };
    if header_len > MAX_HEADER {
        return Err(format!(
            "npy header length {header_len} exceeds {MAX_HEADER}"
        ));
    }
    let header_end = header_start + header_len;
    if header_end > data.len() {
        return Err("truncated npy header".into());
    }
    let header = String::from_utf8_lossy(&data[header_start..header_end]).into_owned();
    let object_dtype = descr_has_object(&header);
    Ok(NpyHeader {
        major,
        minor,
        header,
        payload_offset: header_end,
        object_dtype,
    })
}

/// Bytes needed to read a complete header, given the first 12 bytes.
pub fn header_span(prefix: &[u8]) -> Option<usize> {
    if prefix.len() < 10 || !prefix.starts_with(MAGIC) {
        return None;
    }
    match prefix[6] {
        1 => Some(10 + u16::from_le_bytes([prefix[8], prefix[9]]) as usize),
        2 | 3 if prefix.len() >= 12 => {
            let n = u32::from_le_bytes([prefix[8], prefix[9], prefix[10], prefix[11]]) as usize;
            Some(12usize.saturating_add(n.min(MAX_HEADER + 1)))
        }
        _ => None,
    }
}

/// True when the `descr` entry names the object type code `O`, either alone
/// (`'|O'`) or inside a structured dtype (`[('a', '|O')]`). In a structured
/// dtype the string right after `(` is a field name, not a type, so a field
/// called `'O'` is not mistaken for an object array.
fn descr_has_object(header: &str) -> bool {
    quoted_strings(descr_value(header))
        .iter()
        .any(|(prev, s)| *prev != Some('(') && is_object_code(s))
}

/// Quoted strings with the last non-space character before each one.
fn quoted_strings(s: &str) -> Vec<(Option<char>, String)> {
    let mut out = Vec::new();
    let mut prev: Option<char> = None;
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\'' && c != '"' {
            if !c.is_whitespace() {
                prev = Some(c);
            }
            continue;
        }
        let mut buf = String::new();
        for d in chars.by_ref() {
            if d == c {
                break;
            }
            buf.push(d);
        }
        out.push((prev, buf));
        prev = Some(c);
    }
    out
}

fn is_object_code(s: &str) -> bool {
    let t = s.trim_start_matches(['<', '>', '|', '=']);
    t == "O" || (t.starts_with('O') && t.len() > 1 && t[1..].chars().all(|c| c.is_ascii_digit()))
}

fn descr_value(header: &str) -> &str {
    let Some(at) = header.find("descr") else {
        return header;
    };
    let after_key = &header[at + 5..];
    let rest = match after_key.find(':') {
        Some(colon) => &after_key[colon + 1..],
        None => after_key,
    };
    // The value ends where the next top-level key starts.
    let end = [
        "'fortran_order'",
        "\"fortran_order\"",
        "'shape'",
        "\"shape\"",
    ]
    .iter()
    .filter_map(|k| rest.find(k))
    .min()
    .unwrap_or(rest.len());
    &rest[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy(header: &str, payload: &[u8]) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&[1, 0]);
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn numeric_array_is_not_object() {
        let raw = npy(
            "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }\n",
            &[0; 4],
        );
        let h = parse_header(&raw).unwrap();
        assert!(!h.object_dtype);
        assert_eq!(h.payload_offset, raw.len() - 4);
    }

    #[test]
    fn object_and_structured_object_dtypes() {
        let raw = npy(
            "{'descr': '|O', 'fortran_order': False, 'shape': (2,), }\n",
            b"\x80\x03.",
        );
        assert!(parse_header(&raw).unwrap().object_dtype);
        let raw = npy(
            "{'descr': [('a', '<f4'), ('b', '|O')], 'fortran_order': False, 'shape': (1,), }\n",
            b"",
        );
        assert!(parse_header(&raw).unwrap().object_dtype);
    }

    #[test]
    fn field_named_o_is_not_object() {
        let raw = npy(
            "{'descr': [('O', '<f4')], 'fortran_order': False, 'shape': (1,), }\n",
            b"",
        );
        assert!(!parse_header(&raw).unwrap().object_dtype);
    }

    #[test]
    fn bad_headers() {
        assert!(parse_header(b"\x93NUMPY").is_err());
        let mut raw = MAGIC.to_vec();
        raw.extend_from_slice(&[9, 0, 0, 0]);
        assert!(parse_header(&raw).is_err());
        let mut raw = MAGIC.to_vec();
        raw.extend_from_slice(&[2, 0, 0xff, 0xff, 0xff, 0x7f]);
        assert!(parse_header(&raw).unwrap_err().contains("exceeds"));
        let raw = npy("{'descr': '<f4'", b"");
        let mut cut = raw.clone();
        cut[8] = 200;
        assert!(parse_header(&cut).is_err());
    }
}
