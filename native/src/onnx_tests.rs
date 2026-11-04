//! ONNX reader tests with hand-encoded protobuf.

use super::*;

fn varint(mut v: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return out;
        }
        out.push(b | 0x80);
    }
}

fn field_len(field: u64, payload: &[u8]) -> Vec<u8> {
    let mut out = varint((field << 3) | 2);
    out.extend(varint(payload.len() as u64));
    out.extend_from_slice(payload);
    out
}

fn field_varint(field: u64, v: u64) -> Vec<u8> {
    let mut out = varint(field << 3);
    out.extend(varint(v));
    out
}

fn entry(key: &str, value: &str) -> Vec<u8> {
    let mut e = field_len(1, key.as_bytes());
    e.extend(field_len(2, value.as_bytes()));
    e
}

fn external_tensor(name: &str, location: &str) -> Vec<u8> {
    let mut t = field_len(8, name.as_bytes());
    t.extend(field_len(13, &entry("location", location)));
    t.extend(field_len(13, &entry("offset", "0")));
    t.extend(field_varint(14, 1));
    t
}

fn node(op: &str, domain: &str) -> Vec<u8> {
    let mut n = field_len(4, op.as_bytes());
    if !domain.is_empty() {
        n.extend(field_len(7, domain.as_bytes()));
    }
    n
}

fn model(graph: &[u8], opsets: &[(&str, u64)]) -> Vec<u8> {
    let mut m = field_varint(1, 8);
    m.extend(field_len(2, b"pytorch"));
    m.extend(field_len(7, graph));
    for (domain, version) in opsets {
        let mut o = field_len(1, domain.as_bytes());
        o.extend(field_varint(2, *version));
        m.extend(field_len(8, &o));
    }
    m
}

#[test]
fn clean_model_with_inline_weights() {
    let mut init = field_len(8, b"w");
    init.extend(field_len(9, &[0u8; 4096])); // raw_data, skipped not loaded
    let mut g = field_len(1, &node("Relu", ""));
    g.extend(field_len(5, &init));
    let raw = model(&g, &[("", 17)]);
    assert!(looks_like_onnx(&raw));
    let r = scan_bytes(&raw).unwrap();
    assert_eq!(r["ir_version"], 8);
    assert_eq!(r["node_count"], 1);
    assert_eq!(r["initializer_count"], 1);
    assert!(r["external_data"].as_array().unwrap().is_empty());
    assert!(r["custom_domains"].as_array().unwrap().is_empty());
}

#[test]
fn external_data_traversal_and_absolute_paths() {
    let mut g = Vec::new();
    for (i, loc) in [
        "weights.bin",
        "../../etc/passwd",
        "/etc/shadow",
        "C:\\Windows\\win.ini",
        "sub\\..\\..\\x",
    ]
    .iter()
    .enumerate()
    {
        g.extend(field_len(5, &external_tensor(&format!("t{i}"), loc)));
    }
    let r = scan_bytes(&model(&g, &[("", 17)])).unwrap();
    let ext = r["external_data"].as_array().unwrap();
    assert_eq!(ext.len(), 5);
    assert_eq!(ext[0]["traversal"], false);
    assert_eq!(ext[0]["absolute"], false);
    assert_eq!(ext[1]["traversal"], true);
    assert_eq!(ext[2]["absolute"], true);
    assert_eq!(ext[3]["absolute"], true);
    assert_eq!(ext[4]["traversal"], true);
}

#[test]
fn external_data_in_subgraph_attribute() {
    let attr = field_len(5, &external_tensor("inner", "../secret"));
    let mut sub_node = node("Identity", "");
    sub_node.extend(field_len(5, &attr));
    let subgraph = field_len(1, &sub_node);
    let mut if_node = node("If", "");
    if_node.extend(field_len(5, &field_len(6, &subgraph)));
    let r = scan_bytes(&model(&field_len(1, &if_node), &[])).unwrap();
    assert_eq!(r["external_data"][0]["traversal"], true);
    assert_eq!(r["node_count"], 2);
}

#[test]
fn custom_domains_and_python_ops() {
    let mut g = field_len(1, &node("PyCustomOp", "ai.onnx.contrib"));
    g.extend(field_len(1, &node("FusedGelu", "com.microsoft")));
    g.extend(field_len(1, &node("Mystery", "evil.ops")));
    let r = scan_bytes(&model(&g, &[("", 17), ("ai.onnx.contrib", 1)])).unwrap();
    let domains: Vec<&str> = r["custom_domains"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .collect();
    assert!(domains.contains(&"ai.onnx.contrib"));
    assert!(domains.contains(&"evil.ops"));
    assert!(!domains.contains(&"com.microsoft"));
    let ops = r["custom_ops"].as_array().unwrap();
    assert!(ops
        .iter()
        .any(|o| o["op_type"] == "PyCustomOp" && o["python_op"] == true));
}

#[test]
fn deep_subgraph_nesting_is_limited() {
    let mut g = field_len(1, &node("Relu", ""));
    for _ in 0..40 {
        let mut n = node("If", "");
        n.extend(field_len(5, &field_len(6, &g)));
        g = field_len(1, &n);
    }
    let r = scan_bytes(&model(&g, &[])).unwrap();
    assert!(r["limit_exceeded"].as_str().unwrap().contains("nesting"));
}

#[test]
fn malformed_lengths_are_errors_not_panics() {
    let mut raw = field_varint(1, 8);
    raw.extend(varint((7 << 3) | 2));
    raw.extend(varint(u64::MAX));
    assert!(scan_bytes(&raw).is_err());
    let mut raw = field_varint(1, 8);
    raw.extend(varint((7 << 3) | 3)); // group wire type
    assert!(scan_bytes(&raw).is_err());
    assert!(scan_bytes(&[0xff; 12]).is_err());
    assert!(scan_bytes(b"").is_err());
}

#[test]
fn deadline_in_the_past_is_reported() {
    let mut g = Vec::new();
    for _ in 0..5000 {
        g.extend(field_len(1, &node("Relu", "")));
    }
    let raw = model(&g, &[]);
    let past = Instant::now();
    let r = scan_reader(&mut std::io::Cursor::new(&raw), Some(past)).unwrap();
    assert!(r["limit_exceeded"].as_str().unwrap().contains("time"));
}

#[test]
fn location_classifier() {
    assert_eq!(classify_location("a/b.bin"), (false, false));
    assert_eq!(classify_location("a..b"), (false, false));
    assert_eq!(classify_location("~/x"), (true, false));
    assert_eq!(classify_location("http://x/y"), (true, false));
    assert_eq!(classify_location(".."), (false, true));
}
