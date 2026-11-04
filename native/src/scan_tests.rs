//! Dispatcher tests: containers, new formats, limits, and a deterministic
//! mini-fuzz. All payloads are inert bytes and nothing is unpickled.

use super::*;
use flate2::write::{DeflateEncoder, GzEncoder};
use flate2::Compression;
use std::io::Write;

const PICKLE: &[u8] =
    b"\x80\x04\x8c\x05posix\x94\x8c\x06system\x94\x93\x94\x8c\x04echo\x94\x85\x94R\x94.";

fn deflate(data: &[u8]) -> Vec<u8> {
    let mut e = DeflateEncoder::new(Vec::new(), Compression::fast());
    e.write_all(data).unwrap();
    e.finish().unwrap()
}

fn gzip(data: &[u8]) -> Vec<u8> {
    let mut e = GzEncoder::new(Vec::new(), Compression::fast());
    e.write_all(data).unwrap();
    e.finish().unwrap()
}

/// Minimal zip writer (no CRCs; the scanner does not check them).
fn zip_with(entries: &[(&str, &[u8])], compress: bool, zip64: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in entries {
        let body = if compress {
            deflate(data)
        } else {
            data.to_vec()
        };
        let method: u16 = if compress { 8 } else { 0 };
        let offset = out.len() as u32;
        out.extend_from_slice(b"PK\x03\x04\x14\x00\x00\x00");
        out.extend_from_slice(&method.to_le_bytes());
        out.extend_from_slice(&[0; 8]);
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&body);
        central.extend_from_slice(b"PK\x01\x02\x14\x00\x14\x00\x00\x00");
        central.extend_from_slice(&method.to_le_bytes());
        central.extend_from_slice(&[0; 8]);
        central.extend_from_slice(&(body.len() as u32).to_le_bytes());
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(name.len() as u16).to_le_bytes());
        central.extend_from_slice(&[0; 12]);
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }
    let cd_offset = out.len() as u64;
    out.extend_from_slice(&central);
    let n = entries.len() as u64;
    if zip64 {
        let rec = out.len() as u64;
        out.extend_from_slice(b"PK\x06\x06");
        out.extend_from_slice(&44u64.to_le_bytes());
        out.extend_from_slice(&[0x2d, 0, 0x2d, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&(central.len() as u64).to_le_bytes());
        out.extend_from_slice(&cd_offset.to_le_bytes());
        out.extend_from_slice(b"PK\x06\x07\x00\x00\x00\x00");
        out.extend_from_slice(&rec.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(
            b"PK\x05\x06\x00\x00\x00\x00\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x00\x00",
        );
    } else {
        out.extend_from_slice(b"PK\x05\x06\x00\x00\x00\x00");
        out.extend_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&(central.len() as u32).to_le_bytes());
        out.extend_from_slice(&(cd_offset as u32).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    out
}

fn zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
    zip_with(entries, true, false)
}

fn npy(descr: &str, payload: &[u8]) -> Vec<u8> {
    let header = format!("{{'descr': {descr}, 'fortran_order': False, 'shape': (1,), }}\n");
    let mut out = npy::MAGIC.to_vec();
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(payload);
    out
}

/// Every pickle global found anywhere in the report tree, as "module.name".
fn all_globals(report: &Value) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(globals) = report.pointer("/pickle/globals").and_then(Value::as_array) {
        for g in globals {
            out.push(format!(
                "{}.{}",
                g["module"].as_str().unwrap(),
                g["name"].as_str().unwrap()
            ));
        }
    }
    for n in report["nested"].as_array().into_iter().flatten() {
        out.extend(all_globals(&n["report"]));
    }
    out
}

fn scan(data: &[u8], ext: &str) -> Value {
    serde_json::to_value(scan_bytes(data, ext)).unwrap()
}

#[test]
fn pickle_is_found_whatever_the_extension() {
    for ext in [".pkl", ".bin", ".pt", ".pth", ".ckpt", ""] {
        let r = scan(PICKLE, ext);
        assert_eq!(r["format"], "pickle", "{ext}");
        assert!(all_globals(&r).contains(&"posix.system".to_string()));
        assert_eq!(r["extension_mismatch"], false);
    }
    for ext in [".safetensors", ".gguf", ".npy", ".onnx", ".h5", ".json"] {
        assert_eq!(scan(PICKLE, ext)["extension_mismatch"], true, "{ext}");
    }
}

#[test]
fn torch_zip_and_unusual_member_paths() {
    let r = scan(
        &zip(&[
            ("archive/data.pkl", PICKLE),
            ("archive/byteorder", b"little"),
        ]),
        ".bin",
    );
    assert_eq!(r["format"], "zip+pickle");
    assert!(all_globals(&r).contains(&"posix.system".to_string()));
    let r = scan(
        &zip(&[
            ("model/extra/state.pickle", PICKLE),
            ("x/y/z/weights.pkl", PICKLE),
        ]),
        ".pt",
    );
    assert_eq!(r["nested"].as_array().unwrap().len(), 2);
}

#[test]
fn nested_zip_inside_torch_zip() {
    let inner = zip(&[("archive/data.pkl", PICKLE)]);
    let outer = zip(&[
        ("archive/data.pkl", b"\x80\x02N."),
        ("archive/extra/inner.zip", &inner),
    ]);
    let r = scan(&outer, ".pt");
    assert!(all_globals(&r).contains(&"posix.system".to_string()), "{r}");
    // Stored (uncompressed) members and a nameless nested zip are found too.
    let outer = zip_with(&[("blob", &inner)], false, false);
    assert!(all_globals(&scan(&outer, ".bin")).contains(&"posix.system".to_string()));
}

#[test]
fn zip_nesting_depth_is_limited() {
    let mut z = zip(&[("data.pkl", PICKLE)]);
    for _ in 0..8 {
        z = zip(&[("inner.zip", &z)]);
    }
    let r = scan(&z, ".zip");
    let limits = r["limits_exceeded"].as_array().unwrap();
    assert!(
        limits
            .iter()
            .any(|l| l.as_str().unwrap().contains("nesting")),
        "{limits:?}"
    );
}

#[test]
fn zip_bomb_member_is_capped() {
    let zeros = vec![0u8; MAX_MEMBER + 4096];
    let r = scan(&zip(&[("data.pkl", &zeros)]), ".pt");
    let limits = r["limits_exceeded"].as_array().unwrap();
    assert!(
        limits
            .iter()
            .any(|l| l.as_str().unwrap().contains("decompression bomb")),
        "{limits:?}"
    );
}

#[test]
fn zip64_end_records_are_followed() {
    let r = scan(
        &zip_with(&[("archive/data.pkl", PICKLE)], false, true),
        ".pt",
    );
    assert!(all_globals(&r).contains(&"posix.system".to_string()), "{r}");
}

#[test]
fn member_count_is_limited() {
    let names: Vec<String> = (0..crate::limits::MAX_MEMBERS + 5)
        .map(|i| format!("f{i}"))
        .collect();
    let entries: Vec<(&str, &[u8])> = names.iter().map(|n| (n.as_str(), &b""[..])).collect();
    let r = scan(&zip_with(&entries, false, false), ".zip");
    let limits = r["limits_exceeded"].as_array().unwrap();
    assert!(limits
        .iter()
        .any(|l| l.as_str().unwrap().contains("member limit")));
}

#[test]
fn gzip_and_tar_wrappers() {
    let r = scan(&gzip(PICKLE), ".gz");
    assert_eq!(r["format"], "gzip+pickle");
    let mut tar = vec![0u8; 512];
    tar[..8].copy_from_slice(b"data.pkl");
    tar[124..135].copy_from_slice(format!("{:011o}", PICKLE.len()).as_bytes());
    tar[156] = b'0';
    tar[257..262].copy_from_slice(b"ustar");
    tar.extend_from_slice(PICKLE);
    tar.resize(512 * 2 + 1024, 0);
    let r = scan(&tar, ".tar");
    assert_eq!(r["format"], "tar+pickle");
    assert!(all_globals(&r).contains(&"posix.system".to_string()));
}

#[test]
fn npy_object_array_payload_is_walked() {
    let r = scan(&npy("'|O'", PICKLE), ".npy");
    assert_eq!(r["format"], "npy+pickle");
    assert!(all_globals(&r).contains(&"posix.system".to_string()));
    let r = scan(&npy("'<f4'", &[0; 4]), ".npy");
    assert_eq!(r["format"], "npy");
    assert!(r["nested"].as_array().unwrap().is_empty());
}

#[test]
fn npz_with_large_object_member_is_scanned() {
    let mut payload = PICKLE.to_vec();
    payload.extend(std::iter::repeat(0u8).take(2_000_000)); // > 1 MB, named .npy
    let r = scan(
        &zip(&[
            ("arr_0.npy", &npy("'|O'", &payload)),
            ("arr_1.npy", &npy("'<f4'", &[0; 4])),
        ]),
        ".npz",
    );
    assert_eq!(r["format"], "npz+pickle");
    assert!(all_globals(&r).contains(&"posix.system".to_string()));
}

const LAMBDA_CONFIG: &str = r#"{"module": "keras", "class_name": "Sequential", "registered_name": null,
  "config": {"layers": [{"module": "keras.layers", "class_name": "Lambda", "registered_name": null,
  "config": {"function": {"class_name": "__lambda__", "config": {"code": "x: __import__('os').system('echo hi')"}}}}]}}"#;

#[test]
fn keras_zip_config_is_analysed() {
    let r = scan(
        &zip(&[
            ("metadata.json", br#"{"keras_version": "3.0.0"}"#),
            ("config.json", LAMBDA_CONFIG.as_bytes()),
            ("model.weights.h5", keras::HDF5_MAGIC),
        ]),
        ".keras",
    );
    assert_eq!(r["format"], "keras");
    let cfg = r["nested"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["member"] == "config.json")
        .unwrap();
    assert_eq!(cfg["report"]["format"], "keras_config");
    assert!(!cfg["report"]["keras"]["dangerous"]
        .as_array()
        .unwrap()
        .is_empty());
}

#[test]
fn hdf5_with_keras_config() {
    let mut raw = keras::HDF5_MAGIC.to_vec();
    raw.extend_from_slice(b"\0\0model_config\0");
    raw.extend_from_slice(LAMBDA_CONFIG.as_bytes());
    raw.extend_from_slice(&[0; 32]);
    let r = scan(&raw, ".h5");
    assert_eq!(r["format"], "keras_h5");
    assert_eq!(r["extension_mismatch"], false);
    assert!(r["keras"]["lambda_layers"].as_array().unwrap().len() >= 1);
}

#[test]
fn onnx_external_data_is_reported() {
    // ModelProto{ir_version: 8, graph: {initializer: {name "w", external_data {location: "../x"}, data_location: 1}}}
    let entry = [b"\x0a\x08location\x12\x04../x".as_slice()].concat();
    let mut tensor = b"\x42\x01w".to_vec();
    tensor.push(0x6a);
    tensor.push(entry.len() as u8);
    tensor.extend_from_slice(&entry);
    tensor.extend_from_slice(b"\x70\x01");
    let mut graph = vec![0x2a, tensor.len() as u8];
    graph.extend_from_slice(&tensor);
    let mut raw = b"\x08\x08\x3a".to_vec();
    raw.push(graph.len() as u8);
    raw.extend_from_slice(&graph);
    let r = scan(&raw, ".onnx");
    assert_eq!(r["format"], "onnx", "{r}");
    assert_eq!(r["onnx"]["external_data"][0]["traversal"], true);
    assert_eq!(r["extension_mismatch"], false);
}

#[test]
fn safetensors_and_gguf_still_dispatch() {
    let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut st = (header.len() as u64).to_le_bytes().to_vec();
    st.extend_from_slice(header);
    st.extend_from_slice(&[0; 4]);
    assert_eq!(scan(&st, ".safetensors")["format"], "safetensors");
    let mut g = b"GGUF".to_vec();
    g.extend_from_slice(&3u32.to_le_bytes());
    g.extend_from_slice(&0u64.to_le_bytes());
    g.extend_from_slice(&0u64.to_le_bytes());
    assert_eq!(scan(&g, ".gguf")["format"], "gguf");
}

// ---- deterministic mini-fuzz -------------------------------------------

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn mutate(rng: &mut XorShift, seed: &[u8]) -> Vec<u8> {
    let mut d = seed.to_vec();
    for _ in 0..=rng.below(8) {
        match rng.below(5) {
            0 if !d.is_empty() => {
                let i = rng.below(d.len());
                d[i] = rng.next() as u8;
            }
            1 if !d.is_empty() => {
                let i = rng.below(d.len());
                d[i] ^= 1 << rng.below(8);
            }
            2 => {
                let i = rng.below(d.len() + 1);
                d.insert(i, rng.next() as u8);
            }
            3 if !d.is_empty() => {
                let cut = rng.below(d.len());
                d.truncate(cut);
            }
            _ if d.len() >= 8 => {
                // Overwrite a 4- or 8-byte field with an extreme length.
                let i = rng.below(d.len() - 7);
                let v = [u64::MAX, 0, 0xFFFF_FFFF, 1 << 31, 0x7fff_ffff][rng.below(5)];
                d[i..i + 8].copy_from_slice(&v.to_le_bytes());
            }
            _ => d.push(rng.next() as u8),
        }
    }
    d
}

fn seeds() -> Vec<(Vec<u8>, &'static str)> {
    let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut st = (header.len() as u64).to_le_bytes().to_vec();
    st.extend_from_slice(header);
    st.extend_from_slice(&[0; 4]);
    let mut g = b"GGUF".to_vec();
    g.extend_from_slice(&3u32.to_le_bytes());
    g.extend_from_slice(&0u64.to_le_bytes());
    g.extend_from_slice(&1u64.to_le_bytes());
    g.extend_from_slice(&4u64.to_le_bytes());
    g.extend_from_slice(b"name");
    g.extend_from_slice(&8u32.to_le_bytes());
    g.extend_from_slice(&2u64.to_le_bytes());
    g.extend_from_slice(b"ok");
    let mut h5 = keras::HDF5_MAGIC.to_vec();
    h5.extend_from_slice(LAMBDA_CONFIG.as_bytes());
    vec![
        (PICKLE.to_vec(), ".pkl"),
        (b"cos\nsystem\n(S'x'\ntR.".to_vec(), ".pkl"),
        (zip(&[("archive/data.pkl", PICKLE)]), ".pt"),
        (zip_with(&[("data.pkl", PICKLE)], false, true), ".pt"),
        (gzip(PICKLE), ".gz"),
        (npy("'|O'", PICKLE), ".npy"),
        (st, ".safetensors"),
        (g, ".gguf"),
        (h5, ".h5"),
        (LAMBDA_CONFIG.as_bytes().to_vec(), ".json"),
        (b"\x08\x08\x3a\x04\x0a\x02\x22\x00".to_vec(), ".onnx"),
    ]
}

#[test]
fn mini_fuzz_mutated_seeds_never_panic() {
    let mut rng = XorShift(0x5eed_1234_abcd_ef01);
    let seeds = seeds();
    for i in 0..4000 {
        let (seed, ext) = &seeds[i % seeds.len()];
        let data = mutate(&mut rng, seed);
        let report = scan_bytes(&data, ext);
        // Serialisation must also hold up (this is what sentinel-parse prints).
        serde_json::to_string(&report).unwrap();
    }
}

#[test]
fn mini_fuzz_random_bytes_never_panic() {
    let mut rng = XorShift(0x0bad_cafe_dead_beef);
    for i in 0..3000 {
        let len = rng.below(600);
        let mut data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        // Give some inputs a real magic so the deeper parsers run.
        let magics: [&[u8]; 8] = [
            b"GGUF",
            b"PK\x03\x04",
            b"\x1f\x8b",
            b"\x93NUMPY",
            b"\x80\x04",
            keras::HDF5_MAGIC,
            b"\x08\x07\x3a",
            b"{",
        ];
        if i % 2 == 0 {
            let m = magics[rng.below(magics.len())];
            data.splice(0..0, m.iter().copied());
        }
        let _ = pickle::walk(&data);
        let _ = crate::gguf::scan_bytes(&data);
        let _ = crate::safetensors::scan_bytes(&data);
        let _ = onnx::scan_bytes(&data);
        let _ = keras::analyze_config_bytes(&data);
        let _ = npy::parse_header(&data);
        let _ = scan_bytes(&data, "");
    }
}
