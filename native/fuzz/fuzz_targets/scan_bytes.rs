#![no_main]
use libfuzzer_sys::fuzz_target;

// Top-level magic-byte dispatcher, including zip/tar/gzip recursion. The first
// byte picks the claimed extension so extension-specific paths run too.
const EXTS: &[&str] = &["", ".pkl", ".pt", ".npy", ".onnx", ".h5", ".keras", ".safetensors", ".gguf"];

fuzz_target!(|data: &[u8]| {
    let (ext, body) = match data.split_first() {
        Some((b, rest)) => (EXTS[*b as usize % EXTS.len()], rest),
        None => ("", data),
    };
    let report = sentinel_native::scan_bytes(body, ext);
    let _ = serde_json_len(&report);
});

fn serde_json_len(report: &sentinel_native::ScanReport) -> usize {
    format!("{report:?}").len()
}
