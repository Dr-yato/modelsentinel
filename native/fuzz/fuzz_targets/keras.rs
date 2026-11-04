#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = sentinel_native::keras::analyze_config_bytes(data);
    let _ = sentinel_native::npy::parse_header(data);
});
