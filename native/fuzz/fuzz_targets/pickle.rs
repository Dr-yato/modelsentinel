#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = sentinel_native::pickle::walk_bounded(data, 200_000, None);
});
