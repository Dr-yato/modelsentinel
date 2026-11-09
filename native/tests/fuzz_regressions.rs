//! Inputs found by cargo-fuzz, kept as regression tests.

use std::time::Instant;

/// 59 bytes of alternating DUP / STACK_GLOBAL. Each STACK_GLOBAL joined its
/// operands into a longer name and DUP copied it, doubling memory every two
/// opcodes: the walker held 2.2 GB before libFuzzer killed it.
#[test]
fn dup_stack_global_chain_stays_small() {
    let data = include_bytes!("data/pickle-dup-stack-global-oom.bin");
    let started = Instant::now();
    let report = sentinel_native::pickle::walk_bounded(data, 200_000, None);
    let json = serde_json::to_string(&report).expect("report serializes");
    assert!(json.len() < 2_000_000, "report is {} bytes", json.len());
    assert!(started.elapsed().as_secs() < 5);
}
