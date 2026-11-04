//! Pickle walker tests. Every vector is inert bytes; nothing is unpickled.

use super::*;

const ARG: &[u8] = b"echo SENTINEL_BENIGN_TEST";

fn has_global(r: &PickleReport, module: &str, name: &str) -> bool {
    r.globals
        .iter()
        .any(|g| g.module == module && g.name == name)
}

fn structured(r: &PickleReport, module: &str, name: &str) -> bool {
    r.globals
        .iter()
        .any(|g| g.module == module && g.name == name && g.confidence == "structured")
}

/// `pickle.dumps(R(), protocol=p)` where `R.__reduce__` returns
/// `(os.system, ("echo SENTINEL_BENIGN_TEST",))`, captured on macOS
/// (os.system pickles as posix.system there).
fn cpython_vector(protocol: u8) -> Vec<u8> {
    let mut v = Vec::new();
    match protocol {
        0 => {
            v.extend_from_slice(b"cposix\nsystem\np0\n(V");
            v.extend_from_slice(ARG);
            v.extend_from_slice(b"\np1\ntp2\nRp3\n.");
        }
        1 => {
            v.extend_from_slice(b"cposix\nsystem\nq\x00(X\x19\x00\x00\x00");
            v.extend_from_slice(ARG);
            v.extend_from_slice(b"q\x01tq\x02Rq\x03.");
        }
        2 | 3 => {
            v.extend_from_slice(&[0x80, protocol]);
            v.extend_from_slice(b"cposix\nsystem\nq\x00X\x19\x00\x00\x00");
            v.extend_from_slice(ARG);
            v.extend_from_slice(b"q\x01\x85q\x02Rq\x03.");
        }
        _ => {
            v.extend_from_slice(&[0x80, protocol]);
            v.extend_from_slice(b"\x95\x34\x00\x00\x00\x00\x00\x00\x00");
            v.extend_from_slice(b"\x8c\x05posix\x94\x8c\x06system\x94\x93\x94\x8c\x19");
            v.extend_from_slice(ARG);
            v.extend_from_slice(b"\x94\x85\x94R\x94.");
        }
    }
    v
}

#[test]
fn proto2_global_os_system() {
    let report = walk(b"\x80\x02cos\nsystem\n.");
    assert!(report.stopped);
    assert_eq!(report.protocol, Some(2));
    assert_eq!(report.globals.len(), 1);
    assert_eq!(report.globals[0].opcode, "GLOBAL");
    assert!(structured(&report, "os", "system"));
}

#[test]
fn stack_global_is_recorded() {
    let report = walk(b"\x80\x04\x8c\x02os\x8c\x06system\x93.");
    assert!(report.stopped, "{:?}", report.warnings);
    assert_eq!(report.globals.len(), 1);
    assert_eq!(report.globals[0].opcode, "STACK_GLOBAL");
    assert!(structured(&report, "os", "system"));
}

#[test]
fn cpython_protocols_0_to_5_resolve_global_and_reduce() {
    for protocol in 0..=5u8 {
        let data = cpython_vector(protocol);
        let r = walk(&data);
        assert!(r.stopped, "protocol {protocol}: {:?}", r.warnings);
        assert!(structured(&r, "posix", "system"), "protocol {protocol}");
        assert_eq!(r.reduces.len(), 1, "protocol {protocol}");
        assert_eq!(r.reduces[0].callable, "posix.system");
        assert!(r.reduces[0].args_preview.contains("SENTINEL_BENIGN_TEST"));
        assert_eq!(r.trailing_bytes, 0);
        if protocol >= 2 {
            assert_eq!(r.protocol, Some(protocol));
        }
    }
}

#[test]
fn inst_records_global_and_call() {
    let r = walk(b"(S'echo'\nios\nsystem\n.");
    assert!(r.stopped);
    assert_eq!(r.globals[0].opcode, "INST");
    assert!(structured(&r, "os", "system"));
    assert_eq!(r.reduces[0].callable, "os.system");
}

#[test]
fn obj_records_class_and_args() {
    let r = walk(b"(cos\nsystem\nS'echo hi'\no.");
    assert!(r.stopped);
    assert!(structured(&r, "os", "system"));
    assert_eq!(r.reduces[0].callable, "os.system");
    assert!(r.reduces[0].args_preview.contains("echo hi"));
}

#[test]
fn newobj_and_newobj_ex() {
    let r = walk(b"\x80\x02cos\nsystem\n)\x81.");
    assert!(r.stopped);
    assert_eq!(r.reduces[0].callable, "os.system");
    let r = walk(b"\x80\x04\x8c\x02os\x8c\x06system\x93)}\x92.");
    assert!(r.stopped, "{:?}", r.warnings);
    assert_eq!(r.reduces[0].callable, "os.system");
}

#[test]
fn build_after_reduce_keeps_going() {
    let r = walk(b"\x80\x02ccollections\nOrderedDict\n)R}b.");
    assert!(r.stopped);
    assert!(structured(&r, "collections", "OrderedDict"));
}

#[test]
fn memoize_and_binget_resolve_stack_global_operands() {
    // MEMOIZE assigns 0 and 1; BINGET pushes them back for STACK_GLOBAL.
    let r = walk(b"\x80\x04\x8c\x02os\x940\x8c\x06system\x940h\x00h\x01\x93.");
    assert!(r.stopped, "{:?}", r.warnings);
    assert!(structured(&r, "os", "system"));
}

#[test]
fn memoize_uses_entry_count_not_max_index() {
    // PUT 5 first, then MEMOIZE twice. CPython assigns indices 1 and 2.
    let r = walk(b"\x80\x04\x8c\x05dummyq\x050\x8c\x02os\x940\x8c\x06system\x940h\x01h\x02\x93.");
    assert!(r.stopped, "{:?}", r.warnings);
    assert!(structured(&r, "os", "system"), "{:?}", r.globals);
}

#[test]
fn large_memo_indices_still_resolve() {
    // LONG_BINPUT/LONG_BINGET with indices above 100k used to be dropped.
    let mut d = b"\x80\x04\x8c\x02osr\x40\x0d\x03\x000".to_vec();
    d.extend_from_slice(b"\x8c\x06systemr\x41\x0d\x03\x000");
    d.extend_from_slice(b"j\x40\x0d\x03\x00j\x41\x0d\x03\x00\x93.");
    let r = walk(&d);
    assert!(r.stopped, "{:?}", r.warnings);
    assert!(structured(&r, "os", "system"));
}

#[test]
fn text_get_and_put() {
    let r = walk(b"S'os'\np7\n0S'system'\np8\n0g7\ng8\n\x93.");
    assert!(r.stopped, "{:?}", r.warnings);
    assert!(structured(&r, "os", "system"));
}

#[test]
fn ext_opcodes_are_recorded() {
    let r = walk(b"\x80\x02\x82\x01\x83\x02\x00\x84\x03\x00\x00\x00.");
    assert!(r.stopped);
    assert_eq!(r.extensions, vec![1, 2, 3]);
}

#[test]
fn persistent_ids_are_recorded() {
    let r = walk(b"Pstorage0\n.");
    assert!(r.stopped);
    assert_eq!(r.persistent_ids, vec!["storage0".to_string()]);
    let r = walk(b"\x80\x02X\x03\x00\x00\x00barQ.");
    assert!(r.stopped);
    assert_eq!(r.persistent_ids, vec!["bar".to_string()]);
}

#[test]
fn protocol5_frame_bytearray_and_buffers() {
    let mut d = b"\x80\x05\x95\x0d\x00\x00\x00\x00\x00\x00\x00".to_vec();
    d.extend_from_slice(b"\x96\x03\x00\x00\x00\x00\x00\x00\x00abc0");
    d.extend_from_slice(b"\x97\x98.");
    let r = walk(&d);
    assert!(r.stopped, "{:?}", r.warnings);
    assert!(r.warnings.iter().any(|w| w.contains("NEXT_BUFFER")));
    assert!(r.warnings.iter().any(|w| w.contains("READONLY_BUFFER")));
}

#[test]
fn scalar_and_container_opcodes() {
    let d = b"\x80\x04(K\x01M\x02\x00J\x03\x00\x00\x00\x8a\x01\x05\x8b\x01\x00\x00\x00\x05G\x00\x00\x00\x00\x00\x00\x00\x00F1.5\nI01\nI00\nI42\nL7L\n\x88\x89Nl]\x8f(\x90\x91}(K\x01K\x02u)\x86\x87C\x01xB\x01\x00\x00\x00y\x8e\x01\x00\x00\x00\x00\x00\x00\x00z0.";
    let r = walk(d);
    assert!(r.stopped, "{:?}", r.warnings);
    assert!(r.unknown_opcodes.is_empty());
}

#[test]
fn every_truncation_is_safe_and_unstopped() {
    for protocol in 0..=5u8 {
        let full = cpython_vector(protocol);
        for cut in 0..full.len() {
            let r = walk(&full[..cut]);
            assert!(!r.stopped, "protocol {protocol} cut {cut}");
        }
    }
}

#[test]
fn concatenated_pickles_are_all_walked() {
    let mut d = b"\x80\x02ccollections\nOrderedDict\n)R.".to_vec();
    d.extend_from_slice(&cpython_vector(4));
    let r = walk(&d);
    assert!(r.stopped);
    assert_eq!(r.pickle_count, 2);
    assert!(structured(&r, "collections", "OrderedDict"));
    assert!(structured(&r, "posix", "system"));
}

#[test]
fn proto0_pickle_after_proto2_is_walked() {
    let mut d = b"\x80\x02N.".to_vec();
    d.extend_from_slice(b"cos\nsystem\n(S'x'\ntR.");
    let r = walk(&d);
    assert_eq!(r.pickle_count, 2);
    assert!(structured(&r, "os", "system"));
}

#[test]
fn trailing_raw_bytes_are_counted_not_walked() {
    let mut d = b"\x80\x02N.".to_vec();
    d.extend_from_slice(&[0u8, 1, 2, 3, 0xff]);
    let r = walk(&d);
    assert!(r.stopped);
    assert_eq!(r.pickle_count, 1);
    assert_eq!(r.trailing_bytes, 5);
    assert!(r.unknown_opcodes.is_empty());
}

#[test]
fn filler_string_larger_than_keep_limit_does_not_end_walk() {
    let n: u32 = 2_000_000;
    let mut d = b"\x80\x04X".to_vec();
    d.extend_from_slice(&n.to_le_bytes());
    d.extend(std::iter::repeat(b'A').take(n as usize));
    d.extend_from_slice(b"0\x8c\x02os\x8c\x06system\x93.");
    let r = walk(&d);
    assert!(r.stopped, "{:?}", r.warnings);
    assert!(structured(&r, "os", "system"));
}

#[test]
fn unicode_module_names_are_kept_verbatim() {
    // Cyrillic "о" + "s": must surface as its own module, not be dropped.
    let r = walk("\u{80}".as_bytes()); // non-pickle noise first: must not panic
    assert!(!r.stopped);
    let r = walk(b"\x80\x04\x8c\x03\xd0\xbes\x8c\x06system\x93.");
    assert!(r.stopped);
    assert!(structured(&r, "\u{43e}s", "system"));
}

#[test]
fn opcode_limit_is_reported() {
    let r = walk_bounded(&cpython_vector(0), 3, None);
    assert!(!r.stopped);
    assert!(r
        .limit_exceeded
        .as_deref()
        .unwrap_or("")
        .contains("opcode limit"));
}

#[test]
fn deadline_in_the_past_stops_long_walks() {
    let mut d = b"\x80\x02".to_vec();
    d.extend(std::iter::repeat(b'N').take(200_000));
    d.push(b'.');
    let r = walk_with_deadline(&d, Instant::now());
    assert!(r.limit_exceeded.is_some());
}

#[test]
fn literal_scan_covers_bytes_after_a_desync() {
    let mut d = b"\x80\x02ccollections\nOrderedDict\n\xff".to_vec();
    d.extend_from_slice(b"junk cos\nsystem\n");
    let r = walk(&d);
    assert!(!r.stopped);
    assert!(structured(&r, "collections", "OrderedDict"));
    assert!(r
        .globals
        .iter()
        .any(|g| g.module == "os" && g.name == "system" && g.confidence == "heuristic"));
}

#[test]
fn literal_scan_skips_walked_argument_strings() {
    // "x\nsystem\n" lives inside a walked string and must not become a global.
    let r = walk(b"\x80\x02cos\npath\nX\x0c\x00\x00\x00cx\nsystem\nab.");
    assert!(r.stopped);
    assert!(!has_global(&r, "x", "system"));
}

#[test]
fn stack_global_with_non_string_operands_is_unresolved() {
    let r = walk(b"\x80\x04K\x01K\x02\x93.");
    assert!(r.stopped);
    assert_eq!(r.globals[0].confidence, "unresolved");
}

#[test]
fn unknown_opcode_stops_with_record() {
    let r = walk(b"\x80\x02\xff.");
    assert!(!r.stopped);
    assert_eq!(r.unknown_opcodes.len(), 1);
}

#[test]
fn huge_declared_lengths_do_not_allocate_or_panic() {
    let r = walk(b"\x80\x04\x8d\xff\xff\xff\xff\xff\xff\xff\x7f");
    assert!(!r.stopped);
    let r = walk(b"\x80\x04\x8e\xff\xff\xff\xff\xff\xff\xff\xff");
    assert!(!r.stopped);
    let r = walk(b"T\xff\xff\xff\xff");
    assert!(!r.stopped);
}
