//! Keras config and HDF5 search tests. The base64 payloads are
//! `marshal.dumps(code)` of tiny lambdas; they are only decoded to bytes.

use super::*;
use std::io::Cursor;
use std::time::Duration;

/// `lambda x: __import__('os').system('echo SENTINEL_BENIGN_TEST')`
const EVIL_CODE: &str = "4wEAAAAAAAAAAAAAAAMAAAADAAAA8zYAAACAAFwBAAAAAAAAAABSADQBAAAAAAAAUAMAAAAAAAAAAAAAAAAAAAAAAABSATQBAAAAAAAAIwApAtoCb3P6GWVjaG8gU0VOVElORUxfQkVOSUdOX1RFU1QpAtoKX19pbXBvcnRfX9oGc3lzdGVtqQHaAXhzAQAAACbaCDxzdHJpbmc+2gg8bGFtYmRhPnIJAAAAAwAAAHMVAAAAgACMSpB00wwc1wwj0Qwj0CQ/1AxA8wAAAAA=";
/// `lambda x: x*2+1`
const BENIGN_CODE: &str = "4wEAAAAAAAAAAAAAAAIAAAADAAAA8yIAAACAAFYAXgIsBQAAAAAAAAAAAABeASwAAAAAAAAAAAAAACMAKQHpAgAAAKkAqQHaAXhzAQAAACbaCDxzdHJpbmc+2gg8bGFtYmRhPnIHAAAABAAAAHMMAAAAgACIQYhhjUOQAY5F8wAAAAA=";

fn keras2_model(code: &str) -> Value {
    json!({
        "class_name": "Sequential",
        "config": {"name": "seq", "layers": [
            {"class_name": "Dense", "config": {"name": "d", "units": 1}},
            {"class_name": "Lambda", "config": {
                "name": "lam", "function": [code, null, null],
                "function_type": "lambda", "module": "__main__"
            }}
        ]}
    })
}

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

#[test]
fn base64_roundtrip_and_rejects() {
    assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
    assert_eq!(base64_decode("aGVsbG8").unwrap(), b"hello");
    assert!(base64_decode("not base64!").is_none());
    assert!(base64_decode("ab").is_none());
}

#[test]
fn marshal_names_found_only_in_evil_lambda() {
    let evil = marshal_names(&base64_decode(EVIL_CODE).unwrap());
    assert!(evil.contains(&"os".to_string()), "{evil:?}");
    assert!(evil.contains(&"system".to_string()));
    assert!(evil.contains(&"__import__".to_string()));
    let benign = marshal_names(&base64_decode(BENIGN_CODE).unwrap());
    assert!(benign.is_empty(), "{benign:?}");
}

#[test]
fn keras2_lambda_with_dangerous_code() {
    let r = analyze_config(&keras2_model(EVIL_CODE));
    assert_eq!(r["lambda_layers"].as_array().unwrap().len(), 1);
    let dangerous = r["dangerous"].as_array().unwrap();
    assert!(dangerous
        .iter()
        .any(|d| d.as_str().unwrap().contains("system")));
}

#[test]
fn keras2_lambda_benign_is_listed_without_danger() {
    let r = analyze_config(&keras2_model(BENIGN_CODE));
    assert_eq!(r["lambda_layers"].as_array().unwrap().len(), 1);
    assert!(r["dangerous"].as_array().unwrap().is_empty());
    // "__main__" is the Lambda's own module field and is reported as custom.
    assert!(!r["custom_objects"].as_array().unwrap().is_empty());
}

#[test]
fn keras3_lambda_and_modules() {
    let cfg = json!({
        "module": "keras", "class_name": "Functional", "registered_name": null,
        "config": {"layers": [
            {"module": "keras.layers", "class_name": "Dense", "registered_name": null,
             "config": {"activation": {"module": "builtins", "class_name": "function",
                                       "config": "relu", "registered_name": "function"}}},
            {"module": "keras.layers", "class_name": "Lambda", "registered_name": null,
             "config": {"function": {"class_name": "__lambda__",
                                     "config": {"code": EVIL_CODE, "defaults": null, "closure": null}}}},
            {"module": "my_pkg.layers", "class_name": "Custom", "registered_name": "MyPkg>Custom", "config": {}},
            {"module": "os", "class_name": "system", "registered_name": null, "config": {}},
            {"module": "builtins", "class_name": "function", "config": "exec", "registered_name": "function"}
        ]}
    });
    let r = analyze_config(&cfg);
    let dangerous: Vec<&str> = r["dangerous"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .collect();
    assert!(
        dangerous.iter().any(|d| d.contains("lambda uses system")),
        "{dangerous:?}"
    );
    assert!(
        dangerous.iter().any(|d| d.starts_with("module os")),
        "{dangerous:?}"
    );
    assert!(
        dangerous.iter().any(|d| d.contains("exec")),
        "{dangerous:?}"
    );
    assert!(
        !dangerous.iter().any(|d| d.contains("relu")),
        "{dangerous:?}"
    );
    let custom = r["custom_objects"].as_array().unwrap();
    assert!(custom.iter().any(|c| c["module"] == "my_pkg.layers"));
    assert!(r["lambda_layers"].as_array().unwrap().len() >= 1);
}

#[test]
fn stock_config_is_clean() {
    let cfg = json!({"module": "keras", "class_name": "Sequential", "registered_name": null,
                     "config": {"layers": [{"module": "keras.layers", "class_name": "Dense",
                                            "config": {"units": 2}, "registered_name": null}]}});
    let r = analyze_config(&cfg);
    assert!(r["lambda_layers"].as_array().unwrap().is_empty());
    assert!(r["custom_objects"].as_array().unwrap().is_empty());
    assert!(r["dangerous"].as_array().unwrap().is_empty());
}

#[test]
fn non_keras_json_is_ignored() {
    assert!(analyze_config_bytes(br#"{"model_type": "llama"}"#).is_none());
    assert!(analyze_config_bytes(b"not json").is_none());
}

fn fake_h5(config: &Value, pad: usize) -> Vec<u8> {
    let mut raw = HDF5_MAGIC.to_vec();
    raw.extend(std::iter::repeat(0u8).take(pad));
    raw.extend_from_slice(b"model_config\0\0\0");
    raw.extend_from_slice(config.to_string().as_bytes());
    raw.extend(std::iter::repeat(0u8).take(64));
    raw
}

#[test]
fn h5_config_is_extracted_and_analysed() {
    let raw = fake_h5(&keras2_model(EVIL_CODE), 300);
    let r = scan_h5(&mut Cursor::new(&raw), raw.len() as u64, deadline()).unwrap();
    assert_eq!(r["configs_parsed"], 1);
    assert_eq!(r["keras_model_config"], true);
    assert!(r["lambda_marker_hits"].as_u64().unwrap() >= 1);
    assert!(!r["dangerous"].as_array().unwrap().is_empty());
}

#[test]
fn h5_without_lambda_is_quiet() {
    let cfg = json!({"class_name": "Sequential", "config": {"layers": [{"class_name": "Dense", "config": {}}]}});
    let raw = fake_h5(&cfg, 10);
    let r = scan_h5(&mut Cursor::new(&raw), raw.len() as u64, deadline()).unwrap();
    assert_eq!(r["lambda_marker_hits"], 0);
    assert!(r["dangerous"].as_array().unwrap().is_empty());
}

#[test]
fn h5_marker_beyond_json_window_is_counted() {
    // Marker straddles the boundary between the first window and a chunk.
    let mut raw = HDF5_MAGIC.to_vec();
    raw.resize(H5_JSON_WINDOW as usize - 5, 0);
    raw.extend_from_slice(b"\"class_name\": \"Lambda\"");
    raw.extend(std::iter::repeat(0u8).take(100));
    let r = scan_h5(&mut Cursor::new(&raw), raw.len() as u64, deadline()).unwrap();
    assert_eq!(r["lambda_marker_hits"], 1);
}

#[test]
fn brace_match_handles_strings_and_escapes() {
    assert_eq!(brace_match(br#"{"a": "}\"{", "b": {}} tail"#), Some(22));
    assert_eq!(brace_match(b"{{"), None);
}
