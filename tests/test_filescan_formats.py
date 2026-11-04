"""File-format coverage for the Rust parser and its Python classification.

Every payload is benign (an ``echo`` into a marker file) and is never loaded:
the tests check that the scanner flags it AND that the marker file does not
exist afterwards. Paths use pathlib so the suite runs on Windows, macOS and
Linux alike.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from modelsentinel import fixtureslib as fx
from modelsentinel.filescan import classify_global, findings_from_report
from modelsentinel.formatscan import keras_findings, limit_findings, onnx_findings
from modelsentinel.native import find_binary, parse_file
from modelsentinel.pipeline import scan_source

HAVE_PARSER = find_binary("sentinel-parse") is not None

needs_parser = pytest.mark.skipif(not HAVE_PARSER, reason="sentinel-parse is not built")


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _scan(path: Path) -> dict:
    return scan_source(str(path), persist=False)


def _codes(result: dict) -> set[str]:
    return {item["code"] for item in result["findings"]}


def _categories(result: dict) -> set[str]:
    return {item["category"] for item in result["findings"]}


# ---------------------------------------------------------------- pickles ---


@needs_parser
@pytest.mark.parametrize("protocol", [0, 1, 2, 3, 4, 5])
def test_every_pickle_protocol_fails_without_running(tmp_path: Path, protocol: int) -> None:
    marker = tmp_path / "marker.txt"
    path = _write(tmp_path, f"model_p{protocol}.bin", fx.marker_pickle_bytes(marker, protocol))
    result = _scan(path)
    assert result["verdict"] == "FAIL"
    assert "malicious_pickle" in _categories(result)
    assert not marker.exists()


@needs_parser
def test_torch_zip_with_nested_zip_payload_is_caught(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    payload = fx.marker_pickle_bytes(marker)
    path = _write(tmp_path, "pytorch_model.bin", fx.nested_torch_zip_bytes(payload))
    result = _scan(path)
    assert result["verdict"] == "FAIL"
    locations = [item["location"] for item in result["findings"] if item["category"] == "malicious_pickle"]
    assert any("inner.zip!archive/data.pkl" in loc for loc in locations), locations
    assert not marker.exists()


@needs_parser
def test_concatenated_pickles_are_all_checked(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    data = pickle.dumps({"ok": 1}, protocol=2) + fx.marker_pickle_bytes(marker, 2)
    result = _scan(_write(tmp_path, "legacy.pt", data))
    assert result["verdict"] == "FAIL"
    assert not marker.exists()


@needs_parser
def test_clean_pickle_of_plain_data_passes_file_checks(tmp_path: Path) -> None:
    data = pickle.dumps({"weights": [1.0, 2.0], "names": ("a", "b")}, protocol=4)
    result = _scan(_write(tmp_path, "state.pkl", data))
    assert result["verdict"] != "FAIL"
    assert "malicious_pickle" not in _categories(result)


def test_lookalike_module_names_are_normalised() -> None:
    # Fullwidth "ｏｓ" NFKC-normalises to "os".
    assert classify_global("\uff4f\uff53", "system")[1] == "malicious_pickle"
    assert classify_global("os\r", "system")[1] == "malicious_pickle"
    assert classify_global("collections", "OrderedDict")[1] == "framework_global"


def test_unresolved_stack_global_is_high() -> None:
    report = {
        "format": "pickle",
        "pickle": {
            "globals": [{"module": "memo-missing:7", "name": "x", "opcode": "STACK_GLOBAL", "offset": 3, "confidence": "unresolved"}]
        },
    }
    findings = findings_from_report(report, "x.pkl")
    assert any(f.code == "unresolved_stack_global" and f.severity == "high" for f in findings)


# ------------------------------------------------------------------ numpy ---


@needs_parser
def test_npy_object_array_with_dangerous_pickle_fails(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    path = _write(tmp_path, "arr.npy", fx.object_npy_bytes(fx.marker_pickle_bytes(marker, 3)))
    result = _scan(path)
    assert result["verdict"] == "FAIL"
    assert "numpy_object_array" in _codes(result)
    assert not marker.exists()


@needs_parser
def test_npz_with_object_member_fails_and_numeric_npy_passes(tmp_path: Path) -> None:
    assert _scan(_write(tmp_path, "a.npz", fx.npz_bytes()))["verdict"] == "FAIL"
    numeric = _scan(_write(tmp_path, "b.npy", fx.npy_bytes("'<f4'", b"\x00" * 16, "(4,)")))
    assert numeric["verdict"] == "PASS", numeric["findings"]


@needs_parser
def test_pickle_renamed_to_npy_is_an_extension_mismatch(tmp_path: Path) -> None:
    result = _scan(_write(tmp_path, "arr.npy", fx.benign_pickle_bytes()))
    assert "extension_mismatch" in _codes(result)
    assert result["verdict"] == "FAIL"


# ------------------------------------------------------------------ keras ---


@needs_parser
def test_keras_zip_with_dangerous_lambda_fails(tmp_path: Path) -> None:
    result = _scan(_write(tmp_path, "model.keras", fx.keras_zip_bytes()))
    assert result["verdict"] == "FAIL"
    assert {"keras_dangerous_code", "keras_lambda_layer"} <= _codes(result)


@needs_parser
def test_keras_benign_lambda_warns(tmp_path: Path) -> None:
    config = fx.keras_lambda_config(dangerous=False)
    result = _scan(_write(tmp_path, "model.keras", fx.keras_zip_bytes(config)))
    assert result["verdict"] == "WARN", result["findings"]
    assert "keras_lambda_layer" in _codes(result)
    assert "keras_dangerous_code" not in _codes(result)


@needs_parser
def test_keras_h5_lambda_is_found_by_byte_search(tmp_path: Path) -> None:
    result = _scan(_write(tmp_path, "model.h5", fx.keras_h5_bytes()))
    assert result["verdict"] == "FAIL"
    assert "keras_dangerous_code" in _codes(result)


def test_keras_custom_module_warns() -> None:
    findings = keras_findings(
        {"lambda_layers": [], "dangerous": [], "custom_objects": [{"module": "evil_pkg.layers", "class_name": "X"}]},
        "config.json",
    )
    assert [f.category for f in findings] == ["keras_custom_object"]
    assert findings[0].severity == "medium"


# ------------------------------------------------------------------- onnx ---


@needs_parser
@pytest.mark.parametrize("location", ["../../etc/passwd", "/etc/shadow", "C:\\Windows\\win.ini", "sub\\..\\..\\x"])
def test_onnx_external_data_escape_fails(tmp_path: Path, location: str) -> None:
    result = _scan(_write(tmp_path, "model.onnx", fx.onnx_bytes([location])))
    assert result["verdict"] == "FAIL", result["findings"]
    assert "onnx_external_data_traversal" in _codes(result)


@needs_parser
def test_onnx_relative_external_data_is_info_only(tmp_path: Path) -> None:
    result = _scan(_write(tmp_path, "model.onnx", fx.onnx_bytes(["model.onnx.data"])))
    assert result["verdict"] == "PASS", result["findings"]
    assert "onnx_external_data" in _codes(result)


@needs_parser
def test_onnx_custom_python_op_warns(tmp_path: Path) -> None:
    result = _scan(_write(tmp_path, "model.onnx", fx.onnx_bytes([], op_domain="ai.onnx.contrib")))
    assert result["verdict"] == "WARN"
    assert "onnx_custom_domain" in _codes(result) or "onnx_python_op" in _codes(result)


def test_onnx_python_op_is_high() -> None:
    findings = onnx_findings({"custom_ops": [{"domain": "ai.onnx.contrib", "op_type": "PyOp", "python_op": True}]}, "m.onnx")
    assert findings[0].code == "onnx_python_op" and findings[0].severity == "high"


# ----------------------------------------------------------------- limits ---


@needs_parser
def test_zip_bomb_is_bounded_and_reported(tmp_path: Path) -> None:
    result = _scan(_write(tmp_path, "bomb.pt", fx.zip_bomb_bytes()))
    assert "scan_limit_exceeded" in _codes(result)
    assert result["verdict"] in {"WARN", "FAIL"}


@needs_parser
def test_deep_zip_nesting_is_reported(tmp_path: Path) -> None:
    data = fx.zip_bytes({"data.pkl": fx.benign_pickle_bytes()})
    for _ in range(8):
        data = fx.zip_bytes({"inner.zip": data})
    report = parse_file(_write(tmp_path, "deep.zip", data))
    assert any("nesting" in item for item in report["limits_exceeded"])


def test_limit_findings_survive_parse_errors() -> None:
    report = {"error": "boom", "limits_exceeded": ["time limit reached"]}
    findings = findings_from_report(report, "x")
    assert {f.code for f in findings} == {"scan_limit_exceeded", "parser_error"}
    assert limit_findings({}, "x") == []


# -------------------------------------------------------------- safetensors ---


@needs_parser
def test_safetensors_with_appended_archive_warns(tmp_path: Path) -> None:
    data = fx.safetensors_bytes() + fx.zip_bytes({"data.pkl": b"\x80\x02N."})
    result = _scan(_write(tmp_path, "model.safetensors", data))
    assert "safetensors_trailing_data" in _codes(result)
    assert result["verdict"] == "WARN"
