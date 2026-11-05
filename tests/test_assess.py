"""Combined risk assessment (feature 7) and the no-overclaiming wording rule."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelsentinel.assess import combine, run_assessment, wording_problems
from modelsentinel.cli import main
from modelsentinel.fixtureslib import write_fixtures
from modelsentinel.pipeline import scan_source
from modelsentinel.redteam import run_redteam
from modelsentinel.refdiff import run_refdiff
from modelsentinel.triggersearch import run_trigger_search


def test_worst_component_wins_and_each_explains_itself():
    clean = run_redteam("clean:coder", runs=3, task_limit=6, progress=False)
    planted = run_redteam("planted:coder", runs=3, task_limit=6, progress=False)
    record = combine([clean, planted], source="mixed")
    assert record["verdict"] == "FAIL"
    kinds = [c["verdict"] for c in record["assessment"]["components"]]
    assert kinds[0] == "FAIL", "failing components are listed first"
    assert all(c["explanation"] for c in record["assessment"]["components"])
    assert "backdoor_effect" in record["assessment"]["components"][0]["explanation"]


def test_pass_says_no_evidence_never_safe():
    record = combine([run_redteam("clean:coder", runs=3, task_limit=6, progress=False)], source="clean")
    assert record["verdict"] == "PASS"
    assert record["summary"].startswith("No evidence of a backdoor found")
    assert "not proof that the model is safe" in record["summary"]
    assert wording_problems(record["summary"]) == []


@pytest.mark.parametrize("text, bad", [
    ("This model is proven safe.", True),
    ("Verified clean by ModelSentinel.", True),
    ("There is no backdoor in this model.", True),
    ("No evidence of a backdoor found by 3 analyses. This is not proof that the model is safe.", False),
    ("No evidence of backdoor found in the searched trigger space.", False),
    ("A non-significant result does not prove the model has no backdoor.", False),
    ("No tool can prove a model is clean; this model is safe.", True),
])
def test_wording_rule(text, bad):
    assert bool(wording_problems(text)) is bad


def test_every_analysis_summary_obeys_the_wording_rule(tmp_path: Path):
    fixtures = write_fixtures(tmp_path / "fx")
    records = [
        scan_source(str(fixtures["clean_gguf"]), persist=False),
        scan_source(str(fixtures["pkl"]), persist=False),
        run_redteam("clean:coder", runs=3, task_limit=6, progress=False),
        run_redteam("planted:coder", runs=3, task_limit=6, progress=False),
        run_trigger_search("clean:coder", budget=400),
        run_refdiff("clean:coder", "clean:coder", task_limit=6, runs=2, progress=False),
    ]
    for record in records:
        assert wording_problems(record["summary"]) == [], (record["kind"], record["summary"])
        assert wording_problems(record.get("limitation") or "") == [], record["kind"]
    assert combine(records, source="all")["verdict"] == "FAIL"


def test_run_assessment_on_fixtures_and_cli(tmp_path: Path, capsys):
    record, components = run_assessment("planted:coder", reference="clean:coder", policy={})
    assert record["verdict"] == "FAIL"
    kinds = {c["kind"] for c in components}
    assert {"redteam", "triggersearch", "refdiff"} <= kinds
    db = str(tmp_path / "s.db")
    assert main(["--db", db, "assess", "clean:coder", "--no-trigger-search"]) == 0
    out = capsys.readouterr().out
    assert "No evidence of a backdoor found" in out and "PASS" in out


def test_reports_render_every_advanced_section():
    from modelsentinel.report import render

    record, components = run_assessment("planted:coder", reference="clean:coder", policy={})
    html = render(record, "html")
    assert "<h2>Assessment</h2>" in html
    by_kind = {c["kind"]: c for c in components}
    assert "<h2>Trigger search</h2>" in render(by_kind["triggersearch"], "html")
    assert "Versus reference" in render(by_kind["refdiff"], "html")
    for component in [record, *components]:
        assert render(component, "sarif") and render(component, "json")


def test_assess_includes_weight_diff_and_renders_it(tmp_path: Path):
    np = pytest.importorskip("numpy")
    import json
    import struct

    from modelsentinel.assess import combine
    from modelsentinel.report import render
    from modelsentinel.weightdiff import weight_diff

    def write(path: Path, tensors: dict) -> None:
        header, blobs, offset = {}, [], 0
        for name, array in tensors.items():
            raw = array.astype("<f4").tobytes()
            header[name] = {"dtype": "F32", "shape": list(array.shape), "data_offsets": [offset, offset + len(raw)]}
            blobs.append(raw)
            offset += len(raw)
        meta = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(meta)) + meta + b"".join(blobs))

    rng = np.random.default_rng(0)
    base = {f"model.layers.{i}.self_attn.q_proj.weight": rng.normal(0, 0.02, (64, 64)) for i in range(6)}
    target = dict(base)
    u, v = rng.normal(0, 1, (64, 2)), rng.normal(0, 1, (2, 64))
    target["model.layers.3.self_attn.q_proj.weight"] = base["model.layers.3.self_attn.q_proj.weight"] + 0.01 * (u @ v)
    write(tmp_path / "base.safetensors", base)
    write(tmp_path / "target.safetensors", target)
    wd = weight_diff(str(tmp_path / "target.safetensors"), str(tmp_path / "base.safetensors"), policy={},
                     persist=False)
    record = combine([wd], source="target")
    assert record["verdict"] in {"WARN", "FAIL"}
    assert "Weight diff" in record["assessment"]["components"][0]["title"]
    html = render(wd, "html")
    assert "<h2>Weight diff</h2>" in html and "layers.3" in html
