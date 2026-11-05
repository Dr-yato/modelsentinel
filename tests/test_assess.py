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
