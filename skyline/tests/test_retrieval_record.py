"""Result-record emission (T018) and the contract violations it must never produce (T028).

Spec FR-002..FR-005, SC-004. The record is the *sole* matcher -> evaluator interface (``DEC-002``), so
every test here reads the written record back through the **unmodified** ``naveval.skyline_record``.
Nothing is asserted against a local re-parse: if ``naveval`` would reject it, the test must fail.

The malformed-record cases matter as much as the well-formed one. A relocalizer that reports a
confident position it has no right to is worse than one that crashes, so the contract's job is to
make that unrepresentable -- and these tests prove the reader actually enforces it rather than
merely documenting it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from hsreloc.retrieval.record import QueryRow, RecordError, write_record

from naveval.errors import ContractViolationError
from naveval.skyline_record import load_skyline_record

REF_SOURCE = {"kind": "historical_imagery", "ref_set_id": "refset-synth-sky-v1",
              "dataset_id": "synthetic", "n_references": 9, "total_bytes": 1,
              "covered_area_km2": 0.1, "reference_spacing_m": 1200.0,
              "extraction_mode": "oracle:sim_exact", "evidence_tier": "T1",
              "evidence_caveat": "synthetic fixture"}


def _write(tmp_path, rows, run_id="r1", n_slots=2, **kwargs):
    return write_record(tmp_path, run_id, rows, query_set_id="skyquery-synth-sky-v1",
                        query_set_revision="synth-v1", reference_source=REF_SOURCE,
                        relocalizer_config={"baseline": "ncc"}, n_candidate_slots=n_slots, **kwargs)


def _success(query_id="0"):
    return QueryRow(query_id, 0.0, "SUCCESS", 1234, confidence=0.9, best_score=0.99,
                    second_best_score=0.4, score_margin=0.59, n_viable_candidates=1,
                    est_east_m=100.0, est_north_m=200.0, matched_east_m=100.0,
                    matched_north_m=200.0, candidates=[(100.0, 200.0, 0.99), (5.0, 6.0, 0.4)])


def _rejected(query_id="1"):
    return QueryRow(query_id, 1.0, "REJECTED", 999, confidence=0.1, best_score=0.3,
                    second_best_score=0.29, score_margin=0.01, n_viable_candidates=0,
                    candidates=[(5.0, 6.0, 0.3)])


# --- the well-formed record --------------------------------------------------------------------

def test_record_round_trips_through_the_unmodified_reader(tmp_path):
    _write(tmp_path, [_success(), _rejected()])
    record = load_skyline_record(tmp_path / "r1")
    assert record.manifest.processed_count == 2
    assert [r.outcome for r in record.results] == ["SUCCESS", "REJECTED"]
    assert record.results[0].est_east_m == 100.0
    assert record.results[1].est_east_m is None
    assert len(record.results[0].candidates) == 2


def test_manifest_carries_exactly_one_source_block(tmp_path):
    _write(tmp_path, [_success()])
    manifest = json.loads((tmp_path / "r1" / "manifest.json").read_text(encoding="utf-8"))
    assert "reference_db" not in manifest
    assert manifest["reference_source"]["kind"] == "historical_imagery"
    assert load_skyline_record(tmp_path / "r1").manifest.reference_source["ref_set_id"]


def test_heading_is_never_emitted(tmp_path):
    _write(tmp_path, [_success()])
    result = load_skyline_record(tmp_path / "r1").results[0]
    assert result.est_heading_deg is None and result.heading_source is None


def test_is_target_hardware_defaults_false_and_true_is_refused(tmp_path):
    _write(tmp_path, [_success()])
    assert load_skyline_record(tmp_path / "r1").manifest.is_target_hardware is False
    with pytest.raises(RecordError, match="is_target_hardware"):
        _write(tmp_path, [_success()], run_id="r2", environment={"is_target_hardware": True})


def test_refuses_to_overwrite_a_recorded_run(tmp_path):
    _write(tmp_path, [_success()])
    with pytest.raises(RecordError, match="already exists"):
        _write(tmp_path, [_success()])
    _write(tmp_path, [_success()], overwrite=True)   # explicit opt-in still works


def test_refuses_an_empty_record(tmp_path):
    with pytest.raises(RecordError, match="zero query results"):
        _write(tmp_path, [])


def test_non_finite_values_are_refused_before_they_reach_the_file(tmp_path):
    row = _success()
    row.confidence = float("nan")
    with pytest.raises(RecordError, match="non-finite"):
        _write(tmp_path, [row])


# --- T028 / SC-004: the reader must reject what the contract forbids -----------------------------

def test_success_without_a_position_is_rejected_by_the_reader(tmp_path):
    row = _success()
    row.est_east_m = None
    row.est_north_m = None
    _write(tmp_path, [row])
    with pytest.raises(ContractViolationError, match="requires est_east_m"):
        load_skyline_record(tmp_path / "r1")


def test_a_refusal_carrying_a_position_is_rejected_by_the_reader(tmp_path):
    row = _rejected()
    row.est_east_m = 10.0
    row.est_north_m = 20.0
    _write(tmp_path, [row])
    with pytest.raises(ContractViolationError, match="must not carry a consumable position"):
        load_skyline_record(tmp_path / "r1")


def test_a_manifest_with_neither_source_block_is_rejected_by_the_reader(tmp_path):
    _write(tmp_path, [_success()])
    manifest_path = tmp_path / "r1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["reference_source"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ContractViolationError, match="reference_db or reference_source"):
        load_skyline_record(tmp_path / "r1")


def test_a_truncated_record_is_rejected_by_the_reader(tmp_path):
    _write(tmp_path, [_success("0"), _rejected("1")])
    queries = tmp_path / "r1" / "queries.csv"
    lines = queries.read_text(encoding="utf-8").splitlines()
    queries.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ContractViolationError, match="truncated"):
        load_skyline_record(tmp_path / "r1")


# --- FR-005: the record must identify the code that produced it ----------------------------------

def test_record_carries_a_code_revision(tmp_path):
    """Not decoration: a result that cannot be traced to a commit cannot be reproduced."""
    from hsreloc.retrieval.record import code_revision

    _write(tmp_path, [_success()])
    manifest = json.loads((tmp_path / "r1" / "manifest.json").read_text(encoding="utf-8"))
    assert "code_revision" in manifest
    revision = manifest["code_revision"]
    # None is the honest answer outside a checkout; inside one it must be a real SHA, optionally
    # marked dirty because a run from a modified tree is not reproducible from the commit alone.
    assert revision is None or (len(revision.split("-")[0]) == 40)
    assert revision == code_revision()


def test_code_revision_is_not_marked_dirty_by_untracked_output(tmp_path):
    """A run writes its record into the repo; counting untracked files would mark every run dirty."""
    import subprocess
    from hsreloc.retrieval.record import code_revision

    root = REPO_ROOT
    tracked = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                             cwd=root, capture_output=True, text=True)
    if tracked.returncode != 0:
        pytest.skip("not a git checkout")
    revision = code_revision()
    if revision is None:
        pytest.skip("git unavailable")
    assert revision.endswith("-dirty") == bool(tracked.stdout.strip())
