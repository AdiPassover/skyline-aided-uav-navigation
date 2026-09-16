"""Contract/validation tests for the skyline result-record + query-set readers (spec 004).

File-based, mostly via tmp_path so no large committed fixtures are needed; the committed
golden fixture under tests/fixtures/skyline/ is also loaded to prove a round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from naveval.errors import ContractViolationError, DatasetMismatchError, SchemaVersionError
from naveval.skyline_record import (
    load_skyline_query_set,
    load_skyline_record,
    verify_record_matches_query_set,
)

FIXTURES = Path(__file__).parent / "fixtures" / "skyline"

_HEADER = ("query_id,timestamp_s,outcome,est_east_m,est_north_m,est_heading_deg,heading_source,"
           "compass_prior_deg,confidence,best_score,second_best_score,score_margin,"
           "n_viable_candidates,alignment_residual,matched_east_m,matched_north_m,process_time_ns")


def _write_record(tmp_path: Path, rows: list[str], schema="1.0.0", processed=None) -> Path:
    root = tmp_path / "run"
    root.mkdir()
    manifest = {
        "schema_version": schema, "run_id": "r", "query_set_id": "qs", "query_set_revision": "1",
        "reference_db": {}, "compass_prior": {}, "environment": {"is_target_hardware": False},
        "run_timestamp": "t", "query_count": len(rows),
        "processed_count": len(rows) if processed is None else processed, "completed": True,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "queries.csv").write_text(_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return root


def test_golden_record_and_query_set_round_trip():
    rec = load_skyline_record(FIXTURES / "run")
    qs = load_skyline_query_set(FIXTURES / "queryset")
    assert len(rec.results) == 10
    assert len(qs.queries) == 10
    verify_record_matches_query_set(rec, qs)  # must not raise
    # heading_source preserved distinctly
    by_id = {r.query_id: r for r in rec.results}
    assert by_id["0"].heading_source == "compass_refined"
    assert by_id["1"].heading_source == "skyline_estimated"
    assert by_id["2"].heading_source == "compass_prior"


def test_schema_major_version_rejected(tmp_path):
    root = _write_record(tmp_path, ["0,0.0,REJECTED,,,,,,0.2,,,,,,,,1000"], schema="2.0.0")
    with pytest.raises(SchemaVersionError):
        load_skyline_record(root)


def test_not_attempted_outcome_rejected(tmp_path):
    root = _write_record(tmp_path, ["0,0.0,NOT_ATTEMPTED,,,,,,,,,,,,,,1000"])
    with pytest.raises(ContractViolationError):
        load_skyline_record(root)


def test_success_without_position_rejected(tmp_path):
    # SUCCESS but est/matched empty -> forbidden
    root = _write_record(tmp_path, ["0,0.0,SUCCESS,,,,,,0.9,,,,,,,,1000"])
    with pytest.raises(ContractViolationError):
        load_skyline_record(root)


def test_rejected_with_position_rejected(tmp_path):
    root = _write_record(tmp_path, ["0,0.0,REJECTED,10,20,,,,0.2,,,,,,,,1000"])
    with pytest.raises(ContractViolationError):
        load_skyline_record(root)


def test_heading_without_source_rejected(tmp_path):
    root = _write_record(tmp_path, ["0,0.0,SUCCESS,10,20,90,,,0.9,,,,,,10,20,1000"])
    with pytest.raises(ContractViolationError):
        load_skyline_record(root)


def test_duplicate_query_id_rejected(tmp_path):
    root = _write_record(tmp_path, [
        "0,0.0,REJECTED,,,,,,0.2,,,,,,,,1000",
        "0,1.0,REJECTED,,,,,,0.2,,,,,,,,1000",
    ])
    with pytest.raises(ContractViolationError):
        load_skyline_record(root)


def test_processed_count_mismatch_rejected(tmp_path):
    root = _write_record(tmp_path, ["0,0.0,REJECTED,,,,,,0.2,,,,,,,,1000"], processed=2)
    with pytest.raises(ContractViolationError):
        load_skyline_record(root)


def test_query_set_mismatch_detected(tmp_path):
    root = _write_record(tmp_path, ["0,0.0,REJECTED,,,,,,0.2,,,,,,,,1000"])
    # manifest above declares query_set_id="qs"; the golden query set is "skyquery-mock-001"
    rec = load_skyline_record(root)
    qs = load_skyline_query_set(FIXTURES / "queryset")
    with pytest.raises(DatasetMismatchError):
        verify_record_matches_query_set(rec, qs)


def test_missing_files_rejected(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ContractViolationError):
        load_skyline_record(empty)
