"""Spec 006 — the generic, matcher-agnostic result record + the additive reference_source block.

File-based via tmp_path (no large committed fixtures). Proves: a historical/generic record with a
`reference_source` block round-trips; the exactly-one-source-block rule; footprint reads the
historical block; existing DEM records still load unchanged; and the evaluator is indifferent to the
producer (matcher-agnostic, FR-002).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from naveval import skyline
from naveval.errors import ContractViolationError
from naveval.skyline_record import (
    QueryGroundTruth,
    QueryResult,
    SkylineQuerySet,
    SkylineResultRecord,
    SkylineRunManifest,
    load_skyline_record,
)

_HEADER = ("query_id,timestamp_s,outcome,est_east_m,est_north_m,est_heading_deg,heading_source,"
           "compass_prior_deg,confidence,best_score,second_best_score,score_margin,"
           "n_viable_candidates,alignment_residual,matched_east_m,matched_north_m,process_time_ns")

_ROW = "0,0.0,SUCCESS,10,20,,,,0.9,0.9,0.1,0.8,3,0.05,10,20,1000000"


def _write(tmp_path: Path, manifest: dict, rows=(_ROW,)) -> Path:
    root = tmp_path / "run"
    root.mkdir()
    manifest = {"schema_version": "1.0.0", "run_id": "r", "query_set_id": "qs",
                "query_set_revision": "1", "environment": {"is_target_hardware": False},
                "run_timestamp": "t", "query_count": len(rows), "processed_count": len(rows),
                "completed": True, **manifest}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "queries.csv").write_text(_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return root


_HIST = {"kind": "historical_imagery", "ref_set_id": "refset-x", "dataset_id": "nordland",
         "reference_spacing_m": 100.0, "extraction_mode": "oracle:manual",
         "n_references": 100, "total_bytes": 51200, "covered_area_km2": 2.0,
         "evidence_tier": "T3", "evidence_caveat": "rural/forward-camera, hand-verified"}


def test_historical_record_round_trips(tmp_path):
    rec = load_skyline_record(_write(tmp_path, {"reference_source": _HIST}))
    assert rec.manifest.reference_source["kind"] == "historical_imagery"
    assert rec.manifest.reference_source["extraction_mode"] == "oracle:manual"
    assert rec.manifest.reference_db == {}
    # ranked-shortlist absent here; raw score signals preserved
    assert rec.results[0].best_score == pytest.approx(0.9)


def test_footprint_reads_reference_source(tmp_path):
    rec = load_skyline_record(_write(tmp_path, {"reference_source": _HIST}))
    fp = skyline.database_footprint(rec)
    assert fp["bytes_per_reference"] == pytest.approx(512.0)          # 51200 / 100
    assert fp["bytes_per_km2"] == pytest.approx(25600.0)              # 51200 / 2.0


def test_both_source_blocks_rejected(tmp_path):
    with pytest.raises(ContractViolationError):
        load_skyline_record(_write(tmp_path, {
            "reference_db": {"n_references": 1, "total_bytes": 10},
            "reference_source": _HIST,
        }))


def test_neither_source_block_rejected(tmp_path):
    # a manifest with neither reference_db nor reference_source key
    with pytest.raises(ContractViolationError):
        load_skyline_record(_write(tmp_path, {}))


def test_existing_dem_record_still_loads(tmp_path):
    # an empty reference_db (spec 004/005 tmp-record convention) is still accepted -> no regression
    rec = load_skyline_record(_write(tmp_path, {"reference_db": {}}))
    assert rec.manifest.reference_source == {}
    assert rec.manifest.reference_db == {}


# --- matcher-agnostic: the evaluator does not branch on the producer (FR-002) ---

def _bundle_with_producer(relocalizer_id, source_kind):
    r = QueryResult(query_id="0", timestamp_s=0.0, outcome="SUCCESS", est_east_m=1.0, est_north_m=1.0,
                    est_heading_deg=None, heading_source=None, compass_prior_deg=None, confidence=0.9,
                    best_score=0.9, second_best_score=0.1, score_margin=0.8, n_viable_candidates=2,
                    alignment_residual=None, matched_east_m=1.0, matched_north_m=1.0,
                    process_time_ns=1000, candidates=((1.0, 1.0, 0.9), (1000.0, 1000.0, 0.1)))
    gt = QueryGroundTruth(query_index=0, east_m=1.0, north_m=1.0, heading_deg=None, roll_deg=None,
                          pitch_deg=None, compass_prior_deg=None, in_coverage=True, conditions={})
    m = SkylineRunManifest(
        schema_version="1.0.0", run_id="r", query_set_id="qs", query_set_revision="1",
        relocalizer_id=relocalizer_id, relocalizer_version="v", relocalizer_config={},
        reference_db={}, compass_prior={}, environment={"is_target_hardware": False},
        run_timestamp="t", query_count=1, processed_count=1, completed=True,
        reference_source={"kind": source_kind, "n_references": 1, "total_bytes": 10, "covered_area_km2": 1.0},
    )
    return SkylineResultRecord(manifest=m, results=[r]), \
        SkylineQuerySet(dataset=None, camera={}, compass_prior_spec={}, queries={0: gt})


def test_evaluator_is_matcher_agnostic():
    # Identical ranking numbers from two different "producers" -> identical metrics. The evaluator
    # reads only outcomes/coords/scores, never who produced them (NCC/L1L2/learned/C++/MATLAB/prod).
    rec_a, qs_a = _bundle_with_producer("mock-ncc", "historical_imagery")
    rec_b, qs_b = _bundle_with_producer("some-future-cpp-system", "historical_imagery")
    for fn, args in [
        (skyline.topological_success, (5.0,)),
        (skyline.score_separation, (5.0, 50.0)),
        (skyline.pr_roc, (5.0,)),
    ]:
        assert fn(rec_a, qs_a, *args) == fn(rec_b, qs_b, *args)
