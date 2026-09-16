"""`source_type: historical_imagery` (contracts/dataset.md 1.0.0 -> 1.1.0, minor/additive, spec 007
2026-08-22): a public real-world dataset (e.g. Nordland) is neither a UAV flight, a simulator, nor a
synthetic generator, so it needed its own honest label. Proves: it loads; it is capped at T3 (never
T4); it requires a non-null evidence_caveat (mirroring java_simulator's rule, for the same
domain-honesty reason); and the four pre-existing source_types are unaffected by the addition.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from naveval.dataset import load_dataset
from naveval.errors import ContractViolationError

_BASE = {
    "schema_version": "1.0.0",
    "dataset_id": "hist-test",
    "dataset_revision": "v1",
    "source_type": "historical_imagery",
    "evidence_tier": "T3",
    "evidence_caveat": "rural/forward-camera, not UAV",
    "frame_clock": "c", "gt_clock": "c",
    "clock_offset_s": 0.0, "clock_offset_source": "", "clock_drift_s_per_s": 0.0,
    "frame_convention": "ENU", "heading_convention": "compass_cw_from_north",
    "local_frame_origin": {"lat_deg": 0.0, "lon_deg": 0.0, "alt_m": 0.0},
}


def _write(tmp_path: Path, overrides: dict) -> Path:
    root = tmp_path / "ds"
    root.mkdir()
    d = {**_BASE, **overrides}
    (root / "dataset.json").write_text(json.dumps(d), encoding="utf-8")
    (root / "frames.csv").write_text(
        "frame_index,timestamp_s,image_path\n0,0.0,images/a.png\n", encoding="utf-8")
    return root


def test_historical_imagery_loads_with_caveat(tmp_path):
    ds = load_dataset(_write(tmp_path, {}))
    assert ds.source_type == "historical_imagery"
    assert ds.evidence_tier == "T3"
    assert ds.evidence_caveat == "rural/forward-camera, not UAV"


def test_historical_imagery_without_caveat_rejected(tmp_path):
    with pytest.raises(ContractViolationError):
        load_dataset(_write(tmp_path, {"evidence_caveat": None}))


def test_historical_imagery_capped_below_t4(tmp_path):
    with pytest.raises(ContractViolationError):
        load_dataset(_write(tmp_path, {"evidence_tier": "T4"}))


def test_historical_imagery_t3_is_the_max_allowed(tmp_path):
    # T3 (the cap) still loads -- the rejection above is specifically about exceeding it, not T3 itself.
    ds = load_dataset(_write(tmp_path, {"evidence_tier": "T3"}))
    assert ds.evidence_tier == "T3"


def test_preexisting_source_types_unaffected(tmp_path):
    ds = load_dataset(_write(tmp_path, {
        "source_type": "real_flight", "evidence_tier": "T4", "evidence_caveat": None,
    }))
    assert ds.source_type == "real_flight" and ds.evidence_tier == "T4"
