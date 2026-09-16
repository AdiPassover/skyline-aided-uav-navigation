"""Seam conformance for the automatic curve source (tasks T031/T032, spec FR-016/017, SC-006).

The battery mirrors the oracle source's contract tests (same validation, same digesting, same
no-ground-truth surface) — parametrized over both sources where the property is shared, so the two
producers are provably interchangeable behind the seam.
"""

from __future__ import annotations

import numpy as np
import pytest

from hsreloc.extraction.auto_source import (
    AUTO_SUBDIR,
    AutomaticCurveSource,
    write_auto_curve,
)
from hsreloc.extraction.curves import (
    CONVERSION_KIND_PRED,
    REASON_METHOD_NO_OUTPUT,
    REASON_NONE,
    make_curve as make_canonical,
)
from hsreloc.extraction.methods.base import STATUS_FAILURE, STATUS_OK, ExtractionOutput
from hsreloc.retrieval.skyline_curve import CurveError, CurveSource
from hsreloc.retrieval.sources import ORACLE_SUBDIR, OracleCurveSource

W, H = 8, 16
SESSIONS = {"obs_a": "sess1", "obs_b": "sess1"}


def _write_seam_csv(root, subdir, obs_id, values, session="sess1"):
    p = root / session / subdir / f"{obs_id}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        f.write("col,row\n")
        for c, v in enumerate(values):
            f.write(f"{c},{float(v)}\n")
    return p


def _auto(root):
    return AutomaticCurveSource(root, SESSIONS, H, W, method_id="poc_robust_dp")


def _oracle(root):
    return OracleCurveSource(root, SESSIONS, H, W, expect_provenance="oracle:manual")


@pytest.mark.parametrize("factory,subdir,provenance", [
    (_auto, AUTO_SUBDIR, "automatic:poc_robust_dp"),
    (_oracle, ORACLE_SUBDIR, "oracle:manual"),
])
class TestSharedBattery:
    def test_satisfies_protocol_and_loads(self, tmp_path, factory, subdir, provenance):
        _write_seam_csv(tmp_path, subdir, "obs_a", np.linspace(2, 9, W))
        src = factory(tmp_path)
        assert isinstance(src, CurveSource)
        curve = src.get("obs_a")
        curve.validate()
        assert curve.provenance == provenance
        assert curve.row_per_col.size == W

    def test_digest_deterministic(self, tmp_path, factory, subdir, provenance):
        _write_seam_csv(tmp_path, subdir, "obs_a", np.linspace(2, 9, W))
        src = factory(tmp_path)
        assert src.get("obs_a").digest == src.get("obs_a").digest

    def test_missing_curve_is_curve_error(self, tmp_path, factory, subdir, provenance):
        with pytest.raises(CurveError, match="obs_a"):
            factory(tmp_path).get("obs_a")

    def test_gap_in_columns_rejected(self, tmp_path, factory, subdir, provenance):
        p = _write_seam_csv(tmp_path, subdir, "obs_a", np.linspace(2, 9, W))
        lines = p.read_text(encoding="utf-8").splitlines()
        p.write_text("\n".join(lines[:-2]) + "\n", encoding="utf-8")
        with pytest.raises(CurveError, match="contiguously"):
            factory(tmp_path).get("obs_a")

    def test_out_of_range_row_rejected(self, tmp_path, factory, subdir, provenance):
        _write_seam_csv(tmp_path, subdir, "obs_a", [2.0] * (W - 1) + [H + 5.0])
        with pytest.raises(CurveError, match="outside"):
            factory(tmp_path).get("obs_a")

    def test_unknown_observation_named(self, tmp_path, factory, subdir, provenance):
        with pytest.raises(CurveError, match="ghost"):
            factory(tmp_path).get("ghost")

    def test_no_ground_truth_surface(self, tmp_path, factory, subdir, provenance):
        _write_seam_csv(tmp_path, subdir, "obs_a", np.linspace(2, 9, W))
        curve = factory(tmp_path).get("obs_a")
        for forbidden in ("east", "north", "lat", "lon", "heading", "in_coverage"):
            assert not any(forbidden in f for f in curve.__dataclass_fields__)
        desc = factory(tmp_path).describe()
        assert not any(k in desc for k in ("groundtruth", "in_coverage", "position"))


class TestInterchangeability:
    def test_same_bytes_same_digest_across_sources(self, tmp_path):
        values = np.linspace(2, 9, W)
        _write_seam_csv(tmp_path, AUTO_SUBDIR, "obs_a", values)
        _write_seam_csv(tmp_path, ORACLE_SUBDIR, "obs_a", values)
        auto = _auto(tmp_path).get("obs_a")
        oracle = _oracle(tmp_path).get("obs_a")
        assert auto.digest == oracle.digest          # digest is over curve bytes only
        assert auto.provenance != oracle.provenance  # provenance still separates them


class TestWriter:
    def _output(self, rows, reasons=None):
        reasons = reasons or (REASON_NONE,) * W
        curve = make_canonical(W, H, rows, reasons, CONVERSION_KIND_PRED, "m")
        return ExtractionOutput(status=STATUS_OK, reason=None, curve=curve, runtime_s=0.0)

    def test_roundtrip_byte_stable(self, tmp_path):
        out = self._output(np.linspace(2, 9, W))
        p1, _ = write_auto_curve(tmp_path, "sess1", "obs_a", out)
        first = p1.read_bytes()
        p2, _ = write_auto_curve(tmp_path, "sess1", "obs_a", out)
        assert p2.read_bytes() == first
        loaded = _auto(tmp_path).get("obs_a")
        assert np.allclose(loaded.row_per_col, out.curve.rows)

    def test_failure_produces_no_file(self, tmp_path):
        out = ExtractionOutput(status=STATUS_FAILURE, reason="indeterminate_scene:dark",
                               curve=None, runtime_s=0.0)
        path, reason = write_auto_curve(tmp_path, "sess1", "obs_a", out)
        assert path is None and "dark" in reason
        assert not (tmp_path / "sess1" / AUTO_SUBDIR / "obs_a.csv").exists()

    def test_partial_curve_not_stored(self, tmp_path):
        rows = [2.0] * (W - 1) + [np.nan]
        reasons = (REASON_NONE,) * (W - 1) + (REASON_METHOD_NO_OUTPUT,)
        path, reason = write_auto_curve(tmp_path, "sess1", "obs_a", self._output(rows, reasons))
        assert path is None and "invalid columns" in reason
