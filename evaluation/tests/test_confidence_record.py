"""Run-record v1.1.0 reading (contracts/run-record-v1.1.md §Reader obligations).

A v1.0.0 record still loads with confidence *absent* — emphatically not low — and the added
invariants are enforced as errors, because a malformed record must not be evaluated.

Evidence tier: T1 (analytical).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from naveval.errors import ContractViolationError
from naveval.runrecord import load_run_record

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "evaluation" / "tests" / "fixtures" / "confidence_agreement" / "run_bootstrap"
GOLDEN = REPO / "evaluation" / "tests" / "fixtures" / "golden_run"


class TestBackwardCompatibility:
    def test_a_v1_0_record_loads_with_confidence_absent(self):
        run = load_run_record(GOLDEN)
        assert run.manifest.confidence is None, "absent, not an empty block"
        for fe in run.frame_estimates:
            assert fe.confidence is None, (
                "a v1.0.0 run has no verdict; it does not have a bad one"
            )

    def test_a_v1_1_record_loads_with_all_columns(self):
        run = load_run_record(FIXTURE)
        assert run.manifest.schema_version == "1.1.0"
        assert run.manifest.confidence["calibration_id"] == "bootstrap-unvalidated-v2"
        assert run.manifest.confidence["calibration_validated"] is False
        first = run.frame_estimates[0].confidence
        assert first.outcome == "rejected" and first.reason == "not_established"
        assert first.score is None
        healthy = run.frame_estimates[1].confidence
        assert healthy.outcome == "usable" and healthy.score is not None
        assert healthy.residual_inlier_count == 180
        assert healthy.inlier_threshold_sq_px == 3.0


def _mutate_fixture(tmp_path: Path, row_index: int, column: str, value: str) -> Path:
    """Copies the fixture and rewrites one cell of frames.csv by header name."""
    run_dir = tmp_path / "run"
    shutil.copytree(FIXTURE, run_dir)
    frames = (run_dir / "frames.csv").read_text("utf-8").splitlines()
    header = frames[0].split(",")
    col = header.index(column)
    cells = frames[1 + row_index].split(",")
    cells[col] = value
    frames[1 + row_index] = ",".join(cells)
    (run_dir / "frames.csv").write_text("\n".join(frames) + "\n", "utf-8")
    return run_dir


class TestInvariantsAreErrors:
    def test_score_on_a_rejected_frame_is_an_error(self, tmp_path):
        run_dir = _mutate_fixture(tmp_path, 0, "confidence_score", "0.5")
        with pytest.raises(ContractViolationError, match="confidence_score"):
            load_run_record(run_dir)

    def test_missing_score_on_a_usable_frame_is_an_error(self, tmp_path):
        run_dir = _mutate_fixture(tmp_path, 1, "confidence_score", "")
        with pytest.raises(ContractViolationError, match="confidence_score"):
            load_run_record(run_dir)

    def test_not_produced_on_a_successful_frame_is_an_error(self, tmp_path):
        run_dir = _mutate_fixture(tmp_path, 1, "confidence_outcome", "not_produced")
        with pytest.raises(ContractViolationError):
            load_run_record(run_dir)

    def test_invalid_outcome_is_an_error(self, tmp_path):
        run_dir = _mutate_fixture(tmp_path, 1, "confidence_outcome", "fine")
        with pytest.raises(ContractViolationError, match="confidence_outcome"):
            load_run_record(run_dir)

    def test_residual_stats_without_a_count_are_an_error(self, tmp_path):
        run_dir = _mutate_fixture(tmp_path, 1, "residual_inlier_count", "")
        with pytest.raises(ContractViolationError, match="residual"):
            load_run_record(run_dir)

    def test_negative_confidence_time_is_an_error(self, tmp_path):
        run_dir = _mutate_fixture(tmp_path, 1, "confidence_time_ns", "-1")
        with pytest.raises(ContractViolationError, match="confidence_time_ns"):
            load_run_record(run_dir)
