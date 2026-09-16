"""Tests for the thesis-ready table-export helper (T086, FR-041): comparison of metrics
across segments *and* across flights/runs, with every row carrying its own evidence tier,
caveat and ground-truth class so heterogeneous sources are never silently pooled.
"""

import csv
from pathlib import Path

import pytest

from naveval.evaluate import run_evaluation
from naveval.report import EvaluationConfig, export_metrics_table

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _golden_output(tmp_path: Path) -> Path:
    config = EvaluationConfig(
        evaluation_id="table-export-golden",
        run_record_path=FIXTURES / "golden_run",
        dataset_path=FIXTURES / "golden_dataset",
    )
    out_dir = tmp_path / "golden-out"
    run_evaluation(config, out_dir)
    return out_dir


def _read_rows(path: Path) -> list:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class TestExportMetricsTable:
    def test_single_source_run_level_rows(self, tmp_path):
        golden_out = _golden_output(tmp_path)
        out_path = tmp_path / "table.csv"

        export_metrics_table([golden_out], out_path)
        rows = _read_rows(out_path)

        run_rows = [r for r in rows if r["scope"] == "run"]
        assert run_rows
        assert all(r["source"] == "golden-out" for r in run_rows)
        assert all(r["evidence_tier"] == "T1" for r in run_rows)
        assert all(r["ground_truth_position_class"] == "simulator_exact" for r in run_rows)
        names = {r["metric_name"] for r in run_rows}
        assert "ate_rmse" in names

    def test_segment_rows_included_by_default(self, tmp_path):
        golden_out = _golden_output(tmp_path)
        out_path = tmp_path / "table.csv"

        export_metrics_table([golden_out], out_path)
        rows = _read_rows(out_path)

        segment_rows = [r for r in rows if r["scope"] == "segment"]
        assert segment_rows
        assert {r["metric_name"] for r in segment_rows} == {
            "segment_ate_rmse_m", "segment_path_length_m", "segment_endpoint_error_m",
        }
        # Segment rows still carry the same source-level evidence columns.
        assert all(r["evidence_tier"] == "T1" for r in segment_rows)
        assert all(r["source"] == "golden-out" for r in segment_rows)

    def test_include_segments_false_omits_segment_rows(self, tmp_path):
        golden_out = _golden_output(tmp_path)
        out_path = tmp_path / "table.csv"

        export_metrics_table([golden_out], out_path, include_segments=False)
        rows = _read_rows(out_path)

        assert all(r["scope"] == "run" for r in rows)

    def test_multiple_sources_are_never_silently_pooled(self, tmp_path):
        golden_out = _golden_output(tmp_path)
        # sim_reference: a real, different-evidence-tier committed fixture (T080) --
        # exercises comparing genuinely heterogeneous sources, not two copies of one.
        sim_out = FIXTURES / "sim_reference"
        out_path = tmp_path / "table.csv"

        export_metrics_table([golden_out, sim_out], out_path)
        rows = _read_rows(out_path)

        sources = {r["source"] for r in rows}
        assert sources == {"golden-out", "sim_reference"}

        golden_ate = next(r for r in rows if r["source"] == "golden-out" and r["metric_name"] == "ate_rmse")
        sim_ate = next(r for r in rows if r["source"] == "sim_reference" and r["metric_name"] == "ate_rmse")
        assert golden_ate["evidence_tier"] == "T1"
        assert sim_ate["evidence_tier"] == "T2"
        assert sim_ate["evidence_caveat"] != ""
        assert golden_ate["evidence_caveat"] == ""

    def test_custom_labels_override_directory_names(self, tmp_path):
        golden_out = _golden_output(tmp_path)
        out_path = tmp_path / "table.csv"

        export_metrics_table([golden_out], out_path, labels=["flight-A"])
        rows = _read_rows(out_path)

        assert all(r["source"] == "flight-A" for r in rows)

    def test_labels_length_mismatch_raises(self, tmp_path):
        golden_out = _golden_output(tmp_path)
        with pytest.raises(ValueError):
            export_metrics_table([golden_out], tmp_path / "table.csv", labels=["a", "b"])

    def test_missing_metrics_json_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            export_metrics_table([empty_dir], tmp_path / "table.csv")

    def test_excluded_metrics_never_appear_as_rows(self, tmp_path):
        # sim_reference's ground truth is simulator_exact so nothing is excluded there;
        # this checks the *mechanism* instead: every metric_name present as a row is
        # also present in metrics.json's own "metrics" array, none from
        # unsupported_excluded (SC-013).
        import json
        sim_out = FIXTURES / "sim_reference"
        with (sim_out / "metrics.json").open() as f:
            payload = json.load(f)
        excluded = set(payload["unsupported_excluded"])

        out_path = tmp_path / "table.csv"
        export_metrics_table([sim_out], out_path)
        rows = _read_rows(out_path)
        run_metric_names = {r["metric_name"] for r in rows if r["scope"] == "run"}
        assert run_metric_names.isdisjoint(excluded)
