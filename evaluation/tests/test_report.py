"""Tests for report.py's output-shaping logic (contracts/evaluation-outputs.md).

The property under test (SC-014): a dataset with no heading ground truth must still
produce the complete position metric set, with yaw explicitly marked `omitted` rather
than absent, failing, or silently degrading the position results (FR-068).

Built on the real, checked-in golden fixtures via a modified copy -- an integration
test through the actual `run_evaluation` pipeline, not a hand-built metrics.json.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from naveval.evaluate import run_evaluation
from naveval.report import EvaluationConfig

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN_DATASET = FIXTURES / "golden_dataset"
GOLDEN_RUN = FIXTURES / "golden_run"


def _make_dataset_without_heading(tmp_path: Path) -> Path:
    """A copy of golden_dataset with every heading sample blanked and
    `heading_quality` declared null -- `has_heading` becomes False, exactly the
    "no heading ground truth in dataset" case SC-014 requires be handled gracefully."""
    dest = tmp_path / "dataset_no_heading"
    shutil.copytree(GOLDEN_DATASET, dest)

    descriptor_path = dest / "dataset.json"
    with descriptor_path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    d["heading_quality"] = None
    with descriptor_path.open("w", encoding="utf-8") as f:
        json.dump(d, f)

    gt_path = dest / "groundtruth.csv"
    with gt_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["heading_deg"] = ""
    with gt_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return dest


@pytest.fixture
def config_without_heading(tmp_path) -> EvaluationConfig:
    dataset_dir = _make_dataset_without_heading(tmp_path)
    return EvaluationConfig(
        evaluation_id="no-heading-test",
        run_record_path=GOLDEN_RUN,
        dataset_path=dataset_dir,
    )


class TestNoHeadingGroundTruthStillProducesPositionMetrics:
    def test_pipeline_does_not_fail(self, config_without_heading, tmp_path):
        # SC-014: absence of heading ground truth must not fail the evaluation.
        result = run_evaluation(config_without_heading, tmp_path / "out")
        assert result["alignment"] is not None
        assert result["ate"]["rmse"] >= 0.0

    def test_complete_position_metric_set_present_with_real_values(self, config_without_heading, tmp_path):
        run_evaluation(config_without_heading, tmp_path / "out")
        with (tmp_path / "out" / "metrics.json").open() as f:
            payload = json.load(f)

        by_name = {m["name"]: m for m in payload["metrics"]}
        position_metrics = ["ate_rmse", "ate_rmse_normalised", "drift_per_distance", "endpoint_error", "rpe_5m", "rpe_10m"]
        for name in position_metrics:
            assert name in by_name, f"{name} missing entirely -- position metrics must not be blocked by absent heading GT"
            entry = by_name[name]
            assert entry["value"] is not None
            assert entry.get("state") != "omitted"
            assert entry["support_level"] == "supported"  # golden fixture position quality is simulator_exact

    def test_yaw_explicitly_marked_omitted_not_absent(self, config_without_heading, tmp_path):
        run_evaluation(config_without_heading, tmp_path / "out")
        with (tmp_path / "out" / "metrics.json").open() as f:
            payload = json.load(f)

        by_name = {m["name"]: m for m in payload["metrics"]}
        for name in ("yaw_rmse", "yaw_drift"):
            assert name in by_name, f"{name} must appear explicitly, not be silently absent (SC-014)"
            entry = by_name[name]
            assert entry["value"] is None
            assert entry["state"] == "omitted"
            assert entry["omission_reason"] == "no heading ground truth in dataset"
            assert entry["support_level"] == "unsupported"

    def test_ground_truth_class_reports_heading_as_none(self, config_without_heading, tmp_path):
        run_evaluation(config_without_heading, tmp_path / "out")
        with (tmp_path / "out" / "metrics.json").open() as f:
            payload = json.load(f)
        assert payload["ground_truth_class"]["heading"] is None
        assert payload["ground_truth_class"]["position"] == "simulator_exact"

    def test_position_values_unaffected_by_missing_heading(self, config_without_heading, tmp_path):
        # The absent heading data must not degrade or perturb the position pipeline --
        # values should match a run against the ORIGINAL golden dataset (which has
        # heading data) to numerical tolerance, since alignment and ATE are position-only.
        from naveval.evaluate import run_evaluation as run_eval

        result_no_heading = run_eval(config_without_heading, tmp_path / "out_a")

        config_with_heading = EvaluationConfig(
            evaluation_id="with-heading", run_record_path=GOLDEN_RUN, dataset_path=GOLDEN_DATASET,
        )
        result_with_heading = run_eval(config_with_heading, tmp_path / "out_b")

        assert result_no_heading["ate"]["rmse"] == pytest.approx(result_with_heading["ate"]["rmse"])
        assert result_no_heading["alignment"].scale == pytest.approx(result_with_heading["alignment"].scale)
