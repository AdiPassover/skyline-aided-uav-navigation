"""G15: `run_evaluation`'s choice of segmentation function is a config-driven, explicit,
reproducible decision (`segments.source` in config.json) rather than an unconditional
call to `segments_from_metadata`. `naveval.segments` already implemented and unit-tested
all three definition functions (Phase 7); this file proves the CLI entry point
(`naveval.evaluate.run_evaluation`) actually reaches each of them, per config.json's own
already-documented `"segments": {"source": "metadata"}` convention
(contracts/evaluation-outputs.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from naveval.evaluate import run_evaluation
from naveval.report import EvaluationConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"

DATASET_PATH = REPO_ROOT / "datasets" / "sim-square"
RUN_RECORD_PATH = REPO_ROOT / "runs" / "sim-square-run-v1"


def _skip_unless_sim_square_present():
    if not DATASET_PATH.exists() or not RUN_RECORD_PATH.exists():
        pytest.skip(
            "datasets/sim-square and runs/sim-square-run-v1 (T080 reference) are not "
            "present in this checkout -- generate them via DatasetRecorderApp/VoRunnerApp first"
        )


class TestSegmentationSourceDefault:
    def test_default_is_metadata_whole_trajectory(self, tmp_path):
        # Unspecified `segments` in config.json must preserve the prior behaviour
        # exactly: one segment spanning the whole evaluated trajectory.
        config = EvaluationConfig(
            evaluation_id="seg-default",
            run_record_path=FIXTURES / "golden_run",
            dataset_path=FIXTURES / "golden_dataset",
        )
        assert config.segmentation_source == "metadata"
        result = run_evaluation(config, tmp_path)
        assert len(result["segments"]) == 1
        assert result["segments"][0].definition_source == "metadata"

        with (tmp_path / "metrics.json").open() as f:
            metrics = json.load(f)
        assert metrics["segmentation_source"] == "metadata"


class TestSegmentationSourceReferenceChanges:
    def test_produces_multiple_segments_on_a_run_with_restarts(self, tmp_path):
        _skip_unless_sim_square_present()
        config = EvaluationConfig(
            evaluation_id="seg-reference-changes",
            run_record_path=RUN_RECORD_PATH,
            dataset_path=DATASET_PATH,
            segmentation_source="reference_changes",
        )
        result = run_evaluation(config, tmp_path)

        # T080's own run record shows dozens of restarts (COMP-002 Limitation 3 / G15);
        # a reference-id-boundary segmentation over it must actually split accordingly,
        # not collapse back to one whole-trajectory segment.
        assert len(result["segments"]) > 1
        assert all(s.definition_source == "derived" for s in result["segments"])

        with (tmp_path / "metrics.json").open() as f:
            metrics = json.load(f)
        assert metrics["segmentation_source"] == "reference_changes"

        with (tmp_path / "segments.csv").open() as f:
            rows = list(f)
        assert len(rows) - 1 == len(result["segments"])  # header + one row per segment

    def test_scale_series_exercises_restart_boundaries(self, tmp_path):
        _skip_unless_sim_square_present()
        config = EvaluationConfig(
            evaluation_id="seg-reference-changes-scale",
            run_record_path=RUN_RECORD_PATH,
            dataset_path=DATASET_PATH,
            segmentation_source="reference_changes",
        )
        result = run_evaluation(config, tmp_path)

        # The per-segment scale-series diagnostic (FR-035) is the thing G15 says was
        # never actually exercised across mosaic-reference discontinuities through this
        # entry point. With reference_changes segmentation, more than one scale point
        # must now exist -- the concrete proof the diagnostic is reachable via the CLI.
        scale_series = result["scale_series"]
        assert len(scale_series) == len(result["segments"])
        assert len(scale_series) > 1

        with (tmp_path / "scale_series.csv").open() as f:
            rows = list(f)
        assert len(rows) - 1 == len(scale_series)


class TestSegmentationSourceManual:
    def test_explicit_ranges_are_applied_and_reproducible(self, tmp_path):
        ranges = [
            {"label": "hover", "start_index": 0, "end_index": 4},
            {"label": "turn", "start_index": 5, "end_index": 9},
        ]
        config = EvaluationConfig(
            evaluation_id="seg-manual",
            run_record_path=FIXTURES / "golden_run",
            dataset_path=FIXTURES / "golden_dataset",
            segmentation_source="manual",
            segmentation_ranges=tuple(tuple(sorted(r.items())) for r in ranges),
        )
        result = run_evaluation(config, tmp_path)

        assert len(result["segments"]) == 2
        assert [s.label for s in result["segments"]] == ["hover", "turn"]
        assert all(s.definition_source == "manual" for s in result["segments"])

        with (tmp_path / "metrics.json").open() as f:
            metrics = json.load(f)
        assert metrics["segmentation_source"] == "manual"

    def test_manual_without_ranges_is_rejected_at_config_construction(self):
        with pytest.raises(ValueError):
            EvaluationConfig(
                evaluation_id="seg-manual-empty",
                run_record_path=FIXTURES / "golden_run",
                dataset_path=FIXTURES / "golden_dataset",
                segmentation_source="manual",
            )


class TestSegmentationSourceValidation:
    def test_unknown_source_is_rejected(self):
        with pytest.raises(ValueError):
            EvaluationConfig(
                evaluation_id="seg-bad",
                run_record_path=FIXTURES / "golden_run",
                dataset_path=FIXTURES / "golden_dataset",
                segmentation_source="not_a_real_source",
            )


class TestSegmentationConfigLoading:
    def test_loads_source_from_config_json(self, tmp_path):
        run_record_rel = FIXTURES / "golden_run"
        dataset_rel = FIXTURES / "golden_dataset"
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "evaluation_id": "seg-loaded",
                    "run_record": str(run_record_rel),
                    "dataset": str(dataset_rel),
                    "segments": {"source": "reference_changes"},
                }
            ),
            encoding="utf-8",
        )
        config = EvaluationConfig.load(config_path)
        assert config.segmentation_source == "reference_changes"

    def test_loads_manual_ranges_from_config_json(self, tmp_path):
        run_record_rel = FIXTURES / "golden_run"
        dataset_rel = FIXTURES / "golden_dataset"
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "evaluation_id": "seg-loaded-manual",
                    "run_record": str(run_record_rel),
                    "dataset": str(dataset_rel),
                    "segments": {
                        "source": "manual",
                        "ranges": [{"label": "hover", "start_index": 0, "end_index": 4}],
                    },
                }
            ),
            encoding="utf-8",
        )
        config = EvaluationConfig.load(config_path)
        assert config.segmentation_source == "manual"
        assert config.segmentation_ranges_as_dicts() == [
            {"label": "hover", "start_index": 0, "end_index": 4}
        ]


class TestSegmentationSourceAffectsDigest:
    def test_different_segmentation_source_changes_config_digest(self):
        # FR-051: any figure must be traceable to the exact configuration that
        # produced it -- the segmentation mode must not be invisible to the digest.
        base = EvaluationConfig(
            evaluation_id="seg-digest",
            run_record_path=FIXTURES / "golden_run",
            dataset_path=FIXTURES / "golden_dataset",
        )
        other = EvaluationConfig(
            evaluation_id="seg-digest",
            run_record_path=FIXTURES / "golden_run",
            dataset_path=FIXTURES / "golden_dataset",
            segmentation_source="reference_changes",
        )
        assert base.digest() != other.digest()
