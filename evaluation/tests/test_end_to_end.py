"""End-to-end test: the full pipeline against the golden fixtures.

Ties every foundational piece together -- dataset loading, run-record loading, time
synchronisation, Sim(2) alignment, ATE, and report writing -- and is the automated form
of quickstart.md Scenario 2. Reference values below were obtained by running the CLI
once against the checked-in golden fixtures and are committed here as the regression
baseline (T032); they are not fabricated -- see the golden_run/golden_dataset fixtures'
own construction (research-log 2026-08-12), where the run record's estimator points are
an exact Sim(2) image of the dataset's ground truth (rotation=15 deg, scale=0.05,
translation=(5.0, 3.0)), each independently rounded to 4 decimal places when the CSVs
were hand-authored -- which is exactly why the residual below is small but not exactly
zero.
"""

from pathlib import Path

import pytest

from naveval.evaluate import run_evaluation
from naveval.report import EvaluationConfig

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "eval-golden.json"

# Reference values from the actual CLI run against the golden fixtures (2026-08-12).
REFERENCE_ROTATION_DEG = 15.00000327346589
REFERENCE_SCALE = 0.04999999607499464
REFERENCE_TRANSLATION = (5.000005013656912, 2.999995491678812)
REFERENCE_ATE_RMSE = 3.574149489061649e-05
REFERENCE_ATE_RMSE_NORMALIZED = 2.278263716636219e-06


class TestEndToEndGoldenPipeline:
    def test_full_pipeline_runs_without_error(self, tmp_path):
        config = EvaluationConfig.load(CONFIG_PATH)
        result = run_evaluation(config, tmp_path)
        assert result["alignment"] is not None
        assert result["ate"]["rmse"] >= 0.0

    def test_alignment_matches_reference_values(self, tmp_path):
        config = EvaluationConfig.load(CONFIG_PATH)
        result = run_evaluation(config, tmp_path)
        alignment = result["alignment"]

        assert alignment.rotation_deg == pytest.approx(REFERENCE_ROTATION_DEG, abs=1e-6)
        assert alignment.scale == pytest.approx(REFERENCE_SCALE, rel=1e-6)
        assert alignment.translation[0] == pytest.approx(REFERENCE_TRANSLATION[0], abs=1e-6)
        assert alignment.translation[1] == pytest.approx(REFERENCE_TRANSLATION[1], abs=1e-6)
        assert alignment.reflection_rejected is False
        assert alignment.n_points == 10

    def test_ate_matches_reference_values(self, tmp_path):
        config = EvaluationConfig.load(CONFIG_PATH)
        result = run_evaluation(config, tmp_path)
        ate = result["ate"]

        assert ate["rmse"] == pytest.approx(REFERENCE_ATE_RMSE, rel=1e-6)
        assert ate["rmse_normalized"] == pytest.approx(REFERENCE_ATE_RMSE_NORMALIZED, rel=1e-6)
        # A near-exact Sim(2) image (residual only from 4-decimal CSV rounding) must
        # stay far below any real drift signature -- sanity bound, not a tight one.
        assert ate["rmse"] < 1e-3

    def test_all_frames_synchronized(self, tmp_path):
        config = EvaluationConfig.load(CONFIG_PATH)
        result = run_evaluation(config, tmp_path)
        assert result["sync"].frame_indices.size == 10
        assert result["sync"].excluded_out_of_span_count == 0
        assert result["sync"].excluded_insufficient_valid_gt_count == 0

    def test_output_files_written(self, tmp_path):
        config = EvaluationConfig.load(CONFIG_PATH)
        run_evaluation(config, tmp_path)
        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "alignment.json").exists()
        assert (tmp_path / "figures" / "trajectory_xy.png").exists()

    def test_metrics_json_structure(self, tmp_path):
        import json

        config = EvaluationConfig.load(CONFIG_PATH)
        run_evaluation(config, tmp_path)
        with (tmp_path / "metrics.json").open() as f:
            payload = json.load(f)

        assert payload["schema_version"] == "1.0.0"
        assert payload["run_id"] == "golden-run-001"
        assert payload["dataset_id"] == "golden-001"
        assert payload["evidence_tier"] == "T1"
        assert payload["evidence_caveat"] is None  # synthetic, not java_simulator
        assert payload["ground_truth_class"]["position"] == "simulator_exact"

        # Phase 5: the primary metric set (FR-030) plus its per-segment-rescaled RPE
        # diagnostics (FR-067). Golden fixture is `simulator_exact` throughout, so
        # nothing is class-unsupported; `unsupported_excluded` stays empty. Its total
        # path length (~15.7m) is short enough that the 25/50m RPE lengths in
        # eval-golden's default config are genuinely omitted (not enough trajectory),
        # which is itself useful coverage of the omission path on a real run.
        by_name = {m["name"]: m for m in payload["metrics"]}
        assert {"ate_rmse", "ate_rmse_normalised", "drift_per_distance", "endpoint_error",
                "yaw_rmse", "yaw_drift", "rpe_5m", "rpe_10m"} <= by_name.keys()
        assert payload["unsupported_excluded"] == []

        for primary_name in ("ate_rmse", "ate_rmse_normalised", "endpoint_error", "rpe_5m", "rpe_10m"):
            assert by_name[primary_name]["scale_basis"] == "global_fitted"
            assert by_name[primary_name]["role"] == "primary"
            assert by_name[primary_name]["support_level"] == "supported"

        # Straight-line golden fixture: closed-loop endpoint error is structurally
        # inapplicable and must be explicitly omitted, never silently absent (SC-014).
        assert by_name["endpoint_error_closed_loop"]["state"] == "omitted"
        assert by_name["endpoint_error_closed_loop"]["value"] is None

        # Diagnostic RPE variants must never be labelled primary (FR-067, DEC-003).
        assert by_name["rpe_5m_diagnostic_per_segment_rescaled"]["role"] == "diagnostic"
        assert by_name["rpe_5m_diagnostic_per_segment_rescaled"]["scale_basis"] == "per_segment_fitted"
        assert by_name["rpe_5m_scale_drift_contribution"]["role"] == "diagnostic"

        assert payload["vertical_motion_discarded"] is not None
        assert payload["vertical_motion_discarded"]["range_m"] == pytest.approx(0.0, abs=1e-9)

    def test_config_digest_present_and_consistent_across_artifacts(self, tmp_path):
        import json

        config = EvaluationConfig.load(CONFIG_PATH)
        run_evaluation(config, tmp_path)
        with (tmp_path / "metrics.json").open() as f:
            metrics = json.load(f)
        with (tmp_path / "alignment.json").open() as f:
            alignment = json.load(f)

        assert metrics["config_digest"] == alignment["config_digest"]
        assert len(metrics["config_digest"]) == 64  # sha256 hex digest length

    def test_reproducibility_identical_runs_give_identical_metrics(self, tmp_path):
        # SC-005: two independent executions of the same configuration on the same
        # dataset must produce identical metric values.
        config = EvaluationConfig.load(CONFIG_PATH)

        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        result1 = run_evaluation(config, out1)
        result2 = run_evaluation(config, out2)

        assert result1["ate"]["rmse"] == result2["ate"]["rmse"]
        assert result1["alignment"].rotation_deg == result2["alignment"].rotation_deg
        assert result1["alignment"].scale == result2["alignment"].scale
        assert result1["alignment"].translation == result2["alignment"].translation
