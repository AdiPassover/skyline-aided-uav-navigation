"""Tests for support-level classification (FR-061, FR-062).

Classification correctness is tested in isolation (`TestClassifyPosition`,
`TestClassifyHeading`) and then again through the actual output-shaping code
(`TestUnsupportedMetricNeverAppearsAsValue`) -- the classifier could be correct while
the wiring around it still leaked a value into `metrics.json` (SC-013).
"""

import json

import pytest

from naveval.support import (
    RPE_SHORT_LONG_THRESHOLD_M,
    SUPPORTED,
    UNSUPPORTED,
    WEAKLY_SUPPORTED,
    classify_heading,
    classify_position,
    rpe_metric_name_for_length,
)


class TestClassifyPosition:
    @pytest.mark.parametrize(
        "metric,quality_class,expected",
        [
            ("ate_rmse", "consumer_gnss", WEAKLY_SUPPORTED),
            ("ate_rmse", "rtk_gnss", SUPPORTED),
            ("ate_rmse", "simulator_exact", SUPPORTED),
            ("ate_rmse_normalised", "consumer_gnss", WEAKLY_SUPPORTED),
            ("drift_per_distance", "consumer_gnss", SUPPORTED),
            ("drift_per_distance", "rtk_gnss", SUPPORTED),
            ("rpe_short", "consumer_gnss", UNSUPPORTED),
            ("rpe_short", "rtk_gnss", SUPPORTED),
            ("rpe_short", "simulator_exact", SUPPORTED),
            ("rpe_long", "consumer_gnss", WEAKLY_SUPPORTED),
            ("rpe_long", "rtk_gnss", SUPPORTED),
            ("endpoint_error", "consumer_gnss", WEAKLY_SUPPORTED),
            ("endpoint_error_closed_loop", "consumer_gnss", SUPPORTED),
            ("endpoint_error_closed_loop", "rtk_gnss", SUPPORTED),
            ("scale_drift_diagnostic", "consumer_gnss", WEAKLY_SUPPORTED),
            ("scale_drift_diagnostic", "rtk_gnss", SUPPORTED),
        ],
    )
    def test_matches_spec_support_matrix(self, metric, quality_class, expected):
        assert classify_position(metric, quality_class) == expected

    def test_unknown_class_fails_closed_for_every_metric(self):
        for metric in [
            "ate_rmse", "ate_rmse_normalised", "drift_per_distance", "rpe_short", "rpe_long",
            "endpoint_error", "endpoint_error_closed_loop", "scale_drift_diagnostic",
        ]:
            assert classify_position(metric, "unknown") == UNSUPPORTED

    def test_none_class_fails_closed(self):
        assert classify_position("ate_rmse", None) == UNSUPPORTED

    def test_undefined_metric_raises(self):
        # A missing classification entry is a programming error (a new metric was
        # added to metrics.py without updating the matrix) -- must fail loudly, not
        # silently default to some support level (FR-062).
        with pytest.raises(KeyError):
            classify_position("not_a_real_metric", "simulator_exact")


class TestClassifyHeading:
    @pytest.mark.parametrize(
        "metric,quality_class,expected",
        [
            ("yaw_error", "fc_heading", WEAKLY_SUPPORTED),
            ("yaw_error", "rtk_gnss", WEAKLY_SUPPORTED),  # conservative default: no dual-antenna flag exists yet
            ("yaw_error", "simulator_exact", SUPPORTED),
            ("yaw_drift", "fc_heading", WEAKLY_SUPPORTED),
            ("yaw_drift", "simulator_exact", SUPPORTED),
        ],
    )
    def test_matches_spec_support_matrix(self, metric, quality_class, expected):
        assert classify_heading(metric, quality_class) == expected

    def test_unknown_class_fails_closed(self):
        assert classify_heading("yaw_error", "unknown") == UNSUPPORTED
        assert classify_heading("yaw_drift", "unknown") == UNSUPPORTED

    def test_none_class_fails_closed(self):
        assert classify_heading("yaw_error", None) == UNSUPPORTED

    def test_undefined_metric_raises(self):
        with pytest.raises(KeyError):
            classify_heading("not_a_real_metric", "simulator_exact")


class TestRpeLengthBucketing:
    def test_below_threshold_is_short(self):
        assert rpe_metric_name_for_length(5.0) == "rpe_short"
        assert rpe_metric_name_for_length(RPE_SHORT_LONG_THRESHOLD_M - 1.0) == "rpe_short"

    def test_at_or_above_threshold_is_long(self):
        assert rpe_metric_name_for_length(RPE_SHORT_LONG_THRESHOLD_M) == "rpe_long"
        assert rpe_metric_name_for_length(500.0) == "rpe_long"


class TestUnsupportedMetricNeverAppearsAsValue:
    """SC-013, exercised through `report.write_metrics` itself, not just the
    classifier in isolation."""

    @staticmethod
    def _minimal_metrics_payload(tmp_path, position_quality_class, rpe_length_m=5.0):
        from naveval.report import EvaluationConfig, write_metrics

        config = EvaluationConfig(
            evaluation_id="support-test",
            run_record_path=tmp_path / "run",
            dataset_path=tmp_path / "dataset",
            rpe_lengths_m=(rpe_length_m,),
        )
        write_metrics(
            tmp_path,
            config,
            run_id="r", dataset_id="d", evidence_tier="T3", evidence_caveat=None,
            ground_truth_class={"position": position_quality_class, "heading": None, "height": None},
            frame_counts={
                "total": 1, "processed": 1, "evaluated": 1,
                "excluded_out_of_span": 0, "excluded_insufficient_valid_gt": 0,
            },
            ate_result={"rmse": 1.0, "rmse_normalized": 0.1},
            drift_result={"drift_rate": 0.01, "total_distance_m": 10.0},
            endpoint_error_m=1.0,
            rpe_results={
                rpe_length_m: {
                    "rmse": 0.5, "n_segments": 1, "length_m": rpe_length_m, "scale_basis": "global_fitted",
                }
            },
            rpe_diagnostic_results={rpe_length_m: None},
            position_quality_class=position_quality_class,
        )
        with (tmp_path / "metrics.json").open() as f:
            return json.load(f)

    def test_short_rpe_excluded_for_consumer_gnss(self, tmp_path):
        payload = self._minimal_metrics_payload(tmp_path, "consumer_gnss", rpe_length_m=5.0)
        names = {m["name"] for m in payload["metrics"]}
        assert "rpe_5m" not in names
        assert "rpe_5m" in payload["unsupported_excluded"]

    def test_same_length_supported_for_simulator_exact(self, tmp_path):
        payload = self._minimal_metrics_payload(tmp_path, "simulator_exact", rpe_length_m=5.0)
        names = {m["name"] for m in payload["metrics"]}
        assert "rpe_5m" in names
        assert "rpe_5m" not in payload["unsupported_excluded"]

    def test_unknown_quality_class_excludes_every_position_metric(self, tmp_path):
        payload = self._minimal_metrics_payload(tmp_path, "unknown", rpe_length_m=5.0)

        # No metric that would have been keyed by the "unknown" position class survives
        # with an actual value -- the only entries with a real value would have to be
        # ones this classifier never touches (there are none here).
        names_with_values = {m["name"] for m in payload["metrics"] if m.get("value") is not None}
        assert names_with_values == set()

        assert set(payload["unsupported_excluded"]) == {
            "ate_rmse", "ate_rmse_normalised", "drift_per_distance", "endpoint_error", "rpe_5m",
        }
