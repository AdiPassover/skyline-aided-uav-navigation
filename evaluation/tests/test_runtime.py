"""Tests for runtime measurement (FR-047 through FR-049)."""

import json

import numpy as np
import pytest

from naveval.metrics import runtime_statistics
from naveval.report import EvaluationConfig, write_metrics


class TestRuntimeStatistics:
    def test_known_distribution_values(self):
        ns = np.arange(1, 101) * 1_000_000.0  # 1..100 ms
        result = runtime_statistics(ns)
        assert result["n_frames"] == 100
        assert result["mean_ms"] == pytest.approx(50.5)
        assert result["median_ms"] == pytest.approx(50.5)
        assert result["min_ms"] == pytest.approx(1.0)
        assert result["max_ms"] == pytest.approx(100.0)
        assert result["p90_ms"] == pytest.approx(float(np.percentile(np.arange(1, 101), 90)))
        assert result["p99_ms"] == pytest.approx(float(np.percentile(np.arange(1, 101), 99)))

    def test_distribution_not_just_mean(self):
        # A long tail must show p99 far above the mean -- a single mean value would
        # hide exactly the occasional slow frames that matter (FR-047).
        ns = np.concatenate([np.full(99, 10_000_000.0), [500_000_000.0]])  # 99x10ms + one 500ms
        result = runtime_statistics(ns)
        assert result["mean_ms"] < result["p99_ms"]
        assert result["max_ms"] == pytest.approx(500.0)
        assert result["median_ms"] == pytest.approx(10.0)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            runtime_statistics(np.array([]))

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            runtime_statistics(np.array([1.0, -1.0]))


class TestRuntimeSectionInMetricsJson:
    """FR-048: runtime reported separately from accuracy. FR-049: never presented as
    onboard performance when `is_target_hardware` is false."""

    @staticmethod
    def _minimal_payload(tmp_path, runtime_result, is_target_hardware, environment=None):
        config = EvaluationConfig(
            evaluation_id="runtime-test", run_record_path=tmp_path / "run", dataset_path=tmp_path / "dataset",
        )
        write_metrics(
            tmp_path, config,
            run_id="r", dataset_id="d", evidence_tier="T3", evidence_caveat=None,
            ground_truth_class={"position": "simulator_exact", "heading": None, "height": None},
            frame_counts={
                "total": 1, "processed": 1, "evaluated": 1,
                "excluded_out_of_span": 0, "excluded_insufficient_valid_gt": 0,
            },
            ate_result={"rmse": 1.0, "rmse_normalized": 0.1},
            drift_result={"drift_rate": 0.0, "total_distance_m": 1.0},
            endpoint_error_m=1.0,
            rpe_results={}, rpe_diagnostic_results={},
            position_quality_class="simulator_exact",
            runtime_result=runtime_result,
            environment=environment,
            is_target_hardware=is_target_hardware,
        )
        with (tmp_path / "metrics.json").open() as f:
            return json.load(f)

    @staticmethod
    def _sample_runtime_result():
        return {
            "n_frames": 10, "mean_ms": 5.0, "median_ms": 5.0,
            "p90_ms": 6.0, "p99_ms": 7.0, "min_ms": 1.0, "max_ms": 8.0, "stddev_ms": 1.0,
        }

    def test_runtime_separate_from_accuracy_metrics(self, tmp_path):
        payload = self._minimal_payload(tmp_path, self._sample_runtime_result(), is_target_hardware=False)
        assert "runtime" in payload
        assert payload["runtime"]["mean_ms"] == pytest.approx(5.0)
        # No runtime figure leaks into the accuracy metrics array (FR-048).
        assert all("mean_ms" not in m for m in payload["metrics"])
        assert all(m["name"] not in {"mean_ms", "runtime", "median_ms"} for m in payload["metrics"])

    def test_non_target_hardware_carries_disclaimer(self, tmp_path):
        payload = self._minimal_payload(
            tmp_path, self._sample_runtime_result(), is_target_hardware=False, environment={"hostname": "dev"},
        )
        assert payload["runtime"]["is_target_hardware"] is False
        assert "onboard" in payload["runtime"]["disclaimer"].lower()
        assert payload["runtime"]["environment"] == {"hostname": "dev"}

    def test_target_hardware_omits_disclaimer(self, tmp_path):
        payload = self._minimal_payload(tmp_path, self._sample_runtime_result(), is_target_hardware=True)
        assert payload["runtime"]["is_target_hardware"] is True
        assert "disclaimer" not in payload["runtime"]

    def test_no_runtime_result_yields_null_section(self, tmp_path):
        payload = self._minimal_payload(tmp_path, None, is_target_hardware=False)
        assert payload["runtime"] is None
