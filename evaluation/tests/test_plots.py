"""Tests for figure generation (FR-052). Phase 3 added trajectory_xy; Phase 10 (T085)
covers the remaining four: error_vs_time, error_vs_distance, yaw_error, rpe_by_length,
plus the config_digest stamping and labelled-empty-panel behaviour common to all of them.

Stamping is verified by rendering to SVG rather than PNG in these tests specifically:
matplotlib embeds text as searchable XML in SVG output (the same `_stamp`/`ax.text` calls
as the PNG the CLI actually writes), so the digest substring can be asserted directly
without OCR or a raster diff.
"""

import numpy as np
import pytest

from naveval.plots import (
    plot_error_vs_distance,
    plot_error_vs_time,
    plot_rpe_by_length,
    plot_trajectory_xy,
    plot_yaw_error,
)


class TestTrajectoryXyPlot:
    def test_writes_a_file(self, tmp_path):
        gt = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])
        aligned = gt + 0.1
        out_path = tmp_path / "figures" / "trajectory_xy.png"

        result_path = plot_trajectory_xy(
            aligned, gt, evaluation_id="test-eval", config_digest="a" * 64, out_path=out_path
        )

        assert result_path == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_creates_parent_directory(self, tmp_path):
        gt = np.array([[0.0, 0.0], [1.0, 0.0]])
        out_path = tmp_path / "does" / "not" / "exist" / "yet" / "plot.png"

        plot_trajectory_xy(gt, gt, "eval-id", "digest", out_path)
        assert out_path.exists()

    def test_stamped_with_evaluation_id_and_config_digest(self, tmp_path):
        gt = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
        out_path = tmp_path / "trajectory_xy.svg"  # SVG: text is searchable, unlike PNG

        plot_trajectory_xy(gt, gt, evaluation_id="stamp-test-eval", config_digest="deadbeef" * 8, out_path=out_path)

        svg_text = out_path.read_text(encoding="utf-8")
        assert "stamp-test-eval" in svg_text
        assert "deadbeef" in svg_text  # _stamp truncates the digest to its first 12 chars


class TestErrorVsTimePlot:
    def test_writes_a_file_with_data(self, tmp_path):
        t = np.array([0.0, 0.1, 0.2, 0.3])
        err = np.array([0.0, 0.5, 0.4, 0.6])
        out_path = tmp_path / "error_vs_time.png"

        plot_error_vs_time(t, err, "eval-id", "digest", out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_empty_input_yields_labelled_panel_not_missing_file(self, tmp_path):
        out_path = tmp_path / "error_vs_time.svg"
        plot_error_vs_time(np.zeros(0), np.zeros(0), "eval-id", "digest", out_path)

        assert out_path.exists()
        svg_text = out_path.read_text(encoding="utf-8")
        assert "no error trace" in svg_text.lower() or "no synchronised" in svg_text.lower()

    def test_custom_omission_reason_is_rendered(self, tmp_path):
        out_path = tmp_path / "error_vs_time.svg"
        plot_error_vs_time(
            np.zeros(0), np.zeros(0), "eval-id", "digest", out_path,
            omission_reason="a specific stated reason for this test",
        )
        svg_text = out_path.read_text(encoding="utf-8")
        assert "a specific stated reason for this test" in svg_text


class TestErrorVsDistancePlot:
    def test_writes_a_file_with_data(self, tmp_path):
        d = np.array([0.0, 1.0, 2.0, 3.0])
        err = np.array([0.0, 0.2, 0.3, 0.5])
        out_path = tmp_path / "error_vs_distance.png"

        plot_error_vs_distance(d, err, "eval-id", "digest", out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_empty_input_yields_labelled_panel(self, tmp_path):
        out_path = tmp_path / "error_vs_distance.svg"
        plot_error_vs_distance(np.zeros(0), np.zeros(0), "eval-id", "digest", out_path)

        assert out_path.exists()
        svg_text = out_path.read_text(encoding="utf-8")
        assert "no error trace" in svg_text.lower() or "no synchronised" in svg_text.lower()


class TestYawErrorPlot:
    def test_writes_a_file_with_data(self, tmp_path):
        t = np.array([0.0, 0.1, 0.2])
        yaw_err = np.array([1.0, -2.0, 0.5])
        out_path = tmp_path / "yaw_error.png"

        plot_yaw_error(t, yaw_err, "eval-id", "digest", out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_no_heading_ground_truth_yields_labelled_panel_with_reason(self, tmp_path):
        out_path = tmp_path / "yaw_error.svg"
        plot_yaw_error(
            np.zeros(0), None, "eval-id", "digest", out_path,
            omission_reason="no heading ground truth in dataset",
        )

        assert out_path.exists()
        svg_text = out_path.read_text(encoding="utf-8")
        assert "no heading ground truth in dataset" in svg_text

    def test_default_message_when_no_reason_given(self, tmp_path):
        out_path = tmp_path / "yaw_error.svg"
        plot_yaw_error(np.zeros(0), None, "eval-id", "digest", out_path)

        svg_text = out_path.read_text(encoding="utf-8")
        assert "unsupported" in svg_text.lower()


class TestRpeByLengthPlot:
    def test_all_lengths_available(self, tmp_path):
        rpe = {5.0: {"rmse": 1.2}, 10.0: {"rmse": 2.1}, 25.0: {"rmse": 4.0}}
        out_path = tmp_path / "rpe_by_length.png"

        plot_rpe_by_length(rpe, "eval-id", "digest", out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_partially_omitted_lengths_are_annotated_not_silently_dropped(self, tmp_path):
        rpe = {5.0: {"rmse": 1.2}, 10.0: {"rmse": 2.1}, 25.0: None, 50.0: None}
        out_path = tmp_path / "rpe_by_length.svg"

        plot_rpe_by_length(
            rpe, "eval-id", "digest", out_path,
            omitted_reasons={25.0: "trajectory shorter than 25.0 m", 50.0: "trajectory shorter than 50.0 m"},
        )

        assert out_path.exists()
        svg_text = out_path.read_text(encoding="utf-8")
        assert "trajectory shorter than 25.0 m" in svg_text
        assert "trajectory shorter than 50.0 m" in svg_text

    def test_all_lengths_omitted_yields_labelled_empty_panel(self, tmp_path):
        rpe = {5.0: None, 10.0: None}
        out_path = tmp_path / "rpe_by_length.svg"

        plot_rpe_by_length(rpe, "eval-id", "digest", out_path)

        assert out_path.exists()
        svg_text = out_path.read_text(encoding="utf-8")
        assert "no rpe length" in svg_text.lower()

    def test_no_lengths_configured_yields_labelled_empty_panel(self, tmp_path):
        out_path = tmp_path / "rpe_by_length.svg"
        plot_rpe_by_length({}, "eval-id", "digest", out_path)

        assert out_path.exists()
        svg_text = out_path.read_text(encoding="utf-8")
        assert "no rpe length" in svg_text.lower()


class TestConfigDigestStampingAcrossNewFigures:
    """T085: every new figure carries the evaluation_id/config_digest stamp, not just
    trajectory_xy (already covered above) -- FR-051's traceability requirement applies
    uniformly, so it is checked uniformly rather than once and assumed."""

    @pytest.mark.parametrize("plot_fn,args", [
        (plot_error_vs_time, (np.array([0.0, 0.1]), np.array([0.0, 0.1]))),
        (plot_error_vs_distance, (np.array([0.0, 1.0]), np.array([0.0, 0.1]))),
        (plot_yaw_error, (np.array([0.0, 0.1]), np.array([1.0, -1.0]))),
    ])
    def test_stamped(self, tmp_path, plot_fn, args):
        out_path = tmp_path / "figure.svg"
        plot_fn(*args, "stamp-check-eval", "cafebabe" * 8, out_path)

        svg_text = out_path.read_text(encoding="utf-8")
        assert "stamp-check-eval" in svg_text
        assert "cafebabe" in svg_text

    def test_rpe_by_length_stamped(self, tmp_path):
        out_path = tmp_path / "figure.svg"
        plot_rpe_by_length({5.0: {"rmse": 1.0}}, "stamp-check-eval", "cafebabe" * 8, out_path)

        svg_text = out_path.read_text(encoding="utf-8")
        assert "stamp-check-eval" in svg_text
        assert "cafebabe" in svg_text
