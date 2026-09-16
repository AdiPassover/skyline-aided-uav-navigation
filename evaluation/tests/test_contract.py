"""Contract tests: golden-fixture round trip and malformed-input rejection.

Per DEC-002, the run record and dataset are the sole interface between estimation and
evaluation. These tests are what prevent writer and reader from silently drifting apart.
"""

from pathlib import Path

import pytest

from naveval.dataset import load_dataset
from naveval.errors import ContractViolationError, DatasetMismatchError, SchemaVersionError
from naveval.runrecord import load_run_record, verify_matches_dataset

FIXTURES = Path(__file__).parent / "fixtures"


class TestGoldenDataset:
    def test_loads_without_error(self):
        ds = load_dataset(FIXTURES / "golden_dataset")
        assert ds.dataset_id == "golden-001"

    def test_scalar_fields_recovered_exactly(self):
        ds = load_dataset(FIXTURES / "golden_dataset")
        assert ds.schema_version == "1.0.0"
        assert ds.dataset_revision == "v1"
        assert ds.source_type == "synthetic"
        assert ds.evidence_tier == "T1"
        assert ds.evidence_caveat is None
        assert ds.frame_convention == "ENU"
        assert ds.heading_convention == "compass_cw_from_north"
        assert ds.clock_offset_s == 0.0

    def test_frames_recovered_exactly(self):
        ds = load_dataset(FIXTURES / "golden_dataset")
        assert len(ds.frame_indices) == 10
        assert ds.frame_indices.tolist() == list(range(10))
        assert ds.frame_timestamps[0] == pytest.approx(0.0)
        assert ds.frame_timestamps[9] == pytest.approx(0.9)
        assert ds.frame_image_paths[3] == "images/frame_000003.png"

    def test_ground_truth_recovered_exactly(self):
        ds = load_dataset(FIXTURES / "golden_dataset")
        assert ds.has_ground_truth
        assert len(ds.gt_timestamps) == 10
        # frame 3 (t=0.3): east=5.0000, north=1.3397, up=40.0, heading=60.0
        assert ds.gt_east[3] == pytest.approx(5.0000)
        assert ds.gt_north[3] == pytest.approx(1.3397)
        assert ds.gt_up[3] == pytest.approx(40.0)
        assert ds.gt_heading[3] == pytest.approx(60.0)
        assert bool(ds.gt_valid[3]) is True

    def test_quality_classes_recovered(self):
        ds = load_dataset(FIXTURES / "golden_dataset")
        assert ds.position_quality.quality_class == "simulator_exact"
        assert ds.heading_quality.quality_class == "simulator_exact"
        assert ds.height_quality.quality_class == "simulator_exact"

    def test_metadata_recovered(self):
        ds = load_dataset(FIXTURES / "golden_dataset")
        assert ds.metadata.flight_id == "golden-001"
        assert ds.metadata.trajectory_type == "straight"
        assert ds.metadata.frame_rate_hz == pytest.approx(10.0)
        assert ds.metadata.image_width == 64


class TestGoldenRunRecord:
    def test_loads_without_error(self):
        run = load_run_record(FIXTURES / "golden_run")
        assert run.manifest.run_id == "golden-run-001"

    def test_manifest_recovered_exactly(self):
        run = load_run_record(FIXTURES / "golden_run")
        assert run.manifest.schema_version == "1.0.0"
        assert run.manifest.dataset_id == "golden-001"
        assert run.manifest.dataset_revision == "v1"
        assert run.manifest.frame_count == 10
        assert run.manifest.processed_count == 10
        assert run.manifest.completed is True
        assert run.manifest.environment.is_target_hardware is False
        assert run.manifest.environment.heap_max_mb == 1024
        assert run.manifest.estimator_config["shrinkScale"] == pytest.approx(0.5)
        assert run.manifest.estimator_config["inlierThresholdSq"] == pytest.approx(3.0)

    def test_frame_estimates_recovered_exactly(self):
        run = load_run_record(FIXTURES / "golden_run")
        assert len(run.frame_estimates) == 10

        first = run.frame_estimates[0]
        assert first.frame_index == 0
        assert first.est_x == pytest.approx(-112.1217)
        assert first.est_y == pytest.approx(-32.0736)
        assert first.est_z is None
        # gt_heading - 15deg (Phase 5, 2026-08-12): yaw-consistent with the fixture's
        # positions (fitted rotation=15deg), so transform_yaw recovers gt_heading[0]=90
        # exactly. Corrects the deliberately-deferred inconsistency documented in the
        # research-log 2026-08-12 addendum, ahead of yaw-metric tests needing it.
        assert first.est_yaw_deg == pytest.approx(75.0)
        assert first.success is True
        assert first.event == "init"
        assert first.reference_id == 0
        assert first.homography == pytest.approx((0.5, 0.0, 100.0, 0.0, 0.5, 100.0, 0.0, 0.0, 1.0))
        assert first.track_count == 285
        assert first.inlier_count == 268
        assert first.process_time_ns == 12000000

    def test_empty_vs_zero_distinguished(self):
        run = load_run_record(FIXTURES / "golden_run")
        # est_z is empty (None), never a numeric zero -- FR-015.
        for fe in run.frame_estimates:
            assert fe.est_z is None

    def test_restart_row_preserved(self):
        run = load_run_record(FIXTURES / "golden_run")
        restart_row = run.frame_estimates[5]
        assert restart_row.event == "restart"
        assert restart_row.success is False
        assert restart_row.reference_id == 1
        assert restart_row.track_count == 22
        assert restart_row.inlier_count == 0
        # Position is still recorded on a failed/restart row (contract example).
        assert restart_row.est_x == pytest.approx(54.3574)

    def test_reference_id_increments_after_restart(self):
        run = load_run_record(FIXTURES / "golden_run")
        ref_ids = [fe.reference_id for fe in run.frame_estimates]
        assert ref_ids == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    def test_numpy_views_match_row_order(self):
        run = load_run_record(FIXTURES / "golden_run")
        assert run.est_x[0] == pytest.approx(-112.1217)
        assert run.success.tolist() == [True, True, True, True, True, False, True, True, True, True]


class TestCrossCheck:
    def test_matching_dataset_passes(self):
        run = load_run_record(FIXTURES / "golden_run")
        ds = load_dataset(FIXTURES / "golden_dataset")
        verify_matches_dataset(run, ds)  # must not raise

    def test_dataset_id_mismatch_rejected(self):
        run = load_run_record(FIXTURES / "bad" / "dataset_mismatch")
        ds = load_dataset(FIXTURES / "golden_dataset")
        with pytest.raises(DatasetMismatchError):
            verify_matches_dataset(run, ds)


class TestMalformedRejection:
    def test_major_version_2_rejected(self):
        with pytest.raises(SchemaVersionError):
            load_run_record(FIXTURES / "bad" / "major_version_2")

    def test_truncated_record_rejected(self):
        # manifest claims processed_count=10 but frames.csv has only 8 data rows.
        with pytest.raises(ContractViolationError, match="truncated|stale|rows"):
            load_run_record(FIXTURES / "bad" / "truncated")

    def test_minor_version_with_unknown_column_loads_and_is_ignored(self):
        # v1.1.0: same major version, one unknown trailing column. Must load
        # successfully and recover the same known values as the golden fixture.
        run = load_run_record(FIXTURES / "bad" / "minor_extra_column")
        assert run.manifest.schema_version == "1.1.0"
        assert len(run.frame_estimates) == 10
        assert run.frame_estimates[0].est_x == pytest.approx(-112.1217)


class TestJavaProducedRunRecord:
    """Closes the writer<->reader loop (T047): `JavaProducedRunFixtureTest` regenerates this
    fixture from the real Java capture path (real StitchingEstimator over synthetic textured
    frames, not a mock) on every `./gradlew test` run; this test proves the Python reader
    recovers what was actually written, not just what a hand-authored fixture claims.
    """

    def test_loads_without_error(self):
        run = load_run_record(FIXTURES / "java_produced_run")
        assert run.manifest.run_id == "java-produced-fixture"

    def test_manifest_recovered_exactly(self):
        run = load_run_record(FIXTURES / "java_produced_run")
        assert run.manifest.schema_version == "1.0.0"
        assert run.manifest.dataset_id == "java-produced-fixture-dataset"
        assert run.manifest.dataset_revision == "v1"
        assert run.manifest.estimator_id == "stitching-vo"
        assert run.manifest.frame_count == 6
        assert run.manifest.processed_count == 6
        assert run.manifest.completed is True
        assert run.manifest.environment.is_target_hardware is False
        assert run.manifest.estimator_config["shrinkScale"] == pytest.approx(0.5)

    def test_frame_estimates_recovered_exactly(self):
        run = load_run_record(FIXTURES / "java_produced_run")
        assert len(run.frame_estimates) == 6

        first = run.frame_estimates[0]
        assert first.frame_index == 0
        assert first.timestamp_s == pytest.approx(0.0)
        assert first.est_x == pytest.approx(0.0)
        assert first.est_y == pytest.approx(0.0)
        assert first.est_z is None
        assert first.est_yaw_deg == pytest.approx(0.0)
        assert first.success is True
        assert first.event == "init"
        assert first.reference_id == 0
        assert first.homography == pytest.approx((0.5, 0.0, 30.0, 0.0, 0.5, 30.0, 0.0, 0.0, 1.0))
        assert first.track_count == 50
        assert first.inlier_count == 50

    def test_real_estimator_recovered_genuine_translation(self):
        # The real StitchingEstimator tracked a real rightward/downward pan across a textured
        # synthetic scene (JavaProducedRunFixtureTest's per-frame integer-pixel crop shift) --
        # this is not fabricated ground truth, so only loose bounds are asserted, not exact
        # injected values.
        run = load_run_record(FIXTURES / "java_produced_run")
        assert run.success.tolist() == [True] * 6
        # Estimated x should grow monotonically as the crop window shifts right frame to frame.
        assert all(a < b for a, b in zip(run.est_x, run.est_x[1:]))
        # Yaw should stay near the [0, 360) boundary -- a near-pure translation, no injected
        # rotation -- and must never violate the invariant runrecord.py already enforces on load.
        for fe in run.frame_estimates:
            assert 0.0 <= fe.est_yaw_deg < 360.0
            assert (fe.est_yaw_deg < 5.0) or (fe.est_yaw_deg > 355.0)

    def test_scale_freedom_columns_present_never_zero_for_missing(self):
        run = load_run_record(FIXTURES / "java_produced_run")
        for fe in run.frame_estimates:
            assert fe.est_z is None  # FR-015: always empty, this feature does not estimate height
            assert fe.homography is not None  # real estimator always has a current homography
            assert fe.track_count is not None and fe.track_count > 0
            assert fe.inlier_count is not None and fe.inlier_count >= 0


class TestDatasetConventionRejection:
    def test_wrong_frame_convention_rejected(self, tmp_path):
        _write_dataset_with_override(tmp_path, "frame_convention", "NED")
        from naveval.errors import ConventionError

        with pytest.raises(ConventionError):
            load_dataset(tmp_path)

    def test_wrong_heading_convention_rejected(self, tmp_path):
        _write_dataset_with_override(tmp_path, "heading_convention", "enu_math_ccw_from_east")
        from naveval.errors import ConventionError

        with pytest.raises(ConventionError):
            load_dataset(tmp_path)

    def test_java_simulator_without_caveat_rejected(self, tmp_path):
        _write_dataset_with_override(
            tmp_path, "source_type", "java_simulator", extra={"evidence_tier": "T2"}
        )
        with pytest.raises(ContractViolationError, match="evidence_caveat"):
            load_dataset(tmp_path)


def _write_dataset_with_override(tmp_path, key, value, extra=None):
    import json
    import shutil

    shutil.copytree(FIXTURES / "golden_dataset", tmp_path, dirs_exist_ok=True)
    descriptor_path = tmp_path / "dataset.json"
    with descriptor_path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    d[key] = value
    if extra:
        d.update(extra)
    with descriptor_path.open("w", encoding="utf-8") as f:
        json.dump(d, f)
