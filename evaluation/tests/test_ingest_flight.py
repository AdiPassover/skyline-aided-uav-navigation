"""T082: verify the real-flight ingest path (naveval.ingest_flight) against *constructed*
telemetry with known geodetic coordinates and a known injected clock offset -- no real flight
data exists yet (tasks.md "Explicitly NOT tasks in this feature"), so every value here is
either hand-derived or independently recomputed from naveval.frames, not taken on faith.
"""

from __future__ import annotations

import numpy as np
import pytest

from naveval.dataset import load_dataset
from naveval.frames import geodetic_to_enu
from naveval.ingest_flight import (
    ClockSync,
    FrameRecord,
    GroundTruthRow,
    IngestError,
    TelemetrySample,
    build_dataset_json,
    classify_heading_quality,
    classify_position_quality,
    convert_geodetic_track_to_enu,
    estimate_clock_sync,
    extract_frames,
    write_dataset,
)

ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT = 32.1093, 34.8555, 42.0


class TestExtractFrames:
    def test_indexes_and_validates(self):
        records = extract_frames([0.0, 0.033, 0.066], ["a.png", "b.png", "c.png"])
        assert [r.frame_index for r in records] == [0, 1, 2]
        assert records[1].timestamp_s == pytest.approx(0.033)
        assert records[2].image_path == "c.png"

    def test_rejects_non_increasing_timestamps(self):
        with pytest.raises(IngestError):
            extract_frames([0.0, 0.033, 0.033], ["a.png", "b.png", "c.png"])

    def test_rejects_empty(self):
        with pytest.raises(IngestError):
            extract_frames([], [])

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(IngestError):
            extract_frames([0.0, 0.1], ["a.png"])


class TestConvertGeodeticTrackToEnu:
    def test_matches_geodetic_to_enu_directly(self):
        samples = [
            TelemetrySample(0.0, ORIGIN_LAT, ORIGIN_LON, 42.0, 90.0, "3D,12sat", True),
            TelemetrySample(0.2, ORIGIN_LAT + 0.0001, ORIGIN_LON + 0.0001, 45.0, 95.0, "3D,12sat", True),
        ]
        rows = convert_geodetic_track_to_enu(samples, ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)

        for s, row in zip(samples, rows):
            expected_east, expected_north, expected_up = geodetic_to_enu(
                s.lat_deg, s.lon_deg, s.alt_m, ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT
            )
            assert row.east_m == pytest.approx(expected_east)
            assert row.north_m == pytest.approx(expected_north)
            assert row.up_m == pytest.approx(expected_up)
            assert row.heading_deg == pytest.approx(s.heading_deg)

    def test_invalid_sample_yields_empty_position_not_fabricated(self):
        samples = [
            TelemetrySample(0.0, ORIGIN_LAT, ORIGIN_LON, 42.0, 90.0, "3D,12sat", True),
            TelemetrySample(0.2, None, None, None, None, "2D,4sat", False),
        ]
        rows = convert_geodetic_track_to_enu(samples, ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)
        assert rows[1].east_m is None
        assert rows[1].north_m is None
        assert rows[1].up_m is None
        assert rows[1].heading_deg is None
        assert rows[1].valid is False

    def test_enu_math_heading_convention_converts_to_compass(self):
        # ENU-math 0 deg (pointing along +East) is compass 90 deg (East).
        samples = [TelemetrySample(0.0, ORIGIN_LAT, ORIGIN_LON, 42.0, 0.0, "3D", True)]
        rows = convert_geodetic_track_to_enu(
            samples, ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT, heading_convention="enu_math_ccw_from_east"
        )
        assert rows[0].heading_deg == pytest.approx(90.0)

    def test_rejects_unsupported_heading_convention(self):
        samples = [TelemetrySample(0.0, ORIGIN_LAT, ORIGIN_LON, 42.0, 0.0, "3D", True)]
        with pytest.raises(IngestError):
            convert_geodetic_track_to_enu(samples, ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT, heading_convention="ned")

    def test_rejects_non_increasing_timestamps(self):
        samples = [
            TelemetrySample(0.5, ORIGIN_LAT, ORIGIN_LON, 42.0, 0.0, "3D", True),
            TelemetrySample(0.5, ORIGIN_LAT, ORIGIN_LON, 42.0, 0.0, "3D", True),
        ]
        with pytest.raises(IngestError):
            convert_geodetic_track_to_enu(samples, ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)


class TestEstimateClockSync:
    def test_single_event_gives_offset_only(self):
        gt_timestamps = np.array([0.0, 1.0, 2.0])
        sync = estimate_clock_sync(gt_timestamps, [(10.412, 10.0)])
        assert sync.offset_s == pytest.approx(0.412)
        assert sync.drift_s_per_s == 0.0
        assert "one event" in sync.source_description or "single" in sync.source_description

    def test_two_events_recover_known_injected_offset_and_drift(self):
        # Construct (frame_time, gt_time) pairs that are *exactly* consistent with
        # naveval.sync.synchronize's own anchor formula, then check estimate_clock_sync
        # recovers the injected offset/drift used to build them.
        injected_offset = -0.412
        injected_drift = 2.5e-4
        gt_timestamps = np.array([100.0, 130.0, 160.0])  # anchor = 100.0
        anchor = gt_timestamps[0]

        def frame_time_at(gt_time):
            return anchor + (gt_time - anchor) * (1.0 + injected_drift) + injected_offset

        g1, g2 = 105.0, 155.0
        f1, f2 = frame_time_at(g1), frame_time_at(g2)

        sync = estimate_clock_sync(gt_timestamps, [(f1, g1), (f2, g2)])
        assert sync.offset_s == pytest.approx(injected_offset, abs=1e-9)
        assert sync.drift_s_per_s == pytest.approx(injected_drift, abs=1e-12)

        # And the recovered (offset, drift) must reproduce f1/f2 through the exact same
        # anchor formula naveval.sync.synchronize uses -- the compatibility this function
        # exists for.
        recovered_f1 = anchor + (g1 - anchor) * (1.0 + sync.drift_s_per_s) + sync.offset_s
        recovered_f2 = anchor + (g2 - anchor) * (1.0 + sync.drift_s_per_s) + sync.offset_s
        assert recovered_f1 == pytest.approx(f1, abs=1e-9)
        assert recovered_f2 == pytest.approx(f2, abs=1e-9)

    def test_rejects_zero_events(self):
        with pytest.raises(IngestError):
            estimate_clock_sync(np.array([0.0]), [])

    def test_rejects_more_than_two_events(self):
        with pytest.raises(IngestError):
            estimate_clock_sync(np.array([0.0]), [(1.0, 0.0), (2.0, 1.0), (3.0, 2.0)])

    def test_rejects_coincident_gt_times(self):
        with pytest.raises(IngestError):
            estimate_clock_sync(np.array([0.0]), [(1.0, 0.0), (1.5, 0.0)])


class TestClassifyQuality:
    def test_all_rtk_fixed_is_rtk_gnss(self):
        assert classify_position_quality(["RTK_FIXED", "rtk_fixed", "RTK,FIXED"]) == "rtk_gnss"

    def test_all_3d_is_consumer_gnss(self):
        assert classify_position_quality(["3D,14sat", "3D,9sat"]) == "consumer_gnss"

    def test_mixed_or_2d_is_unknown(self):
        assert classify_position_quality(["3D,14sat", "2D,4sat"]) == "unknown"

    def test_empty_is_unknown(self):
        assert classify_position_quality([]) == "unknown"

    def test_heading_quality_reflects_availability(self):
        assert classify_heading_quality(True) == "fc_heading"
        assert classify_heading_quality(False) == "unknown"


class TestBuildDatasetJsonAndWrite:
    def test_full_round_trip_loads_via_naveval_dataset(self, tmp_path):
        clock_sync = ClockSync(offset_s=-0.412, drift_s_per_s=0.0, source_description="test sync")
        dataset_json = build_dataset_json(
            dataset_id="flight-test-001",
            dataset_revision="v1",
            evidence_tier="T3",
            clock_sync=clock_sync,
            position_quality_class="consumer_gnss",
            heading_quality_class="fc_heading",
            metadata={
                "flight_id": "flight-test-001", "trajectory_type": "straight",
                "frame_rate_hz": 30.0, "image_width": 1920, "image_height": 1080,
                "environment": "test", "nominal_altitude_m": 40.0, "nominal_speed_ms": 4.0,
                "capture_date": "2026-08-13", "notes": "constructed for T082",
            },
            local_frame_origin={"lat_deg": ORIGIN_LAT, "lon_deg": ORIGIN_LON, "alt_m": ORIGIN_ALT},
        )

        frame_records = [
            FrameRecord(0, 0.000, "images/frame_000000.png"),
            FrameRecord(1, 0.033, "images/frame_000001.png"),
        ]
        samples = [
            TelemetrySample(0.412, ORIGIN_LAT, ORIGIN_LON, 42.0, 90.0, "3D,12sat", True),
            TelemetrySample(0.612, ORIGIN_LAT + 0.0001, ORIGIN_LON, 42.0, 90.0, "3D,12sat", True),
        ]
        gt_rows = convert_geodetic_track_to_enu(samples, ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)

        write_dataset(tmp_path, dataset_json, frame_records, gt_rows)

        loaded = load_dataset(tmp_path)
        assert loaded.source_type == "real_flight"
        assert loaded.evidence_tier == "T3"
        assert loaded.evidence_caveat is None
        assert loaded.clock_offset_s == pytest.approx(-0.412)
        assert loaded.position_quality.quality_class == "consumer_gnss"
        assert loaded.heading_quality.quality_class == "fc_heading"
        assert loaded.height_quality is None
        assert loaded.frame_indices.tolist() == [0, 1]
        assert loaded.gt_east.size == 2
        expected_north = geodetic_to_enu(
            ORIGIN_LAT + 0.0001, ORIGIN_LON, 42.0, ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT
        )[1]
        assert loaded.gt_north[1] == pytest.approx(expected_north)

    def test_rtk_gnss_heading_quality_class_is_accepted_and_round_trips(self, tmp_path):
        """Found 2026-08-14 while wiring naveval.ingest_mars_lvig's MCAP source: the loader
        (naveval.dataset.VALID_QUALITY_CLASSES) and the support matrix
        (naveval.support._HEADING_METRIC_MATRIX) already treated "rtk_gnss" as a valid heading
        class -- e.g. a DJI M300's dual-antenna RTK heading -- but this writer's own local set
        had never been updated to match, so no producer could actually declare it."""
        clock_sync = ClockSync(0.0, 0.0, "test")
        dataset_json = build_dataset_json(
            dataset_id="rtk-heading-test", dataset_revision="v1", evidence_tier="T3",
            clock_sync=clock_sync, position_quality_class="rtk_gnss",
            heading_quality_class="rtk_gnss",
            metadata={
                "flight_id": "d", "trajectory_type": "custom", "frame_rate_hz": 10.0,
                "image_width": 100, "image_height": 100, "environment": "", "nominal_altitude_m": None,
                "nominal_speed_ms": None, "capture_date": "", "notes": "",
            },
            local_frame_origin={"lat_deg": ORIGIN_LAT, "lon_deg": ORIGIN_LON, "alt_m": ORIGIN_ALT},
        )
        write_dataset(tmp_path, dataset_json, [FrameRecord(0, 0.0, "images/f0.png")], [])
        loaded = load_dataset(tmp_path)
        assert loaded.heading_quality.quality_class == "rtk_gnss"

    def test_rejects_invalid_position_quality_class(self):
        clock_sync = ClockSync(0.0, 0.0, "test")
        with pytest.raises(IngestError):
            build_dataset_json(
                "d", "v1", "T3", clock_sync, "military_grade", "fc_heading",
                metadata={}, local_frame_origin=None,
            )

    def test_no_groundtruth_csv_written_when_no_samples(self, tmp_path):
        clock_sync = ClockSync(0.0, 0.0, "test")
        dataset_json = build_dataset_json(
            "d", "v1", "T3", clock_sync, "unknown", "unknown",
            metadata={
                "flight_id": "d", "trajectory_type": "hover", "frame_rate_hz": 30.0,
                "image_width": 100, "image_height": 100, "environment": "", "nominal_altitude_m": None,
                "nominal_speed_ms": None, "capture_date": "", "notes": "",
            },
        )
        write_dataset(tmp_path, dataset_json, [FrameRecord(0, 0.0, "images/f0.png")], [])
        assert not (tmp_path / "groundtruth.csv").exists()

        loaded = load_dataset(tmp_path)
        assert not loaded.has_ground_truth
