"""T083: the same evaluation config shape runs unchanged across synthetic, simulator, and
constructed-flight datasets (SC-009) -- nothing in `run_evaluation` special-cases
`source_type` -- and declared ground-truth quality propagates into differing support levels
(FR-061), demonstrated end to end rather than only at the `support.py` unit level.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from naveval.evaluate import run_evaluation
from naveval.ingest_flight import (
    ClockSync,
    FrameRecord,
    TelemetrySample,
    build_dataset_json,
    convert_geodetic_track_to_enu,
    write_dataset,
)
from naveval.report import EvaluationConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"

ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT = 32.1093, 34.8555, 42.0


def _synthetic_config() -> EvaluationConfig:
    return EvaluationConfig(
        evaluation_id="source-agnostic-synthetic",
        run_record_path=FIXTURES / "golden_run",
        dataset_path=FIXTURES / "golden_dataset",
    )


def _simulator_config() -> EvaluationConfig:
    dataset_path = REPO_ROOT / "datasets" / "sim-square"
    run_record_path = REPO_ROOT / "runs" / "sim-square-run-v1"
    if not dataset_path.exists() or not run_record_path.exists():
        pytest.skip(
            "datasets/sim-square and runs/sim-square-run-v1 (T080 reference) are not present "
            "in this checkout -- generate them via DatasetRecorderApp/VoRunnerApp first"
        )
    return EvaluationConfig(
        evaluation_id="source-agnostic-simulator",
        run_record_path=run_record_path,
        dataset_path=dataset_path,
    )


def _write_constructed_flight_dataset(root: Path) -> None:
    """A hand-constructed `real_flight`-shaped dataset with `consumer_gnss` quality --
    deliberately weaker ground truth than the synthetic/simulator sources, to exercise
    differing support levels (FR-061/SC-009), not just differing `source_type` labels."""
    n = 6
    samples = [
        TelemetrySample(
            timestamp_s=0.1 * i,
            lat_deg=ORIGIN_LAT + 0.00002 * i,
            lon_deg=ORIGIN_LON,
            alt_m=42.0,
            heading_deg=0.0,
            fix_quality="3D,9sat",
            valid=True,
        )
        for i in range(n)
    ]
    gt_rows = convert_geodetic_track_to_enu(samples, ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)

    clock_sync = ClockSync(offset_s=0.0, drift_s_per_s=0.0, source_description="constructed, same clock")
    dataset_json = build_dataset_json(
        dataset_id="constructed-flight-001",
        dataset_revision="v1",
        evidence_tier="T3",
        clock_sync=clock_sync,
        position_quality_class="consumer_gnss",
        heading_quality_class="fc_heading",
        metadata={
            "flight_id": "constructed-flight-001", "trajectory_type": "straight",
            "frame_rate_hz": 10.0, "image_width": 640, "image_height": 512,
            "environment": "constructed for T083, no real flight data", "nominal_altitude_m": 42.0,
            "nominal_speed_ms": 2.0, "capture_date": "2026-08-13",
            "notes": "constructed telemetry, no real flight data (tasks.md scope boundary)",
        },
        local_frame_origin={"lat_deg": ORIGIN_LAT, "lon_deg": ORIGIN_LON, "alt_m": ORIGIN_ALT},
    )
    frame_records = [FrameRecord(i, 0.1 * i, f"images/frame_{i:06d}.png") for i in range(n)]
    write_dataset(root, dataset_json, frame_records, gt_rows)


def _write_constructed_flight_run(root: Path) -> None:
    """A matching run record -- estimator positions as a known similarity transform of the
    ground truth (rotation + scale + translation), so the Sim(2) fit is well-conditioned
    rather than degenerate. No real capture exists; this is deliberately synthetic-but-shaped
    like a real run record, the same way `golden_run` was hand-authored."""
    manifest = {
        "schema_version": "1.0.0",
        "run_id": "constructed-flight-run-001",
        "dataset_id": "constructed-flight-001",
        "dataset_revision": "v1",
        "estimator_id": "stitching-vo",
        "estimator_version": "test-fixture",
        "estimator_config": {},
        "evaluator_capture_version": "test-fixture",
        "environment": {
            "hostname": "test-host", "os": "test-os", "cpu_model": "test-cpu",
            "jvm_version": "19.0.2", "heap_max_mb": 1024, "is_target_hardware": False,
        },
        "run_timestamp": "2026-08-13T00:00:00Z",
        "frame_count": 6,
        "processed_count": 6,
        "completed": True,
    }
    with (root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    header = (
        "frame_index,timestamp_s,est_x,est_y,est_z,est_yaw_deg,success,event,reference_id,"
        "h00,h01,h02,h10,h11,h12,h20,h21,h22,track_count,inlier_count,process_time_ns"
    )
    lines = [header]
    for i in range(6):
        event = "init" if i == 0 else "none"
        # est_y grows with i, matching the north-only ground-truth track above scaled and offset --
        # a well-conditioned similarity relationship for Sim(2) fitting.
        est_x = 100.0 + i * 0.02
        est_y = i * 5.0
        lines.append(
            f"{i},{0.1 * i},{est_x},{est_y},,0.0,true,{event},0,"
            "0.5,0.0,100.0,0.0,0.5,100.0,0.0,0.0,1.0,200,190,10000000"
        )

    (root / "frames.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _constructed_flight_config(tmp_path: Path) -> EvaluationConfig:
    dataset_root = tmp_path / "dataset"
    run_root = tmp_path / "run"
    run_root.mkdir()
    _write_constructed_flight_dataset(dataset_root)
    _write_constructed_flight_run(run_root)
    return EvaluationConfig(
        evaluation_id="source-agnostic-constructed-flight",
        run_record_path=run_root,
        dataset_path=dataset_root,
    )


class TestSourceAgnosticEvaluation:
    def test_synthetic_and_constructed_flight_run_through_identical_code_path(self, tmp_path):
        # Same call, same function, two different source_types -- no source_type branch
        # anywhere in run_evaluation (SC-009). A crash or special-casing here is the failure
        # mode this test exists to catch.
        synthetic_result = run_evaluation(_synthetic_config(), tmp_path / "out-synthetic")
        flight_result = run_evaluation(_constructed_flight_config(tmp_path), tmp_path / "out-flight")

        assert synthetic_result["ate"]["rmse"] >= 0.0
        assert flight_result["ate"]["rmse"] >= 0.0

    def test_simulator_source_runs_through_identical_code_path(self, tmp_path):
        result = run_evaluation(_simulator_config(), tmp_path / "out-simulator")
        assert result["ate"]["rmse"] >= 0.0

    def test_quality_class_propagates_to_differing_support_levels(self, tmp_path):
        """FR-061/SC-009: simulator_exact ground truth supports every RPE length; the
        weaker consumer_gnss ground truth used for the constructed flight must exclude
        short-baseline RPE as unsupported (spec.md's KITTI-derived support matrix,
        naveval.support.RPE_SHORT_LONG_THRESHOLD_M = 100.0, all of this feature's default
        RPE lengths are below it)."""
        run_evaluation(_synthetic_config(), tmp_path / "out-synthetic")
        run_evaluation(_constructed_flight_config(tmp_path), tmp_path / "out-flight")

        with (tmp_path / "out-synthetic" / "metrics.json").open() as f:
            synthetic_metrics = json.load(f)
        with (tmp_path / "out-flight" / "metrics.json").open() as f:
            flight_metrics = json.load(f)

        assert synthetic_metrics["ground_truth_class"]["position"] == "simulator_exact"
        assert flight_metrics["ground_truth_class"]["position"] == "consumer_gnss"

        synthetic_names = {m["name"] for m in synthetic_metrics["metrics"]}
        flight_names = {m["name"] for m in flight_metrics["metrics"]}

        # rpe_5m/rpe_10m are short-baseline (< naveval.support.RPE_SHORT_LONG_THRESHOLD_M =
        # 100 m) and both trajectories are long enough to actually compute them (unlike
        # rpe_25m/rpe_50m here, which the short constructed-flight trajectory is too short
        # for regardless of quality -- a *different*, orthogonal reason for omission this
        # assertion deliberately does not conflate with a support-classification exclusion).
        rpe_short_names = {"rpe_5m", "rpe_10m"}
        assert rpe_short_names.issubset(synthetic_names)
        assert rpe_short_names.isdisjoint(flight_names)
        assert rpe_short_names.issubset(set(flight_metrics["unsupported_excluded"]))

        # ATE is at least reported under consumer_gnss, just weakly rather than fully.
        flight_ate = next(m for m in flight_metrics["metrics"] if m["name"] == "ate_rmse")
        synthetic_ate = next(m for m in synthetic_metrics["metrics"] if m["name"] == "ate_rmse")
        assert flight_ate["support_level"] == "weakly_supported"
        assert synthetic_ate["support_level"] == "supported"
