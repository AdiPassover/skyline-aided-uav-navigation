"""CLI entry point: evaluate a VO run record against its dataset's ground truth.

    python -m naveval.evaluate --config configs/eval-golden.json --out /tmp/eval-golden

Phase 3 built the vertical slice: load a run record and dataset (DEC-002's file
boundary), synchronize frames to ground truth, fit the Sim(2) alignment (DEC-003), and
compute absolute trajectory error. Phase 5 adds the rest of the primary metric set
(FR-030) -- RPE, drift-per-distance, yaw error/drift, endpoint error -- plus the
per-segment-rescaled RPE diagnostic (FR-067) and support-level classification
(FR-061) that decide what actually reaches metrics.json. Phase 6 adds failure/event
statistics (FR-042 through FR-046); Phase 7 adds segmentation and the diagnostic
per-segment scale series (FR-035, FR-038 through FR-041); Phase 8 adds runtime
measurement (FR-047 through FR-049).

Known scope boundary (Phase 6): `events.py`'s success-rate and `from_run_record` event
extraction take only the run record and are independently tested without any ground
truth (FR-045). This CLI's overall `run_evaluation`, however, still requires a
synchronizable dataset -- it raises before reaching the event/segment/scale wiring
below if fewer than 2 frames sync to ground truth. Running the full CLI against a
dataset with NO ground truth at all is not yet wired end-to-end; the module-level
capability FR-045 asks for is real and tested (`tests/test_events.py`), but exercising
it through this specific entry point on a truly ground-truth-free dataset is future
work, not claimed complete here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from naveval.alignment import fit_sim2
from naveval.dataset import load_dataset
from naveval.events import (
    attribute_error,
    detect_large_error_excursions,
    detect_pose_discontinuities,
    extract_run_record_events,
    success_rate_stats,
)
from naveval.metrics import (
    _cumulative_distance,
    absolute_trajectory_error,
    endpoint_error,
    endpoint_error_closed_loop,
    relative_pose_error,
    relative_pose_error_per_segment_rescaled,
    runtime_statistics,
    translational_drift_rate,
    vertical_motion_discarded,
    yaw_drift_rate,
    yaw_error_metric,
)
from naveval.plots import (
    plot_error_vs_distance,
    plot_error_vs_time,
    plot_event_timeline,
    plot_rpe_by_length,
    plot_scale_series,
    plot_trajectory_xy,
    plot_yaw_error,
)
from naveval.report import (
    EvaluationConfig,
    write_alignment,
    write_events,
    write_metrics,
    write_scale_series,
    write_segments,
)
from naveval.runrecord import load_run_record, verify_matches_dataset
from naveval.scale import compute_scale_series
from naveval.segments import (
    mark_segments_containing_events,
    segment_metrics,
    segments_from_config,
    segments_from_metadata,
    segments_from_reference_changes,
)
from naveval.sync import synchronize


def run_evaluation(config: EvaluationConfig, out_dir: Path | str) -> dict:
    """Run one evaluation end to end. Returns the alignment result and per-metric
    dicts for programmatic use (e.g. tests); also writes the JSON artifacts to out_dir."""
    run_record = load_run_record(config.run_record_path)
    dataset = load_dataset(config.dataset_path)
    verify_matches_dataset(run_record, dataset)

    gt_heading = dataset.gt_heading if dataset.has_heading else None
    sync_result = synchronize(
        run_record.timestamps_s,
        dataset.gt_timestamps,
        dataset.gt_east,
        dataset.gt_north,
        gt_heading,
        dataset.gt_valid,
        clock_offset_s=dataset.clock_offset_s,
        clock_drift_s_per_s=dataset.clock_drift_s_per_s,
    )

    if sync_result.frame_indices.size < 2:
        raise ValueError(
            f"Only {sync_result.frame_indices.size} frame(s) could be synchronized to "
            f"ground truth; at least 2 are required to fit a Sim(2) alignment"
        )

    est_points_synced = np.stack(
        [
            run_record.est_x[sync_result.frame_indices],
            run_record.est_y[sync_result.frame_indices],
        ],
        axis=1,
    )
    gt_points_synced = np.stack([sync_result.east, sync_result.north], axis=1)

    alignment_result = fit_sim2(est_points_synced, gt_points_synced)
    aligned = alignment_result.apply(est_points_synced)
    ate_result = absolute_trajectory_error(aligned, gt_points_synced)
    drift_result = translational_drift_rate(aligned, gt_points_synced)
    endpoint_error_m = endpoint_error(aligned, gt_points_synced)

    is_closed_loop = dataset.metadata.trajectory_type == "loop"
    endpoint_closed_loop_m = None
    endpoint_closed_loop_omission_reason = None
    if is_closed_loop:
        endpoint_closed_loop_m = endpoint_error_closed_loop(aligned, gt_points_synced)
    else:
        endpoint_closed_loop_omission_reason = (
            f"dataset metadata declares trajectory_type={dataset.metadata.trajectory_type!r}, not a closed loop"
        )

    rpe_results = {length_m: relative_pose_error(aligned, gt_points_synced, length_m) for length_m in config.rpe_lengths_m}
    rpe_diagnostic_results = {
        length_m: relative_pose_error_per_segment_rescaled(est_points_synced, gt_points_synced, length_m)
        for length_m in config.rpe_lengths_m
    }

    yaw_error_result = None
    yaw_drift_result = None
    yaw_omission_reason = None
    if dataset.has_heading and sync_result.has_heading:
        aligned_yaw = alignment_result.transform_yaw(run_record.est_yaw_deg[sync_result.frame_indices])
        yaw_error_result = yaw_error_metric(aligned_yaw, sync_result.heading)
        elapsed_time_s = run_record.timestamps_s[sync_result.frame_indices]
        yaw_drift_result = yaw_drift_rate(aligned_yaw, sync_result.heading, elapsed_time_s)
    else:
        yaw_omission_reason = "no heading ground truth in dataset"

    vertical_motion_result = vertical_motion_discarded(dataset.gt_up[dataset.gt_valid])

    # --- Phase 6: failure and event statistics (FR-042 through FR-046) ---
    success_stats = success_rate_stats(run_record)  # no ground truth required (FR-045)
    from_run_record_events = extract_run_record_events(run_record)

    timestamps_synced = run_record.timestamps_s[sync_result.frame_indices]
    real_frame_index_synced = run_record.frame_indices[sync_result.frame_indices]

    derived_events = detect_pose_discontinuities(real_frame_index_synced, timestamps_synced, aligned)
    derived_events += detect_large_error_excursions(real_frame_index_synced, timestamps_synced, ate_result["per_point"])

    all_events = from_run_record_events + derived_events
    all_events = attribute_error(all_events, real_frame_index_synced, ate_result["per_point"])

    # --- Phase 7: segmentation and the diagnostic per-segment scale series (FR-035, FR-038-041) ---
    # G15: which of segments.py's three definition functions the CLI applies is an
    # explicit, reproducible config choice (`segments.source` in config.json, folded
    # into config.digest()) rather than the unconditional whole-trajectory default it
    # used to be.
    n_evaluated = int(sync_result.frame_indices.size)
    reference_ids_synced = [
        run_record.frame_estimates[pos].reference_id for pos in sync_result.frame_indices
    ]
    if config.segmentation_source == "metadata":
        segments = segments_from_metadata(dataset.metadata.trajectory_type, n_evaluated)
    elif config.segmentation_source == "reference_changes":
        segments = segments_from_reference_changes(reference_ids_synced)
    elif config.segmentation_source == "manual":
        segments = segments_from_config(config.segmentation_ranges_as_dicts())
    else:
        raise ValueError(f"Unknown segments.source: {config.segmentation_source!r}")

    event_positions = []
    recenter_positions = []
    for idx, pos in enumerate(sync_result.frame_indices):
        fe = run_record.frame_estimates[pos]
        if fe.event in ("restart", "recenter"):
            recenter_positions.append(idx)
    for e in all_events:
        p = int(np.searchsorted(real_frame_index_synced, e.frame_index))
        if p < real_frame_index_synced.size and real_frame_index_synced[p] == e.frame_index:
            event_positions.append(p)
    segments = mark_segments_containing_events(segments, event_positions)

    segment_metric_dicts = [segment_metrics(seg, aligned, gt_points_synced) for seg in segments]
    scale_series = compute_scale_series(
        segments, est_points_synced, gt_points_synced, timestamps_synced, alignment_result.scale,
        recenter_positions=recenter_positions,
    )

    # --- Phase 8: runtime, separate from accuracy (FR-047 through FR-049) ---
    all_process_time_ns = np.array([fe.process_time_ns for fe in run_record.frame_estimates])
    runtime_result = runtime_statistics(all_process_time_ns)

    out_dir = Path(out_dir)
    write_alignment(out_dir, config, alignment_result)
    write_metrics(
        out_dir,
        config,
        run_id=run_record.manifest.run_id,
        dataset_id=dataset.dataset_id,
        evidence_tier=dataset.evidence_tier,
        evidence_caveat=dataset.evidence_caveat,
        ground_truth_class={
            "position": dataset.position_quality.quality_class if dataset.position_quality else None,
            "heading": dataset.heading_quality.quality_class if dataset.heading_quality else None,
            "height": dataset.height_quality.quality_class if dataset.height_quality else None,
        },
        frame_counts={
            "total": int(dataset.frame_indices.size),
            "processed": run_record.manifest.processed_count,
            "evaluated": int(sync_result.frame_indices.size),
            "excluded_out_of_span": sync_result.excluded_out_of_span_count,
            "excluded_insufficient_valid_gt": sync_result.excluded_insufficient_valid_gt_count,
        },
        ate_result=ate_result,
        drift_result=drift_result,
        endpoint_error_m=endpoint_error_m,
        endpoint_closed_loop_m=endpoint_closed_loop_m,
        endpoint_closed_loop_omission_reason=endpoint_closed_loop_omission_reason,
        rpe_results=rpe_results,
        rpe_diagnostic_results=rpe_diagnostic_results,
        position_quality_class=dataset.position_quality.quality_class if dataset.position_quality else None,
        heading_quality_class=dataset.heading_quality.quality_class if dataset.heading_quality else None,
        yaw_error_result=yaw_error_result,
        yaw_drift_result=yaw_drift_result,
        yaw_omission_reason=yaw_omission_reason,
        vertical_motion_result=vertical_motion_result,
        synchronization_uncertainty_s=sync_result.synchronization_uncertainty_s,
        runtime_result=runtime_result,
        environment=run_record.manifest.environment.__dict__,
        is_target_hardware=run_record.manifest.environment.is_target_hardware,
        event_stats=success_stats,
        segmentation_source=config.segmentation_source,
    )
    write_events(out_dir, all_events)
    write_segments(out_dir, segment_metric_dicts)
    write_scale_series(out_dir, scale_series)
    plot_trajectory_xy(
        aligned, gt_points_synced,
        evaluation_id=config.evaluation_id,
        config_digest=config.digest(),
        out_path=out_dir / "figures" / "trajectory_xy.png",
    )
    plot_event_timeline(
        timestamps_synced, ate_result["per_point"], all_events,
        evaluation_id=config.evaluation_id,
        config_digest=config.digest(),
        out_path=out_dir / "figures" / "event_timeline.png",
    )
    plot_scale_series(
        scale_series,
        evaluation_id=config.evaluation_id,
        config_digest=config.digest(),
        out_path=out_dir / "figures" / "scale_series.png",
    )
    plot_error_vs_time(
        timestamps_synced, ate_result["per_point"],
        evaluation_id=config.evaluation_id,
        config_digest=config.digest(),
        out_path=out_dir / "figures" / "error_vs_time.png",
    )
    plot_error_vs_distance(
        _cumulative_distance(gt_points_synced), ate_result["per_point"],
        evaluation_id=config.evaluation_id,
        config_digest=config.digest(),
        out_path=out_dir / "figures" / "error_vs_distance.png",
    )
    plot_yaw_error(
        timestamps_synced if yaw_error_result is not None else np.zeros(0),
        yaw_error_result["per_frame"] if yaw_error_result is not None else None,
        evaluation_id=config.evaluation_id,
        config_digest=config.digest(),
        out_path=out_dir / "figures" / "yaw_error.png",
        omission_reason=yaw_omission_reason,
    )
    plot_rpe_by_length(
        rpe_results,
        evaluation_id=config.evaluation_id,
        config_digest=config.digest(),
        out_path=out_dir / "figures" / "rpe_by_length.png",
        omitted_reasons={
            length_m: f"trajectory shorter than the configured {length_m} m sub-trajectory length"
            for length_m in config.rpe_lengths_m
        },
    )

    return {
        "alignment": alignment_result,
        "ate": ate_result,
        "sync": sync_result,
        "drift": drift_result,
        "endpoint_error_m": endpoint_error_m,
        "endpoint_closed_loop_m": endpoint_closed_loop_m,
        "rpe": rpe_results,
        "rpe_diagnostic": rpe_diagnostic_results,
        "yaw_error": yaw_error_result,
        "yaw_drift": yaw_drift_result,
        "vertical_motion": vertical_motion_result,
        "success_stats": success_stats,
        "events": all_events,
        "segments": segments,
        "segment_metrics": segment_metric_dicts,
        "scale_series": scale_series,
        "runtime": runtime_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a stitching-VO run record against its dataset's ground truth."
    )
    parser.add_argument("--config", required=True, help="Path to an EvaluationConfig JSON file")
    parser.add_argument("--out", required=True, help="Output directory for metrics.json / alignment.json")
    args = parser.parse_args()

    config = EvaluationConfig.load(args.config)
    result = run_evaluation(config, Path(args.out))

    print(f"Evaluated {config.evaluation_id}")
    print(f"  Alignment: rotation={result['alignment'].rotation_deg:.4f} deg, "
          f"scale={result['alignment'].scale:.6g}, "
          f"translation={result['alignment'].translation}")
    print(f"  ATE RMSE: {result['ate']['rmse']:.6g} m "
          f"({result['ate']['rmse_normalized']:.6g} of path length)")


if __name__ == "__main__":
    main()
