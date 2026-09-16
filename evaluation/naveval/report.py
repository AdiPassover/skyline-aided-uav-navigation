"""Evaluation configuration and reporting (FR-050, FR-051; contracts/evaluation-outputs.md).

Every generated artifact is traceable to the exact configuration that produced it via
`config_digest` (FR-051), embedded in every output file.

Phase 3 built absolute trajectory error only. Phase 5 adds RPE, drift-per-distance, yaw
error/drift and endpoint error to the `metrics` array -- additively, per
contracts/evaluation-outputs.md's own evolution rule -- plus the support-level filtering
(FR-061/FR-062) and explicit-omission (SC-014) structure those metrics require.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from naveval import support

SCHEMA_VERSION = "1.0.0"

# spec.md's Deferred Decisions table: the defensible sub-trajectory length set depends
# on the first campaign's measured ground-truth accuracy and is not yet known. This
# placeholder matches contracts/evaluation-outputs.md's own example values -- callers
# should override via config.json's `metrics.rpe.sub_trajectory_lengths_m` once real
# figures exist (FR-034).
DEFAULT_RPE_LENGTHS_M = (5.0, 10.0, 25.0, 50.0)

# contracts/evaluation-outputs.md's `segments.source` (G15 fix): which of
# `naveval.segments`' three definition functions the CLI applies. "metadata" (the
# prior unconditional default) keeps the whole-trajectory behaviour; "reference_changes"
# is the mosaic-restart-aware derivation (`segments_from_reference_changes`) that makes
# the per-segment scale-series diagnostic actually exercise restart/recenter boundaries;
# "manual" reads explicit ranges from this same config (`segments_from_config`), already
# supported and preserved unchanged.
VALID_SEGMENTATION_SOURCES = {"metadata", "reference_changes", "manual"}


@dataclass(frozen=True)
class EvaluationConfig:
    evaluation_id: str
    run_record_path: Path
    dataset_path: Path
    naveval_version: str = "dev"
    rpe_lengths_m: tuple = DEFAULT_RPE_LENGTHS_M
    segmentation_source: str = "metadata"
    segmentation_ranges: tuple = ()

    def __post_init__(self):
        if self.segmentation_source not in VALID_SEGMENTATION_SOURCES:
            raise ValueError(
                f"segments.source must be one of {sorted(VALID_SEGMENTATION_SOURCES)}, "
                f"got {self.segmentation_source!r}"
            )
        if self.segmentation_source == "manual" and not self.segmentation_ranges:
            raise ValueError("segments.source == 'manual' requires a non-empty segments.ranges list")

    @staticmethod
    def load(path: Path | str) -> "EvaluationConfig":
        """Load an evaluation config. `run_record`/`dataset` paths are resolved
        relative to the config file's own directory, not the process's working
        directory, so a config remains valid regardless of where it is invoked from."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        base = path.parent
        rpe_lengths = tuple(
            d.get("metrics", {}).get("rpe", {}).get("sub_trajectory_lengths_m", DEFAULT_RPE_LENGTHS_M)
        )
        segments_cfg = d.get("segments", {})
        segmentation_ranges = tuple(
            tuple(sorted(r.items())) for r in segments_cfg.get("ranges", ())
        )
        return EvaluationConfig(
            evaluation_id=d["evaluation_id"],
            run_record_path=(base / d["run_record"]).resolve(),
            dataset_path=(base / d["dataset"]).resolve(),
            naveval_version=d.get("naveval_version", "dev"),
            rpe_lengths_m=rpe_lengths,
            segmentation_source=segments_cfg.get("source", "metadata"),
            segmentation_ranges=segmentation_ranges,
        )

    def segmentation_ranges_as_dicts(self) -> list:
        """`segmentation_ranges` is stored as a tuple of sorted-item tuples so the
        (frozen, hashable) config can be digested; `segments_from_config` wants plain
        dicts back."""
        return [dict(r) for r in self.segmentation_ranges]

    def digest(self) -> str:
        """SHA-256 over the config's identifying fields, embedded in every output
        artifact so a figure or number can always be traced back to what produced it."""
        payload = json.dumps(
            {
                "evaluation_id": self.evaluation_id,
                "run_record": str(self.run_record_path),
                "dataset": str(self.dataset_path),
                "naveval_version": self.naveval_version,
                "rpe_lengths_m": list(self.rpe_lengths_m),
                "segmentation_source": self.segmentation_source,
                "segmentation_ranges": [list(r) for r in self.segmentation_ranges],
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def write_alignment(out_dir: Path | str, config: EvaluationConfig, alignment_result) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config_digest": config.digest(),
        "type": "sim2",
        "rotation_deg": alignment_result.rotation_deg,
        "translation_e": alignment_result.translation[0],
        "translation_n": alignment_result.translation[1],
        "scale": alignment_result.scale,
        "n_points": alignment_result.n_points,
        "reflection_rejected": alignment_result.reflection_rejected,
        "near_straight": alignment_result.near_straight,
    }
    path = out_dir / "alignment.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def _fmt_length(length_m: float) -> str:
    """5.0 -> '5', 12.5 -> '12.5' -- metric names read as `rpe_5m`, not `rpe_5.0m`."""
    return str(int(length_m)) if float(length_m).is_integer() else str(length_m)


def _metric_entry(
    name: str,
    value,
    units: str,
    support_level: str,
    scale_basis: Optional[str] = None,
    role: str = "primary",
    demonstrates: Optional[str] = None,
    omission_reason: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Shape one `metrics.json` entry per contracts/evaluation-outputs.md's three
    structural honesty rules.

    `value=None` always means "could not be computed" (SC-014, rule 2): emitted as a
    visible entry with `state: "omitted"`, `support_level` forced to `unsupported`
    regardless of the declared ground-truth class, and a required reason -- a null
    value can never be evidence for anything, so it cannot inherit whatever support
    level the class would otherwise have earned. This mirrors the contract's own
    example, where a dataset's `simulator_exact`-quality yaw metric is still shown
    `unsupported`/omitted when no heading samples exist at all.

    A non-null value whose `support_level` classifies as `unsupported` is instead
    *excluded from the array entirely* by the caller (rule 1, SC-013) -- this function
    only shapes the entry; `write_metrics` decides whether it survives into the output.
    """
    if value is None:
        entry = {
            "name": name,
            "value": None,
            "units": units,
            "support_level": "unsupported",
            "state": "omitted",
            "omission_reason": omission_reason or "insufficient data",
            "role": role,
        }
        if scale_basis is not None:
            entry["scale_basis"] = scale_basis
        return entry

    entry = {"name": name, "value": value, "units": units, "support_level": support_level, "role": role}
    if scale_basis is not None:
        entry["scale_basis"] = scale_basis
    if demonstrates is not None:
        entry["demonstrates"] = demonstrates
    if extra:
        entry.update(extra)
    return entry


def write_metrics(
    out_dir: Path | str,
    config: EvaluationConfig,
    run_id: str,
    dataset_id: str,
    evidence_tier: str,
    evidence_caveat: Optional[str],
    ground_truth_class: dict,
    frame_counts: dict,
    ate_result: dict,
    drift_result: dict,
    endpoint_error_m: float,
    rpe_results: dict,
    rpe_diagnostic_results: dict,
    position_quality_class: Optional[str],
    endpoint_closed_loop_m: Optional[float] = None,
    endpoint_closed_loop_omission_reason: Optional[str] = None,
    yaw_error_result: Optional[dict] = None,
    yaw_drift_result: Optional[dict] = None,
    yaw_omission_reason: Optional[str] = None,
    heading_quality_class: Optional[str] = None,
    vertical_motion_result: Optional[dict] = None,
    synchronization_uncertainty_s: float = 0.0,
    runtime_result: Optional[dict] = None,
    environment: Optional[dict] = None,
    is_target_hardware: bool = False,
    event_stats: Optional[dict] = None,
    segmentation_source: str = "metadata",
) -> Path:
    """Write metrics.json per contracts/evaluation-outputs.md.

    Every scale-aligned metric carries a `demonstrates` string attached to the number
    itself (FR-032) -- attached to prose elsewhere would let it get lost when the
    number is copied into a table. Support-level classification (FR-061) and explicit
    omission (SC-014) are applied uniformly to every entry via `_metric_entry`, so no
    metric can silently appear as a misleading zero.

    `rpe_results` / `rpe_diagnostic_results`: `{length_m: dict | None}`, one entry per
    configured sub-trajectory length; `None` means the trajectory was too short for
    that length (contracts/evaluation-outputs.md).

    `runtime_result` (FR-047) is written under its own top-level `"runtime"` key,
    separate from `"metrics"` (FR-048) -- accuracy and performance are different kinds
    of evidence and must never be read from the same list. `is_target_hardware` gates
    an explicit disclaimer: development-machine timing MUST NOT be presented as
    onboard performance (FR-049).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        _metric_entry(
            "ate_rmse", ate_result["rmse"], "m",
            support.classify_position("ate_rmse", position_quality_class),
            scale_basis="global_fitted",
            demonstrates=(
                "trajectory shape fidelity after removal of unobservable gauge "
                "freedoms (global position, heading, scale); not metric accuracy"
            ),
        ),
        _metric_entry(
            "ate_rmse_normalised", ate_result["rmse_normalized"], "fraction of ground-truth path length",
            support.classify_position("ate_rmse_normalised", position_quality_class),
            scale_basis="global_fitted",
            demonstrates="ATE RMSE normalised by ground-truth path length, so the non-metric estimator scale stays interpretable",
        ),
        _metric_entry(
            "drift_per_distance", drift_result["drift_rate"], "fraction of distance travelled",
            support.classify_position("drift_per_distance", position_quality_class),
            scale_basis="global_fitted",
            demonstrates=(
                "rate at which position error grows with distance actually travelled "
                "(path length, not displacement); a growth rate, not an aggregate dispersion figure"
            ),
            extra={"total_distance_m": drift_result["total_distance_m"]},
        ),
        _metric_entry(
            "endpoint_error", endpoint_error_m, "m",
            support.classify_position("endpoint_error", position_quality_class),
            scale_basis="global_fitted",
            demonstrates="position error at the final evaluated frame, after global alignment",
        ),
        _metric_entry(
            "endpoint_error_closed_loop", endpoint_closed_loop_m, "m",
            support.classify_position("endpoint_error_closed_loop", position_quality_class),
            scale_basis="global_fitted",
            demonstrates="final position vs. the physically known launch point of a closed-loop flight",
            omission_reason=endpoint_closed_loop_omission_reason,
        ),
    ]

    for length_m in sorted(rpe_results):
        result = rpe_results[length_m]
        bucket = support.rpe_metric_name_for_length(length_m)
        entries.append(_metric_entry(
            f"rpe_{_fmt_length(length_m)}m",
            result["rmse"] if result else None,
            "m",
            support.classify_position(bucket, position_quality_class),
            scale_basis="global_fitted",
            demonstrates="relative pose error over a fixed sub-trajectory length, under the single global fitted scale (FR-067)",
            omission_reason=(
                f"trajectory shorter than the configured {length_m} m sub-trajectory length" if result is None else None
            ),
            extra={"length_m": length_m, "n_segments": result["n_segments"]} if result else {"length_m": length_m},
        ))

    entries.append(_metric_entry(
        "yaw_rmse", yaw_error_result["rmse"] if yaw_error_result else None, "deg",
        support.classify_heading("yaw_error", heading_quality_class),
        demonstrates="RMS yaw error using shortest-angle differences, independent of position error (FR-031)",
        omission_reason=yaw_omission_reason,
    ))
    entries.append(_metric_entry(
        "yaw_drift", yaw_drift_result["drift_deg_per_s"] if yaw_drift_result else None, "deg/s",
        support.classify_heading("yaw_drift", heading_quality_class),
        demonstrates="rate of yaw-error growth over elapsed processing time",
        omission_reason=yaw_omission_reason,
    ))

    # Diagnostics only (FR-067, DEC-003): never role="primary". Reported alongside the
    # primary RPE so the scale-drift contribution -- their difference -- is visible.
    for length_m in sorted(rpe_diagnostic_results):
        diag = rpe_diagnostic_results[length_m]
        primary = rpe_results.get(length_m)
        entries.append(_metric_entry(
            f"rpe_{_fmt_length(length_m)}m_diagnostic_per_segment_rescaled",
            diag["rmse"] if diag else None, "m",
            support.classify_position("scale_drift_diagnostic", position_quality_class),
            scale_basis="per_segment_fitted", role="diagnostic",
            demonstrates=(
                "shape fidelity over the same sub-trajectory length with LOCAL scale drift "
                "removed by a fresh per-segment fit -- NOT a primary error figure (FR-067)"
            ),
            omission_reason=(
                f"trajectory shorter than the configured {length_m} m sub-trajectory length" if diag is None else None
            ),
            extra={"length_m": length_m},
        ))
        if diag is not None and primary is not None:
            entries.append(_metric_entry(
                f"rpe_{_fmt_length(length_m)}m_scale_drift_contribution",
                primary["rmse"] - diag["rmse"], "m",
                support.classify_position("scale_drift_diagnostic", position_quality_class),
                scale_basis="not_applicable", role="diagnostic",
                demonstrates=(
                    "difference between global-scale and per-segment-rescaled RPE at this length -- "
                    "the scale-drift contribution to error (FR-067's last sentence)"
                ),
                extra={"length_m": length_m},
            ))

    metrics_out = []
    unsupported_excluded = []
    for entry in entries:
        if entry.get("state") == "omitted":
            metrics_out.append(entry)  # SC-014: always visible, never silently dropped
            continue
        if entry["support_level"] == "unsupported":
            unsupported_excluded.append(entry["name"])  # SC-013: named, not valued
            continue
        metrics_out.append(entry)

    runtime_section = None
    if runtime_result is not None:
        runtime_section = dict(runtime_result)
        runtime_section["environment"] = environment
        runtime_section["is_target_hardware"] = is_target_hardware
        if not is_target_hardware:
            runtime_section["disclaimer"] = (
                "development-machine timing -- MUST NOT be presented as onboard "
                "performance (FR-049)"
            )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "config_digest": config.digest(),
        "run_id": run_id,
        "dataset_id": dataset_id,
        "evidence_tier": evidence_tier,
        "evidence_caveat": evidence_caveat,
        "ground_truth_class": ground_truth_class,
        "frames": frame_counts,
        "metrics": metrics_out,
        "unsupported_excluded": unsupported_excluded,
        "segmentation_source": segmentation_source,
        "synchronisation_uncertainty_s": synchronization_uncertainty_s,
        "vertical_motion_discarded": vertical_motion_result,
        "runtime": runtime_section,
        "event_stats": event_stats,
    }
    path = out_dir / "metrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def write_events(out_dir: Path | str, events: list) -> Path:
    """Write events.csv per contracts/evaluation-outputs.md: one row per detected
    event, `from_run_record` events and derived events side by side, each carrying its
    own `detection_rule` so events from different configurations are never silently
    compared as if they meant the same thing."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "events.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "timestamp_s", "type", "detection_rule", "error_before_m", "error_after_m"])
        for e in events:
            writer.writerow([
                e.frame_index,
                e.timestamp_s,
                e.type,
                e.detection_rule,
                "" if e.error_before_m is None else e.error_before_m,
                "" if e.error_after_m is None else e.error_after_m,
            ])
    return path


def write_segments(out_dir: Path | str, segment_metric_dicts: list) -> Path:
    """Write segments.csv: the applied segment definitions (FR-040) alongside their
    per-segment metrics (FR-041), so a segment breakdown is reproducible from the file
    alone without recomputing anything."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "segments.csv"
    columns = [
        "segment_id", "label", "definition_source", "start_index", "end_index",
        "n_points", "contains_events", "ate_rmse_m", "path_length_m", "endpoint_error_m",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for m in segment_metric_dicts:
            writer.writerow([
                m["segment_id"], m["label"], m["definition_source"], m["start_index"], m["end_index"],
                m["n_points"], m["contains_events"],
                "" if m["ate_rmse_m"] is None else m["ate_rmse_m"],
                m["path_length_m"],
                "" if m["endpoint_error_m"] is None else m["endpoint_error_m"],
            ])
    return path


def write_scale_series(out_dir: Path | str, scale_points: list) -> Path:
    """Write scale_series.csv per contracts/evaluation-outputs.md -- the measurement
    behind SC-012. Always a DIAGNOSTIC (FR-067, DEC-003): nothing reads this file back
    into a primary metric."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scale_series.csv"
    columns = [
        "segment_id", "time_start_s", "distance_start_m", "scale",
        "scale_ratio_to_global", "spans_recenter", "n_poses", "refusal_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for p in scale_points:
            writer.writerow([
                p.segment_id, p.time_start_s, p.distance_start_m,
                "" if p.scale is None else p.scale,
                "" if p.scale_ratio_to_global is None else p.scale_ratio_to_global,
                p.spans_recenter, p.n_poses,
                p.refusal_reason or "",
            ])
    return path


_TABLE_COLUMNS = [
    "source", "scope", "evaluation_id", "run_id", "dataset_id",
    "evidence_tier", "evidence_caveat",
    "ground_truth_position_class", "ground_truth_heading_class", "ground_truth_height_class",
    "segment_id", "segment_label",
    "metric_name", "value", "units", "support_level", "role", "state",
]


def _run_level_rows(source: str, payload: dict) -> list:
    gt_class = payload.get("ground_truth_class", {})
    common = {
        "source": source,
        "scope": "run",
        "evaluation_id": payload.get("evaluation_id", ""),
        "run_id": payload.get("run_id", ""),
        "dataset_id": payload.get("dataset_id", ""),
        "evidence_tier": payload.get("evidence_tier", ""),
        "evidence_caveat": payload.get("evidence_caveat") or "",
        "ground_truth_position_class": gt_class.get("position") or "",
        "ground_truth_heading_class": gt_class.get("heading") or "",
        "ground_truth_height_class": gt_class.get("height") or "",
        "segment_id": "",
        "segment_label": "",
    }
    rows = []
    for m in payload.get("metrics", []):
        row = dict(common)
        row["metric_name"] = m["name"]
        row["value"] = "" if m.get("value") is None else m["value"]
        row["units"] = m.get("units", "")
        row["support_level"] = m.get("support_level", "")
        row["role"] = m.get("role", "")
        row["state"] = m.get("state", "")
        rows.append(row)
    return rows


def _segment_level_rows(source: str, payload: dict, segments_csv_path: Path) -> list:
    gt_class = payload.get("ground_truth_class", {})
    common = {
        "source": source,
        "scope": "segment",
        "evaluation_id": payload.get("evaluation_id", ""),
        "run_id": payload.get("run_id", ""),
        "dataset_id": payload.get("dataset_id", ""),
        "evidence_tier": payload.get("evidence_tier", ""),
        "evidence_caveat": payload.get("evidence_caveat") or "",
        "ground_truth_position_class": gt_class.get("position") or "",
        "ground_truth_heading_class": gt_class.get("heading") or "",
        "ground_truth_height_class": gt_class.get("height") or "",
    }
    segment_metric_columns = ["ate_rmse_m", "path_length_m", "endpoint_error_m"]
    rows = []
    with segments_csv_path.open("r", encoding="utf-8", newline="") as f:
        for seg_row in csv.DictReader(f):
            for metric_name in segment_metric_columns:
                row = dict(common)
                row["segment_id"] = seg_row["segment_id"]
                row["segment_label"] = seg_row["label"]
                row["metric_name"] = f"segment_{metric_name}"
                row["value"] = seg_row.get(metric_name, "")
                row["units"] = "m"
                # Segments are not support-classified (naveval.segments has no support.py
                # dependency) -- left blank rather than guessing a level, per FR-062's
                # "never relax a classification to look adequate" logic applied to its
                # absence too: no classification is not the same as "supported".
                row["support_level"] = ""
                row["role"] = "segment_diagnostic"
                row["state"] = "" if seg_row.get(metric_name) not in (None, "") else "omitted"
                rows.append(row)
    return rows


def export_metrics_table(
    output_dirs,
    out_path: Path | str,
    labels: Optional[list] = None,
    include_segments: bool = True,
) -> Path:
    """Flatten `metrics.json` (and, if present alongside it, `segments.csv`) from one or
    more evaluation output directories into a single thesis-ready comparison table
    (FR-041: "comparison of metrics across segments and across flights").

    One row per (source, metric): a `metrics.json` from a single flight/run contributes
    its whole-run "primary" metrics as `scope=run` rows, and, when `include_segments` and
    a sibling `segments.csv` exists, its per-segment breakdown as `scope=segment` rows --
    the same table therefore supports both axes FR-041 names, not two separate exports.

    Every row repeats `evidence_tier`, `evidence_caveat` and the three
    `ground_truth_*_class` columns from its own source. This is deliberate, not
    redundant: pooling a T2 near-T1-caveated simulator run against a T3/T4 real-flight
    run into one column-less number is exactly the silent-pooling failure mode this task
    exists to prevent (Constitution Principle VIII, spec.md's evidence-tier discipline).
    Unsupported-and-excluded metrics (`unsupported_excluded` in `metrics.json`) are not
    present in `metrics.json`'s own `metrics` array and so cannot appear here either --
    consistent with SC-013, not a separate omission.
    """
    output_dirs = [Path(d) for d in output_dirs]
    if labels is not None and len(labels) != len(output_dirs):
        raise ValueError("labels must have the same length as output_dirs")

    rows = []
    for i, out_dir in enumerate(output_dirs):
        metrics_path = out_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"No metrics.json in {out_dir}")
        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        source = labels[i] if labels is not None else out_dir.name
        rows.extend(_run_level_rows(source, payload))

        if include_segments:
            segments_path = out_dir / "segments.csv"
            if segments_path.exists():
                rows.extend(_segment_level_rows(source, payload, segments_path))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TABLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path
