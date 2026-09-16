"""Run-record loading and validation (contracts/run-record.md).

The run record is the sole interface between estimation and evaluation (DEC-002).
Malformed input is an error, not a warning: a truncated or version-mismatched record must
not be silently analysed as if it were a complete, valid run.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from naveval.dataset import Dataset
from naveval.errors import ContractViolationError, DatasetMismatchError, SchemaVersionError

SUPPORTED_SCHEMA_MAJOR = 1

VALID_EVENTS = {"none", "init", "recenter", "restart"}

# Columns read by name (contracts/run-record.md); order in the file is not relied upon,
# so appended columns in a later minor version are ignored automatically.
_FLOAT_COLUMNS = (
    "timestamp_s", "est_x", "est_y", "est_z", "est_yaw_deg",
    "h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22",
)
_OPTIONAL_FLOAT_COLUMNS = ("est_z", "h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22")
_OPTIONAL_INT_COLUMNS = ("track_count", "inlier_count")


def _major_version(schema_version: str) -> int:
    try:
        return int(schema_version.split(".", 1)[0])
    except (ValueError, IndexError, AttributeError) as exc:
        raise SchemaVersionError(f"Malformed schema_version: {schema_version!r}") from exc


def _parse_float_or_empty(s: str) -> Optional[float]:
    if s == "":
        return None
    low = s.strip().lower()
    if low in ("nan", "inf", "-inf", "+inf", "infinity", "-infinity"):
        raise ContractViolationError(f"NaN/Inf is not a valid field value: {s!r}")
    return float(s)


def _parse_int_or_empty(s: str) -> Optional[int]:
    if s == "":
        return None
    return int(s)


def _parse_bool(s: str) -> bool:
    if s == "true":
        return True
    if s == "false":
        return False
    raise ContractViolationError(f"Invalid boolean field value: {s!r} (expected 'true' or 'false')")


_VALID_CONFIDENCE_OUTCOMES = {"usable", "degraded", "rejected", "not_produced"}


def _parse_confidence(row: dict, success: bool) -> "FrameConfidence":
    """One row's v1.1.0 confidence columns, with the reader-obligation invariants as ERRORS
    (contracts/run-record-v1.1.md §Reader obligations — a malformed record must not be evaluated).
    """
    fi = row.get("frame_index", "?")
    outcome = row.get("confidence_outcome", "")
    if outcome not in _VALID_CONFIDENCE_OUTCOMES:
        raise ContractViolationError(
            f"Invalid confidence_outcome {outcome!r} at frame_index {fi}"
        )
    reason = row.get("confidence_reason", "")
    if reason == "":
        raise ContractViolationError(f"confidence_reason must not be empty at frame_index {fi}")

    score = _parse_float_or_empty(row.get("confidence_score", ""))
    scoreless = outcome in ("rejected", "not_produced")
    if scoreless != (score is None):
        raise ContractViolationError(
            f"confidence_score must be empty exactly when outcome is rejected/not_produced; "
            f"frame_index {fi} has outcome {outcome!r} with score {score!r}"
        )
    if (outcome == "not_produced") != (not success):
        raise ContractViolationError(
            f"not_produced must coincide with success == false at frame_index {fi}"
        )
    if outcome == "rejected" and reason in ("ok", "low_score"):
        raise ContractViolationError(
            f"a rejected frame must carry a named condition, got reason {reason!r} at "
            f"frame_index {fi}"
        )

    residual_inlier_count = _parse_int_or_empty(row.get("residual_inlier_count", ""))
    residual_mean_sq_px = _parse_float_or_empty(row.get("residual_mean_sq_px", ""))
    residual_rms_px = _parse_float_or_empty(row.get("residual_rms_px", ""))
    if residual_mean_sq_px is not None:
        if residual_inlier_count is None or residual_inlier_count < 1:
            raise ContractViolationError(
                f"residual statistics present without a positive residual_inlier_count at "
                f"frame_index {fi}"
            )
        if residual_rms_px is not None and not math.isclose(
                residual_rms_px * residual_rms_px, residual_mean_sq_px, rel_tol=1e-12, abs_tol=0.0):
            raise ContractViolationError(
                f"residual_rms_px^2 != residual_mean_sq_px beyond float round-trip at "
                f"frame_index {fi}"
            )

    time_raw = row.get("confidence_time_ns", "")
    if time_raw == "":
        raise ContractViolationError(f"confidence_time_ns must not be empty at frame_index {fi}")
    confidence_time_ns = int(time_raw)
    if confidence_time_ns < 0:
        raise ContractViolationError(
            f"confidence_time_ns must be >= 0, got {confidence_time_ns} at frame_index {fi}"
        )

    return FrameConfidence(
        residual_inlier_count=residual_inlier_count,
        residual_mean_sq_px=residual_mean_sq_px,
        residual_rms_px=residual_rms_px,
        residual_median_sq_px=_parse_float_or_empty(row.get("residual_median_sq_px", "")),
        residual_max_sq_px=_parse_float_or_empty(row.get("residual_max_sq_px", "")),
        inlier_threshold_sq_px=_parse_float_or_empty(row.get("inlier_threshold_sq_px", "")),
        inlier_coverage=_parse_float_or_empty(row.get("inlier_coverage", "")),
        keyframe_age=_parse_int_or_empty(row.get("keyframe_age", "")),
        relative_support=_parse_float_or_empty(row.get("relative_support", "")),
        inc_flow_px=_parse_float_or_empty(row.get("inc_flow_px", "")),
        inc_log_scale=_parse_float_or_empty(row.get("inc_log_scale", "")),
        inc_log_scale_dispersion=_parse_float_or_empty(row.get("inc_log_scale_dispersion", "")),
        outcome=outcome,
        reason=reason,
        score=score,
        confidence_time_ns=confidence_time_ns,
    )


@dataclass(frozen=True)
class Environment:
    hostname: str
    os: str
    cpu_model: str
    jvm_version: str
    heap_max_mb: Optional[int]
    is_target_hardware: bool

    @staticmethod
    def from_dict(d: dict) -> "Environment":
        return Environment(
            hostname=d.get("hostname", ""),
            os=d.get("os", ""),
            cpu_model=d.get("cpu_model", ""),
            jvm_version=d.get("jvm_version", ""),
            heap_max_mb=d.get("heap_max_mb"),
            is_target_hardware=bool(d.get("is_target_hardware", False)),
        )


@dataclass(frozen=True)
class RunManifest:
    schema_version: str
    run_id: str
    dataset_id: str
    dataset_revision: str
    estimator_id: str
    estimator_version: str
    estimator_config: dict
    evaluator_capture_version: str
    environment: Environment
    run_timestamp: str
    frame_count: int
    processed_count: int
    completed: bool
    # v1.1.0 (contracts/run-record-v1.1.md): the confidence block, or None for a run captured
    # without confidence. Absent != calibration_validated=false — the first means no verdict was
    # computed, the second that one was, under an unvalidated calibration.
    confidence: Optional[dict] = None


@dataclass(frozen=True)
class FrameConfidence:
    """Columns 22-37 of a v1.1.0 row (contracts/run-record-v1.1.md).

    Every optional field is None for *absent*, never zero — a missing residual read as 0.0 would
    look like a perfect fit. ``outcome``/``reason`` are the lowercase wire names; ``score`` is
    None exactly when the outcome is ``rejected`` or ``not_produced`` (never computed, not
    computed-as-zero).
    """
    residual_inlier_count: Optional[int]
    residual_mean_sq_px: Optional[float]
    residual_rms_px: Optional[float]
    residual_median_sq_px: Optional[float]
    residual_max_sq_px: Optional[float]
    inlier_threshold_sq_px: Optional[float]
    inlier_coverage: Optional[float]
    keyframe_age: Optional[int]           # specified-and-empty until the VO exposes keyframe state
    relative_support: Optional[float]
    inc_flow_px: Optional[float]
    inc_log_scale: Optional[float]
    inc_log_scale_dispersion: Optional[float]
    outcome: str
    reason: str
    score: Optional[float]
    confidence_time_ns: int


@dataclass(frozen=True)
class FrameEstimate:
    frame_index: int
    timestamp_s: float
    est_x: float
    est_y: float
    est_z: Optional[float]
    est_yaw_deg: float
    success: bool
    event: str
    reference_id: int
    homography: Optional[tuple]  # 9 floats, row-major, or None if any component missing
    track_count: Optional[int]
    inlier_count: Optional[int]
    process_time_ns: int
    # v1.1.0: the persisted per-frame confidence columns, or None for a v1.0.0 record. A v1.0.0
    # run has no verdict; it does not have a bad one (reader obligation 1).
    confidence: Optional[FrameConfidence] = None


@dataclass
class RunRecord:
    manifest: RunManifest
    frame_estimates: list  # list[FrameEstimate]

    # Convenience numpy views, in file order (== frame_index order, since strictly increasing)
    frame_indices: np.ndarray
    timestamps_s: np.ndarray
    est_x: np.ndarray
    est_y: np.ndarray
    est_yaw_deg: np.ndarray
    success: np.ndarray  # bool


def load_run_record(root: Path | str) -> RunRecord:
    """Load and validate a run record directory per contracts/run-record.md."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    frames_path = root / "frames.csv"

    if not manifest_path.exists():
        raise ContractViolationError(f"Missing manifest.json in {root}")
    if not frames_path.exists():
        raise ContractViolationError(f"Missing frames.csv in {root}")

    with manifest_path.open("r", encoding="utf-8") as f:
        m = json.load(f)

    schema_version = m.get("schema_version", "")
    if _major_version(schema_version) != SUPPORTED_SCHEMA_MAJOR:
        raise SchemaVersionError(
            f"Unsupported run-record schema major version: {schema_version!r} "
            f"(this reader supports major version {SUPPORTED_SCHEMA_MAJOR})"
        )

    manifest = RunManifest(
        schema_version=schema_version,
        run_id=m["run_id"],
        dataset_id=m["dataset_id"],
        dataset_revision=m["dataset_revision"],
        estimator_id=m.get("estimator_id", ""),
        estimator_version=m.get("estimator_version", ""),
        estimator_config=m.get("estimator_config", {}),
        evaluator_capture_version=m.get("evaluator_capture_version", ""),
        environment=Environment.from_dict(m.get("environment", {})),
        run_timestamp=m.get("run_timestamp", ""),
        frame_count=int(m["frame_count"]),
        processed_count=int(m["processed_count"]),
        completed=bool(m["completed"]),
        confidence=m.get("confidence"),
    )

    frame_estimates = []
    with frames_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        has_confidence_columns = reader.fieldnames is not None and \
            "confidence_outcome" in reader.fieldnames
        for row in reader:
            floats = {}
            for col in _FLOAT_COLUMNS:
                raw = row.get(col, "")
                if col in _OPTIONAL_FLOAT_COLUMNS:
                    floats[col] = _parse_float_or_empty(raw)
                else:
                    parsed = _parse_float_or_empty(raw)
                    if parsed is None:
                        raise ContractViolationError(f"Column {col!r} must not be empty")
                    floats[col] = parsed

            success = _parse_bool(row["success"])
            event = row["event"]
            if event not in VALID_EVENTS:
                raise ContractViolationError(
                    f"Invalid event {event!r}; expected one of {sorted(VALID_EVENTS)}"
                )

            h_components = [floats[f"h{i}{j}"] for i in range(3) for j in range(3)]
            homography = tuple(h_components) if all(v is not None for v in h_components) else None

            track_count = _parse_int_or_empty(row.get("track_count", ""))
            inlier_count = _parse_int_or_empty(row.get("inlier_count", ""))

            process_time_raw = row.get("process_time_ns", "")
            if process_time_raw == "":
                raise ContractViolationError("process_time_ns must not be empty")
            process_time_ns = int(process_time_raw)
            if process_time_ns < 0:
                raise ContractViolationError(f"process_time_ns must be >= 0, got {process_time_ns}")

            yaw = floats["est_yaw_deg"]
            if not (0.0 <= yaw < 360.0):
                raise ContractViolationError(
                    f"est_yaw_deg must be in [0, 360), got {yaw} at frame_index {row['frame_index']}"
                )

            confidence = (
                _parse_confidence(row, success) if has_confidence_columns else None
            )

            frame_estimates.append(
                FrameEstimate(
                    frame_index=int(row["frame_index"]),
                    timestamp_s=floats["timestamp_s"],
                    est_x=floats["est_x"],
                    est_y=floats["est_y"],
                    est_z=floats["est_z"],
                    est_yaw_deg=yaw,
                    success=success,
                    event=event,
                    reference_id=int(row["reference_id"]),
                    homography=homography,
                    track_count=track_count,
                    inlier_count=inlier_count,
                    process_time_ns=process_time_ns,
                    confidence=confidence,
                )
            )

    if not frame_estimates:
        raise ContractViolationError("Run record has zero frame estimates")

    # Invariant: frame_index strictly increasing
    indices = [fe.frame_index for fe in frame_estimates]
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise ContractViolationError("frame_index must be strictly increasing in frames.csv")

    # Invariant: row count equals manifest processed_count
    if len(frame_estimates) != manifest.processed_count:
        raise ContractViolationError(
            f"frames.csv has {len(frame_estimates)} rows but manifest.processed_count is "
            f"{manifest.processed_count} -- the record is truncated or the manifest is stale"
        )

    # Invariant: exactly one init row, and it is first
    init_positions = [i for i, fe in enumerate(frame_estimates) if fe.event == "init"]
    if len(init_positions) != 1:
        raise ContractViolationError(
            f"Expected exactly one row with event=init, found {len(init_positions)}"
        )
    if init_positions[0] != 0:
        raise ContractViolationError("The event=init row must be the first row")

    frame_indices_arr = np.array(indices, dtype=np.int64)
    timestamps_arr = np.array([fe.timestamp_s for fe in frame_estimates], dtype=np.float64)
    est_x_arr = np.array([fe.est_x for fe in frame_estimates], dtype=np.float64)
    est_y_arr = np.array([fe.est_y for fe in frame_estimates], dtype=np.float64)
    est_yaw_arr = np.array([fe.est_yaw_deg for fe in frame_estimates], dtype=np.float64)
    success_arr = np.array([fe.success for fe in frame_estimates], dtype=bool)

    return RunRecord(
        manifest=manifest,
        frame_estimates=frame_estimates,
        frame_indices=frame_indices_arr,
        timestamps_s=timestamps_arr,
        est_x=est_x_arr,
        est_y=est_y_arr,
        est_yaw_deg=est_yaw_arr,
        success=success_arr,
    )


def verify_matches_dataset(run_record: RunRecord, dataset: Dataset) -> None:
    """Cross-check a run record against the dataset it claims to have been run over.

    Requires both objects because the check is inherently relational: a run record
    cannot validate its own dataset reference in isolation (contracts/run-record.md
    invariant 2 and the manifest's dataset_id/dataset_revision fields).
    """
    if run_record.manifest.dataset_id != dataset.dataset_id:
        raise DatasetMismatchError(
            f"Run record references dataset_id={run_record.manifest.dataset_id!r} "
            f"but was evaluated against dataset_id={dataset.dataset_id!r}"
        )
    if run_record.manifest.dataset_revision != dataset.dataset_revision:
        raise DatasetMismatchError(
            f"Run record references dataset_revision={run_record.manifest.dataset_revision!r} "
            f"but the dataset is at revision={dataset.dataset_revision!r} -- "
            f"the dataset may have changed since this run was captured"
        )

    dataset_frame_set = set(dataset.frame_indices.tolist())
    missing = [fi for fi in run_record.frame_indices.tolist() if fi not in dataset_frame_set]
    if missing:
        raise ContractViolationError(
            f"Run record references frame_index values not present in the dataset: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
