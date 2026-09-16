"""Dataset loading and validation (contracts/dataset.md, data-model.md).

A dataset is source-agnostic by construction (FR-002, SC-009): nothing downstream of this
loader knows whether the ground truth came from a real flight, the Java simulator, UE5, or
a synthetic generator. That is enforced here, once, rather than trusted of every caller.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from naveval.errors import ContractViolationError, ConventionError, SchemaVersionError

SUPPORTED_SCHEMA_MAJOR = 1

CANONICAL_FRAME_CONVENTION = "ENU"
CANONICAL_HEADING_CONVENTION = "compass_cw_from_north"

# "historical_imagery" (spec 006/007, added 2026-08-22): a public real-world dataset used as
# cross-season skyline-relocalization evidence (e.g. Nordland) -- neither a UAV flight, a
# simulator, nor a synthetic generator, so none of the other four labels would be honest. Additive
# (minor) per contracts/dataset.md's own versioning rule: an appended enum value, no field
# removed/renamed/retyped, existing readers unaffected. See research.md (007) for the rationale.
VALID_SOURCE_TYPES = {"real_flight", "java_simulator", "ue5_simulator", "synthetic", "historical_imagery"}
VALID_EVIDENCE_TIERS = {"T1", "T2", "T3", "T4"}
VALID_QUALITY_CLASSES = {"consumer_gnss", "rtk_gnss", "simulator_exact", "fc_heading", "unknown"}

# contracts/dataset.md: "a java_simulator dataset declares T2 at most" and the near-T1
# caveat is mandatory, so the evidence tier's meaning cannot drift by forgetting it.
# "historical_imagery" is capped at T3 for the same reason: it is real photography of a real
# place, but from a different platform/domain than the target UAV use case (contracts/
# reference-set.md's "rural/forward-camera, not UAV" caveat) -- never claimable as T4.
_SOURCE_TYPE_MAX_TIER_RANK = {
    "synthetic": 1,
    "java_simulator": 2,
    "ue5_simulator": 2,
    "historical_imagery": 3,
    "real_flight": 4,
}
_TIER_RANK = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}


def _major_version(schema_version: str) -> int:
    try:
        return int(schema_version.split(".", 1)[0])
    except (ValueError, IndexError, AttributeError) as exc:
        raise SchemaVersionError(f"Malformed schema_version: {schema_version!r}") from exc


@dataclass(frozen=True)
class GroundTruthQuality:
    quality_class: str
    nominal_accuracy: Optional[float]
    accuracy_source: Optional[str]
    notes: str = ""

    @staticmethod
    def from_dict(d: Optional[dict]) -> Optional["GroundTruthQuality"]:
        if d is None:
            return None
        cls = d.get("class")
        if cls not in VALID_QUALITY_CLASSES:
            raise ContractViolationError(
                f"Unknown ground-truth quality class {cls!r}; expected one of {sorted(VALID_QUALITY_CLASSES)}"
            )
        return GroundTruthQuality(
            quality_class=cls,
            nominal_accuracy=d.get("nominal_accuracy"),
            accuracy_source=d.get("accuracy_source"),
            notes=d.get("notes", ""),
        )


@dataclass(frozen=True)
class Metadata:
    flight_id: str = ""
    trajectory_type: str = ""
    frame_rate_hz: Optional[float] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    environment: str = ""
    nominal_altitude_m: Optional[float] = None
    nominal_speed_ms: Optional[float] = None
    capture_date: str = ""
    notes: str = ""

    @staticmethod
    def from_dict(d: Optional[dict]) -> "Metadata":
        d = d or {}
        return Metadata(
            flight_id=d.get("flight_id", ""),
            trajectory_type=d.get("trajectory_type", ""),
            frame_rate_hz=d.get("frame_rate_hz"),
            image_width=d.get("image_width"),
            image_height=d.get("image_height"),
            environment=d.get("environment", ""),
            nominal_altitude_m=d.get("nominal_altitude_m"),
            nominal_speed_ms=d.get("nominal_speed_ms"),
            capture_date=d.get("capture_date", ""),
            notes=d.get("notes", ""),
        )


@dataclass
class Dataset:
    """A loaded, validated dataset. Numeric series are exposed as numpy arrays."""

    schema_version: str
    dataset_id: str
    dataset_revision: str
    source_type: str
    evidence_tier: str
    evidence_caveat: Optional[str]

    frame_clock: str
    gt_clock: str
    clock_offset_s: float
    clock_offset_source: str
    clock_drift_s_per_s: float

    frame_convention: str
    heading_convention: str
    local_frame_origin: Optional[dict]

    position_quality: Optional[GroundTruthQuality]
    heading_quality: Optional[GroundTruthQuality]
    height_quality: Optional[GroundTruthQuality]

    metadata: Metadata

    # Frames
    frame_indices: np.ndarray  # int
    frame_timestamps: np.ndarray  # float
    frame_image_paths: list

    # Ground truth (all empty arrays if no groundtruth.csv)
    gt_timestamps: np.ndarray  # float
    gt_east: np.ndarray  # float
    gt_north: np.ndarray  # float
    gt_up: np.ndarray  # float, NaN where absent
    gt_heading: np.ndarray  # float, NaN where absent
    gt_fix_quality: list
    gt_valid: np.ndarray  # bool

    root: Path = field(repr=False, default=None)

    @property
    def has_ground_truth(self) -> bool:
        return self.gt_timestamps.size > 0

    @property
    def has_heading(self) -> bool:
        return self.has_ground_truth and not np.all(np.isnan(self.gt_heading))

    @property
    def has_height(self) -> bool:
        return self.has_ground_truth and not np.all(np.isnan(self.gt_up))


def _parse_float_or_empty(s: str) -> float:
    """Parse a CSV field to float, mapping the empty string to NaN.

    Rejects literal NaN/Inf tokens per contract -- "not available" must be spelled with
    an empty field, never with a value that could be mistaken for a real (if unusual)
    measurement.
    """
    if s == "":
        return float("nan")
    low = s.strip().lower()
    if low in ("nan", "inf", "-inf", "+inf", "infinity", "-infinity"):
        raise ContractViolationError(f"NaN/Inf is not a valid field value: {s!r}")
    return float(s)


def _parse_bool(s: str) -> bool:
    if s == "true":
        return True
    if s == "false":
        return False
    raise ContractViolationError(f"Invalid boolean field value: {s!r} (expected 'true' or 'false')")


def load_dataset(root: Path | str) -> Dataset:
    """Load and validate a dataset directory per contracts/dataset.md.

    Raises SchemaVersionError, ConventionError, or ContractViolationError on malformed
    input rather than returning a partially-valid object -- a dataset that fails to load
    correctly must not be silently evaluated.
    """
    root = Path(root)
    descriptor_path = root / "dataset.json"
    if not descriptor_path.exists():
        raise ContractViolationError(f"Missing dataset.json in {root}")

    with descriptor_path.open("r", encoding="utf-8") as f:
        d = json.load(f)

    schema_version = d.get("schema_version", "")
    if _major_version(schema_version) != SUPPORTED_SCHEMA_MAJOR:
        raise SchemaVersionError(
            f"Unsupported dataset schema major version: {schema_version!r} "
            f"(this reader supports major version {SUPPORTED_SCHEMA_MAJOR})"
        )

    source_type = d.get("source_type")
    if source_type not in VALID_SOURCE_TYPES:
        raise ContractViolationError(
            f"Unknown source_type {source_type!r}; expected one of {sorted(VALID_SOURCE_TYPES)}"
        )

    evidence_tier = d.get("evidence_tier")
    if evidence_tier not in VALID_EVIDENCE_TIERS:
        raise ContractViolationError(
            f"Unknown evidence_tier {evidence_tier!r}; expected one of {sorted(VALID_EVIDENCE_TIERS)}"
        )
    max_rank = _SOURCE_TYPE_MAX_TIER_RANK.get(source_type)
    if max_rank is not None and _TIER_RANK[evidence_tier] > max_rank:
        raise ContractViolationError(
            f"source_type {source_type!r} declares evidence_tier {evidence_tier!r}, "
            f"which exceeds the maximum this source type may claim"
        )

    evidence_caveat = d.get("evidence_caveat")
    if source_type == "java_simulator" and not evidence_caveat:
        raise ContractViolationError(
            "java_simulator datasets must declare a non-null evidence_caveat "
            "(the near-T1 VO-front-end caveat; see contracts/dataset.md and "
            "research-log 2026-08-10) -- this is what mechanically prevents a near-T1 "
            "result from being reported as plain T2"
        )
    if source_type == "historical_imagery" and not evidence_caveat:
        raise ContractViolationError(
            "historical_imagery datasets must declare a non-null evidence_caveat "
            "(the platform/domain-mismatch caveat -- e.g. rural/forward-camera, not UAV) "
            "-- this is what "
            "mechanically prevents a different-domain result from being read as an unqualified T3"
        )

    frame_convention = d.get("frame_convention")
    if frame_convention != CANONICAL_FRAME_CONVENTION:
        raise ConventionError(
            f"Dataset declares frame_convention={frame_convention!r}; "
            f"only {CANONICAL_FRAME_CONVENTION!r} is accepted (DEC-004). "
            f"Convert to canonical ENU at ingest before evaluating."
        )
    heading_convention = d.get("heading_convention")
    if heading_convention != CANONICAL_HEADING_CONVENTION:
        raise ConventionError(
            f"Dataset declares heading_convention={heading_convention!r}; "
            f"only {CANONICAL_HEADING_CONVENTION!r} is accepted (DEC-004)."
        )

    position_quality = GroundTruthQuality.from_dict(d.get("position_quality"))
    heading_quality = GroundTruthQuality.from_dict(d.get("heading_quality"))
    height_quality = GroundTruthQuality.from_dict(d.get("height_quality"))

    metadata = Metadata.from_dict(d.get("metadata"))

    # --- frames.csv ---
    frames_path = root / "frames.csv"
    if not frames_path.exists():
        raise ContractViolationError(f"Missing frames.csv in {root}")

    frame_indices = []
    frame_timestamps = []
    frame_image_paths = []
    with frames_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_indices.append(int(row["frame_index"]))
            frame_timestamps.append(float(row["timestamp_s"]))
            frame_image_paths.append(row["image_path"])

    if not frame_indices:
        raise ContractViolationError("Dataset has zero frames; at least one is required (FR-001)")

    frame_indices_arr = np.array(frame_indices, dtype=np.int64)
    frame_timestamps_arr = np.array(frame_timestamps, dtype=np.float64)

    if not np.array_equal(frame_indices_arr, np.arange(len(frame_indices_arr))):
        raise ContractViolationError(
            "frame_index must be 0-based and contiguous "
            f"(got {frame_indices_arr[:5].tolist()}...)"
        )
    if np.any(np.diff(frame_timestamps_arr) <= 0):
        raise ContractViolationError("Frame timestamps must be strictly increasing")

    # --- groundtruth.csv (optional) ---
    gt_path = root / "groundtruth.csv"
    gt_timestamps: list = []
    gt_east: list = []
    gt_north: list = []
    gt_up: list = []
    gt_heading: list = []
    gt_fix_quality: list = []
    gt_valid: list = []

    if gt_path.exists():
        with gt_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gt_timestamps.append(float(row["timestamp_s"]))
                valid = _parse_bool(row["valid"])
                gt_valid.append(valid)
                gt_east.append(_parse_float_or_empty(row["east_m"]))
                gt_north.append(_parse_float_or_empty(row["north_m"]))
                gt_up.append(_parse_float_or_empty(row["up_m"]))
                gt_heading.append(_parse_float_or_empty(row["heading_deg"]))
                gt_fix_quality.append(row.get("fix_quality", ""))

        gt_timestamps_arr = np.array(gt_timestamps, dtype=np.float64)
        if gt_timestamps_arr.size and np.any(np.diff(gt_timestamps_arr) <= 0):
            raise ContractViolationError("Ground-truth timestamps must be strictly increasing")
    else:
        gt_timestamps_arr = np.array([], dtype=np.float64)

    return Dataset(
        schema_version=schema_version,
        dataset_id=d["dataset_id"],
        dataset_revision=d["dataset_revision"],
        source_type=source_type,
        evidence_tier=evidence_tier,
        evidence_caveat=evidence_caveat,
        frame_clock=d.get("frame_clock", ""),
        gt_clock=d.get("gt_clock", ""),
        clock_offset_s=float(d.get("clock_offset_s", 0.0)),
        clock_offset_source=d.get("clock_offset_source", ""),
        clock_drift_s_per_s=float(d.get("clock_drift_s_per_s", 0.0)),
        frame_convention=frame_convention,
        heading_convention=heading_convention,
        local_frame_origin=d.get("local_frame_origin"),
        position_quality=position_quality,
        heading_quality=heading_quality,
        height_quality=height_quality,
        metadata=metadata,
        frame_indices=frame_indices_arr,
        frame_timestamps=frame_timestamps_arr,
        frame_image_paths=frame_image_paths,
        gt_timestamps=gt_timestamps_arr,
        gt_east=np.array(gt_east, dtype=np.float64),
        gt_north=np.array(gt_north, dtype=np.float64),
        gt_up=np.array(gt_up, dtype=np.float64),
        gt_heading=np.array(gt_heading, dtype=np.float64),
        gt_fix_quality=gt_fix_quality,
        gt_valid=np.array(gt_valid, dtype=bool),
        root=root,
    )
