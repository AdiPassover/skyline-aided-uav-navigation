"""Real-flight ingest (contracts/dataset.md "Real-flight ingest -- requires collected data").

Converts recorded telemetry into a conforming dataset directory: frame indexing with
timestamps, geodetic -> ENU conversion about a chosen origin, flight-controller heading ->
canonical conversion, clock-offset/drift estimation from bracketing synchronisation events,
and an honest quality-class declaration derived from observed fix quality (never assumed).

**Scope boundary, stated rather than silently assumed**: this module does not decode video.
`evaluation/requirements.txt` deliberately carries no video/image-I/O dependency (none has been
evaluated or approved here). `extract_frames` therefore takes an already-produced sequence of
(timestamp, image_path) pairs -- the output of whatever frame-extraction step a future flight
produces -- and is responsible only for validating and indexing them per contracts/dataset.md.
No real flight has been flown for this feature (tasks.md "Explicitly NOT tasks in this
feature"); T082 exercises every function here against constructed telemetry with known
injected values.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from naveval.frames import geodetic_to_enu, normalize_heading_deg, enu_math_to_compass_deg

# contracts/dataset.md field rules. Must stay a subset of naveval.dataset.VALID_QUALITY_CLASSES
# (the loader's shared, axis-agnostic set) -- found out of sync for heading on 2026-08-14 while
# wiring naveval.ingest_mars_lvig's MCAP path: naveval.support._HEADING_METRIC_MATRIX already
# classifies "rtk_gnss" for yaw metrics (dual-antenna RTK heading, e.g. DJI M300), and the loader
# already accepted it, but this writer-side set had never been updated to match, so no producer
# could declare it. Added rather than left inconsistent (a concrete incompatibility, not a
# redesign).
_VALID_POSITION_QUALITY_CLASSES = {"consumer_gnss", "rtk_gnss", "unknown"}
_VALID_HEADING_QUALITY_CLASSES = {"fc_heading", "rtk_gnss", "unknown"}


class IngestError(ValueError):
    """Raised when constructed/collected telemetry cannot be honestly converted."""


# --- Frame extraction (contract: "extract frames with timestamps") ---


@dataclass(frozen=True)
class FrameRecord:
    frame_index: int
    timestamp_s: float
    image_path: str


def extract_frames(frame_timestamps_s: Sequence[float], image_paths: Sequence[str]) -> list:
    """Index already-extracted frames per contracts/dataset.md: 0-based, contiguous,
    strictly increasing timestamps. Does not read or write image files."""
    if len(frame_timestamps_s) != len(image_paths):
        raise IngestError(
            f"frame_timestamps_s ({len(frame_timestamps_s)}) and image_paths "
            f"({len(image_paths)}) must have the same length"
        )
    if len(frame_timestamps_s) == 0:
        raise IngestError("At least one frame is required (FR-001)")

    records = []
    last_ts = float("-inf")
    for i, (ts, path) in enumerate(zip(frame_timestamps_s, image_paths)):
        ts = float(ts)
        if ts <= last_ts:
            raise IngestError(f"Frame timestamps must be strictly increasing; frame {i} has timestamp_s={ts}")
        last_ts = ts
        records.append(FrameRecord(frame_index=i, timestamp_s=ts, image_path=path))
    return records


# --- Geodetic -> ENU ground truth (contract: "convert geodetic to ENU about a chosen origin") ---


@dataclass(frozen=True)
class TelemetrySample:
    """One flight-controller telemetry sample, in whatever units/convention the FC emits."""

    timestamp_s: float  # on the FC's own clock (gt_clock)
    lat_deg: Optional[float]
    lon_deg: Optional[float]
    alt_m: Optional[float]
    heading_deg: Optional[float]  # in `heading_convention`, or None if unavailable
    fix_quality: str
    valid: bool


@dataclass(frozen=True)
class GroundTruthRow:
    timestamp_s: float
    east_m: Optional[float]
    north_m: Optional[float]
    up_m: Optional[float]
    heading_deg: Optional[float]
    fix_quality: str
    valid: bool


_SUPPORTED_HEADING_CONVENTIONS = {"compass_cw_from_north", "enu_math_ccw_from_east"}


def convert_geodetic_track_to_enu(
    samples: Sequence[TelemetrySample],
    origin_lat_deg: float,
    origin_lon_deg: float,
    origin_alt_m: float,
    heading_convention: str = "compass_cw_from_north",
) -> list:
    """Convert a telemetry track to canonical-ENU ground-truth rows (DEC-004).

    `heading_convention` names the *source* telemetry's own heading convention, converted
    here to the canonical `compass_cw_from_north`, once, at ingest -- FR-006's "raw geodetic
    coordinates never reach the evaluator" and DEC-004's "no source format's convention is
    assumed downstream". `east_m`/`north_m`/`heading_deg` are None (written empty) wherever
    the sample itself is invalid or missing that field, never fabricated.
    """
    if heading_convention not in _SUPPORTED_HEADING_CONVENTIONS:
        raise IngestError(
            f"Unsupported source heading_convention {heading_convention!r}; "
            f"expected one of {sorted(_SUPPORTED_HEADING_CONVENTIONS)}"
        )

    rows = []
    last_ts = float("-inf")
    for i, s in enumerate(samples):
        if s.timestamp_s <= last_ts:
            raise IngestError(f"Ground-truth timestamps must be strictly increasing; sample {i}")
        last_ts = s.timestamp_s

        if s.valid and s.lat_deg is not None and s.lon_deg is not None:
            east_m, north_m, _ = geodetic_to_enu(
                s.lat_deg, s.lon_deg, s.alt_m if s.alt_m is not None else origin_alt_m,
                origin_lat_deg, origin_lon_deg, origin_alt_m,
            )
        else:
            east_m, north_m = None, None

        up_m = None
        if s.valid and s.alt_m is not None:
            _, _, up_m = geodetic_to_enu(
                s.lat_deg if s.lat_deg is not None else origin_lat_deg,
                s.lon_deg if s.lon_deg is not None else origin_lon_deg,
                s.alt_m, origin_lat_deg, origin_lon_deg, origin_alt_m,
            )

        heading_deg = None
        if s.heading_deg is not None:
            heading_deg = (
                normalize_heading_deg(s.heading_deg) if heading_convention == "compass_cw_from_north"
                else enu_math_to_compass_deg(s.heading_deg)
            )

        rows.append(GroundTruthRow(
            timestamp_s=s.timestamp_s, east_m=east_m, north_m=north_m, up_m=up_m,
            heading_deg=heading_deg, fix_quality=s.fix_quality, valid=s.valid,
        ))
    return rows


# --- Clock offset/drift estimation (contract: "estimate clock offset from sync events") ---


@dataclass(frozen=True)
class ClockSync:
    offset_s: float
    drift_s_per_s: float
    source_description: str


def estimate_clock_sync(
    gt_timestamps_s: np.ndarray, sync_events: Sequence[tuple],
) -> ClockSync:
    """Estimate `clock_offset_s`/`clock_drift_s_per_s` (contracts/dataset.md field rules:
    ``frame_time = gt_time + clock_offset_s``, drift applied about `gt_timestamps_s[0]` --
    matching `naveval.sync.synchronize`'s anchor exactly, so a dataset built from this
    estimate synchronizes correctly).

    `sync_events` is a sequence of `(frame_time_s, gt_time_s)` pairs, each identifying the
    same physical instant on both clocks (contract example: "start/end LED sync events").
    One event fixes the offset only -- drift is not estimable from a single point (FR-011),
    and `drift_s_per_s` is reported as exactly 0.0 with that limitation stated in
    `source_description` rather than silently implied to be measured. Two events bracketing
    the flight fix both.
    """
    if len(sync_events) == 0:
        raise IngestError("At least one synchronisation event is required to estimate clock offset")
    if len(sync_events) > 2:
        raise IngestError(
            f"estimate_clock_sync takes at most 2 bracketing sync events, got {len(sync_events)}"
        )

    gt_timestamps_s = np.asarray(gt_timestamps_s, dtype=np.float64)
    if gt_timestamps_s.size == 0:
        raise IngestError("gt_timestamps_s must be non-empty (drift is anchored on its first sample)")
    anchor = float(gt_timestamps_s[0])

    if len(sync_events) == 1:
        f1, g1 = sync_events[0]
        offset_s = float(f1) - float(g1)
        return ClockSync(
            offset_s=offset_s, drift_s_per_s=0.0,
            source_description=(
                f"single synchronisation event at gt_time={g1}s; offset correctable, "
                "drift not estimable from one event (FR-011)"
            ),
        )

    (f1, g1), (f2, g2) = sync_events
    g1, g2, f1, f2 = float(g1), float(g2), float(f1), float(f2)
    if g1 == g2:
        raise IngestError("The two synchronisation events must occur at different gt_times")

    x1, x2 = g1 - anchor, g2 - anchor
    y1, y2 = f1 - anchor, f2 - anchor
    drift_s_per_s = ((y1 - x1) - (y2 - x2)) / (x1 - x2)
    offset_s = (y1 - x1) - x1 * drift_s_per_s

    return ClockSync(
        offset_s=offset_s, drift_s_per_s=drift_s_per_s,
        source_description=(
            f"two bracketing synchronisation events at gt_time={g1}s and gt_time={g2}s"
        ),
    )


# --- Honest quality-class declaration (contract: "declare quality classes honestly") ---


def classify_position_quality(fix_qualities: Sequence[str]) -> str:
    """Map observed `fix_quality` strings to a contract quality class, failing closed to
    `"unknown"` on anything not confidently recognised (contracts/dataset.md: "Quality
    class: unknown maps every dependent metric to unsupported. Undeclared quality fails
    closed.") -- never guessed generous."""
    if not fix_qualities:
        return "unknown"
    lowered = [str(q).lower() for q in fix_qualities]
    if all(("rtk" in q and "fixed" in q) or "rtk_fixed" in q for q in lowered):
        return "rtk_gnss"
    if all("3d" in q for q in lowered):
        return "consumer_gnss"
    return "unknown"


def classify_heading_quality(has_heading: bool) -> str:
    """`"fc_heading"` when the telemetry carries a flight-controller heading channel at all,
    else `"unknown"` -- there is no intermediate honest claim to make from ingest alone."""
    return "fc_heading" if has_heading else "unknown"


# --- Dataset assembly and writing ---


def build_dataset_json(
    dataset_id: str,
    dataset_revision: str,
    evidence_tier: str,
    clock_sync: ClockSync,
    position_quality_class: str,
    heading_quality_class: str,
    metadata: dict,
    local_frame_origin: Optional[dict] = None,
    height_quality_class: Optional[str] = None,
) -> dict:
    """Assemble the `dataset.json` payload (contracts/dataset.md). `evidence_tier` is a
    caller-supplied judgment (real_flight may claim up to T4), never defaulted here --
    inventing a tier would be exactly the kind of unmeasured claim Principle IX forbids."""
    if position_quality_class not in _VALID_POSITION_QUALITY_CLASSES:
        raise IngestError(
            f"position_quality_class {position_quality_class!r} invalid; "
            f"expected one of {sorted(_VALID_POSITION_QUALITY_CLASSES)}"
        )
    if heading_quality_class not in _VALID_HEADING_QUALITY_CLASSES:
        raise IngestError(
            f"heading_quality_class {heading_quality_class!r} invalid; "
            f"expected one of {sorted(_VALID_HEADING_QUALITY_CLASSES)}"
        )

    return {
        "schema_version": "1.0.0",
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "source_type": "real_flight",
        "evidence_tier": evidence_tier,
        "evidence_caveat": None,
        "frame_clock": "camera_monotonic",
        "gt_clock": "fc_boot_monotonic",
        "clock_offset_s": clock_sync.offset_s,
        "clock_offset_source": clock_sync.source_description,
        "clock_drift_s_per_s": clock_sync.drift_s_per_s,
        "frame_convention": "ENU",
        "heading_convention": "compass_cw_from_north",
        "local_frame_origin": local_frame_origin,
        "position_quality": {
            "class": position_quality_class,
            "nominal_accuracy": None,
            "accuracy_source": "observed fix_quality, see groundtruth.csv",
            "notes": "",
        },
        "heading_quality": {
            "class": heading_quality_class,
            "nominal_accuracy": None,
            "accuracy_source": "flight-controller heading channel",
            "notes": "",
        } if heading_quality_class != "unknown" else {
            "class": "unknown", "nominal_accuracy": None, "accuracy_source": None, "notes": "",
        },
        "height_quality": (
            {"class": height_quality_class, "nominal_accuracy": None,
             "accuracy_source": "observed fix_quality, see groundtruth.csv", "notes": ""}
            if height_quality_class else None
        ),
        "metadata": metadata,
    }


def _fmt(v: Optional[float]) -> str:
    return "" if v is None else repr(float(v))


def write_dataset(
    root: Path,
    dataset_json: dict,
    frame_records: Sequence[FrameRecord],
    groundtruth_rows: Sequence[GroundTruthRow],
) -> None:
    """Write `dataset.json`, `frames.csv`, and (if non-empty) `groundtruth.csv` per
    contracts/dataset.md. Does not write `images/` -- frame extraction (per this module's
    documented scope boundary) supplies image paths, not image bytes."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    with (root / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)
        f.write("\n")

    with (root / "frames.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "timestamp_s", "image_path"])
        for r in frame_records:
            w.writerow([r.frame_index, repr(r.timestamp_s), r.image_path])

    if groundtruth_rows:
        with (root / "groundtruth.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_s", "east_m", "north_m", "up_m", "heading_deg", "fix_quality", "valid"])
            for r in groundtruth_rows:
                w.writerow([
                    repr(r.timestamp_s), _fmt(r.east_m), _fmt(r.north_m), _fmt(r.up_m),
                    _fmt(r.heading_deg), r.fix_quality, "true" if r.valid else "false",
                ])
