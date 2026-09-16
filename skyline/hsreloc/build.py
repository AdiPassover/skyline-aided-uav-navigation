"""Observations + oracle curves + split -> the `006` reference set + `004`/`006` query set (spec 007,
T023; `contracts/reference-set.md`, `contracts/skyline-query-set.md`, `contracts/observation-record.md`).

Down-projects the source-agnostic observation records into the two frozen consumer contracts spec `008`
reads. No matcher, no ranking, no scoring -- this module only assembles already-validated, already
QC'd, already human-annotated data into the shape the evaluator expects (`DEC-013`).

Every curve placed in either output MUST already be `stored=True` in its session's
`skylines_oracle/annotations.json` (`store.py`'s safeguard) -- this module does not re-derive or
re-accept anything; a missing curve for a requested observation is a hard error, not silently skipped.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from hsreloc.observation import Observation


class BuildError(Exception):
    """The requested reference/query set could not be assembled from validated inputs."""


def local_frame_origin_from_sessions(oracle_dirs: dict) -> dict:
    """The ENU origin the observations were actually normalized against, read from their sessions.

    ``oracle_dirs`` maps ``session_id -> <session_dir>/skylines_oracle``, so each session's
    ``session.json`` is its parent. Every session involved in one build MUST declare the same origin
    -- their ``pos_east_m``/``pos_north_m`` are only comparable if they share a frame, and a reference
    set and query set built from disagreeing sessions could not be matched against each other at all.

    Returns the origin in the *dataset* contract's key names (``lat_deg``/``lon_deg``/``alt_m``);
    ``session.json`` uses ``lat``/``lon``/``alt_m`` (`contracts/observation-record.md`).

    Raises ``BuildError`` rather than guessing: a session that declares no origin (or a null one,
    e.g. an adapter's placeholder before ``ingest.normalize_to_enu`` has run) means the frame its ENU
    values live in is unknown, and inventing one produces coordinates that are silently wrong in
    absolute terms while looking perfectly consistent relatively -- exactly the defect this function
    exists to prevent (research R5, SKY lane 2026-08-23).
    """
    if not oracle_dirs:
        raise BuildError("cannot determine the local frame origin: no sessions given")
    origins: dict = {}
    for session_id, oracle_dir in sorted(oracle_dirs.items()):
        session_path = Path(oracle_dir).parent / "session.json"
        if not session_path.exists():
            raise BuildError(f"session {session_id!r}: no session.json at {session_path}")
        with session_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        origin = meta.get("local_frame_origin") or {}
        lat, lon, alt = origin.get("lat"), origin.get("lon"), origin.get("alt_m")
        if lat is None or lon is None:
            raise BuildError(
                f"session {session_id!r}: session.json declares no local_frame_origin -- the frame its "
                f"ENU coordinates live in is unknown, so no origin can be recorded honestly "
                f"(run ingest.normalize_to_enu first)"
            )
        origins[session_id] = {"lat_deg": float(lat), "lon_deg": float(lon),
                               "alt_m": float(alt if alt is not None else 0.0)}

    ids = sorted(origins)
    first = origins[ids[0]]
    for sid in ids[1:]:
        other = origins[sid]
        if (abs(other["lat_deg"] - first["lat_deg"]) > 1e-9
                or abs(other["lon_deg"] - first["lon_deg"]) > 1e-9
                or abs(other["alt_m"] - first["alt_m"]) > 1e-6):
            raise BuildError(
                f"sessions declare different local frame origins and cannot be built into one "
                f"comparable set: {ids[0]}={first}, {sid}={other}"
            )
    return dict(first)


def _load_curve(oracle_dir: Path, observation_id: str, expected_width: int) -> np.ndarray:
    curve_path = Path(oracle_dir) / f"{observation_id}.csv"
    if not curve_path.exists():
        raise BuildError(
            f"{observation_id}: no stored oracle curve at {curve_path} -- only observations with "
            "store.py's stored=True are eligible for a build (structural rejects and "
            "review_required-pending curves are excluded by construction, not by this check)"
        )
    rows = {}
    with curve_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["col"])] = float(row["row"])
    if sorted(rows) != list(range(expected_width)):
        raise BuildError(
            f"{observation_id}: stored curve does not cover columns 0..{expected_width - 1} contiguously"
        )
    return np.array([rows[c] for c in range(expected_width)], dtype=np.float64)


def build_reference_set(
    reference_obs: list,
    oracle_dirs: dict,
    out_dir: Path | str,
    ref_set_id: str,
    *,
    dataset_id: str,
    extraction_mode: str = "oracle:manual",
    evidence_tier: str = "T3",
    evidence_caveat: str,
    reference_spacing_m: float,
    local_frame_origin: Optional[dict] = None,
) -> dict:
    """Write ``<out_dir>/<ref_set_id>/`` (``manifest.json`` + ``references.csv`` + ``curves.npz``).
    ``oracle_dirs`` maps ``session_id -> <session_dir>/skylines_oracle``. Returns the manifest dict.

    ``local_frame_origin`` (``lat_deg``/``lon_deg``/``alt_m``) is the geodetic anchor of the ENU
    coordinates in ``references.csv``; when omitted it is read from the involved sessions
    (`local_frame_origin_from_sessions`). It is recorded in the manifest so the set is self-describing
    -- ``references.csv``'s own contract already defines ``east_m``/``north_m`` as "ENU vs the set's
    local_frame_origin", and until 2026-08-23 no origin was actually emitted, leaving the reference
    coordinates with no declared frame at all."""
    if not reference_obs:
        raise BuildError("reference set is empty -- nothing to build")
    if local_frame_origin is None:
        local_frame_origin = local_frame_origin_from_sessions(oracle_dirs)
    out_dir = Path(out_dir) / ref_set_id
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = {}
    rows = []
    for o in reference_obs:
        oracle_dir = oracle_dirs.get(o.session_id)
        if oracle_dir is None:
            raise BuildError(f"no oracle_dirs entry for session {o.session_id!r} ({o.observation_id})")
        curves[o.observation_id] = _load_curve(oracle_dir, o.observation_id, o.image_width_px)
        rows.append({
            "reference_id": o.observation_id,
            "east_m": o.pos_east_m, "north_m": o.pos_north_m,
            "up_m": ("" if o.up_m is None else o.up_m),
            "heading_deg": ("" if o.yaw_deg is None else o.yaw_deg),
            "traversal_id": o.session_id,
            "extraction_mode": extraction_mode,
            "condition_season": o.extra.get("condition_season", ""),
        })

    curves_path = out_dir / "curves.npz"
    np.savez(curves_path, **curves)

    with (out_dir / "references.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    east = [o.pos_east_m for o in reference_obs if o.pos_east_m is not None]
    north = [o.pos_north_m for o in reference_obs if o.pos_north_m is not None]
    covered_area_km2 = (
        ((max(east) - min(east)) * (max(north) - min(north)) / 1.0e6) if east and north else 0.0
    )

    manifest = {
        "schema_version": "1.1.0",
        "local_frame_origin": dict(local_frame_origin),
        "reference_source": {
            "kind": "historical_imagery",
            "ref_set_id": ref_set_id,
            "dataset_id": dataset_id,
            "traversal_ids": sorted({o.session_id for o in reference_obs}),
            "reference_spacing_m": reference_spacing_m,
            "extraction_mode": extraction_mode,
            "n_references": len(reference_obs),
            "total_bytes": curves_path.stat().st_size,
            "covered_area_km2": round(covered_area_km2, 4),
            "evidence_tier": evidence_tier,
            "evidence_caveat": evidence_caveat,
        },
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def build_query_set(
    query_obs: list,
    oracle_dirs: dict,
    images_dirs: dict,
    out_dir: Path | str,
    query_set_id: str,
    *,
    dataset_id: str,
    dataset_revision: str,
    query_in_coverage: dict,
    evidence_tier: str = "T3",
    evidence_caveat: str,
    source_type: str = "historical_imagery",
    extraction_mode: str = "oracle:manual",
    local_frame_origin: Optional[dict] = None,
) -> dict:
    """Write ``<out_dir>/<query_set_id>/`` (``dataset.json`` + ``frames.csv`` + ``groundtruth.csv`` +
    ``skyline_queries.csv`` + ``images/``). ``oracle_dirs``/``images_dirs`` map
    ``session_id -> <session_dir>/skylines_oracle`` / ``<session_dir>/images``. ``query_in_coverage``
    maps ``observation_id -> bool`` (from ``split.build_split``). Returns the ``dataset.json`` dict.

    ``local_frame_origin`` (``lat_deg``/``lon_deg``/``alt_m``) is the geodetic anchor the observations'
    ENU coordinates were normalized against; when omitted it is read from the involved sessions
    (`local_frame_origin_from_sessions`).

    **This used to be derived wrongly.** Until 2026-08-23 the origin was taken from the *first query
    observation's own* lat/lon, which is not the frame ``pos_east_m``/``pos_north_m`` live in: for
    `skyquery-nordland-sf-oracle-v1` the two differ by ~86 km, so every relative quantity (retrieval,
    tolerances, position error) stayed correct while any absolute ENU->geodetic conversion was badly
    wrong. Found by the SKY lane while planning the relocalizer (research R5, 2026-08-23)."""
    if not query_obs:
        raise BuildError("query set is empty -- nothing to build")
    if local_frame_origin is None:
        local_frame_origin = local_frame_origin_from_sessions(oracle_dirs)
    missing_coverage = {o.observation_id for o in query_obs} - set(query_in_coverage)
    if missing_coverage:
        raise BuildError(f"query_in_coverage missing entries for {sorted(missing_coverage)}")

    out_dir = Path(out_dir) / query_set_id
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # chronological order within each session's own numbering (frame_index is chronological for
    # Nordland, R8) so timestamps stay strictly increasing after reindexing to 0-based query_index.
    ordered = sorted(query_obs, key=lambda o: (o.session_id, o.frame_index))

    frame_rows, gt_rows, sidecar_rows = [], [], []
    widths, heights = set(), set()
    for qi, o in enumerate(ordered):
        images_dir = images_dirs.get(o.session_id)
        oracle_dir = oracle_dirs.get(o.session_id)
        if images_dir is None or oracle_dir is None:
            raise BuildError(f"no images_dirs/oracle_dirs entry for session {o.session_id!r}")
        _load_curve(oracle_dir, o.observation_id, o.image_width_px)  # existence/coverage check only

        src_img = Path(images_dir) / o.image_path
        dst_img = images_out / f"{o.observation_id}{Path(o.image_path).suffix}"
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        widths.add(o.image_width_px); heights.add(o.image_height_px)

        ts = float(qi)  # a strictly increasing, dataset-relative index-clock (no shared frame_rate_hz
                        # is known for Nordland's train camera, research R8) -- honestly not wall time.
        frame_rows.append({"frame_index": qi, "timestamp_s": ts,
                           "image_path": f"images/{dst_img.name}"})
        gt_rows.append({
            "timestamp_s": ts,
            "east_m": o.pos_east_m, "north_m": o.pos_north_m,
            "up_m": ("" if o.up_m is None else o.up_m),
            "heading_deg": ("" if o.yaw_deg is None else o.yaw_deg),
            "fix_quality": "gnss:nordland_annotation_log", "valid": "true",
        })
        sidecar_rows.append({
            "query_index": qi,
            "roll_deg": "", "pitch_deg": "",
            # Nordland has no compass-prior sensor distinct from the derived travel-direction
            # heading in groundtruth.csv; setting this to the GT heading would be an oracle leak
            # (a zero-error "prior"). Left empty = the honest prior-absent case (002 FR-003).
            "compass_prior_deg": "",
            "in_coverage": ("true" if query_in_coverage[o.observation_id] else "false"),
            "condition_season": o.extra.get("condition_season", ""),
            "traversal_id": o.session_id,
            "oracle_provenance": extraction_mode.split(":", 1)[-1],  # "manual" from "oracle:manual"
        })

    if len(widths) != 1 or len(heights) != 1:
        raise BuildError(f"query images have inconsistent dimensions: widths={widths} heights={heights}")

    with (out_dir / "frames.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_index", "timestamp_s", "image_path"])
        w.writeheader(); w.writerows(frame_rows)
    with (out_dir / "groundtruth.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp_s", "east_m", "north_m", "up_m",
                                          "heading_deg", "fix_quality", "valid"])
        w.writeheader(); w.writerows(gt_rows)
    with (out_dir / "skyline_queries.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_index", "roll_deg", "pitch_deg", "compass_prior_deg",
                                          "in_coverage", "condition_season", "traversal_id",
                                          "oracle_provenance"])
        w.writeheader(); w.writerows(sidecar_rows)

    dataset_json = {
        "schema_version": "1.0.0",
        "dataset_id": query_set_id,
        "dataset_revision": dataset_revision,
        "source_type": source_type,
        "evidence_tier": evidence_tier,
        "evidence_caveat": evidence_caveat,
        "frame_clock": "nordland_annotation_index",
        "gt_clock": "nordland_annotation_index",
        "clock_offset_s": 0.0,
        "clock_offset_source": "frames and GT share one fused GNSS-indexed log (Nordland); no separate clocks to sync",
        "clock_drift_s_per_s": 0.0,
        "frame_convention": "ENU",
        "heading_convention": "compass_cw_from_north",
        "local_frame_origin": dict(local_frame_origin),
        "position_quality": {
            "class": "consumer_gnss",
            "nominal_accuracy": None,
            "accuracy_source": "not published by Sunderhauf et al. 2013; treated conservatively as consumer-grade",
            "notes": "Nordland train-mounted GNSS log, ~1 Hz interpolated to frame rate (research R8)",
        },
        "heading_quality": {
            "class": "unknown",
            "nominal_accuracy": None,
            "accuracy_source": "derived from consecutive-fix GPS track bearing, not a measured heading (ingest.derive_headings)",
            "notes": "direction of travel, not a sensor reading; treat heading metrics as unsupported",
        },
        "height_quality": None,
        "metadata": {
            "flight_id": "", "trajectory_type": "linear_traversal",
            "frame_rate_hz": None, "image_width": widths.pop(), "image_height": heights.pop(),
            "environment": "rural/natural, train-mounted forward camera (Nordland)",
            "nominal_altitude_m": None, "nominal_speed_ms": None, "capture_date": "",
            "notes": "public dataset (Sunderhauf, Neubert & Protzel 2013), not UAV/own-area capture",
        },
        "skyline": {
            "camera": {"fov_deg": None, "calibration_id": None, "orientation": "forward"},
            "compass_prior": {
                "mode": "absent",
                "noise_model": None, "seed": None,
                "notes": ("Nordland has no compass-prior sensor distinct from the derived "
                         "travel-direction heading; every skyline_queries.csv row leaves "
                         "compass_prior_deg empty rather than leak the GT heading as a "
                         "zero-error prior (prior-absent case, 002 FR-003)"),
            },
            "condition_factors": ["condition_season"],
        },
    }
    with (out_dir / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2, sort_keys=True)
    return dataset_json
