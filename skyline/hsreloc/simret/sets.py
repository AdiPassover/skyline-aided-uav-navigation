"""spec-004 query sets and spec-006 reference sets for simulator sessions (PROT-SKY-001 §7 f, D10).

The contracts are frozen and unchanged; this writes into them. It mirrors ``hsreloc.placeret.sets``
rather than extending ``hsreloc.build.build_query_set``, and that is the ``PROT-SKY-001`` §8 **D10
decision taken**: ``build_query_set`` is spec-007's Nordland path — it hard-codes
``fix_quality: gnss:nordland_annotation_log`` and ``frame_clock: nordland_annotation_index``, emits
only ``condition_season``, blanks roll and pitch, and is a dependency of frozen experiments. Adding
simulator keyword arguments to it would edit a module three closed experiments rest on, to serve a
source whose fields it does not have. A sibling writer costs one file and touches nothing frozen.

What differs from the ECL writer, and why
-----------------------------------------
* **Heading is known and is written.** ECL had no compass convention in its SfM frame, so
  ``heading_deg`` was blank everywhere. A world-locked-North simulator camera has an exact heading,
  so it goes into ``groundtruth.csv`` and ``references.csv`` — this is the known-heading assumption
  (``PROT-SKY-001`` §6) appearing in the data rather than only in prose.
* **Pitch and roll are written.** They are exact here, and Stage 3 varies them deliberately.
* **The position class is ``simulator_exact``**, with zero sigma, because it is.
* **The tier is T2** with a non-null caveat, for every session, and ``tier_pooling: forbidden``
  remains the evaluator's rule.
* **The geodetic origin is declared synthetic.** There is no anchor and none is invented; the same
  declaration must appear in the query set, the reference set and the run config, or the matcher's
  own R5 preflight guard refuses the run.

One query set is written per (task, spacing, source). The three sources' materialisations are
byte-identical except for ``dataset_id`` and the provenance column — asserted by
``assert_source_materialisations_identical``, so a difference between two sources' results can only
come from their curves.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from hsreloc.retrieval.skyline_curve import CurveError
from hsreloc.simret.sources import PROVENANCE_BY_KEY

SETS_VERSION = "1.0.0"

#: No geodetic anchor exists for a simulator scene, so the origin is declared synthetic rather than
#: invented (the same resolution ``placeret`` used for ECL's SfM frame; PROT-SKY-001 §5).
SYNTHETIC_ORIGIN = {"lat_deg": 0.0, "lon_deg": 0.0, "alt_m": 0.0, "is_synthetic": True}

SOURCE_TYPE = "ue5_simulator"
EVIDENCE_TIER = "T2"
CAVEAT = (
    "UE5 simulator render: real 3D geometry, synthetic appearance and lighting. Poses are exact by "
    "construction in a DECLARED SYNTHETIC local ENU frame (no geodetic anchor — no absolute geographic "
    "quantity is meaningful); lengths are simulator metres. Camera is forward-looking with a known "
    "world heading. Skyline curves are the simulator's own sky mask (ground truth), a frozen classical "
    "extractor, or a SILVER semantic segmentor — labelled per run, never pooled. No UAV-flight claim, "
    "no real-sensor claim, no onboard or real-time claim."
)

QUERY_CONDITION_FIELDS = [
    "condition_stage", "condition_translation_m", "condition_along_m", "condition_lateral_m",
    "condition_up_diff_m", "condition_yaw_diff_deg", "condition_abs_yaw_diff_deg",
    "condition_pitch_diff_deg", "condition_roll_diff_deg", "condition_translation_bin",
    "condition_yaw_bin", "condition_pitch_bin", "condition_roll_bin", "condition_n_references",
    "condition_nearest_reference_id", "condition_reference_spacing_m", "condition_attainable",
    "condition_level", "condition_time_of_day", "condition_hour", "condition_clouds",
    "condition_session_id", "condition_observation_id", "condition_anchor_id",
]


class SetError(Exception):
    """A task set cannot be written as the contract requires."""


def query_set_id(task: str, spacing: float, source_key: str, population: str = "all") -> str:
    prefix = "sim" if population == "all" else f"sim-{population}"
    return f"{prefix}-{task}-s{spacing:g}-{source_key}"


def reference_set_id(task: str, spacing: float, source_key: str, population: str = "all") -> str:
    prefix = "sim" if population == "all" else f"sim-{population}"
    return f"{prefix}-refs-{task}-s{spacing:g}-{source_key}"


def _fmt(v) -> str:
    if v is None or v == "":
        return ""
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def write_query_set(tasks: dict, task_name: str, spacing: float, source_key: str, out_root: Path,
                    observations: dict, image_width_px: int, image_height_px: int,
                    session_meta: dict | None = None, population: str = "all",
                    anchors: dict | None = None) -> Path:
    """One spec-004 query set from the pose-only task rows.

    ``observations`` maps ``observation_id -> Observation`` so the condition columns (level, time of
    day, cloud state) travel with the query without the geometry module ever having read them.
    """
    rows = [q for q in tasks["queries"] if float(q["spacing_m"]) == float(spacing)]
    if not rows:
        raise SetError(f"no queries at spacing {spacing} m")
    rows = sorted(rows, key=lambda q: (q["session_id"], int(q["frame_index"])))
    qs_id = query_set_id(task_name, spacing, source_key, population)
    out = Path(out_root) / qs_id
    out.mkdir(parents=True, exist_ok=True)
    provenance_suffix = PROVENANCE_BY_KEY[source_key].split(":", 1)[-1]
    tau_pos = float(spacing) / 2.0
    anchors = anchors or {}

    frame_rows, gt_rows, side_rows = [], [], []
    for qi, q in enumerate(rows):
        oid = q["observation_id"]
        obs = observations.get(oid)
        if obs is None:
            raise SetError(f"query {oid!r} is not among the ingested observations")
        ts = float(qi)
        frame_rows.append({"frame_index": qi, "timestamp_s": ts, "image_path": f"images/{oid}.png"})
        gt_rows.append({"timestamp_s": ts, "east_m": _fmt(q["east_m"]), "north_m": _fmt(q["north_m"]),
                        "up_m": _fmt(q["up_m"]), "heading_deg": _fmt(q["yaw_deg"]),
                        "fix_quality": "simulator_exact", "valid": "true"})
        side_rows.append({
            "query_index": qi, "roll_deg": _fmt(q["roll_deg"]), "pitch_deg": _fmt(q["pitch_deg"]),
            "compass_prior_deg": _fmt(q["yaw_deg"]),
            "in_coverage": "true" if q["in_coverage"] else "false",
            "traversal_id": q["session_id"], "oracle_provenance": provenance_suffix,
            "condition_stage": q["condition_stage"],
            "condition_translation_m": _fmt(q["condition_translation_m"]),
            "condition_along_m": _fmt(q["condition_along_m"]),
            "condition_lateral_m": _fmt(q["condition_lateral_m"]),
            "condition_up_diff_m": _fmt(q["condition_up_diff_m"]),
            "condition_yaw_diff_deg": _fmt(q["condition_yaw_diff_deg"]),
            "condition_abs_yaw_diff_deg": _fmt(q["condition_abs_yaw_diff_deg"]),
            "condition_pitch_diff_deg": _fmt(q["condition_pitch_diff_deg"]),
            "condition_roll_diff_deg": _fmt(q["condition_roll_diff_deg"]),
            "condition_translation_bin": _fmt(q["condition_translation_bin"]),
            "condition_yaw_bin": _fmt(q["condition_yaw_bin"]),
            "condition_pitch_bin": _fmt(q["condition_pitch_bin"]),
            "condition_roll_bin": _fmt(q["condition_roll_bin"]),
            "condition_n_references": q["condition_n_references"],
            "condition_nearest_reference_id": q["nearest_reference_id"],
            "condition_reference_spacing_m": f"{float(spacing):g}",
            "condition_attainable": "true" if float(q["condition_translation_m"]) <= tau_pos else "false",
            "condition_level": obs.extra.get("condition_level", ""),
            "condition_time_of_day": obs.extra.get("condition_time_of_day", ""),
            "condition_hour": obs.extra.get("condition_hour", ""),
            "condition_clouds": obs.extra.get("condition_clouds", ""),
            "condition_session_id": q["session_id"], "condition_observation_id": oid,
            "condition_anchor_id": anchors.get(oid, ""),
        })

    with (out / "frames.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_index", "timestamp_s", "image_path"])
        w.writeheader(); w.writerows(frame_rows)
    with (out / "groundtruth.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp_s", "east_m", "north_m", "up_m", "heading_deg",
                                          "fix_quality", "valid"])
        w.writeheader(); w.writerows(gt_rows)
    with (out / "skyline_queries.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_index", "roll_deg", "pitch_deg", "compass_prior_deg",
                                          "in_coverage", "traversal_id", "oracle_provenance"]
                           + QUERY_CONDITION_FIELDS)
        w.writeheader(); w.writerows(side_rows)

    cam = (session_meta or {}).get("camera", {})
    dataset_json = {
        "schema_version": "1.0.0", "dataset_id": qs_id,
        "dataset_revision": f"sim-{tasks['manifest'].get('content_digest', 'unwritten')[:12]}",
        "source_type": SOURCE_TYPE, "evidence_tier": EVIDENCE_TIER, "evidence_caveat": CAVEAT,
        "frame_clock": "sim_query_index", "gt_clock": "sim_query_index",
        "clock_offset_s": 0.0,
        "clock_offset_source": "queries are individual rendered frames with exact poses; the clock is "
                               "the query index",
        "clock_drift_s_per_s": 0.0,
        "frame_convention": "ENU", "heading_convention": "compass_cw_from_north",
        "local_frame_origin": dict(SYNTHETIC_ORIGIN),
        "position_quality": {"class": "simulator_exact", "nominal_accuracy": 0.0,
                             "accuracy_source": "renderer camera transform; exact by construction",
                             "notes": "the ENU frame is local and declared synthetic; no geodetic anchor"},
        "heading_quality": {"class": "simulator_exact", "nominal_accuracy": 0.0,
                            "accuracy_source": "renderer camera transform; world-locked camera",
                            "notes": "known-heading assumption (PROT-SKY-001 §6) holds by construction"},
        "height_quality": {"class": "simulator_exact", "nominal_accuracy": 0.0,
                           "accuracy_source": "renderer camera transform relative to the declared datum",
                           "notes": ""},
        "metadata": {
            "flight_id": "", "trajectory_type": "scripted_simulator", "frame_rate_hz": None,
            "image_width": int(image_width_px), "image_height": int(image_height_px),
            "environment": (session_meta or {}).get("scene", {}).get("level_id", ""),
            "nominal_altitude_m": None, "nominal_speed_ms": None, "capture_date": "",
            "notes": json.dumps({
                "is_synthetic_frame": True, "units": "simulator metres",
                "task": task_name, "spacing_m": float(spacing),
                "tau_pos_m": tau_pos, "tau_near_m": float(spacing),
                "population": population, "sets_version": SETS_VERSION,
                "task_digest": tasks["manifest"].get("content_digest"),
                "grid_digest": _grid(tasks, spacing)["digest"], "images_virtual": True}),
        },
        "skyline": {
            "camera": {"fov_deg": cam.get("fov_deg"), "calibration_id": None,
                       "orientation": cam.get("heading_mode", "forward")},
            "compass_prior": {"mode": "exact", "noise_model": None, "seed": None,
                              "notes": "world-locked camera; the prior is the true heading"},
            "condition_factors": QUERY_CONDITION_FIELDS,
        },
    }
    (out / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8")
    return out


def _grid(tasks: dict, spacing: float) -> dict:
    key = f"{float(spacing):g}"
    if key in tasks["grids"]:
        return tasks["grids"][key]
    raise SetError(f"no grid at spacing {spacing}; have {sorted(tasks['grids'])}")


def write_reference_set(tasks: dict, task_name: str, spacing: float, source_key: str, source,
                        out_root: Path, population: str = "all") -> dict:
    """One spec-006 reference set from the source's own curves. A missing curve is a database hole."""
    grid = _grid(tasks, spacing)
    rs_id = reference_set_id(task_name, spacing, source_key, population)
    out = Path(out_root) / rs_id
    out.mkdir(parents=True, exist_ok=True)
    curves, rows, holes = {}, [], []
    for g in grid["references"]:
        rid = g["reference_id"]
        try:
            c = source.get(rid)
        except CurveError as exc:
            holes.append({"reference_id": rid, "reason": str(exc)})
            continue
        curves[rid] = c.row_per_col
        rows.append({"reference_id": rid, "east_m": _fmt(g["east_m"]), "north_m": _fmt(g["north_m"]),
                     "up_m": _fmt(g["up_m"]), "heading_deg": _fmt(g["yaw_deg"]),
                     "traversal_id": g["session_id"],
                     "extraction_mode": PROVENANCE_BY_KEY[source_key]})
    if not rows:
        raise SetError(f"{rs_id}: the source produced no reference curve at all "
                       f"({len(holes)} holes) — there is nothing to search")
    np.savez(out / "curves.npz", **curves)
    with (out / "references.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    east = [float(r["east_m"]) for r in rows]
    north = [float(r["north_m"]) for r in rows]
    manifest = {
        "schema_version": "1.1.0", "local_frame_origin": dict(SYNTHETIC_ORIGIN),
        "reference_source": {
            "kind": SOURCE_TYPE, "ref_set_id": rs_id, "dataset_id": "sim",
            "traversal_ids": sorted({r["traversal_id"] for r in rows}),
            "reference_spacing_m": float(spacing),
            "extraction_mode": PROVENANCE_BY_KEY[source_key],
            "n_references": len(rows), "n_grid": len(grid["references"]),
            "grid_digest": grid["digest"], "database_holes": holes, "n_holes": len(holes),
            "tau_pos_m": grid["tau_pos_m"], "tau_near_m": grid["tau_near_m"],
            "covered_area_km2": ((max(east) - min(east)) * (max(north) - min(north)) / 1e6),
            "total_bytes": int((out / "curves.npz").stat().st_size),
            "evidence_tier": EVIDENCE_TIER, "evidence_caveat": CAVEAT,
            "source": source.describe(), "population": population, "task": task_name,
            "sets_version": SETS_VERSION,
            "task_digest": tasks["manifest"].get("content_digest"),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8")
    return manifest


def assert_source_materialisations_identical(datasets_root: Path, task: str, spacing: float, keys,
                                             population: str = "all") -> None:
    """The sources' query sets differ only in ``dataset_id`` and the provenance column.

    Without this, a difference between two sources' results could come from the sets rather than from
    the curves, and the experiment's only controlled variable would not be controlled.
    """
    paths = [Path(datasets_root) / query_set_id(task, spacing, k, population) for k in keys]
    for name in ("frames.csv", "groundtruth.csv"):
        contents = {p.name: (p / name).read_bytes() for p in paths}
        if len(set(contents.values())) != 1:
            raise SetError(f"{task} s{spacing:g}: {name} differs between source materialisations")
    sidecars = []
    for p in paths:
        with (p / "skyline_queries.csv").open(encoding="utf-8", newline="") as f:
            sidecars.append([{k: v for k, v in r.items() if k != "oracle_provenance"}
                             for r in csv.DictReader(f)])
    if any(s != sidecars[0] for s in sidecars[1:]):
        raise SetError(f"{task} s{spacing:g}: sidecars differ beyond the provenance column")
    descs = []
    for p in paths:
        d = json.loads((p / "dataset.json").read_text(encoding="utf-8")); d.pop("dataset_id")
        descs.append(d)
    if any(d != descs[0] for d in descs[1:]):
        raise SetError(f"{task} s{spacing:g}: dataset.json differs beyond dataset_id")
