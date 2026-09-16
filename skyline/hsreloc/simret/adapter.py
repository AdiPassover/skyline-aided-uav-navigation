"""Simulator run directory → spec-006 observation session (PROT-SKY-001 §2, §7 a; Part A).

The **only** format-dependent module in the simulator path. Everything downstream — the geometry, the
curve sources, the matcher, the evaluator — consumes the frozen observation record and does not know
a simulator exists. There is no parallel observation model.

The run format it reads (delivered 2026-09-01; recorded in ``PROT-SKY-001`` §10)::

    <run>/settings.json            run_id, level, environment, north_camera, skyline
    <run>/observations.csv         one row per capture; north-camera pose columns + image/mask paths
    <run>/skyline/images/*.png     RGB frames
    <run>/skyline/sim/*_sky.png    binary simulator GT sky masks (white = sky)

The first real export batch (2026-09-02) places ``observations.csv`` under ``skyline/`` rather than
at the run root; both locations are accepted (root first) and the one found is recorded in the
session provenance. Row-level ``image_path`` / ``sim_sky_mask_path`` values resolve from the run
root in both layouts.

Nothing about ``256×256``, ``90°`` or those exact subdirectory names is assumed: the resolution, the
field of view and the intrinsics are read from each run's own ``settings.json``, and the image and
mask locations come from each row's own ``image_path`` / ``sim_sky_mask_path``.

Validation posture
------------------
``validate_run`` returns a **report**, not a boolean: every check that ran, what it measured, and
which checks failed. ``ingest_run`` refuses on any error and writes nothing — a half-ingested session
is worse than none. Nothing is repaired: a mask that is not binary, a quaternion that does not match
its Euler angles, a camera whose ``fx`` does not follow from its FOV, and a "world-locked North"
camera whose yaw is not North are all refusals with the measured numbers in the message.

Warnings are for things that are *not* wrong but must not be read as verified — chiefly the two
that matter early:

* every logged attitude being identity-like, so the export's quaternion convention cannot yet be
  discriminated (the normal state of a level, north-locked Stage-0/1 run); and
* pitch and roll signs still resting on a declaration rather than on the ``PROT-SKY-001`` §4 R0-S
  renders.

Both ride into ``session.json`` so a later record cannot claim a verification that never happened.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from hsreloc.observation import Observation, write_session
from hsreloc.simret import simgt
from hsreloc.simret.conventions import (SimConventions, identify_quaternion_convention,
                                        normalise_0_360, quaternion_norm_error,
                                        ue_attitude_to_enu, ue_position_to_enu)

ADAPTER_VERSION = "1.1.0"      # 1.1.0: the `view` option (north | west); the default is unchanged

SETTINGS_FILE = "settings.json"
OBSERVATIONS_FILE = "observations.csv"
#: Accepted locations of the observations table, tried in order, relative to the run root.
OBSERVATIONS_LOCATIONS = ("observations.csv", "skyline/observations.csv")

#: Columns the SKY experiment cannot run without. Nadir columns are preserved but never required.
REQUIRED_OBS_COLUMNS = (
    "observation_id", "sim_time_s",
    "north_ue_x_cm", "north_ue_y_cm", "north_ue_z_cm",
    "north_ue_yaw_deg", "north_ue_pitch_deg", "north_ue_roll_deg",
    "north_ue_quat_w", "north_ue_quat_x", "north_ue_quat_y", "north_ue_quat_z",
    "image_path", "sim_sky_mask_path",
)

#: The world-locked skyline views a run may carry (2026-09-06 dual-direction batch: North + West at
#: the same callback). Each view names its pose-column prefix, its image / mask columns, its camera
#: block and the compass direction its ``heading_mode`` claims. ``north`` is the delivered 2026-09-01
#: format, unchanged; every check and the ingest read the *selected* view's columns and nothing else.
VIEWS = {
    "north": {"prefix": "north_ue_", "image": "image_path", "mask": "sim_sky_mask_path",
              "camera": "north_camera", "heading_mode": "world_locked_north", "compass_deg": 0.0,
              "name": "North"},
    "west": {"prefix": "west_ue_", "image": "west_image_path", "mask": "west_sim_sky_mask_path",
             "camera": "west_camera", "heading_mode": "world_locked_west", "compass_deg": 270.0,
             "name": "West"},
}
DEFAULT_VIEW = "north"
#: Compass direction a world-locked heading mode claims (checked against the logged yaw).
WORLD_LOCKED_COMPASS = {"world_locked_north": ("North", 0.0), "world_locked_east": ("East", 90.0),
                        "world_locked_south": ("South", 180.0), "world_locked_west": ("West", 270.0)}


def required_columns(view: str = DEFAULT_VIEW) -> tuple:
    """The observation columns one view cannot be ingested without."""
    v = VIEWS[view]
    p = v["prefix"]
    return ("observation_id", "sim_time_s",
            f"{p}x_cm", f"{p}y_cm", f"{p}z_cm", f"{p}yaw_deg", f"{p}pitch_deg", f"{p}roll_deg",
            f"{p}quat_w", f"{p}quat_x", f"{p}quat_y", f"{p}quat_z", v["image"], v["mask"])


assert required_columns("north") == REQUIRED_OBS_COLUMNS
REQUIRED_CAMERA_KEYS = ("width_px", "height_px", "projection", "horizontal_fov_deg",
                        "vertical_fov_deg", "intrinsics_px", "distortion", "heading_mode")
REQUIRED_INTRINSICS = ("fx", "fy", "cx", "cy")

HEADING_WORLD_LOCKED_NORTH = "world_locked_north"
HEADING_BODY_FIXED = "body_fixed"

IMAGE_MODE_COPY = "copy"
IMAGE_MODE_REFERENCE = "reference"

EVIDENCE_TIER = "T2"
SOURCE_TYPE = "ue5_simulator"
EVIDENCE_CAVEAT = (
    "UE5 simulator render: real 3D geometry, synthetic appearance and lighting; poses are exact by "
    "construction (gt_source=sim_exact) in a declared local ENU frame with no geodetic anchor. Skyline "
    "curves are either the simulator's own sky mask (oracle:sim_exact), an automatic extractor, or a "
    "SILVER semantic segmentor — never a real-sensor measurement. No UAV-flight claim and no onboard "
    "or real-time claim follows from any number produced on this data."
)


class SimRunError(Exception):
    """A simulator run cannot be read or ingested as declared."""


# --------------------------------------------------------------------------------------------------
# validation report
# --------------------------------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """What was checked, what it measured, and what failed. Never a bare boolean."""

    checks: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def record(self, name: str, ok: bool, detail=None, error: Optional[str] = None,
               warning: Optional[str] = None) -> None:
        self.checks[name] = {"ok": bool(ok), "detail": detail}
        if not ok and error:
            self.errors.append(f"[{name}] {error}")
        if warning:
            self.warnings.append(f"[{name}] {warning}")

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self, where: str) -> None:
        if self.errors:
            joined = "\n  - ".join(self.errors)
            raise SimRunError(
                f"{where}: {len(self.errors)} validation error(s); nothing was written.\n  - {joined}")

    def as_dict(self) -> dict:
        return {"adapter_version": ADAPTER_VERSION, "ok": self.ok, "checks": self.checks,
                "errors": list(self.errors), "warnings": list(self.warnings)}


# --------------------------------------------------------------------------------------------------
# reading the run
# --------------------------------------------------------------------------------------------------

def png_size(path: Path) -> tuple:
    """(width, height) from a PNG's IHDR — no decode, so a whole run's frames cost one seek each."""
    with Path(path).open("rb") as f:
        head = f.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise SimRunError(f"{path}: not a PNG (the run format declares lossless PNG frames)")
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


@dataclass(frozen=True)
class SimRun:
    run_dir: Path
    settings: dict
    rows: list
    columns: list
    observations_file: str = OBSERVATIONS_FILE      # which accepted location the table was read from
    view: str = DEFAULT_VIEW                          # which world-locked skyline view is being read

    @property
    def camera(self) -> dict:
        return self.settings.get(self.camera_key) or {}       # absence is reported by check_settings

    @property
    def camera_key(self) -> str:
        return VIEWS[self.view]["camera"]

    def col(self, name: str) -> str:
        """The selected view's pose column for ``name`` (``yaw_deg``, ``quat_w``, ...)."""
        return VIEWS[self.view]["prefix"] + name

    @property
    def image_col(self) -> str:
        return VIEWS[self.view]["image"]

    @property
    def mask_col(self) -> str:
        return VIEWS[self.view]["mask"]

    @property
    def required_columns(self) -> tuple:
        return required_columns(self.view)

    @property
    def run_id(self) -> str:
        return self.settings["run_id"]


def read_run(run_dir: Path | str, view: str = DEFAULT_VIEW) -> SimRun:
    """Read ``settings.json`` and ``observations.csv``; structural checks only, no semantics.

    ``view`` selects which world-locked skyline camera the run is read for (``north`` — the default
    and the only view before 2026-09-06 — or ``west``). A run carrying both is read twice, once per
    view, into two sessions; nothing pairs them here.
    """
    if view not in VIEWS:
        raise SimRunError(f"unknown skyline view {view!r}; expected one of {sorted(VIEWS)}")
    run_dir = Path(run_dir)
    settings_path = run_dir / SETTINGS_FILE
    if not settings_path.exists():
        raise SimRunError(f"{run_dir} is not a simulator run: no {SETTINGS_FILE}")
    obs_rel = next((rel for rel in OBSERVATIONS_LOCATIONS if (run_dir / rel).exists()), None)
    if obs_rel is None:
        raise SimRunError(f"{run_dir} is not a simulator run: no observations table at any accepted "
                          f"location {list(OBSERVATIONS_LOCATIONS)}")
    obs_path = run_dir / obs_rel
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SimRunError(f"{settings_path}: not valid JSON — {exc}") from exc
    with obs_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    if not rows:
        raise SimRunError(f"{obs_path}: no observation rows")
    return SimRun(run_dir=run_dir, settings=settings, rows=rows, columns=columns,
                  observations_file=obs_rel, view=view)


def _resolve(run_dir: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (run_dir / p)


def _float(row: dict, key: str, where: str) -> float:
    raw = row.get(key, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise SimRunError(f"{where}: column {key!r} is not a number ({raw!r})") from None


# --------------------------------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------------------------------

def check_settings(run: SimRun, report: ValidationReport) -> None:
    s = run.settings
    missing = [k for k in ("run_id", "level", "environment", run.camera_key) if k not in s]
    report.record("settings_keys", not missing, {"present": sorted(s)},
                  error=f"settings.json is missing {missing}" if missing else None)
    if missing:
        return
    cam = s[run.camera_key]
    cam_missing = [k for k in REQUIRED_CAMERA_KEYS if k not in cam]
    intr = cam.get("intrinsics_px", {}) or {}
    intr_missing = [k for k in REQUIRED_INTRINSICS if k not in intr]
    report.record("camera_keys", not (cam_missing or intr_missing),
                  {"missing": cam_missing, "missing_intrinsics": intr_missing},
                  error=(f"{run.camera_key} is missing {cam_missing} / intrinsics_px missing {intr_missing}"
                         if (cam_missing or intr_missing) else None))
    env = s.get("environment", {}) or {}
    env_missing = [k for k in ("time_of_day", "clouds") if k not in env]
    report.record("environment_keys", not env_missing, {"environment": env},
                  error=(f"environment is missing {env_missing} — the condition a session carries must "
                         f"be explicit, not inferred" if env_missing else None))


def check_calibration(run: SimRun, report: ValidationReport, tol_px: float = 1.0,
                      tol_rel: float = 0.01) -> None:
    """``fx`` must follow from the horizontal FOV, ``fy`` from the vertical, the principal point from
    the image centre — otherwise the column↔azimuth mapping every angular representation needs is not
    the one the settings advertise."""
    cam = run.camera
    if any(k not in cam for k in REQUIRED_CAMERA_KEYS) or not isinstance(cam.get("intrinsics_px"), dict):
        return                                     # already reported by check_settings
    w, h = int(cam["width_px"]), int(cam["height_px"])
    intr = cam["intrinsics_px"]
    detail = {"width_px": w, "height_px": h}
    problems = []

    if cam.get("projection") != "perspective":
        problems.append(f"projection is {cam.get('projection')!r}, not 'perspective' — the pinhole "
                        f"column↔azimuth mapping does not apply")
    if str(cam.get("distortion", "none")).lower() not in ("none", "null", ""):
        problems.append(f"distortion is {cam.get('distortion')!r}; the matcher's angular seam assumes "
                        f"a distortion-free camera")
    if w <= 0 or h <= 0:
        problems.append(f"non-positive resolution {w}x{h}")
    for name, fov_key, size, f_key in (("fx", "horizontal_fov_deg", w, "fx"),
                                       ("fy", "vertical_fov_deg", h, "fy")):
        fov = float(cam[fov_key])
        if not 0.0 < fov < 180.0:
            problems.append(f"{fov_key} = {fov} is outside (0, 180)")
            continue
        expected = (size / 2.0) / math.tan(math.radians(fov) / 2.0)
        got = float(intr[f_key])
        detail[f"{name}_expected"] = expected
        detail[f"{name}_declared"] = got
        if expected <= 0 or abs(got - expected) > max(tol_px, tol_rel * expected):
            problems.append(f"{name} = {got:.4f} but {fov_key} = {fov} implies {expected:.4f} "
                            f"(tolerance {max(tol_px, tol_rel * expected):.4f} px)")
    for name, expected in (("cx", w / 2.0), ("cy", h / 2.0)):
        got = float(intr[name])
        detail[f"{name}_expected"] = expected
        detail[f"{name}_declared"] = got
        if abs(got - expected) > tol_px:
            problems.append(f"{name} = {got} is {abs(got - expected):.3f} px from the image centre "
                            f"{expected} — an off-centre principal point is legal but must be declared "
                            f"deliberately, because the azimuth of column 0 depends on it")
    report.record("camera_calibration", not problems, detail,
                  error="; ".join(problems) if problems else None)


def check_columns(run: SimRun, report: ValidationReport) -> None:
    missing = [c for c in run.required_columns if c not in run.columns]
    report.record("observation_columns", not missing,
                  {"n_columns": len(run.columns), "missing": missing, "view": run.view,
                   "nadir_columns": [c for c in run.columns if c.startswith("nadir")]},
                  error=(f"observations.csv is missing required {run.view}-camera columns {missing}; "
                         f"present: {run.columns}" if missing else None))


def check_identity_and_clock(run: SimRun, report: ValidationReport) -> None:
    ids = [r.get("observation_id", "") for r in run.rows]
    dupes = sorted({i for i in ids if ids.count(i) > 1}) if len(set(ids)) != len(ids) else []
    blank = sum(1 for i in ids if not i)
    report.record("observation_ids", not dupes and not blank,
                  {"n_rows": len(ids), "n_unique": len(set(ids))},
                  error=(f"{len(dupes)} duplicated and {blank} blank observation_id(s): {dupes[:5]}"
                         if (dupes or blank) else None))
    if "sim_time_s" not in run.columns:
        return
    try:
        times = [float(r["sim_time_s"]) for r in run.rows]
    except (TypeError, ValueError):
        report.record("sim_clock", False, None, error="sim_time_s contains a non-numeric value")
        return
    non_monotone = sum(1 for a, b in zip(times[:-1], times[1:]) if b <= a)
    report.record("sim_clock", all(math.isfinite(t) for t in times),
                  {"t_first": times[0], "t_last": times[-1], "n_non_monotone": non_monotone},
                  error=None if all(math.isfinite(t) for t in times) else "sim_time_s has a non-finite value",
                  warning=(f"{non_monotone} row(s) do not advance sim_time_s; rows are ingested in "
                           f"file order and frame_index follows that order"
                           if non_monotone else None))


def check_attitude(run: SimRun, report: ValidationReport, conventions: SimConventions,
                   quat_norm_tol: float = 1e-3, quat_angle_tol_deg: float = 0.5) -> None:
    """Quaternion finiteness, normalisation, and which convention the export is actually using."""
    if any(c not in run.columns for c in run.required_columns):
        return
    samples, worst_norm, bad = [], 0.0, []
    for i, r in enumerate(run.rows):
        where = f"row {i} ({r.get('observation_id')})"
        q = np.array([_float(r, run.col(f"quat_{k}"), where) for k in ("w", "x", "y", "z")])
        err = quaternion_norm_error(q)
        worst_norm = max(worst_norm, err) if math.isfinite(err) else float("inf")
        if not math.isfinite(err) or err > quat_norm_tol:
            bad.append({"observation_id": r.get("observation_id"), "quat": q.tolist(),
                        "norm_error": None if not math.isfinite(err) else err})
        samples.append((_float(r, run.col("yaw_deg"), where), _float(r, run.col("pitch_deg"), where),
                        _float(r, run.col("roll_deg"), where), q))
    report.record("quaternion_norm", not bad,
                  {"worst_norm_error": worst_norm, "tolerance": quat_norm_tol, "n_bad": len(bad),
                   "examples": bad[:5]},
                  error=(f"{len(bad)} quaternion(s) are non-finite or off unit length by more than "
                         f"{quat_norm_tol} (worst {worst_norm}); e.g. {bad[:3]}" if bad else None))

    ident = identify_quaternion_convention(samples, tolerance_deg=quat_angle_tol_deg)
    declared = conventions.quaternion_convention
    if declared is not None:
        ok = ident["residual_deg"].get(declared, float("inf")) <= quat_angle_tol_deg
        report.record("quaternion_convention", ok, ident,
                      error=(f"the run declares quaternion convention {declared!r} but the logged "
                             f"quaternions disagree with the logged Euler angles under it by up to "
                             f"{ident['residual_deg'].get(declared)}° (tolerance {quat_angle_tol_deg}°). "
                             f"Residuals per candidate: {ident['residual_deg']}" if not ok else None),
                      warning=(f"attitudes in this run cannot discriminate the candidate quaternion "
                               f"conventions (all fit); the declaration is carried, not confirmed"
                               if ok and not ident["discriminating"] else None))
        return
    if not ident["fits"]:
        report.record("quaternion_convention", False, ident,
                      error=(f"the logged quaternions match the logged Euler angles under NONE of the "
                             f"candidate conventions {list(ident['residual_deg'])} within "
                             f"{quat_angle_tol_deg}°. Worst-case residual per candidate: "
                             f"{ident['residual_deg']}. Either the Euler angles or the quaternions are "
                             f"in a convention this adapter does not know; the simulator owner must say "
                             f"which, and it must not be guessed."))
        return
    report.record("quaternion_convention", True, ident,
                  warning=(f"every logged attitude is compatible with all of {ident['fits']} — this run "
                           f"cannot establish the export's quaternion convention (expected for a level, "
                           f"north-locked run where yaw = pitch = roll = 0). A run with non-zero pitch, "
                           f"roll and yaw (PROT-SKY-001 §4 R0-S2/S3) is what settles it."
                           if not ident["discriminating"] else None))


def check_heading_lock(run: SimRun, report: ValidationReport, conventions: SimConventions,
                       tolerance_deg: float = 0.5) -> None:
    """``heading_mode: world_locked_north`` is a claim about the pose columns, so check it there.

    This is also the empirical test of the axis declaration: if the camera really faces North and the
    logged compass yaw is 90 rather than 0, the UE project's ``+X = North`` declaration is not what was
    rendered, and no downstream number would be trustworthy.
    """
    yaw_col = run.col("yaw_deg")
    if yaw_col not in run.columns or "heading_mode" not in run.camera:
        return
    mode = run.camera["heading_mode"]
    yaws = np.array([normalise_0_360(_float(r, yaw_col, f"row {i}"))
                     for i, r in enumerate(run.rows)])
    centred = (yaws + 180.0) % 360.0 - 180.0            # to [-180, 180) so 359.9 reads as -0.1
    detail = {"heading_mode": mode, "view": run.view, "compass_yaw_mean_deg": float(centred.mean()),
              "compass_yaw_min_deg": float(centred.min()), "compass_yaw_max_deg": float(centred.max()),
              "compass_yaw_span_deg": float(centred.max() - centred.min()),
              "tolerance_deg": float(tolerance_deg)}
    if mode not in WORLD_LOCKED_COMPASS:
        report.record("heading_lock", True, detail,
                      warning=(f"heading_mode is {mode!r}, not {HEADING_WORLD_LOCKED_NORTH!r}: the "
                               f"known-heading isolation of PROT-SKY-001 §6 does not hold for this run "
                               f"and stage assignment will see a varying yaw"))
        return
    direction, expected = WORLD_LOCKED_COMPASS[mode]
    # the yaw the camera's mode claims, measured as a signed offset from that direction
    rel = ((yaws - expected + 180.0) % 360.0) - 180.0
    detail.update({"expected_compass_deg": expected, "offset_from_expected_mean_deg": float(rel.mean()),
                   "offset_from_expected_span_deg": float(rel.max() - rel.min())})
    problems = []
    if detail["offset_from_expected_span_deg"] > tolerance_deg:
        problems.append(f"compass yaw varies by {detail['offset_from_expected_span_deg']:.4f}° across the run")
    if VIEWS[run.view]["heading_mode"] != mode:
        problems.append(f"the {run.view} view's camera block declares heading_mode {mode!r}, not "
                        f"{VIEWS[run.view]['heading_mode']!r} — the view and its camera disagree")
    if abs(detail["offset_from_expected_mean_deg"]) > tolerance_deg:
        mean_c = float(normalise_0_360(float(np.degrees(np.arctan2(np.sin(np.radians(yaws)).mean(),
                                                                    np.cos(np.radians(yaws)).mean())))))
        near = min((0.0, 90.0, 180.0, 270.0), key=lambda v: abs(((mean_c - v + 180) % 360) - 180))
        problems.append(
            f"mean compass yaw is {detail['compass_yaw_mean_deg']:.4f}°, not {expected:.0f}° — a camera declared "
            f"world-locked to {direction} is not pointing {direction} under the {conventions.ue_world_axes!r} "
            f"declaration (nearest cardinal {near:.0f}°). Either the camera is locked to a different "
            f"direction, or the UE project does not declare world +X = North (PROT-001 §2.2). Do not "
            f"proceed until the simulator owner says which")
    report.record("heading_lock", not problems, detail, error="; ".join(problems) if problems else None)


def check_assets(run: SimRun, report: ValidationReport, mask_binarity_tol: float,
                 max_reported: int = 5) -> None:
    """Every RGB and every mask exists, and the three declared resolutions agree."""
    if any(c not in run.columns for c in (run.image_col, run.mask_col)):
        return
    cam = run.camera
    want = (int(cam["width_px"]), int(cam["height_px"])) if {"width_px", "height_px"} <= set(cam) else None
    missing_img, missing_mask, wrong_img, wrong_mask = [], [], [], []
    for r in run.rows:
        oid = r.get("observation_id")
        img = _resolve(run.run_dir, r[run.image_col])
        msk = _resolve(run.run_dir, r[run.mask_col])
        if not img.exists():
            missing_img.append(f"{oid}: {r[run.image_col]}")
            continue
        if not msk.exists():
            missing_mask.append(f"{oid}: {r[run.mask_col]}")
            continue
        size_i, size_m = png_size(img), png_size(msk)
        if want and size_i != want:
            wrong_img.append(f"{oid}: image is {size_i[0]}x{size_i[1]}, settings say {want[0]}x{want[1]}")
        if size_m != size_i:
            wrong_mask.append(f"{oid}: mask is {size_m[0]}x{size_m[1]} but its image is {size_i[0]}x{size_i[1]}")
    report.record("images_present", not missing_img, {"n_missing": len(missing_img)},
                  error=(f"{len(missing_img)} referenced RGB image(s) do not exist: "
                         f"{missing_img[:max_reported]}" if missing_img else None))
    report.record("masks_present", not missing_mask, {"n_missing": len(missing_mask)},
                  error=(f"{len(missing_mask)} referenced simulator mask(s) do not exist: "
                         f"{missing_mask[:max_reported]}" if missing_mask else None))
    report.record("image_resolution", not wrong_img, {"n_wrong": len(wrong_img), "declared": want},
                  error=(f"{len(wrong_img)} image(s) do not match the resolution in settings.json: "
                         f"{wrong_img[:max_reported]}" if wrong_img else None))
    report.record("mask_resolution", not wrong_mask, {"n_wrong": len(wrong_mask)},
                  error=(f"{len(wrong_mask)} mask(s) do not match their own RGB frame: "
                         f"{wrong_mask[:max_reported]}" if wrong_mask else None))
    report.checks.setdefault("mask_binarity", {"ok": True, "detail": {"tolerance": mask_binarity_tol,
                                                                     "measured_at": "ingest"}})


# --------------------------------------------------------------------------------------------------
# validate + ingest
# --------------------------------------------------------------------------------------------------

def validate_run(run: SimRun, conventions: SimConventions, options: Optional[dict] = None) -> ValidationReport:
    """Every structural, calibration, convention and asset check. Writes nothing."""
    options = dict(options or {})
    conventions.validate()
    report = ValidationReport()
    check_settings(run, report)
    check_calibration(run, report,
                      tol_px=float(options.get("calibration_tolerance_px", 1.0)),
                      tol_rel=float(options.get("calibration_tolerance_rel", 0.01)))
    check_columns(run, report)
    check_identity_and_clock(run, report)
    check_attitude(run, report, conventions,
                   quat_norm_tol=float(options.get("quaternion_norm_tolerance", 1e-3)),
                   quat_angle_tol_deg=float(options.get("quaternion_angle_tolerance_deg", 0.5)))
    check_heading_lock(run, report, conventions,
                       tolerance_deg=float(options.get("heading_lock_tolerance_deg", 0.5)))
    check_assets(run, report,
                 mask_binarity_tol=float(options.get("mask_binarity_tolerance",
                                                     simgt.DEFAULT_MAX_INTERMEDIATE_FRAC)))
    if not conventions.attitude_verified:
        report.warnings.append(
            "[conventions] pitch and roll signs are DECLARED but not verified: no R0-S convention-check "
            "recording is named in conventions.verified_by (PROT-SKY-001 §4). Stage-3 results must "
            "carry that caveat; Stage 0/1 with pitch = roll = 0 is unaffected.")
    return report


def _session_meta(run: SimRun, session_id: str, conventions: SimConventions, report: ValidationReport,
                  extra_meta: Optional[dict] = None) -> dict:
    cam = run.camera
    intr = cam.get("intrinsics_px", {})
    env = run.settings.get("environment", {})
    meta = {
        "session_id": session_id,
        "platform": "uav_sim",
        "frame_convention": "ENU",
        "heading_convention": "compass_cw_from_north",
        "evidence_tier": EVIDENCE_TIER,
        "source_type": SOURCE_TYPE,
        "evidence_caveat": EVIDENCE_CAVEAT,
        "gt_source": "sim_exact",
        "auto_extraction_method": None,
        "local_frame_origin": None,
        "adapter": {"module": "hsreloc.simret.adapter", "version": ADAPTER_VERSION,
                    "gt_convert_version": simgt.GT_CONVERT_VERSION},
        "simulator_run": {"run_id": run.run_id, "level": run.settings.get("level"),
                          "run_dir": str(run.run_dir), "n_rows": len(run.rows),
                          "observations_file": run.observations_file,
                          "skyline_settings": run.settings.get("skyline"),
                          "raw_settings": run.settings},
        "skyline_view": run.view,
        "camera": {"orientation": "forward", "view": run.view,
                   "world_direction_deg": VIEWS[run.view]["compass_deg"],
                   "heading_mode": cam.get("heading_mode"),
                   "fov_deg": cam.get("horizontal_fov_deg"), "vfov_deg": cam.get("vertical_fov_deg"),
                   "intrinsics": {k: intr.get(k) for k in REQUIRED_INTRINSICS},
                   "distortion": cam.get("distortion"), "projection": cam.get("projection"),
                   "resolution_px": [cam.get("width_px"), cam.get("height_px")]},
        "condition": {"time_of_day": env.get("time_of_day"), "hour": env.get("hour"),
                      "clouds": env.get("clouds"), "time_speed": env.get("time_speed")},
        "scene": {"level_id": run.settings.get("level")},
        "conventions": conventions.as_dict(),
        "validation": report.as_dict(),
    }
    meta.update(extra_meta or {})
    return meta


def ingest_run(run_dir: Path | str, out_root: Path | str, session_id: Optional[str] = None,
               conventions: Optional[SimConventions] = None, options: Optional[dict] = None) -> dict:
    """Validate a simulator run and write one spec-006 observation session. Refuses on any error.

    ``options["view"]`` selects the skyline view (``north`` by default; ``west`` for the
    dual-direction batch). Only that view's pose columns, image and mask are read; the session
    records the view in ``session.json`` (``skyline_view``) and on every observation (``sim_view``).

    Produces, under ``<out_root>/<session_id>/``:

    * ``session.json`` / ``observations.csv`` — the frozen observation record, with every raw UE value
      preserved as a ``sim_ue_*`` extra (PROT-001 §3.3: a convention error must be fixable by
      re-conversion, never by re-rendering);
    * ``sim/<observation_id>_skyline_gt.csv`` — the full-fidelity GT curve, invalid columns included;
    * ``skylines_oracle/<observation_id>.csv`` — the matcher-seam curve, **only** where every column
      has sky above it;
    * ``images/<observation_id>.png`` — copied by default, or left in the run directory and referenced
      when ``image_mode`` is ``"reference"``.
    """
    options = dict(options or {})
    conventions = conventions or SimConventions()
    view = str(options.get("view", DEFAULT_VIEW))
    run = read_run(run_dir, view=view)
    session_id = session_id or run.run_id
    report = validate_run(run, conventions, options)
    report.raise_if_errors(f"simulator run {run.run_dir}")

    out_root = Path(out_root)
    session_dir = out_root / session_id
    if session_dir.exists() and not options.get("overwrite", False):
        raise SimRunError(f"{session_dir} already exists — refusing to overwrite an ingested session "
                          f"(pass overwrite to replace it deliberately)")
    image_mode = options.get("image_mode", IMAGE_MODE_COPY)
    if image_mode not in (IMAGE_MODE_COPY, IMAGE_MODE_REFERENCE):
        raise SimRunError(f"image_mode must be {IMAGE_MODE_COPY!r} or {IMAGE_MODE_REFERENCE!r}")
    min_valid_frac = float(options.get("gt_min_valid_frac", 1.0))
    binarity_tol = float(options.get("mask_binarity_tolerance", simgt.DEFAULT_MAX_INTERMEDIATE_FRAC))
    preserve_nadir = bool(options.get("preserve_nadir", True))

    import cv2                                       # a project dependency; imported at use, as elsewhere

    cam = run.camera
    width, height = int(cam["width_px"]), int(cam["height_px"])
    env = run.settings.get("environment", {})
    observations, gt_stats, seam_written, holes = [], [], [], []

    for frame_index, r in enumerate(run.rows):
        raw_id = r["observation_id"]
        oid = raw_id if options.get("observation_id_mode") == "raw" else f"{session_id}__{raw_id}"
        where = f"{run.run_dir}:{raw_id}"

        x_cm = _float(r, run.col("x_cm"), where)
        y_cm = _float(r, run.col("y_cm"), where)
        z_cm = _float(r, run.col("z_cm"), where)
        east, north, up = ue_position_to_enu(x_cm, y_cm, z_cm, conventions)
        yaw, pitch, roll = ue_attitude_to_enu(_float(r, run.col("yaw_deg"), where),
                                              _float(r, run.col("pitch_deg"), where),
                                              _float(r, run.col("roll_deg"), where), conventions)

        mask_path = _resolve(run.run_dir, r[run.mask_col])
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if raw_mask is None:
            raise SimRunError(f"{oid}: mask at {mask_path} could not be decoded")
        try:
            sky, binarity_report = simgt.sky_mask_from_image(raw_mask, binarity_tol)
        except simgt.SimGtError as exc:
            raise SimRunError(f"{oid}: {exc}") from exc
        gt = simgt.sim_gt_curve(sky, min_valid_frac=min_valid_frac)
        gt["binarity"] = binarity_report
        simgt.write_gt_curve(session_dir / "sim" / f"{oid}_skyline_gt.csv", gt)
        oracle_rel = None
        if np.asarray(gt["valid"], dtype=bool).all():
            simgt.write_seam_curve(session_dir / "skylines_oracle" / f"{oid}.csv", gt)
            oracle_rel = f"skylines_oracle/{oid}.csv"
            seam_written.append(oid)
        else:
            holes.append({"observation_id": oid, "status": gt["status"],
                          "valid_frac": gt["valid_frac"]})
        gt_stats.append({"observation_id": oid, "status": gt["status"], "valid_frac": gt["valid_frac"],
                         "sky_frac_raw": gt["sky_frac_raw"],
                         "n_enclosed_sky_px": gt["n_enclosed_sky_px"],
                         "intermediate_frac": binarity_report["intermediate_frac"]})

        src_image = _resolve(run.run_dir, r[run.image_col])
        if image_mode == IMAGE_MODE_COPY:
            dst = session_dir / "images" / f"{oid}{src_image.suffix}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_image, dst)
            image_path = f"images/{oid}{src_image.suffix}"
        else:
            image_path = f"images/{oid}{src_image.suffix}"      # virtual; the real file stays put

        extra = {
            # raw UE values, preserved verbatim (PROT-001 §3.3)
            "sim_ue_x_cm": x_cm, "sim_ue_y_cm": y_cm, "sim_ue_z_cm": z_cm,
            "sim_ue_yaw_deg": r[run.col("yaw_deg")], "sim_ue_pitch_deg": r[run.col("pitch_deg")],
            "sim_ue_roll_deg": r[run.col("roll_deg")],
            "sim_ue_quat_w": r[run.col("quat_w")], "sim_ue_quat_x": r[run.col("quat_x")],
            "sim_ue_quat_y": r[run.col("quat_y")], "sim_ue_quat_z": r[run.col("quat_z")],
            "sim_view": run.view,
            "sim_view_compass_deg": VIEWS[run.view]["compass_deg"],
            "sim_observation_id_raw": raw_id,
            "sim_run_id": run.run_id,
            "sim_time_s": r["sim_time_s"],
            "sim_image_source": r[run.image_col],
            "sim_sky_mask_path": r[run.mask_col],
            "sim_skyline_gt_path": f"sim/{oid}_skyline_gt.csv",
            "sim_gt_status": gt["status"],
            "sim_gt_valid_frac": f"{gt['valid_frac']:.6f}",
            "sky_fraction": f"{gt['sky_frac_raw']:.6f}",
            "sim_heading_mode": cam.get("heading_mode"),
            "sim_level": run.settings.get("level"),
            # condition axes the unchanged evaluator can slice by
            "condition_level": run.settings.get("level"),
            "condition_time_of_day": env.get("time_of_day"),
            "condition_hour": env.get("hour"),
            "condition_clouds": env.get("clouds"),
        }
        if preserve_nadir:
            for c in run.columns:
                if c.startswith("nadir"):
                    extra[f"sim_{c}"] = r.get(c, "")

        obs = Observation(
            observation_id=oid, session_id=session_id, frame_index=frame_index,
            timestamp_s=_float(r, "sim_time_s", where), image_path=image_path,
            image_width_px=width, image_height_px=height, gt_source="sim_exact",
            pos_east_m=east, pos_north_m=north, up_m=up,
            yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
            fov_deg=float(cam["horizontal_fov_deg"]),
            gt_pos_sigma_m=0.0, gt_yaw_sigma_deg=0.0,
            oracle_skyline_path=oracle_rel,
            oracle_provenance=simgt.SIM_EXACT_PROVENANCE if oracle_rel else None,
            extra=extra,
        )
        obs.validate()
        observations.append(obs)

    summary = {
        "session_id": session_id, "session_dir": str(session_dir), "run_id": run.run_id,
        "view": run.view, "n_observations": len(observations), "image_mode": image_mode,
        "n_seam_curves": len(seam_written), "n_database_holes": len(holes), "holes": holes,
        "gt_status_counts": _count(gt_stats, "status"),
        "worst_mask_intermediate_frac": max(g["intermediate_frac"] for g in gt_stats),
        "sky_fraction": {"min": min(g["sky_frac_raw"] for g in gt_stats),
                         "median": float(np.median([g["sky_frac_raw"] for g in gt_stats])),
                         "max": max(g["sky_frac_raw"] for g in gt_stats)},
        "warnings": list(report.warnings),
    }
    meta = _session_meta(run, session_id, conventions, report, {"ingest_summary": summary})
    write_session(session_dir, meta, observations)
    (session_dir / "sim" / "gt_index.csv").parent.mkdir(parents=True, exist_ok=True)
    with (session_dir / "sim" / "gt_index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gt_stats[0].keys()))
        w.writeheader()
        w.writerows(gt_stats)
    return summary


def _count(rows: list, key: str) -> dict:
    out: dict = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return dict(sorted(out.items()))
