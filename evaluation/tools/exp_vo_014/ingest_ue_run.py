"""`EXP-VO-014`: the minimal deterministic adapter from a UE5 capture folder to the dataset contract.

A UE run folder (`PROT-001` shape) looks like this:

    Run_<stamp>/
      settings.json          # world frame, intrinsics, capture policy, height datums
      vo/frames.csv          # one row per IMAGE (10 Hz): pose + height channels
      vo/groundtruth.csv     # one row per SIM TICK (60 Hz): same columns, no image
      vo/images/frame_NNNNNN.png
      skyline/...            # not read here

and `contracts/dataset.md` wants this:

    datasets/<id>/
      dataset.json  frames.csv  groundtruth.csv  images/

This module writes that, plus two sidecars the contract does not define but this experiment needs:
`terrain.csv` (the `bvo-*` convention from `EXP-VO-012`, carrying the height channels per frame) and
`ue_body.csv` (the airframe pose, preserved but deliberately quarantined -- see below).

**It converts nothing it can avoid converting.** The simulator already emits `east_m`/`north_m`/`up_m`
in ENU metres, so the adapter's job is to *check* that they are what `settings.json` says they are,
not to recompute them. Everything it does compute is one subtraction: the contract's `up_m` is height
above the take-off ground plane (the `bvo-*` convention, so `up_m[0] == h0`), which is
`cam_ue_z_cm/100 - terrain_elevation_m[0]`.

**Three things are quarantined rather than dropped.**

`ue_body.csv`  The airframe pose. `EXP-VO-013` R6 measured what conflating camera yaw with airframe
               yaw costs on real data (~66 deg from the first metre), and on THIS data the two are not
               even related by a constant: the nadir camera is world-stabilised, its quaternion is
               constant across every frame, while body yaw spans +/-180 deg. So the body pose never
               reaches `groundtruth.csv`, and `heading_deg` there is the CAMERA's.
`terrain.csv`  `up_m` (take-off datum, what a barometer reports) and `agl_m` (height above the
               imaged surface) are DIFFERENT QUANTITIES wherever terrain moves, and here terrain
               moves by 33 m. Keeping them in one file with both names visible is the cheapest
               defence against the substitution `LIT-VO-006` section 3 warns about.
`ingest_provenance.json`  Source digests and every verification result, so a later reader can tell
               whether the checks below actually passed on the bytes in front of them.

**Verification, not repair.** Every check in `verify()` raises. If the simulator ever flips an axis,
changes a datum or renames a channel, this refuses to build a dataset rather than building a quietly
mirrored one -- `contracts/dataset.md`'s stated reason for recording `frame_convention` instead of
assuming it.

    PY=.venv/Scripts/python.exe
    $PY evaluation/tools/exp_vo_014/ingest_ue_run.py \
        --source simulator_vo_data/Run_20260904_172033 \
        --dataset-id uevo-fig8-const-v1 --role constant_height
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVALUATION_DIR = HERE.parents[1]
REPO = EVALUATION_DIR.parent

#: Tolerances. Deliberately tight: these are exact algebraic identities on noise-free simulator
#: telemetry, not measurements. The observed residuals are ~1e-6 m and ~1e-8 deg, so a failure here
#: means a convention changed, never that the numbers drifted.
TOL_M = 1e-4
TOL_DEG = 1e-3
TOL_AXIS = 1e-5

#: The exact ENU mapping this adapter is written against. Compared as strings, so a simulator-side
#: change to the mapping breaks the build instead of silently producing a mirrored dataset.
EXPECTED_ENU_MAPPING = {
    "east_m": "ue_y_cm / 100",
    "north_m": "ue_x_cm / 100",
    "up_m": "ue_z_cm / 100",
}
EXPECTED_POSITION_SOURCE = "nadir_camera_center"
EXPECTED_QUAT_ORDER = "wxyz"


# --------------------------------------------------------------------------- reading


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def col(rows: list[dict], name: str) -> np.ndarray:
    return np.array([float(r[name]) for r in rows], dtype=float)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quat_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Rotation matrix for a `wxyz` quaternion. Columns are the images of the local axes."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# --------------------------------------------------------------------------- verification


class ConventionError(RuntimeError):
    """A frame, unit or datum convention in the source does not match what this adapter assumes."""


def verify(settings: dict, frames: list[dict], gt: list[dict], images_dir: Path) -> dict:
    """Every assumption this adapter makes, checked against the bytes. Raises on any failure.

    Returns the measured residuals so `ingest_provenance.json` can record what the checks actually
    saw rather than merely that they passed.
    """
    out: dict = {}
    wf = settings["world_frame"]

    # -- C1 declared conventions -------------------------------------------------------------
    if wf.get("enu_mapping") != EXPECTED_ENU_MAPPING:
        raise ConventionError(f"enu_mapping is {wf.get('enu_mapping')}, expected {EXPECTED_ENU_MAPPING}")
    if wf.get("enu_position_source") != EXPECTED_POSITION_SOURCE:
        raise ConventionError(
            f"enu_position_source is {wf.get('enu_position_source')!r}; this adapter evaluates the "
            f"CAMERA trajectory and requires {EXPECTED_POSITION_SOURCE!r}")
    if wf.get("ue_units") != "cm":
        raise ConventionError(f"ue_units is {wf.get('ue_units')!r}, expected 'cm'")
    if wf.get("enu_handedness") != "right":
        raise ConventionError(f"enu_handedness is {wf.get('enu_handedness')!r}, expected 'right'")
    if settings["quaternion_convention"]["order"] != EXPECTED_QUAT_ORDER:
        raise ConventionError("quaternion order is not wxyz")
    out["declared_conventions"] = "ok"

    # -- C2 the ENU columns really are the declared function of the camera's UE coordinates ----
    e, n, u = col(frames, "east_m"), col(frames, "north_m"), col(frames, "up_m")
    cx, cy, cz = (col(frames, f"cam_ue_{a}_cm") / 100.0 for a in "xyz")
    r = {"east": float(np.abs(e - cy).max()), "north": float(np.abs(n - cx).max()),
         "up": float(np.abs(u - cz).max())}
    if max(r.values()) > TOL_M:
        raise ConventionError(f"ENU columns disagree with cam_ue_*_cm/100: {r}")
    out["enu_mapping_residual_m"] = r

    # -- C3 the ENU columns are the CAMERA's, not the body's ----------------------------------
    bx, by = (col(frames, f"body_ue_{a}_cm") / 100.0 for a in "xy")
    sep = float(np.abs(np.hypot(cx - bx, cy - by)).max())
    if np.abs(n - bx).max() <= TOL_M and np.abs(e - by).max() <= TOL_M and sep > TOL_M:
        raise ConventionError("east_m/north_m track the BODY, not the camera")
    out["camera_body_max_horizontal_separation_m"] = sep

    # -- C4 the camera's optical frame maps as OpenCV-on-ENU --------------------------------
    # Local +X optical axis -> world down; +Y image right -> East; +Z image up -> North. This is
    # what makes `east = +T_x, north = -T_y` (LIT-VO-003 eq. 10) the correct readout mapping.
    q = np.stack([col(frames, f"cam_ue_q{c}") for c in "wxyz"], axis=1)
    norms = np.linalg.norm(q, axis=1)
    if np.abs(norms - 1.0).max() > TOL_AXIS:
        raise ConventionError(f"camera quaternions not normalised (max |‖q‖-1| = {np.abs(norms - 1).max():.3e})")
    uniq = np.unique(np.round(q, 9), axis=0)
    out["camera_quaternion_unique_rows"] = int(len(uniq))
    # Columns of the rotation matrix, vectorised over EVERY frame -- not a sample. A camera that is
    # nadir at frame 0 and drifts afterwards is exactly the failure a sampled check would miss.
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    axis_x = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], axis=1)
    axis_y = np.stack([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], axis=1)
    axis_z = np.stack([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)], axis=1)
    # UE axes: +X North, +Y East, +Z Up.
    worst = {
        "down": float(np.linalg.norm(axis_x - np.array([0.0, 0.0, -1.0]), axis=1).max()),
        "right_east": float(np.linalg.norm(axis_y - np.array([0.0, 1.0, 0.0]), axis=1).max()),
        "up_north": float(np.linalg.norm(axis_z - np.array([1.0, 0.0, 0.0]), axis=1).max()),
    }
    if max(worst.values()) > 1e-4:
        raise ConventionError(
            "camera optical axes are not the assumed nadir/East-right/North-up frame: " + str(worst))
    out["camera_axis_residual"] = worst

    # -- C5 logged heading is the camera's image-up azimuth, compass CW from North ------------
    # Checked at EVERY frame. Checking only frame 0 would accept a heading column carrying AIRFRAME
    # yaw whenever the airframe happens to start pointing north -- which is the common case, and is
    # exactly how EXP-VO-013 R6's mistake would slip through a weaker check.
    heading_from_quat = np.degrees(np.arctan2(axis_z[:, 1], axis_z[:, 0])) % 360.0
    logged = col(frames, "heading_deg")
    dev = np.abs(((logged - heading_from_quat + 180.0) % 360.0) - 180.0)
    if float(dev.max()) > TOL_DEG:
        k = int(dev.argmax())
        raise ConventionError(
            f"heading_deg is not the camera image-up azimuth at frame {k} "
            f"(logged {logged[k]:.6f}, camera {heading_from_quat[k]:.6f}, "
            f"max deviation {dev.max():.6f} deg over {len(dev)} frames); "
            f"it may be the AIRFRAME yaw -- see EXP-VO-013 R6")
    out["heading_vs_camera_quaternion_max_deg"] = float(dev.max())
    out["camera_heading_deg_range"] = [float(logged.min()), float(logged.max())]

    # -- C6 the airframe yaw is a DIFFERENT quantity, and is measured so it cannot be assumed --
    bq = np.stack([col(frames, f"body_ue_q{c}") for c in "wxyz"], axis=1)
    byaw = np.degrees(np.arctan2(2 * (bq[:, 0] * bq[:, 3] + bq[:, 1] * bq[:, 2]),
                                 1 - 2 * (bq[:, 2] ** 2 + bq[:, 3] ** 2)))
    out["body_yaw_deg_range"] = [float(byaw.min()), float(byaw.max())]
    out["body_yaw_deg_sd"] = float(byaw.std())
    out["camera_body_yaw_is_fixed_offset"] = bool(
        np.std(((byaw - logged + 180.0) % 360.0) - 180.0) < 1.0)

    # -- C7 the height identities ------------------------------------------------------------
    agl = col(frames, "true_agl_m")
    terr = col(frames, "terrain_elevation_m")
    baro = col(frames, "baro_relative_alt_m")
    h0 = float(settings["height"]["h0_agl_m"])
    res = {
        "h0_minus_true_agl0": float(abs(h0 - agl[0])),
        "up_minus_terrain_minus_agl": float(np.abs(u - terr - agl).max()),
        "h0_plus_baro_minus_up_minus_terrain0": float(np.abs(h0 + baro - (u - terr[0])).max()),
        "baro_datum_identity": float(np.abs((h0 + baro - agl) - (terr - terr[0])).max()),
    }
    if max(res.values()) > TOL_M:
        raise ConventionError(f"height datum identities violated: {res}")
    out["height_identity_residual_m"] = res
    if float(np.abs(baro[0])) > TOL_M:
        raise ConventionError(f"baro_relative_alt_m[0] = {baro[0]}, expected 0 at the reference frame")

    # -- C8 the AGL trace is usable at every frame -------------------------------------------
    hit = col(frames, "ground_hit")
    if not np.all(hit == 1):
        raise ConventionError(f"{int((hit != 1).sum())} frames have no ground_hit; true_agl invalid there")
    if float(agl.min()) <= 0:
        raise ConventionError("non-positive true_agl_m")
    out["true_agl_m_range"] = [float(agl.min()), float(agl.max())]

    # -- C9 the camera is level (the nadir/planar readout assumes it) -------------------------
    tilt = col(frames, "camera_tilt_deg")
    if float(np.abs(tilt).max()) > TOL_DEG:
        raise ConventionError(f"camera_tilt_deg reaches {np.abs(tilt).max():.6f}; readout assumes nadir")
    out["camera_tilt_deg_max"] = float(np.abs(tilt).max())

    # -- C10 frame index / timestamp integrity ------------------------------------------------
    fid = np.array([int(r["frame_id"]) for r in frames])
    if not np.array_equal(fid, np.arange(len(frames))):
        raise ConventionError("frame_id is not 0-based contiguous")
    t = col(frames, "sim_time_s")
    if not np.all(np.diff(t) > 0):
        raise ConventionError("frame timestamps are not strictly increasing")
    dt = np.diff(t)
    out["frame_dt_s"] = [float(dt.min()), float(dt.max())]
    out["image_rate_hz"] = float(1.0 / dt.mean())

    # -- C11 frame timestamps are a subset of the telemetry grid, so no offset is estimated ---
    tg = col(gt, "sim_time_s")
    if not np.all(np.diff(tg) > 0):
        raise ConventionError("groundtruth timestamps are not strictly increasing")
    idx = np.clip(np.searchsorted(tg, t), 0, len(tg) - 1)
    dtmax = float(np.abs(t - tg[idx]).max())
    if dtmax > 1e-6:
        raise ConventionError(f"frame timestamps are not a subset of the telemetry grid (max {dtmax:.3e} s)")
    pose = {c: float(np.abs(col(gt, c)[idx] - col(frames, c)).max())
            for c in ("east_m", "north_m", "up_m", "true_agl_m", "baro_relative_alt_m")}
    if max(pose.values()) > TOL_M:
        raise ConventionError(f"frames.csv and groundtruth.csv disagree at matched timestamps: {pose}")
    out["frame_gt_timestamp_residual_s"] = dtmax
    out["frame_gt_pose_residual"] = pose

    # -- C12 every image exists -------------------------------------------------------------
    missing = [r["image_path"] for r in frames if not (images_dir.parent.parent / r["image_path"]).exists()]
    if missing:
        raise ConventionError(f"{len(missing)} image(s) referenced by frames.csv are absent, e.g. {missing[0]}")
    out["images_present"] = len(frames)

    return out


# --------------------------------------------------------------------------- writing


def link_or_copy(src: Path, dst: Path) -> str:
    """Hard-link the image, falling back to a copy. Never modifies the source.

    13 GB of PNG across the two runs, so copying would be wasteful and would also create a second
    set of bytes that could drift from the source. A hard link is the same inode: the dataset and
    `simulator_vo_data/` are provably the same image.
    """
    if dst.exists():
        return "present"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def ingest(source: Path, dataset_id: str, role: str, out_root: Path,
           link_images: bool = True) -> dict:
    source = Path(source).resolve()
    settings = json.loads((source / "settings.json").read_text())
    frames = read_csv(source / "vo" / "frames.csv")
    gt = read_csv(source / "vo" / "groundtruth.csv")
    images_dir = source / "vo" / "images"

    checks = verify(settings, frames, gt, images_dir)

    out = out_root / dataset_id
    (out / "images").mkdir(parents=True, exist_ok=True)

    h0 = float(settings["height"]["h0_agl_m"])
    terr0 = col(frames, "terrain_elevation_m")[0]
    intr = settings["camera_intrinsics"]

    # ---- frames.csv
    with (out / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["frame_index", "timestamp_s", "image_path"])
        for r in frames:
            name = Path(r["image_path"]).name
            w.writerow([r["frame_id"], "%.9f" % float(r["sim_time_s"]), f"images/{name}"])

    # ---- groundtruth.csv, at the FULL 60 Hz telemetry rate, camera pose, camera heading.
    # `up_m` is height above the TAKE-OFF GROUND PLANE (the bvo-* convention, so up_m[0] == h0),
    # which is what a barometer with a known h0 reports -- NOT height above the imaged surface.
    with (out / "groundtruth.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["timestamp_s", "east_m", "north_m", "up_m", "heading_deg", "fix_quality", "valid"])
        for r in gt:
            w.writerow(["%.9f" % float(r["sim_time_s"]), "%.9f" % float(r["east_m"]),
                        "%.9f" % float(r["north_m"]),
                        "%.9f" % (float(r["up_m"]) - terr0), "%.9f" % float(r["heading_deg"]),
                        "simulator_exact", "true"])

    # ---- terrain.csv, one row per IMAGE frame (bvo-* columns + an explicit frame_index).
    with (out / "terrain.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["frame_index", "timestamp_s", "terrain_m", "agl_m", "up_m", "baro_relative_m"])
        for r in frames:
            w.writerow([r["frame_id"], "%.9f" % float(r["sim_time_s"]),
                        "%.9f" % (float(r["terrain_elevation_m"]) - terr0),
                        "%.9f" % float(r["true_agl_m"]),
                        "%.9f" % (float(r["up_m"]) - terr0),
                        "%.9f" % float(r["baro_relative_alt_m"])])

    # ---- ue_body.csv -- the AIRFRAME pose. Preserved, quarantined, never scored against.
    with (out / "ue_body.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["frame_index", "timestamp_s", "body_east_m", "body_north_m", "body_up_m",
                    "body_yaw_deg", "body_qw", "body_qx", "body_qy", "body_qz"])
        for r in frames:
            bw, bx, by, bz = (float(r[f"body_ue_q{c}"]) for c in "wxyz")
            yaw = math.degrees(math.atan2(2 * (bw * bz + bx * by), 1 - 2 * (by * by + bz * bz))) % 360.0
            w.writerow([r["frame_id"], "%.9f" % float(r["sim_time_s"]),
                        "%.9f" % (float(r["body_ue_y_cm"]) / 100.0),
                        "%.9f" % (float(r["body_ue_x_cm"]) / 100.0),
                        "%.9f" % (float(r["body_ue_z_cm"]) / 100.0 - terr0),
                        "%.9f" % yaw, "%.9f" % bw, "%.9f" % bx, "%.9f" % by, "%.9f" % bz])

    # ---- images
    modes: dict[str, int] = {}
    for r in frames:
        name = Path(r["image_path"]).name
        mode = (link_or_copy(images_dir / name, out / "images" / name) if link_images
                else "skipped")
        modes[mode] = modes.get(mode, 0) + 1

    # ---- dataset.json
    agl = col(frames, "true_agl_m")
    baro = col(frames, "baro_relative_alt_m")
    terr = col(frames, "terrain_elevation_m")
    e, n = col(frames, "east_m"), col(frames, "north_m")
    path_len = float(np.sum(np.hypot(np.diff(e), np.diff(n))))
    t = col(frames, "sim_time_s")

    desc = {
        "schema_version": "1.0.0",
        "dataset_id": dataset_id,
        "dataset_revision": "v1",
        "source_type": "ue5_simulator",
        "evidence_tier": "T2",
        "evidence_caveat": (
            "EXP-VO-014 UE5 capture (Unreal " + settings["engine_version"] + ", level '"
            + settings["level"] + "'). T2 SIMULATOR evidence. Unlike the Java simulator this is NOT "
            "model-matched to the estimator's planar assumption -- DEC-005: UE5 renders real 3D "
            "geometry -- and the scene is a village on hills, so the ground footprint is emphatically "
            "not a plane. But the imagery has no motion blur, no auto-exposure, no lens distortion, "
            "no sensor noise, fixed noon lighting and lossless PNG, and the nadir camera is "
            "world-stabilised in yaw and exactly level (camera_tilt_deg == 0 at every frame), so no "
            "rotation is exercised at all. CRITICALLY: baro_relative_alt_m is an IDEAL simulated "
            "channel -- derived exactly from the physics body's Z, hence noise-free, drift-free, "
            "bias-free, unquantised and zero-latency. Results from it validate the IMPLEMENTATION "
            "and GEOMETRY of an altitude-constrained metric readout, and say NOTHING about a real "
            "barometer's error behaviour (EXP-VO-012 R3-R6 characterises that separately)."),
        "frame_clock": "sim_time_s",
        "gt_clock": "sim_time_s",
        "clock_offset_s": 0.0,
        "clock_offset_source": (
            "none required: settings.json capture.pose_instant records that pose, altitude trace and "
            "image are taken from the same callback, and every image timestamp is bit-identical to a "
            "telemetry sample time (verified, max |dt| = %.1e s)" % checks["frame_gt_timestamp_residual_s"]),
        "clock_drift_s_per_s": 0.0,
        "frame_convention": "ENU",
        "heading_convention": "compass_cw_from_north",
        "local_frame_origin": None,
        "position_quality": {
            "class": "simulator_exact", "nominal_accuracy": 0.0,
            "accuracy_source": "exact rendered camera pose (nadir_camera_center)",
            "notes": "east_m/north_m are the NADIR CAMERA optical centre in the UE world's ENU "
                     "mapping, not re-origined to frame 0, so the paired runs share one frame."},
        "heading_quality": {
            "class": "simulator_exact", "nominal_accuracy": 0.0,
            "accuracy_source": "camera image-up azimuth, verified against the camera quaternion",
            "notes": "THE CAMERA'S heading, NOT the airframe's. The nadir camera is world-stabilised: "
                     "its quaternion is constant across every frame (%d unique row(s)) while body yaw "
                     "spans %.1f..%.1f deg with sd %.1f deg. They are not related by a fixed offset. "
                     "The airframe pose is in ue_body.csv and must never be used to initialise a "
                     "camera-frame evaluation -- EXP-VO-013 R6."
                     % (checks["camera_quaternion_unique_rows"], checks["body_yaw_deg_range"][0],
                        checks["body_yaw_deg_range"][1], checks["body_yaw_deg_sd"])},
        "height_quality": {
            "class": "simulator_exact", "nominal_accuracy": 0.0,
            "accuracy_source": "exact rendered pose and principal-ray ground trace",
            "notes": "up_m is altitude above the TAKE-OFF GROUND PLANE -- what a barometer reports "
                     "given h0. Height above the IMAGED SURFACE is agl_m in terrain.csv and is a "
                     "DIFFERENT quantity wherever terrain_m is nonzero; here it moves by %.1f m. "
                     "LIT-VO-006 section 3." % float(terr.max() - terr.min())},
        "metadata": {
            "flight_id": settings["run_id"],
            "trajectory_type": "figure8",
            "frame_rate_hz": round(checks["image_rate_hz"], 6),
            "image_width": intr["width"],
            "image_height": intr["height"],
            "environment": settings["level"],
            "nominal_altitude_m": h0,
            "nominal_speed_ms": settings["path"]["cruise_speed_mps"],
            "capture_date": "2026-09-04",
            "notes": ("EXP-VO-014 paired figure-8, role=%s. Path '%s'. %d frames, %.1f s, "
                      "%.1f m horizontal path." % (role, settings["path"]["path_id"], len(frames),
                                                   float(t[-1] - t[0]), path_len)),
            "provenance": ("EXP-VO-014: ingested from %s by evaluation/tools/exp_vo_014/"
                           "ingest_ue_run.py. Imagery hard-linked from the source, not copied; the "
                           "source folder is preserved byte-for-byte." % source.name),
            "camera_intrinsics": {k: intr[k] for k in
                                  ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2",
                                   "width", "height", "source")},
            "role": role,
            "h0_agl_m": h0,
            "h0_source": settings["height"]["h0_source"],
            "gsd0_m_per_px": settings["height"]["gsd0_m_per_px"],
            "baro_relative_alt_definition": settings["height"]["baro_relative_alt_definition"],
            "agl_definition": settings["height"]["agl_definition"],
            "terrain_datum_offset_m": float(terr0),
            "terrain_datum_note": ("terrain.csv/groundtruth.csv up_m and terrain_m are relative to "
                                   "the take-off ground elevation, UE world Z = %.6f m." % terr0),
            "gt_horizontal_path_length_m": path_len,
            "gt_start_end_xy_m": float(math.hypot(e[-1] - e[0], n[-1] - n[0])),
            "camera_altitude_range_m": [float(col(frames, "up_m").min()), float(col(frames, "up_m").max())],
            "baro_relative_alt_range_m": [float(baro.min()), float(baro.max())],
            "true_agl_range_m": [float(agl.min()), float(agl.max())],
            "terrain_elevation_range_m": [float(terr.min()), float(terr.max())],
            "ue_rendering_baseline": settings["vo_rendering_baseline"],
            "ue_capture": settings["capture"],
            "ue_world_frame": settings["world_frame"],
            "ue_quaternion_convention": settings["quaternion_convention"],
        },
    }
    (out / "dataset.json").write_text(json.dumps(desc, indent=2) + "\n")

    prov = {
        "experiment": "EXP-VO-014",
        "tool": "evaluation/tools/exp_vo_014/ingest_ue_run.py",
        "source_dir": str(source),
        "dataset_id": dataset_id,
        "role": role,
        "source_digests": {
            "settings.json": sha256(source / "settings.json"),
            "vo/frames.csv": sha256(source / "vo" / "frames.csv"),
            "vo/groundtruth.csv": sha256(source / "vo" / "groundtruth.csv"),
        },
        "n_image_frames": len(frames),
        "n_groundtruth_samples": len(gt),
        "image_link_modes": modes,
        "verification": checks,
    }
    (out / "ingest_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    return prov


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", required=True, help="UE run folder (contains settings.json and vo/)")
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--role", required=True, choices=("constant_height", "varying_height"))
    p.add_argument("--out-root", default=str(REPO / "datasets"))
    p.add_argument("--no-images", action="store_true", help="descriptors only (tests use this)")
    a = p.parse_args(argv)
    prov = ingest(Path(a.source), a.dataset_id, a.role, Path(a.out_root), link_images=not a.no_images)
    print(json.dumps({k: prov[k] for k in
                      ("dataset_id", "role", "n_image_frames", "n_groundtruth_samples",
                       "image_link_modes")}, indent=2))
    print("verification: all checks passed")
    for k, v in prov["verification"].items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
