"""INT smoke-test adapter: a UE capture folder -> the dataset contract, for a world-stabilised nadir
camera whose heading is NOT North.

This is the VO lane's ``evaluation/tools/exp_vo_014/ingest_ue_run.py`` with exactly one check
generalised. That adapter's C4 requires the camera's image-right and image-up axes to be East and
North *exactly*, which is true of the ``EXP-VO-014`` captures (heading 359.99 deg) and false of any
capture whose nadir camera holds another azimuth -- the ``important simulator runs`` batch holds
~10 deg, 3.3 deg and 353.3 deg, and the VO adapter refuses all three with a residual equal to
``sin(heading)``. C4 is replaced here by the general statement of the same convention: the optical
axis points world-down, image-right and image-up are horizontal and orthonormal, and image-right is
image-up rotated a quarter turn clockwise; C5 (the logged ``heading_deg`` IS the image-up azimuth,
at every frame) is what pins the azimuth itself and is kept verbatim. Every other check (C1-C3,
C5-C12), every output file and every convention are the VO adapter's own, called through it --
nothing about the height datums, the ENU mapping or the body-yaw quarantine is reimplemented.

Kept as a wrapper rather than an edit of the VO adapter, so that adapter is unchanged.

    PY=<repo>/.venv/Scripts/python.exe
    $PY evaluation/tools/int/ingest_ue_run_int.py --source <UE run folder with settings.json and vo/> \\
        --dataset-id intsmoke-mtn-190438-v1 --out-root <scratch>/datasets
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
if str(TOOLS / "exp_vo_014") not in sys.path:
    sys.path.insert(0, str(TOOLS / "exp_vo_014"))

import ingest_ue_run as vo  # noqa: E402  (the VO lane's adapter, unmodified)

TOOL = "evaluation/tools/int/ingest_ue_run_int.py"


def verify_heading_aware(settings: dict, frames: list[dict], gt: list[dict], images_dir: Path) -> dict:
    """The VO adapter's ``verify`` with C4 generalised to a non-North world-stabilised camera."""
    out: dict = {}
    wf = settings["world_frame"]
    col = vo.col

    # -- C1 declared conventions (VO adapter, verbatim) ---------------------------------------
    if wf.get("enu_mapping") != vo.EXPECTED_ENU_MAPPING:
        raise vo.ConventionError(f"enu_mapping is {wf.get('enu_mapping')}, expected {vo.EXPECTED_ENU_MAPPING}")
    if wf.get("enu_position_source") != vo.EXPECTED_POSITION_SOURCE:
        raise vo.ConventionError(
            f"enu_position_source is {wf.get('enu_position_source')!r}; this adapter evaluates the "
            f"CAMERA trajectory and requires {vo.EXPECTED_POSITION_SOURCE!r}")
    if wf.get("ue_units") != "cm":
        raise vo.ConventionError(f"ue_units is {wf.get('ue_units')!r}, expected 'cm'")
    if wf.get("enu_handedness") != "right":
        raise vo.ConventionError(f"enu_handedness is {wf.get('enu_handedness')!r}, expected 'right'")
    if settings["quaternion_convention"]["order"] != vo.EXPECTED_QUAT_ORDER:
        raise vo.ConventionError("quaternion order is not wxyz")
    out["declared_conventions"] = "ok"

    # -- C2 / C3 (verbatim) -----------------------------------------------------------------
    e, n, u = col(frames, "east_m"), col(frames, "north_m"), col(frames, "up_m")
    cx, cy, cz = (col(frames, f"cam_ue_{a}_cm") / 100.0 for a in "xyz")
    r = {"east": float(np.abs(e - cy).max()), "north": float(np.abs(n - cx).max()),
         "up": float(np.abs(u - cz).max())}
    if max(r.values()) > vo.TOL_M:
        raise vo.ConventionError(f"ENU columns disagree with cam_ue_*_cm/100: {r}")
    out["enu_mapping_residual_m"] = r
    bx, by = (col(frames, f"body_ue_{a}_cm") / 100.0 for a in "xy")
    sep = float(np.abs(np.hypot(cx - bx, cy - by)).max())
    if np.abs(n - bx).max() <= vo.TOL_M and np.abs(e - by).max() <= vo.TOL_M and sep > vo.TOL_M:
        raise vo.ConventionError("east_m/north_m track the BODY, not the camera")
    out["camera_body_max_horizontal_separation_m"] = sep

    # -- C4' the camera's optical frame is nadir with HORIZONTAL image axes, at every frame ----
    # Local +X optical axis -> world down; +Y image right and +Z image up horizontal, orthonormal,
    # with image-right = image-up rotated a quarter turn clockwise. The azimuth of image-up is NOT
    # assumed to be North: it is whatever C5 confirms the logged heading to be. UE axes: +X North,
    # +Y East, +Z Up.
    q = np.stack([col(frames, f"cam_ue_q{c}") for c in "wxyz"], axis=1)
    norms = np.linalg.norm(q, axis=1)
    if np.abs(norms - 1.0).max() > vo.TOL_AXIS:
        raise vo.ConventionError(f"camera quaternions not normalised (max |‖q‖-1| = {np.abs(norms - 1).max():.3e})")
    uniq = np.unique(np.round(q, 9), axis=0)
    out["camera_quaternion_unique_rows"] = int(len(uniq))
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    axis_x = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], axis=1)
    axis_y = np.stack([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], axis=1)
    axis_z = np.stack([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)], axis=1)
    up_az = np.arctan2(axis_z[:, 1], axis_z[:, 0])                     # image-up azimuth, CW from North
    right_expected = np.stack([-np.sin(up_az), np.cos(up_az), np.zeros_like(up_az)], axis=1)
    worst = {
        "down": float(np.linalg.norm(axis_x - np.array([0.0, 0.0, -1.0]), axis=1).max()),
        "up_horizontal": float(np.abs(axis_z[:, 2]).max()),
        "right_is_up_rotated_cw": float(np.linalg.norm(axis_y - right_expected, axis=1).max()),
    }
    if max(worst.values()) > 1e-4:
        raise vo.ConventionError(
            "camera optical axes are not a nadir frame with horizontal, orthonormal image axes: " + str(worst))
    out["camera_axis_residual"] = worst
    out["camera_axis_check"] = ("C4 generalised (INT): nadir, horizontal image axes, right = up rotated "
                                "90 deg CW; the azimuth is pinned by C5, not assumed North")

    # -- C5 .. C12 (verbatim from the VO adapter) --------------------------------------------
    heading_from_quat = np.degrees(up_az) % 360.0
    logged = col(frames, "heading_deg")
    dev = np.abs(((logged - heading_from_quat + 180.0) % 360.0) - 180.0)
    if float(dev.max()) > vo.TOL_DEG:
        k = int(dev.argmax())
        raise vo.ConventionError(
            f"heading_deg is not the camera image-up azimuth at frame {k} "
            f"(logged {logged[k]:.6f}, camera {heading_from_quat[k]:.6f}, "
            f"max deviation {dev.max():.6f} deg over {len(dev)} frames); "
            f"it may be the AIRFRAME yaw -- see EXP-VO-013 R6")
    out["heading_vs_camera_quaternion_max_deg"] = float(dev.max())
    out["camera_heading_deg_range"] = [float(logged.min()), float(logged.max())]

    bq = np.stack([col(frames, f"body_ue_q{c}") for c in "wxyz"], axis=1)
    byaw = np.degrees(np.arctan2(2 * (bq[:, 0] * bq[:, 3] + bq[:, 1] * bq[:, 2]),
                                 1 - 2 * (bq[:, 2] ** 2 + bq[:, 3] ** 2)))
    out["body_yaw_deg_range"] = [float(byaw.min()), float(byaw.max())]
    out["body_yaw_deg_sd"] = float(byaw.std())
    out["camera_body_yaw_is_fixed_offset"] = bool(
        np.std(((byaw - logged + 180.0) % 360.0) - 180.0) < 1.0)

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
    if max(res.values()) > vo.TOL_M:
        raise vo.ConventionError(f"height datum identities violated: {res}")
    out["height_identity_residual_m"] = res
    if float(np.abs(baro[0])) > vo.TOL_M:
        raise vo.ConventionError(f"baro_relative_alt_m[0] = {baro[0]}, expected 0 at the reference frame")

    hit = col(frames, "ground_hit")
    if not np.all(hit == 1):
        raise vo.ConventionError(f"{int((hit != 1).sum())} frames have no ground_hit; true_agl invalid there")
    if float(agl.min()) <= 0:
        raise vo.ConventionError("non-positive true_agl_m")
    out["true_agl_m_range"] = [float(agl.min()), float(agl.max())]

    tilt = col(frames, "camera_tilt_deg")
    if float(np.abs(tilt).max()) > vo.TOL_DEG:
        raise vo.ConventionError(f"camera_tilt_deg reaches {np.abs(tilt).max():.6f}; readout assumes nadir")
    out["camera_tilt_deg_max"] = float(np.abs(tilt).max())

    fid = np.array([int(r_["frame_id"]) for r_ in frames])
    if not np.array_equal(fid, np.arange(len(frames))):
        raise vo.ConventionError("frame_id is not 0-based contiguous")
    t = col(frames, "sim_time_s")
    if not np.all(np.diff(t) > 0):
        raise vo.ConventionError("frame timestamps are not strictly increasing")
    dt = np.diff(t)
    out["frame_dt_s"] = [float(dt.min()), float(dt.max())]
    out["image_rate_hz"] = float(1.0 / dt.mean())

    tg = col(gt, "sim_time_s")
    if not np.all(np.diff(tg) > 0):
        raise vo.ConventionError("groundtruth timestamps are not strictly increasing")
    idx = np.clip(np.searchsorted(tg, t), 0, len(tg) - 1)
    dtmax = float(np.abs(t - tg[idx]).max())
    if dtmax > 1e-6:
        raise vo.ConventionError(f"frame timestamps are not a subset of the telemetry grid (max {dtmax:.3e} s)")
    pose = {c: float(np.abs(col(gt, c)[idx] - col(frames, c)).max())
            for c in ("east_m", "north_m", "up_m", "true_agl_m", "baro_relative_alt_m")}
    if max(pose.values()) > vo.TOL_M:
        raise vo.ConventionError(f"frames.csv and groundtruth.csv disagree at matched timestamps: {pose}")
    out["frame_gt_timestamp_residual_s"] = dtmax
    out["frame_gt_pose_residual"] = pose

    missing = [r_["image_path"] for r_ in frames if not (images_dir.parent.parent / r_["image_path"]).exists()]
    if missing:
        raise vo.ConventionError(f"{len(missing)} image(s) referenced by frames.csv are absent, e.g. {missing[0]}")
    out["images_present"] = len(frames)
    return out


def ingest(source: Path, dataset_id: str, out_root: Path, link_images: bool = True,
           role: str = "constant_height") -> dict:
    """The VO adapter's ``ingest`` with the heading-aware verify, plus an honest provenance stamp.
    ``role`` is the VO adapter's height-profile label (``constant_height`` / ``varying_height``)."""
    original = vo.verify
    vo.verify = verify_heading_aware
    try:
        prov = vo.ingest(source, dataset_id, role, out_root, link_images=link_images)
    finally:
        vo.verify = original

    out = out_root / dataset_id
    settings = json.loads((Path(source) / "settings.json").read_text())
    desc = json.loads((out / "dataset.json").read_text())
    md = desc["metadata"]
    md["trajectory_type"] = settings["path"].get("path_id", md.get("trajectory_type"))
    md["capture_date"] = str(settings["run_id"])[4:8] + "-" + str(settings["run_id"])[8:10] + "-" + str(settings["run_id"])[10:12]
    md["notes"] = (f"INT ingest of {settings['run_id']} (path '{settings['path'].get('path_id')}', "
                   f"level '{settings['level']}'): {prov['n_image_frames']} frames. Not an EXP-VO-014 run; "
                   f"role '{role}' is the VO adapter's height-profile label, declared by the operator.")
    md["provenance"] = (f"Ingested from {Path(source).name} by {TOOL}, which calls "
                        "evaluation/tools/exp_vo_014/ingest_ue_run.py with its C4 camera-axis check generalised "
                        "to a non-North world-stabilised nadir camera (the azimuth is pinned by C5). Imagery "
                        "hard-linked, source preserved byte-for-byte.")
    md["camera_axis_check"] = prov["verification"]["camera_axis_check"]
    desc["evidence_caveat"] = desc["evidence_caveat"].replace("EXP-VO-014 UE5 capture", "UE5 capture (INT smoke test)")
    (out / "dataset.json").write_text(json.dumps(desc, indent=2) + "\n")
    prov["tool"] = TOOL
    prov["experiment"] = "INT smoke test (no experiment record; mechanism check only)"
    (out / "ingest_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    return prov


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", required=True, help="UE run folder (contains settings.json and vo/)")
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--out-root", required=True, help="never datasets/ by default: this is scratch tooling")
    p.add_argument("--no-images", action="store_true")
    p.add_argument("--role", default="constant_height", choices=("constant_height", "varying_height"),
                   help="the VO adapter's height-profile label for dataset.json (a declaration, not a measurement)")
    a = p.parse_args(argv)
    prov = ingest(Path(a.source), a.dataset_id, Path(a.out_root), link_images=not a.no_images, role=a.role)
    print(json.dumps({k: prov[k] for k in ("dataset_id", "n_image_frames", "n_groundtruth_samples",
                                           "image_link_modes")}, indent=2))
    print("verification: all checks passed")
    for k, v in prov["verification"].items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
