"""EXP-VO-011: load the committed per-frame scale series and the physical reference it has to be
compared against.

One module so that every other tool in `exp_vo_011/` reads the same columns, applies the same row
filter, and uses the same sign convention. Three facts this module encodes, all verified in source
and recorded in the experiment's *What the source says* section:

- `inc_log_scale` is `log sqrt(|det J|)` of **`D_k^-1 : C_k -> C_{k-1}`**, evaluated at the image
  centre (`RigidNavigationState.observe`). Under `LIT-VO-003` eq. (2) `D_k` has scale `h_{k-1}/h_k`,
  so a **positive** increment means the estimator believes the camera is climbing.
- The `init` row carries no motion estimate (the increment is 0 by construction) and is dropped.
- A `restart` row's increment is not a motion estimate of that frame either -- the estimator
  re-initialises inside `processFrame` and `RigidNavigationState` is fed the post-restart
  accumulation. Dropped, and the count is reported rather than hidden.
- `recenter` rows ARE kept: `EXP-VO-004` R3 measured their increments to be ordinary-sized and
  ordinary-signed, because the canvas fold is exact and the frame still moves.
"""
from __future__ import annotations

import csv
import json
import math
from bisect import bisect_left
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]

# The six real (window, model) arms the record scores, plus the similarity control arms.
REAL_ARMS = [
    ("hkairport01-a", "affine"), ("hkairport01-a", "homography"),
    ("hkairport01-b", "affine"), ("hkairport01-b", "homography"),
    ("amtown01-c", "affine"), ("amtown01-c", "homography"),
]
CONTROL_ARMS = [("hkairport01-a", "similarity"), ("hkairport01-b", "similarity"),
                ("amtown01-c", "similarity")]

# Which sequence in `evaluations/exp-vo-007/candidate_probe.json` each window is a piece of.
SCENE_OF = {"hkairport01-a": "HKairport01", "hkairport01-b": "HKairport01",
            "amtown01-c": "AMtown01"}

DROPPED_EVENTS = ("init", "restart")


def run_dir(window: str, model: str) -> Path:
    return REPO / "runs" / f"{window}-{model}-rigid-v1"


def dataset_dir(window: str) -> Path:
    return REPO / "datasets" / window


def load_arm(window: str, model: str) -> dict:
    """-> per-frame arrays for one (window, model) arm, with the pre-declared row filter applied."""
    path = run_dir(window, model) / "logical_transform.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} -- this arm's committed run record is missing")

    cols: dict[str, list] = {}
    events: list[str] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            events.append(row["event"])
            for k in ("frame_index", "inc_log_scale", "inc_scale", "inc_anisotropy",
                      "inc_deformation", "inc_stretch_axis_deg", "inc_perspective",
                      "inc_flow_px", "inc_rotation_deg", "rigid_accum_log_scale",
                      "rigid_x", "rigid_y", "rigid_yaw_deg"):
                cols.setdefault(k, []).append(float(row[k]))

    events_arr = np.array(events)
    keep = ~np.isin(events_arr, DROPPED_EVENTS)
    out = {k: np.asarray(v, dtype=float)[keep] for k, v in cols.items()}
    out["event"] = events_arr[keep]
    out["n_total_rows"] = int(len(events_arr))
    out["n_dropped"] = int((~keep).sum())
    out["n_restart"] = int((events_arr == "restart").sum())
    out["n_recenter"] = int((events_arr == "recenter").sum())
    out["window"] = window
    out["model"] = model

    # Timestamps and tracking support come from the run record's frames.csv, same frame schedule.
    ts, tracks, inliers = {}, {}, {}
    with (run_dir(window, model) / "frames.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            i = int(row["frame_index"])
            ts[i] = float(row["timestamp_s"])
            tracks[i] = float(row["track_count"] or "nan")
            inliers[i] = float(row["inlier_count"] or "nan")
    idx = [int(i) for i in out["frame_index"]]
    out["timestamp_s"] = np.array([ts[i] for i in idx])
    out["track_count"] = np.array([tracks[i] for i in idx])
    out["inlier_count"] = np.array([inliers[i] for i in idx])
    return out


def _interp_series(t_query: np.ndarray, t_ref: list, v_ref: list) -> np.ndarray:
    """Linear interpolation, clamped at both ends. numpy.interp already clamps; kept explicit so the
    clamping is a documented choice rather than a default."""
    return np.interp(t_query, np.asarray(t_ref, dtype=float), np.asarray(v_ref, dtype=float))


def load_lidar(window: str) -> dict:
    """-> {'t', 'height_m', 'footprint_spread_m', 'source'} for the scene this window belongs to.

    `height_m` is height above **the imaged surface** (`LIT-VO-004` section 4), which is the quantity
    that sets the pixel-to-metre factor -- NOT `groundtruth.csv`'s `up_m`, which is height above the
    takeoff datum. The two differ by the terrain, and on no MARS-LVIG sequence do they coincide.
    """
    path = REPO / "evaluations" / "exp-vo-007" / "candidate_probe.json"
    obj = json.loads(path.read_text())
    scene = SCENE_OF[window]
    prof = obj["sequences"][scene]["lidar"]["height_profile"]
    return {"t": prof["t"], "height_m": prof["height_m"],
            "footprint_spread_m": prof["footprint_spread_m"],
            "scene": scene,
            "source": f"{path.relative_to(REPO)} :: sequences.{scene}.lidar.height_profile"}


def load_groundtruth(window: str) -> dict:
    """-> {'t', 'east_m', 'north_m', 'up_m', 'heading_deg'} over the valid RTK samples."""
    t, e, n, u, h = [], [], [], [], []
    with (dataset_dir(window) / "groundtruth.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["valid"].strip().lower() != "true":
                continue
            t.append(float(row["timestamp_s"]))
            e.append(float(row["east_m"]))
            n.append(float(row["north_m"]))
            u.append(float(row["up_m"]))
            h.append(float(row["heading_deg"]))
    return {"t": np.array(t), "east_m": np.array(e), "north_m": np.array(n),
            "up_m": np.array(u), "heading_deg": np.array(h)}


def load_attitude(window: str) -> dict:
    """-> {'t', 'roll_deg', 'pitch_deg', 'tilt_deg', 'yaw_compass_deg'}."""
    keys = ("roll_deg", "pitch_deg", "tilt_deg", "yaw_compass_deg")
    t: list[float] = []
    cols: dict[str, list[float]] = {k: [] for k in keys}
    with (dataset_dir(window) / "attitude.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["timestamp_s"]))
            for k in keys:
                cols[k].append(float(row[k]))
    out = {k: np.array(v) for k, v in cols.items()}
    out["t"] = np.array(t)
    return out


def covariates_for(arm: dict) -> dict:
    """Every per-frame covariate the record's Phase 6 uses, interpolated onto the arm's frames.

    Heading is interpolated through its unit vector so that the 0/360 wrap cannot manufacture a
    spurious mid-flight excursion.
    """
    window = arm["window"]
    t = arm["timestamp_s"]
    lid = load_lidar(window)
    gt = load_groundtruth(window)
    att = load_attitude(window)

    hx = np.interp(t, gt["t"], np.cos(np.radians(gt["heading_deg"])))
    hy = np.interp(t, gt["t"], np.sin(np.radians(gt["heading_deg"])))
    heading = (np.degrees(np.arctan2(hy, hx)) + 360.0) % 360.0

    east = np.interp(t, gt["t"], gt["east_m"])
    north = np.interp(t, gt["t"], gt["north_m"])
    step = np.hypot(np.diff(east, prepend=east[0]), np.diff(north, prepend=north[0]))

    tilt = np.interp(t, att["t"], att["tilt_deg"])
    pitch = np.interp(t, att["t"], att["pitch_deg"])

    return {
        "lidar_height_m": _interp_series(t, lid["t"], lid["height_m"]),
        # Estimator-side per-frame covariates, straight from the same committed rows.
        "flow_px": arm["inc_flow_px"],
        "anisotropy": arm["inc_anisotropy"],
        "perspective": arm["inc_perspective"],
        "inlier_count": arm["inlier_count"],
        "abs_rotation_deg": np.abs(arm["inc_rotation_deg"]),
        # Rates rather than levels, for the mechanisms whose prediction is about CHANGE of attitude.
        "abs_d_tilt_deg": np.abs(np.diff(tilt, prepend=tilt[0])),
        "d_pitch_deg": np.diff(pitch, prepend=pitch[0]),
        "lidar_spread_m": _interp_series(t, lid["t"], lid["footprint_spread_m"]),
        "lidar_spread_frac": (_interp_series(t, lid["t"], lid["footprint_spread_m"])
                              / _interp_series(t, lid["t"], lid["height_m"])),
        "up_m": np.interp(t, gt["t"], gt["up_m"]),
        "heading_deg": heading,
        "tilt_deg": np.interp(t, att["t"], att["tilt_deg"]),
        "roll_deg": np.interp(t, att["t"], att["roll_deg"]),
        "pitch_deg": np.interp(t, att["t"], att["pitch_deg"]),
        "distance_m": np.cumsum(step),
        "elapsed_s": t - t[0],
        "lidar_source": lid["source"],
    }


# ------------------------------------------------------------------ statistics


def mad_sigma(x: np.ndarray) -> float:
    """Robust scale: 1.4826 * median(|x - median x|), consistent with a Gaussian sd."""
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def lag1(x: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    a, b = x[:-1] - x[:-1].mean(), x[1:] - x[1:].mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def describe(x: np.ndarray) -> dict:
    """Mean, spread and significance of a per-frame increment series.

    `t_naive` treats the increments as independent. They are not -- the lag-1 autocorrelation is
    reported alongside -- so `t_eff` rescales the sample size by the standard AR(1) effective-sample
    factor `n (1 - rho) / (1 + rho)`, and it is `t_eff` the record quotes.
    """
    n = int(len(x))
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    rho = lag1(x)
    n_eff = n * (1.0 - rho) / (1.0 + rho) if n > 1 and -1 < rho < 1 else float(n)
    return {
        "n": n,
        "mean": mean,
        "median": float(np.median(x)),
        "sd": sd,
        "mad_sigma": mad_sigma(x),
        "lag1_autocorr": rho,
        "n_eff": float(n_eff),
        "t_naive": mean / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan"),
        "t_eff": mean / (sd / math.sqrt(n_eff)) if n_eff > 1 and sd > 0 else float("nan"),
        "sum": float(np.sum(x)),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
    }
