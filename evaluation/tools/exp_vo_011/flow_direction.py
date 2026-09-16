"""EXP-VO-011: recover the per-frame travel direction **in the image frame**, from committed
columns only, and test the odd-symmetry prediction against the right variable.

**Why this module exists, and what it corrects.** The pre-registration's discriminator (H5c) says
that lens distortion and terrain relief both produce an apparent scale proportional to
`(centroid . travel)` and `(travel . grad z)` -- odd in image position, and therefore sign-flipping
with the **direction of travel**. `covariates.py` tested that against **compass heading**, which is
the wrong variable: the camera is rigidly mounted to the airframe, so what an odd mechanism responds
to is the travel direction expressed in the IMAGE frame, and the compass tells you that only if the
aircraft's attitude relative to its track is known and constant. A mechanism odd in image position
can therefore contribute a compass-INVARIANT constant while being entirely direction-dependent,
which is what the compass test found and could not interpret.

The time-reversal arm (`reversal_pair`) measures the odd component directly instead, with no model
and no ground truth: replay the identical frames backwards and every quantity odd in the travel
direction reverses, while anything carrying an arrow of time does not. This module supplies the
image-frame travel direction that says whether that odd component cancels over a flight or
accumulates.

The image-frame displacement is recoverable exactly from what the run record already stores.
`RigidNavigationState.observe` computes `q = D_k^-1(c) - c` in the image frame and integrates

    T_k = T_{k-1} + R(theta_{k-1}) q ,   theta_k = theta_{k-1} + polarRotation

so inverting the rotation recovers `q` from the stored pose series:

    q = R(-theta_{k-1}) (T_k - T_{k-1})

with `theta_{k-1}` the previous row's `rigid_yaw_deg`. `hypot(q)` must equal the stored
`inc_flow_px`, which is the module's own validation gate.
"""
from __future__ import annotations

import math

import numpy as np

import scale_series as ss


def image_flow(arm: dict) -> dict:
    """-> {'qx', 'qy', 'angle_deg', 'flow_px', 'max_abs_error_px'} in the image frame.

    `angle_deg` is measured in the image convention (x right, y down), in [0, 360).
    """
    # `rigid_y` is the POSE's y, which is `-centreY` -- `RigidNavigationState.pose()` negates it for
    # Pose3D's image-down to pose-forward convention (`COMP-001` section 5). Undo that before
    # inverting the rotation, or the recovered vector is a reflection of the true one: its length
    # still matches `inc_flow_px` exactly, so the magnitude gate below would pass while every angle
    # was wrong. That is precisely the trap this comment exists to mark.
    x, y = arm["rigid_x"], -arm["rigid_y"]
    yaw = np.radians(arm["rigid_yaw_deg"])

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    prev = np.concatenate([[yaw[0]], yaw[:-1]])
    c, s = np.cos(prev), np.sin(prev)
    qx = c * dx + s * dy          # R(-theta) applied to the world-frame increment
    qy = -s * dx + c * dy

    flow = np.hypot(qx, qy)
    err = float(np.max(np.abs(flow[1:] - arm["inc_flow_px"][1:]))) if len(flow) > 1 else 0.0
    return {
        "qx": qx, "qy": qy,
        "angle_deg": (np.degrees(np.arctan2(qy, qx)) + 360.0) % 360.0,
        "flow_px": flow,
        "max_abs_error_px": err,
    }


def circular_stats(angle_deg: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Circular mean and concentration. `R` near 1 means the direction is essentially constant."""
    a = np.radians(angle_deg)
    w = np.ones_like(a) if weights is None else weights
    cx = float(np.sum(w * np.cos(a)) / np.sum(w))
    cy = float(np.sum(w * np.sin(a)) / np.sum(w))
    R = math.hypot(cx, cy)
    return {
        "mean_deg": (math.degrees(math.atan2(cy, cx)) + 360.0) % 360.0,
        "resultant_R": R,
        # Circular sd in degrees, the standard sqrt(-2 ln R) form.
        "circular_sd_deg": math.degrees(math.sqrt(max(0.0, -2.0 * math.log(max(R, 1e-12))))),
    }


def travel_direction_summary(arm: dict) -> dict:
    """How concentrated is the direction of travel in the IMAGE frame, over this run?

    This is the quantity that decides whether an odd-in-image-position mechanism cancels over a
    flight or accumulates. A resultant `R` near 0 means the aircraft presents every travel direction
    to the camera equally, and an odd mechanism averages away; `R` near 1 means it does not.

    No regression is fitted on the travel components here. They are strongly collinear whenever the
    direction is concentrated -- which is exactly the regime of interest -- so a regression of the
    increment on `(qx, qy)` is ill-conditioned and its coefficients are not interpretable. The
    time-reversal arm measures the same thing directly and without that problem.
    """
    fl = image_flow(arm)
    return {
        "concentration": circular_stats(fl["angle_deg"]),
        "concentration_flow_weighted": circular_stats(fl["angle_deg"], weights=fl["flow_px"]),
        "flow_px": {"median": float(np.median(fl["flow_px"])),
                    "p95": float(np.percentile(fl["flow_px"], 95))},
        "recovery_max_error_px": fl["max_abs_error_px"],
    }


def reversal_pair(window: str, model: str) -> dict:
    """Forward against the time-reversed replay of the same frames (Phase 6b).

    Under reversal the aircraft retraces the identical path the other way, so every quantity that is
    odd in the direction of travel reverses and everything with an arrow of time does not. The
    comparison is therefore a sign test with no ground truth in it at all.
    """
    fwd = ss.load_arm(window, model)
    rev = ss.load_arm(f"{window}-reversed", model)
    f_y, r_y = fwd["inc_log_scale"], rev["inc_log_scale"]

    fs, rs = ss.describe(f_y), ss.describe(r_y)
    # If the mechanism were purely odd in travel direction, mean(rev) = -mean(fwd). If it were
    # purely an arrow-of-time property of the estimator, mean(rev) = +mean(fwd). The measured value
    # sits somewhere between, and where it sits is the attribution.
    odd_fraction = (fs["mean"] - rs["mean"]) / (2 * fs["mean"]) if fs["mean"] else float("nan")
    return {
        "window": window, "model": model,
        "forward": fs, "reversed": rs,
        "forward_travel": travel_direction_summary(fwd),
        "reversed_travel": travel_direction_summary(rev),
        "sign_flipped": bool(fs["mean"] * rs["mean"] < 0),
        "odd_fraction": odd_fraction,
        "even_fraction": 1.0 - odd_fraction if odd_fraction == odd_fraction else float("nan"),
        "even_component": (fs["mean"] + rs["mean"]) / 2.0,
        "odd_component": (fs["mean"] - rs["mean"]) / 2.0,
    }
